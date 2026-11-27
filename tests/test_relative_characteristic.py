"""Tests for characteristics.characteristic.RelativeCharacteristic: a deviation judged
against a datum frame fitted to measured datum targets, rather than in the anchor
station's frame.

The physical claims tested:
  * it reproduces what the cell planners already compute for a profile tolerance
    (`DatumCellScorer`, `PartFrameScorer`) -- the same S-transformation, reached through
    the characteristic layer;
  * errors shared by a point and its datum targets cancel in it, and by a lot when the
    tracker's geometric-parameter errors are modelled (nearby points see nearly the same
    parameter error), where the anchor-frame uncertainty does not;
  * a network solve and the fast design covariance give the same answer;
  * it fails loudly rather than returning a number when the datum frame was never
    realised.
"""
from dataclasses import replace

import numpy as np
import pytest

from characteristics.characteristic import (
    Characteristic,
    RelativeCharacteristic,
    evaluate_characteristic,
    evaluate_relative_characteristic,
)
from characteristics.tolerances import ParallelismTolerance, PositionTolerance, ProfileTolerance
from characteristics.datum import Datum
from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import NPLErrorParameters
from network.design import design_joint_covariance
from network.solve import exact_observations, perturbed_initial_guess, solve_network
from planning.cell import (
    DatumCellScorer, PartFrameScorer, factory_cell_scenario, layout_poses, part_datum_indices,
)
from planning.experiment import long_part_track

TRACK = long_part_track()
LOOPS_4 = [(15.9, 1.28), (27.5, 1.45), (45.0, 1.17), (54.4, 1.18)]


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def bare():
    return factory_cell_scenario(track=TRACK, with_reference_network=False)


# -- agreement with the planning scorers ---------------------------------------------------------
def cell_joint(scenario, poses):
    """What the scorers do: design joint covariance over the measured characteristic
    targets followed by the observed datum targets."""
    visible = DatumCellScorer(scenario).visible(poses)  # every station planned; no separate anchor row
    seen = visible.any(axis=0)
    char_targets = [c.target_indices[0] for c in scenario.characteristics]
    datum = [i for i in part_datum_indices(scenario) if seen[i]]
    subset = char_targets + datum
    joint, _, connected = design_joint_covariance(scenario.target_points_m, list(poses), scenario.tracker, visible, subset, 0)
    assert connected and seen[char_targets].all()
    return joint, subset, datum


@pytest.mark.parametrize("floor_m", [0.0, 10e-6])
def test_profile_characteristic_reproduces_the_datum_cell_and_part_frame_scorers(bare, floor_m):
    """With a placement floor the scorers add it, in quadrature, on top of the
    datum-referenced variance; the characteristic layer does not (the floor is a property
    of the tracker, applied where the planner chooses to), so the test adds it back."""
    bare = replace(bare, tracker=LaserTracker(systematic_std_m=floor_m))  # noise-only, so the floor is not counted twice
    poses = layout_poses(LOOPS_4, bare, TRACK)
    joint, subset, datum = cell_joint(bare, poses)
    old = DatumCellScorer(bare).score(poses)
    new = PartFrameScorer(bare, tuple(part_datum_indices(bare))).score(poses)
    assert old.feasible and new.feasible

    sys2 = floor_m ** 2
    for k, c in enumerate(bare.characteristics):
        rel = RelativeCharacteristic(c.name, c.target_indices[0], tuple(datum), c.tolerance)
        result = evaluate_relative_characteristic(rel, joint, bare.target_points_m, subset)
        u = np.sqrt(result.worst_direction_std_m ** 2 + sys2)
        assert u == pytest.approx(old.u_m[k], rel=1e-9)
        assert u == pytest.approx(new.u_m[k], rel=1e-9)
        assert result.uncertainty_m == pytest.approx(result.worst_direction_std_m, rel=1e-12)  # one direction
        assert result.projected_covariance_m2.shape == (1, 1)
        assert result.datum_referenced_covariance_m2.shape == (3, 3)


def test_datum_referenced_uncertainty_is_below_the_anchor_frame_uncertainty_for_a_distant_feature(bare):
    """The reason the layer exists: in the anchor's frame a characteristic far from the
    anchor inherits the anchor's long-range angular error; relative to the part's own datum
    targets it does not."""
    poses = layout_poses(LOOPS_4, bare, TRACK)
    joint, subset, datum = cell_joint(bare, poses)
    from planning.objectives import evaluate_plan

    reference = evaluate_plan(poses, bare)
    n_smaller = 0
    for k, c in enumerate(bare.characteristics):
        anchor_frame = evaluate_characteristic(c, reference.network_solve)[0].uncertainty_m
        rel = RelativeCharacteristic(c.name, c.target_indices[0], tuple(datum), c.tolerance)
        relative = evaluate_relative_characteristic(rel, joint, bare.target_points_m, subset).uncertainty_m
        n_smaller += relative < anchor_frame
    assert n_smaller >= 1


def test_the_part_frame_may_include_the_toleranced_point_itself(bare):
    poses = layout_poses(LOOPS_4, bare, TRACK)
    joint, subset, datum = cell_joint(bare, poses)
    c = bare.characteristics[0]
    t = c.target_indices[0]
    separate = evaluate_relative_characteristic(
        RelativeCharacteristic(c.name, t, tuple(datum), c.tolerance), joint, bare.target_points_m, subset)
    included = evaluate_relative_characteristic(
        RelativeCharacteristic(c.name, t, tuple(datum) + (t,), c.tolerance), joint, bare.target_points_m, subset)
    assert np.isfinite(included.uncertainty_m)
    # fitting the frame through the point itself pulls the frame towards the point's own error
    assert included.uncertainty_m < separate.uncertainty_m


# -- common-mode cancellation with the NPL geometric model ---------------------------------------------
def small_cluster_scene():
    """Three trackers around a 0.3 m cluster: one toleranced point and four datum
    targets, all within about 0.3 m of each other, 3-5 m from every station."""
    rng = np.random.default_rng(11)
    centre = np.array([0.4, 0.2, 1.0])
    datum = centre + rng.uniform(-0.15, 0.15, (4, 3))
    point = centre + np.array([0.05, -0.1, 0.12])
    targets = np.vstack([point, datum])
    poses = [
        InstrumentPose(np.array([4.0, 0.5, 1.4]), yaw(3.2)),
        InstrumentPose(np.array([-3.5, 2.5, 1.1]), yaw(-0.4)),
        InstrumentPose(np.array([0.5, -4.0, 1.7]), yaw(1.5)),
    ]
    return targets, poses, np.ones((3, len(targets)), bool)


def test_shared_tracker_parameter_errors_cancel_relative_to_the_datum_but_not_in_the_anchor_frame():
    targets, poses, visible = small_cluster_scene()
    quiet = LaserTracker(systematic_std_m=0.0)
    consider = LaserTracker(systematic_std_m=0.0, geometric_params=NPLErrorParameters.hughes_2011_residual())
    subset = list(range(len(targets)))
    normal = np.array([0.0, 0.0, 1.0])
    characteristic = RelativeCharacteristic("cluster point", 0, (1, 2, 3, 4), ProfileTolerance(50e-6, normal))

    def anchor_and_relative(tracker):
        joint, _, connected = design_joint_covariance(targets, poses, tracker, visible, subset, 0)
        assert connected
        anchor = float(normal @ joint[:3, :3] @ normal)
        relative = evaluate_relative_characteristic(characteristic, joint, targets, subset).projected_covariance_m2[0, 0]
        return anchor, relative

    anchor_quiet, relative_quiet = anchor_and_relative(quiet)
    anchor_consider, relative_consider = anchor_and_relative(consider)
    anchor_parameter_part = anchor_consider - anchor_quiet
    relative_parameter_part = relative_consider - relative_quiet
    assert anchor_parameter_part > 0 and relative_parameter_part > 0
    # most of the parameter error is common to all five points and moves the fitted datum frame with it
    assert relative_parameter_part < 0.1 * anchor_parameter_part
    # the sensor noise does not cancel: relative uncertainty is not simply smaller everywhere
    assert relative_quiet > 0.05 * anchor_quiet


def test_without_cross_covariances_the_cancellation_cannot_be_seen():
    """A block-diagonal covariance (per-point blocks only, what `evaluate_characteristic`
    works from) overstates the parameter-induced relative variance: the cancellation lives
    in the off-diagonal blocks."""
    targets, poses, visible = small_cluster_scene()
    consider = LaserTracker(systematic_std_m=0.0, geometric_params=NPLErrorParameters.hughes_2011_residual())
    subset = list(range(len(targets)))
    joint, _, _ = design_joint_covariance(targets, poses, consider, visible, subset, 0)
    blockdiag = np.zeros_like(joint)
    for i in range(len(subset)):
        blockdiag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = joint[3 * i:3 * i + 3, 3 * i:3 * i + 3]
    c = RelativeCharacteristic("p", 0, (1, 2, 3, 4), ProfileTolerance(50e-6, np.array([0.0, 0.0, 1.0])))
    full = evaluate_relative_characteristic(c, joint, targets, subset).uncertainty_m
    per_point = evaluate_relative_characteristic(c, blockdiag, targets, subset).uncertainty_m
    assert per_point > 1.5 * full


# -- the network solve path ---------------------------------------------------------------------------------
def test_network_solve_and_design_covariance_give_the_same_relative_uncertainty():
    targets, poses, visible = small_cluster_scene()
    tracker = LaserTracker(systematic_std_m=0.0)
    rng = np.random.default_rng(5)
    observations = exact_observations(targets, poses)
    initial_targets, initial_poses = perturbed_initial_guess(targets, poses, 0.002, 0.0002, rng)
    solved = solve_network(observations, initial_targets, initial_poses, tracker, anchor_index=0)
    joint, _, _ = design_joint_covariance(targets, poses, tracker, visible, list(range(len(targets))), 0)

    c = RelativeCharacteristic("p", 0, (1, 2, 3, 4), ProfileTolerance(50e-6, np.array([0.3, -0.5, 0.8])))
    from_solve = evaluate_relative_characteristic(c, solved, targets)
    from_design = evaluate_relative_characteristic(c, joint, targets)
    assert from_solve.uncertainty_m == pytest.approx(from_design.uncertainty_m, rel=1e-4)
    np.testing.assert_allclose(
        from_solve.datum_referenced_covariance_m2, from_design.datum_referenced_covariance_m2,
        rtol=1e-3, atol=1e-6 * np.abs(from_design.datum_referenced_covariance_m2).max(),
    )


def test_joint_point_covariance_is_the_requested_block_of_the_full_covariance():
    targets, poses, _ = small_cluster_scene()
    tracker = LaserTracker(systematic_std_m=0.0)
    observations = exact_observations(targets, poses)
    solved = solve_network(observations, targets, poses, tracker, anchor_index=0)
    block = solved.joint_point_covariance([3, 1])
    full = solved.target_covariance_m2
    assert block.shape == (6, 6)
    np.testing.assert_array_equal(block[:3, :3], full[9:12, 9:12])
    np.testing.assert_array_equal(block[:3, 3:], full[9:12, 3:6])
    np.testing.assert_array_equal(block[3:, :3], full[3:6, 9:12])


# -- tolerance types ------------------------------------------------------------------------------------
def test_two_tolerances_on_one_point_have_different_relative_uncertainties():
    """Step 4's gate, restated relative to a datum frame: the constrained direction still
    picks which part of the (datum-referenced) ellipsoid the tolerance sees."""
    targets, poses, visible = small_cluster_scene()
    tracker = LaserTracker(systematic_std_m=0.0)
    subset = list(range(len(targets)))
    joint, _, _ = design_joint_covariance(targets, poses, tracker, visible, subset, 0)
    sphere = RelativeCharacteristic("pos", 0, (1, 2, 3, 4), PositionTolerance(50e-6))
    profile = RelativeCharacteristic("prof", 0, (1, 2, 3, 4), ProfileTolerance(50e-6, np.array([0.0, 0.0, 1.0])))
    u_sphere = evaluate_relative_characteristic(sphere, joint, targets, subset)
    u_profile = evaluate_relative_characteristic(profile, joint, targets, subset)
    assert u_sphere.projected_covariance_m2.shape == (3, 3) and u_profile.projected_covariance_m2.shape == (1, 1)
    assert u_sphere.uncertainty_m > u_profile.uncertainty_m
    assert u_sphere.worst_direction_std_m <= u_sphere.uncertainty_m + 1e-18


def test_parallelism_datum_establishment_variance_is_added_as_in_evaluate_characteristic():
    targets, poses, visible = small_cluster_scene()
    subset = list(range(len(targets)))
    joint, _, _ = design_joint_covariance(targets, poses, LaserTracker(systematic_std_m=0.0), visible, subset, 0)
    datum = Datum("plane", np.array([0.0, 0.0, 1.0]), establishment_uncertainty_m=10e-6)
    c = RelativeCharacteristic("par", 0, (1, 2, 3, 4), ParallelismTolerance(50e-6), datum)
    plain = RelativeCharacteristic("par", 0, (1, 2, 3, 4), ProfileTolerance(50e-6, np.array([0.0, 0.0, 1.0])))
    with_datum = evaluate_relative_characteristic(c, joint, targets, subset)
    without = evaluate_relative_characteristic(plain, joint, targets, subset)
    assert with_datum.projected_covariance_m2[0, 0] == pytest.approx(without.projected_covariance_m2[0, 0] + (10e-6) ** 2)


# -- refusing to invent a datum frame -----------------------------------------------------------------------------
def test_too_few_or_collinear_datum_targets_raise():
    targets, poses, visible = small_cluster_scene()
    subset = list(range(len(targets)))
    joint, _, _ = design_joint_covariance(targets, poses, LaserTracker(systematic_std_m=0.0), visible, subset, 0)
    tol = ProfileTolerance(50e-6, np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="six degrees of freedom"):
        evaluate_relative_characteristic(RelativeCharacteristic("p", 0, (1, 2), tol), joint, targets, subset)
    line = targets.copy()
    line[1:4] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="six degrees of freedom"):
        evaluate_relative_characteristic(RelativeCharacteristic("p", 0, (1, 2, 3), tol), joint, line, subset)


def test_an_unobserved_target_has_no_relative_uncertainty():
    targets, poses, visible = small_cluster_scene()
    visible = visible.copy()
    visible[:, 2] = False  # a datum target nobody sees: NaN rows in the design covariance
    subset = list(range(len(targets)))
    joint, observed, _ = design_joint_covariance(targets, poses, LaserTracker(systematic_std_m=0.0), visible, subset, 0)
    assert not observed[2]
    tol = ProfileTolerance(50e-6, np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="not observed"):
        evaluate_relative_characteristic(RelativeCharacteristic("p", 0, (1, 2, 3, 4), tol), joint, targets, subset)


def test_covariance_shape_and_subset_membership_are_checked():
    targets, poses, visible = small_cluster_scene()
    subset = list(range(len(targets)))
    joint, _, _ = design_joint_covariance(targets, poses, LaserTracker(systematic_std_m=0.0), visible, subset, 0)
    tol = ProfileTolerance(50e-6, np.array([0.0, 0.0, 1.0]))
    c = RelativeCharacteristic("p", 0, (1, 2, 3, 4), tol)
    with pytest.raises(ValueError, match="shape"):
        evaluate_relative_characteristic(c, joint[:12, :12], targets, subset)
    with pytest.raises(ValueError, match="not in the covariance's subset"):
        evaluate_relative_characteristic(c, joint[:12, :12], targets, [0, 1, 2, 3])
