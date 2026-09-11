"""Tests for characteristics.characteristic.

test_position_profile_parallelism_differ_on_one_point_one_station is the
gate for build step 4 (CLAUDE.md §5): two (here, three) different
tolerance types on one feature, measured from one station, produce
different characteristic uncertainties.
"""
import numpy as np
import pytest

from characteristics.characteristic import Characteristic, evaluate_characteristic
from characteristics.datum import Datum
from characteristics.tolerances import ProfileTolerance, ParallelismTolerance, PositionTolerance
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker, scene_point_covariances
from network.solve import perturbed_initial_guess, simulate_observations, solve_network


def test_position_profile_parallelism_differ_on_one_point_one_station():
    """One target, one laser tracker station, boresight geometry (theta =
    phi = 0) -- so the covariance's three principal axes line up exactly
    with the global x (radial/along-beam), y (azimuth-lateral) and z
    (elevation-lateral) axes (docs/step1_laser_tracker_physics.md). Three
    tolerance types are evaluated against that one covariance:

      - position (spherical, no axis): sees all three axes at once.
      - profile, surface normal = y: sees only the azimuth-lateral axis.
      - parallelism, datum normal = x (the beam direction): sees only the
        tiny radial axis.

    These must come out different, and in a specific, physically
    predictable order: parallelism (along the beam) < profile (one
    lateral axis) < position (all three combined).
    """
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([2.5, 0.0, 0.0])
    scene = Scene(target_points_m=target, instrument_pose=pose)
    covariances = scene_point_covariances(scene, tracker)

    position = Characteristic(
        name="position", target_indices=[0], tolerance=PositionTolerance(zone_diameter_m=1e-4)
    )
    profile = Characteristic(
        name="profile",
        target_indices=[0],
        tolerance=ProfileTolerance(zone_width_m=5e-5, surface_normal=np.array([0.0, 1.0, 0.0])),
    )
    datum_along_beam = Datum(name="A", normal=np.array([1.0, 0.0, 0.0]))
    parallelism = Characteristic(
        name="parallelism",
        target_indices=[0],
        tolerance=ParallelismTolerance(zone_width_m=5e-5),
        datum=datum_along_beam,
    )

    position_result = evaluate_characteristic(position, covariances)[0]
    profile_result = evaluate_characteristic(profile, covariances)[0]
    parallelism_result = evaluate_characteristic(parallelism, covariances)[0]

    # Match the exact per-axis standard deviations step 1 predicts
    # (CLAUDE.md §4b): radial ~1.216 um, azimuth-lateral ~5.878 um,
    # elevation-lateral ~8.413 um, RSS of all three ~10.33 um.
    assert parallelism_result.uncertainty_m * 1e6 == pytest.approx(1.216, rel=1e-3)
    assert profile_result.uncertainty_m * 1e6 == pytest.approx(5.878, rel=1e-3)
    assert position_result.uncertainty_m * 1e6 == pytest.approx(10.334, rel=1e-3)

    # The gate itself: they differ, substantially, and in the predicted order.
    assert parallelism_result.uncertainty_m < profile_result.uncertainty_m < position_result.uncertainty_m
    assert position_result.uncertainty_m > 5 * parallelism_result.uncertainty_m


def test_position_tolerance_trace_is_rotation_invariant():
    """A spherical position tolerance's projected variance should equal
    the covariance's full trace regardless of which orthonormal basis
    happens to be used to represent 'all three directions' -- projecting
    onto any complete orthonormal basis keeps the total variance, it just
    redistributes it across the k x k projected matrix."""
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([1.3, 2.1, 0.6])  # off-boresight, non-diagonal covariance
    scene = Scene(target_points_m=target, instrument_pose=pose)
    covariance = scene_point_covariances(scene, tracker)[0]

    characteristic = Characteristic(
        name="position", target_indices=[0], tolerance=PositionTolerance(zone_diameter_m=1e-4)
    )
    result = evaluate_characteristic(characteristic, covariance[None, :, :])[0]
    assert result.uncertainty_m == pytest.approx(np.sqrt(np.trace(covariance)))


def test_cylindrical_position_tolerance_excludes_its_own_axis():
    """A position tolerance with an axis along the beam should be *less*
    uncertain than the full spherical version, since it drops the radial
    axis's (small, here) contribution -- but still bigger than either
    single lateral axis alone, since it keeps both."""
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([2.5, 0.0, 0.0])
    scene = Scene(target_points_m=target, instrument_pose=pose)
    covariance = scene_point_covariances(scene, tracker)[0]

    spherical = PositionTolerance(zone_diameter_m=1e-4)
    cylindrical = PositionTolerance(zone_diameter_m=1e-4, axis_direction=np.array([1.0, 0.0, 0.0]))

    spherical_result = evaluate_characteristic(
        Characteristic("p", [0], spherical), covariance[None, :, :]
    )[0]
    cylindrical_result = evaluate_characteristic(
        Characteristic("p", [0], cylindrical), covariance[None, :, :]
    )[0]

    assert cylindrical_result.uncertainty_m < spherical_result.uncertainty_m
    # Excluding only the tiny radial axis barely changes anything here.
    assert cylindrical_result.uncertainty_m == pytest.approx(spherical_result.uncertainty_m, rel=0.02)


def test_evaluate_characteristic_accepts_network_solve_result():
    """The same Characteristic/tolerance machinery should work directly
    off a step 3 NetworkSolveResult, not just a raw per-point covariance
    array -- exercising _point_covariance's duck-typed dispatch."""
    rng = np.random.default_rng(0)
    tracker = LaserTracker()
    true_targets_m = np.array([[2.5, 0.0, 0.0]])
    true_poses = [InstrumentPose(position_m=np.zeros(3))]
    observations = simulate_observations(true_targets_m, true_poses, tracker, rng)
    init_targets_m, init_poses = perturbed_initial_guess(true_targets_m, true_poses, 0.01, 0.001, rng)
    result = solve_network(observations, init_targets_m, init_poses, tracker)

    characteristic = Characteristic(
        name="position", target_indices=[0], tolerance=PositionTolerance(zone_diameter_m=1e-4)
    )
    evaluated = evaluate_characteristic(characteristic, result)[0]
    assert evaluated.uncertainty_m > 0
    assert evaluated.projected_covariance_m2.shape == (3, 3)


def test_multi_point_characteristic_returns_one_result_per_point():
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    points = np.array([[2.5, 0.0, 0.0], [0.0, 3.0, 0.0]])
    scene = Scene(target_points_m=points, instrument_pose=pose)
    covariances = scene_point_covariances(scene, tracker)

    characteristic = Characteristic(
        name="profile",
        target_indices=[0, 1],
        tolerance=ProfileTolerance(zone_width_m=5e-5, surface_normal=np.array([0.0, 0.0, 1.0])),
    )
    results = evaluate_characteristic(characteristic, covariances)
    assert [r.target_index for r in results] == [0, 1]
    assert results[0].uncertainty_m != results[1].uncertainty_m  # different geometry per point


def test_parallelism_uncertainty_grows_with_datum_establishment_uncertainty():
    """CLAUDE.md referee note M5: a parallelism callout's reported
    uncertainty must reflect the datum's own establishment uncertainty,
    not just the measured point's -- an exact datum (0.0, the default)
    must not silently understate the true characteristic uncertainty."""
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([2.5, 0.0, 0.0])
    scene = Scene(target_points_m=target, instrument_pose=pose)
    covariances = scene_point_covariances(scene, tracker)

    exact_datum = Datum(name="A", normal=np.array([1.0, 0.0, 0.0]))
    uncertain_datum = Datum(name="A", normal=np.array([1.0, 0.0, 0.0]), establishment_uncertainty_m=5e-6)

    exact_result = evaluate_characteristic(
        Characteristic("parallelism", [0], ParallelismTolerance(zone_width_m=5e-5), datum=exact_datum),
        covariances,
    )[0]
    uncertain_result = evaluate_characteristic(
        Characteristic("parallelism", [0], ParallelismTolerance(zone_width_m=5e-5), datum=uncertain_datum),
        covariances,
    )[0]

    assert uncertain_result.uncertainty_m > exact_result.uncertainty_m
    # Exact combination: point variance (along x, the datum normal here)
    # plus the datum's own variance, added in quadrature.
    point_variance_m2 = covariances[0][0, 0]
    expected_m2 = point_variance_m2 + 5e-6**2
    assert uncertain_result.projected_covariance_m2[0, 0] == pytest.approx(expected_m2)
