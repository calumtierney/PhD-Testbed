"""Tests for network.solve.

test_mean_uncertainty_falls_and_flattens_with_station_count is the gate
for build step 3 (CLAUDE.md §5): for a fixed target set, mean uncertainty
falls as stations are added and the improvement flattens after 4-5
stations.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker, scene_point_covariances
from network.solve import (
    Observation,
    exact_observations,
    mean_positional_uncertainty_m,
    perturbed_initial_guess,
    simulate_observations,
    solve_network,
    _build_jac_sparsity,
)


def _look_at_pose(position_m: np.ndarray, aim_point_m: np.ndarray) -> InstrumentPose:
    """A station pose whose local +x axis (this codebase's boresight
    direction, see geometry.spherical) points from `position_m` at
    `aim_point_m`. Just test-scaffolding for building a plausible-looking
    network, not something the solver itself needs."""
    x_axis = aim_point_m - position_m
    x_axis = x_axis / np.linalg.norm(x_axis)
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference_up, x_axis)) > 0.99:
        reference_up = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference_up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    orientation = np.column_stack([x_axis, y_axis, z_axis])
    return InstrumentPose(position_m=position_m, orientation=orientation)


# A fixed synthetic network scene reused across several tests: 12 targets
# scattered around (4, 0, 1), and up to 5 candidate station positions at
# varied angles around them (the angular spread is what actually drives
# the uncertainty reduction -- see docs/step3_network_solve_physics.md).
_TARGET_CENTRE_M = np.array([4.0, 0.0, 1.0])
_RNG_SEED = 0


def _make_targets():
    rng = np.random.default_rng(_RNG_SEED)
    return rng.uniform(low=[-1, -1, 0], high=[1, 1, 2], size=(12, 3)) + _TARGET_CENTRE_M


_CANDIDATE_STATION_POSITIONS_M = [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 4.0, 1.0]),
    np.array([0.0, -4.0, 1.0]),
    np.array([-3.0, 0.0, 3.0]),
    np.array([-3.0, 3.0, -1.0]),
]


def _run_solve(n_stations: int, rng: np.random.Generator):
    true_targets_m = _make_targets()
    true_poses = [
        _look_at_pose(p, _TARGET_CENTRE_M) for p in _CANDIDATE_STATION_POSITIONS_M[:n_stations]
    ]
    tracker = LaserTracker()
    observations = simulate_observations(true_targets_m, true_poses, tracker, rng)
    initial_targets_m, initial_poses = perturbed_initial_guess(
        true_targets_m, true_poses, position_noise_m=0.05, rotation_noise_rad=0.01, rng=rng
    )
    result = solve_network(observations, initial_targets_m, initial_poses, tracker, anchor_index=0)
    return true_targets_m, result


def test_mean_uncertainty_falls_and_flattens_with_station_count():
    rng = np.random.default_rng(_RNG_SEED)
    mean_uncertainties_um = []
    for n_stations in range(1, 6):
        _, result = _run_solve(n_stations, rng)
        assert result.success
        mean_uncertainties_um.append(mean_positional_uncertainty_m(result) * 1e6)

    # Falls monotonically as stations are added (CLAUDE.md §5 step 3 gate).
    deltas = -np.diff(mean_uncertainties_um)
    assert np.all(deltas > 0), (
        f"mean uncertainty should strictly decrease with station count, got {mean_uncertainties_um}"
    )

    # ... and flattens: the last improvement (4 -> 5 stations) is much
    # smaller than the first (1 -> 2 stations), not still falling at the
    # same rate.
    assert deltas[-1] < 0.5 * deltas[0], (
        f"improvement should be flattening by 5 stations, got deltas {deltas} um "
        f"(from mean uncertainties {mean_uncertainties_um} um)"
    )


@pytest.mark.parametrize("systematic_std_m", [0.0, 10e-6])
def test_single_station_network_solve_matches_step1_covariance(systematic_std_m):
    """With one station and no redundancy (3 unknowns per target, exactly
    3 observations), the network solve should reduce to step 1's direct
    per-point covariance -- the same physics, just reached by a nonlinear
    solve instead of a closed-form Jacobian propagation. This ties step 3
    back to step 1 rather than treating them as unrelated code paths.

    Parametrized over `systematic_std_m` (instruments.laser_tracker.
    LaserTracker's placement-irreducible isotropic term): step 1's
    `covariance_local`/`covariance_global` already adds this term, so a
    correct network solve must reduce to *that* -- including the added
    term -- in the one-station limit too, not just the sensor-noise-only
    case."""
    rng = np.random.default_rng(1)
    tracker = LaserTracker(systematic_std_m=systematic_std_m)
    true_targets_m = np.array([[2.5, 0.0, 0.0], [3.0, 1.0, 0.5]])
    anchor_pose = InstrumentPose(position_m=np.zeros(3))

    observations = simulate_observations(true_targets_m, [anchor_pose], tracker, rng)
    initial_targets_m, initial_poses = perturbed_initial_guess(
        true_targets_m, [anchor_pose], position_noise_m=0.01, rotation_noise_rad=0.001, rng=rng
    )
    result = solve_network(observations, initial_targets_m, initial_poses, tracker, anchor_index=0)

    scene = Scene(target_points_m=true_targets_m, instrument_pose=anchor_pose)
    step1_covariances = scene_point_covariances(scene, tracker)

    for i in range(len(true_targets_m)):
        assert np.allclose(
            result.target_point_covariance(i), step1_covariances[i], rtol=0.02, atol=1e-13
        )
    # Fully determined (3 unknowns, 3 observations per target): the
    # solve should hit the noisy observation exactly, zero residual.
    assert result.residual_rms == pytest.approx(0.0, abs=1e-6)


def test_systematic_std_survives_a_multi_station_network_solve():
    """Regression test for a real bug: `LaserTracker.systematic_std_m`
    was added only to the single-station `covariance_local`/
    `covariance_global` path; `solve_network` never applied it, so a
    `planning.PlanningScenario` built with a nonzero-systematic tracker
    silently changed nothing about a multi-station plan's covariance.

    Uses `exact_observations` (noiseless, at the true geometry) so the
    only difference between the two solves is the tracker's
    `systematic_std_m` -- the extra per-target variance must come out as
    exactly the isotropic term the tracker declares, on every target,
    with no cross-target leakage (the term is modelled as independent per
    point, not shared -- see `LaserTracker.systematic_std_m`'s
    docstring)."""
    target_points_m = np.array([[3.0, 0.0, 1.0], [3.0, 1.0, 0.5]])
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 0.0, 1.0])),
        InstrumentPose(position_m=np.array([0.0, 3.0, 1.0])),
        InstrumentPose(position_m=np.array([0.0, -3.0, 1.0])),
    ]
    geometric_only = LaserTracker()
    with_systematic = LaserTracker(systematic_std_m=10e-6)

    result_geometric = solve_network(
        exact_observations(target_points_m, station_poses),
        target_points_m.copy(),
        station_poses,
        geometric_only,
        anchor_index=0,
    )
    result_systematic = solve_network(
        exact_observations(target_points_m, station_poses),
        target_points_m.copy(),
        station_poses,
        with_systematic,
        anchor_index=0,
    )

    expected_extra_variance_m2 = with_systematic.systematic_std_m**2
    n_targets = target_points_m.shape[0]
    for i in range(n_targets):
        extra = result_systematic.target_point_covariance(i) - result_geometric.target_point_covariance(i)
        assert np.allclose(extra, expected_extra_variance_m2 * np.eye(3), atol=1e-20)

    # No cross-target leakage: the off-diagonal (target 0, target 1) block
    # should be identical whether or not the systematic term is present,
    # since it's added once per target, not shared between targets.
    cross_geometric = result_geometric.target_covariance_m2[0:3, 3:6]
    cross_systematic = result_systematic.target_covariance_m2[0:3, 3:6]
    assert np.allclose(cross_geometric, cross_systematic)


def test_systematic_std_does_not_shrink_with_more_stations():
    """`systematic_std_m` is *irreducible by placement* (the field's own
    docstring): adding a station reduces the sensor-noise-driven part of
    the covariance, but must never reduce this term below its own value,
    since no amount of redundancy corrects a per-point floor that isn't
    modelled as arising from range/angle noise in the first place."""
    target_points_m = np.array([[3.0, 0.0, 1.0]])
    tracker = LaserTracker(systematic_std_m=10e-6)
    two_stations = [
        InstrumentPose(position_m=np.array([0.0, 0.0, 1.0])),
        InstrumentPose(position_m=np.array([0.0, 3.0, 1.0])),
    ]
    three_stations = two_stations + [InstrumentPose(position_m=np.array([0.0, -3.0, 1.0]))]

    result_two = solve_network(
        exact_observations(target_points_m, two_stations),
        target_points_m.copy(),
        two_stations,
        tracker,
        anchor_index=0,
    )
    result_three = solve_network(
        exact_observations(target_points_m, three_stations),
        target_points_m.copy(),
        three_stations,
        tracker,
        anchor_index=0,
    )

    # Adding a station should not increase the covariance's trace...
    assert np.trace(result_three.target_point_covariance(0)) <= np.trace(
        result_two.target_point_covariance(0)
    ) + 1e-20
    # ...but it can never fall below the systematic floor alone (3 axes,
    # each carrying at least systematic_std_m**2), however many stations
    # are added.
    systematic_floor_m2 = 3 * tracker.systematic_std_m**2
    assert np.trace(result_three.target_point_covariance(0)) >= systematic_floor_m2 - 1e-20


def test_anchor_pose_is_held_fixed_not_solved_for():
    """The anchor's pose comes back exactly as given in the initial guess
    -- untouched by the optimiser -- because it was never in the
    parameter vector to begin with (see _build_parameter_vector)."""
    rng = np.random.default_rng(2)
    true_targets_m, result = _run_solve(n_stations=3, rng=rng)
    true_anchor_pose = _look_at_pose(_CANDIDATE_STATION_POSITIONS_M[0], _TARGET_CENTRE_M)
    # perturbed_initial_guess leaves the anchor (index 0) unperturbed, so
    # the solved anchor pose should match the true one exactly.
    assert np.array_equal(result.station_poses[0].position_m, true_anchor_pose.position_m)
    assert np.array_equal(result.station_poses[0].orientation, true_anchor_pose.orientation)


def test_solved_targets_are_close_to_the_truth():
    rng = np.random.default_rng(3)
    true_targets_m, result = _run_solve(n_stations=4, rng=rng)
    # Generous tolerance (sub-millimetre) -- this just checks the
    # optimiser actually converges to the right answer, not a precision
    # claim; precision is what target_covariance_m2 is for.
    assert np.allclose(result.target_points_m, true_targets_m, atol=2e-4)


def test_target_covariance_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(4)
    _, result = _run_solve(n_stations=3, rng=rng)
    cov = result.target_covariance_m2
    assert np.allclose(cov, cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > -1e-18)  # allow tiny negative numerical noise at 0


def test_jac_sparsity_reflects_one_target_one_station_per_observation():
    """Direct check on CLAUDE.md §3's block-diagonal structure claim: each
    observation's 3 residual rows touch exactly one target's 3 columns
    and (for a non-anchor station) one station's 6 columns -- never more."""
    observations = [
        Observation(station_index=0, target_index=0, d_m=1.0, theta_rad=0.0, phi_rad=0.0),
        Observation(station_index=1, target_index=0, d_m=1.0, theta_rad=0.0, phi_rad=0.0),
        Observation(station_index=1, target_index=2, d_m=1.0, theta_rad=0.0, phi_rad=0.0),
    ]
    n_targets, n_stations, anchor_index = 3, 2, 0
    sparsity = _build_jac_sparsity(observations, n_targets, n_stations, anchor_index)

    n_target_params = 3 * n_targets  # 9
    n_station_params = 6 * (n_stations - 1)  # 6 (only station 1 is free)
    assert sparsity.shape == (3 * len(observations), n_target_params + n_station_params)

    nonzero_per_row = np.asarray(sparsity.sum(axis=1)).ravel()
    # Observation 0: station 0 is the anchor (no free params) -> only 3 target columns.
    assert list(nonzero_per_row[0:3]) == [3, 3, 3]
    # Observations 1 and 2: station 1 is free -> 3 target columns + 6 station columns.
    assert list(nonzero_per_row[3:6]) == [9, 9, 9]
    assert list(nonzero_per_row[6:9]) == [9, 9, 9]
