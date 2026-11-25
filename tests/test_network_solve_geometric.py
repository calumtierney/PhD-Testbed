"""Faces, instrument identity, the consider covariance and estimate mode of
`network.solve` (docs/step1b_npl_geometric_model.md)."""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import BACK, FRONT, INDEX, N_PARAMETERS, PARAMETER_NAMES, NPLErrorParameters
from network.design import design_covariances_for_poses
from network.solve import NetworkRankError, ParameterPrior, simulate_observations, solve_network

HUGHES = NPLErrorParameters.hughes_2011_uncompensated()
RESIDUAL = NPLErrorParameters.hughes_2011_residual()
NOISE_FREE = LaserTracker(sigma_d_m=1e-30, sigma_theta_rad=1e-30, sigma_phi_rad=1e-30)  # exact readings, same weights below
TRACKER = LaserTracker()


def close(actual, desired, tolerance=1e-4):
    """Largest difference relative to the largest entry, the convention of
    tests/test_design_covariance.py: the dense solve's Jacobian is a finite
    difference, so tiny off-diagonal entries agree to ~1e-5 of the largest."""
    assert np.abs(np.asarray(actual) - np.asarray(desired)).max() <= tolerance * np.abs(desired).max()


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rich_scene(n_targets=18, seed=1):
    """Targets spread over a room and five stations at different positions and
    heights (two on one stand, low and high): enough variety of range and
    elevation for the parameters to be separable."""
    rng = np.random.default_rng(seed)
    targets = rng.uniform([-3.0, -2.0, 0.0], [3.0, 2.0, 2.5], (n_targets, 3))
    poses = [InstrumentPose(np.array([4.0, 0.5, 1.4]), yaw(3.2)), InstrumentPose(np.array([-3.5, 2.5, 1.1]), yaw(-0.4)),
             InstrumentPose(np.array([0.5, -4.0, 1.7]), yaw(1.5)), InstrumentPose(np.array([0.3, 0.2, 0.9]), yaw(0.2)),
             InstrumentPose(np.array([0.3, 0.2, 2.4]), yaw(2.0))]
    return targets, poses


def exact_readings(targets, poses, params, faces):
    return simulate_observations(targets, poses, NOISE_FREE, np.random.default_rng(0), instrument_params=[params], faces=faces)


def test_default_noise_only_solve_is_untouched_by_the_new_arguments():
    targets, poses = rich_scene()
    observations = simulate_observations(targets, poses, TRACKER, np.random.default_rng(3))
    plain = solve_network(observations, targets, poses, TRACKER)
    same = solve_network(observations, targets, poses, LaserTracker(geometric_params=NPLErrorParameters.zero()))
    np.testing.assert_array_equal(plain.target_covariance_m2, same.target_covariance_m2)
    assert plain.parameter_estimates is None and plain.full_covariance_m2 is None


def test_back_face_readings_are_solved_and_two_faces_halve_the_covariance():
    targets, poses = rich_scene()
    both = [(FRONT, BACK)] * 5
    observations = simulate_observations(targets, poses, NOISE_FREE, np.random.default_rng(0), faces=both)
    assert {o.face for o in observations} == {FRONT, BACK}
    fit = solve_network(observations, targets, poses, TRACKER)
    assert np.abs(fit.target_points_m - targets).max() < 1e-9
    front_only = solve_network(simulate_observations(targets, poses, NOISE_FREE, np.random.default_rng(0), faces=[(FRONT,)] * 5),
                               targets, poses, TRACKER)
    close(fit.target_covariance_m2, front_only.target_covariance_m2 / 2.0)
    design = design_covariances_for_poses(targets, poses, TRACKER, np.ones((5, len(targets)), bool), faces=both)
    for t in range(len(targets)):
        close(fit.target_point_covariance(t), design.point_covariance_m2[t])


def test_consider_covariance_from_the_full_solve_matches_the_fast_design_path():
    """The prompt's agreement test: with geometric-parameter uncertainty the
    dense solve and the low-rank analytic update give the same target
    covariances, noise and parameters together."""
    targets, poses = rich_scene()
    tracker = LaserTracker(geometric_params=RESIDUAL)
    for faces in ([(FRONT,)] * 5, [(FRONT, BACK)] * 5):
        observations = simulate_observations(targets, poses, NOISE_FREE, np.random.default_rng(0), faces=faces)
        fit = solve_network(observations, targets, poses, tracker)
        design = design_covariances_for_poses(targets, poses, tracker, np.ones((5, len(targets)), bool), faces=faces)
        for t in range(len(targets)):
            close(fit.target_point_covariance(t), design.point_covariance_m2[t])
    noise_only = design_covariances_for_poses(targets, poses, TRACKER, np.ones((5, len(targets)), bool), faces=faces).point_covariance_m2
    assert np.all(np.einsum("tii->t", design.point_covariance_m2) > np.einsum("tii->t", noise_only))  # parameter uncertainty only adds


def test_consider_covariance_follows_instrument_identity():
    targets, poses = rich_scene()
    tracker = LaserTracker(geometric_params=RESIDUAL)
    ids = [0, 0, 1, 1, 1]
    observations = simulate_observations(targets, poses, NOISE_FREE, np.random.default_rng(0),
                                         instrument_params=[RESIDUAL, RESIDUAL], instrument_ids=ids)
    fit = solve_network(observations, targets, poses, tracker, instrument_ids=ids)
    design = design_covariances_for_poses(targets, poses, tracker, np.ones((5, len(targets)), bool), instrument_ids=ids)
    for t in range(len(targets)):
        close(fit.target_point_covariance(t), design.point_covariance_m2[t])


# -- estimate mode ----------------------------------------------------------------------------------------
def test_estimate_mode_recovers_every_parameter_from_exact_readings():
    targets, poses = rich_scene()
    observations = exact_readings(targets, poses, HUGHES, [(FRONT, BACK)] * 5)
    fit = solve_network(observations, targets, poses, TRACKER, estimate_parameters=ParameterPrior.flat_except({"mu": 5e-7}))
    assert fit.success and fit.n_parameters_estimated == N_PARAMETERS
    estimate = fit.parameter_estimates[0]
    sigma = estimate.standard_uncertainties()
    assert np.all(sigma > 0)
    assert np.all(np.abs(estimate.values - HUGHES.values) < 1e-4 * sigma)
    assert np.abs(fit.target_points_m - targets).max() < 1e-9


def test_estimate_mode_posterior_is_never_worse_than_leaving_the_parameters_unestimated():
    """Estimating the parameters under their prior uses the data's information
    about them: the target covariance can only shrink against the consider
    covariance (same prior), in the matrix sense."""
    targets, poses = rich_scene()
    tracker = LaserTracker(geometric_params=RESIDUAL)
    observations = exact_readings(targets, poses, RESIDUAL, [(FRONT, BACK)] * 5)
    consider = solve_network(observations, targets, poses, tracker).target_covariance_m2
    estimate = solve_network(observations, targets, poses, tracker, estimate_parameters=True).target_covariance_m2
    gap = consider - estimate
    assert np.linalg.eigvalsh(0.5 * (gap + gap.T)).min() > -1e-6 * np.abs(consider).max()
    assert np.trace(estimate) < np.trace(consider)


def test_estimate_mode_with_a_vanishing_prior_reduces_to_the_noise_only_covariance():
    targets, poses = rich_scene()
    observations = exact_readings(targets, poses, NPLErrorParameters.zero(), [(FRONT,)] * 5)
    prior = ParameterPrior(np.zeros(N_PARAMETERS), np.eye(N_PARAMETERS) * 1e-30, np.zeros(N_PARAMETERS, bool), np.zeros(N_PARAMETERS, bool))
    pinned = solve_network(observations, targets, poses, TRACKER, estimate_parameters=prior).target_covariance_m2
    noise_only = solve_network(observations, targets, poses, TRACKER).target_covariance_m2
    close(pinned, noise_only)
    fixed = ParameterPrior.flat_except(fixed=PARAMETER_NAMES)
    close(solve_network(observations, targets, poses, TRACKER, estimate_parameters=fixed).target_covariance_m2, noise_only)


def test_a_network_that_cannot_determine_the_scale_reports_it_instead_of_returning_zero_variance():
    """With no length reference and no prior, the range scale mu cannot be
    separated from the size of the network. `pinv` would give the undetermined
    direction zero variance; the estimate mode raises, naming it."""
    targets, poses = rich_scene()
    observations = exact_readings(targets, poses, HUGHES, [(FRONT, BACK)] * 5)
    with pytest.raises(NetworkRankError) as caught:
        solve_network(observations, targets, poses, TRACKER, estimate_parameters=ParameterPrior.flat_except())
    assert any("mu" in label for label in caught.value.weak_directions)
    assert "does not determine every unknown" in str(caught.value)
    # a prior on mu (a calibrated wavelength) is what makes it determined
    solve_network(observations, targets, poses, TRACKER, estimate_parameters=ParameterPrior.flat_except({"mu": 5e-7}))


def test_estimate_mode_rejects_bad_arguments():
    targets, poses = rich_scene()
    observations = exact_readings(targets, poses, HUGHES, [(FRONT,)] * 5)
    with pytest.raises(ValueError, match="geometric_params as the prior"):
        solve_network(observations, targets, poses, TRACKER, estimate_parameters=True)
    with pytest.raises(ValueError, match="one entry per station"):
        solve_network(observations, targets, poses, LaserTracker(geometric_params=RESIDUAL), instrument_ids=[0, 1])
    with pytest.raises(ValueError, match="one entry per instrument"):
        simulate_observations(targets, poses, TRACKER, np.random.default_rng(0), instrument_params=[HUGHES, HUGHES])
