"""Geometric-parameter uncertainty in the fast design covariance
(`network.design`, the 'consider term'), checked against independent
calculations: nonlinear network solves of data generated with a perturbed
tracker, and physical limits (what two faces cancel and what they cannot).
"""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import (
    ARCSEC_TO_RAD, BACK, FRONT, INDEX, N_PARAMETERS, PARAMETER_STEPS, NPLErrorParameters, true_to_observed_batch,
)
from network import design as design_module
from network.design import design_covariances_for_poses, design_joint_covariance, design_parameter_sensitivity
from network.solve import Observation, solve_network

HUGHES_RESIDUAL = NPLErrorParameters.hughes_2011_residual()


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def scene(n_targets=9, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.uniform([-2.0, -2.0, 0.0], [2.0, 2.0, 2.0], (n_targets, 3))
    poses = [InstrumentPose(np.array([4.0, 0.5, 1.4]), yaw(3.2)), InstrumentPose(np.array([-3.5, 2.5, 1.1]), yaw(-0.4)),
             InstrumentPose(np.array([0.5, -4.0, 1.7]), yaw(1.5))]
    return targets, poses, np.ones((3, n_targets), bool)


def tracker_with(params):
    return LaserTracker(geometric_params=params)


def only_parameter_uncertain(name, sigma):
    cov = np.zeros((N_PARAMETERS, N_PARAMETERS))
    cov[INDEX[name], INDEX[name]] = sigma**2
    return NPLErrorParameters(np.zeros(N_PARAMETERS), cov)


# -- the extended algebra reduces to the old one -----------------------------------------------------
def test_extended_algebra_reproduces_the_noise_only_functions_to_rounding():
    targets, poses, visible = scene()
    positions = np.array([p.position_m for p in poses])
    orientations = np.array([p.orientation for p in poses])
    tracker = LaserTracker()
    original = design_covariances_for_poses(targets, poses, tracker, visible).point_covariance_m2
    extended = design_module._extended_design(targets, positions, orientations, tracker, visible, 0, None, None, None)[0]
    np.testing.assert_allclose(extended, original, rtol=1e-12, atol=0.0)
    subset = [0, 3, 5]
    joint = design_joint_covariance(targets, poses, tracker, visible, subset)[0]
    extended_joint = design_module._extended_design(targets, positions, orientations, tracker, visible, 0, None, None, np.array(subset))[0]
    np.testing.assert_allclose(extended_joint, joint, rtol=1e-12, atol=0.0)


def test_without_geometric_parameters_or_with_zero_covariance_nothing_changes():
    targets, poses, visible = scene()
    base = design_covariances_for_poses(targets, poses, LaserTracker(), visible).point_covariance_m2
    for tracker in (tracker_with(NPLErrorParameters.zero()), tracker_with(NPLErrorParameters.hughes_2011_uncompensated().__class__(
            NPLErrorParameters.hughes_2011_uncompensated().values, np.zeros((16, 16))))):
        np.testing.assert_array_equal(design_covariances_for_poses(targets, poses, tracker, visible).point_covariance_m2, base)
    assert design_parameter_sensitivity(targets, poses, LaserTracker(), visible) is None


# -- faces ------------------------------------------------------------------------------------------
def test_a_second_face_doubles_the_information_and_halves_the_noise_only_covariance():
    targets, poses, visible = scene()
    front = design_covariances_for_poses(targets, poses, LaserTracker(), visible).point_covariance_m2
    both = design_covariances_for_poses(targets, poses, LaserTracker(), visible, faces=[(FRONT, BACK)] * 3).point_covariance_m2
    np.testing.assert_allclose(both, front / 2.0, rtol=1e-10)
    with pytest.raises(ValueError, match="face"):
        design_covariances_for_poses(targets, poses, LaserTracker(), visible, faces=[("top",)] * 3)
    with pytest.raises(ValueError, match="one entry per station"):
        design_covariances_for_poses(targets, poses, LaserTracker(), visible, faces=[(FRONT, BACK)])


# -- the consider term against nonlinear solves ---------------------------------------------------------
def solve_error_for_parameters(targets, poses, values, faces=None):
    """Network-solve error of every target when the tracker really has
    `values` as its parameter errors but the solver assumes a perfect one:
    noise-free readings generated with the NPL model, fitted with the
    existing ideal-geometry solver started at the truth."""
    observations = []
    for s, pose in enumerate(poses):
        local = np.array([pose.to_local(t) for t in targets])
        for face in (faces[s] if faces else (FRONT,)):
            readings = true_to_observed_batch(local, values, face)
            observations.extend(Observation(s, t, *readings[t], **({"face": face} if faces else {})) for t in range(len(targets)))
    fit = solve_network(observations, targets, poses, LaserTracker(), anchor_index=0)
    return fit.target_points_m - targets


# step sizes for the numerical derivative: 100x the ones used inside the model, so that the
# solver's own 1e-8 tolerance is a small fraction of the response, yet still deep in the linear range
NUMERICAL_STEP = np.array([1e-4] * 5 + [5e-4] * 11)


def test_consider_sensitivity_matches_nonlinear_network_solves_parameter_by_parameter():
    targets, poses, visible = scene()
    tracker = tracker_with(NPLErrorParameters(np.zeros(16), np.eye(16) * 1e-12))
    s_design = design_parameter_sensitivity(targets, poses, tracker, visible)
    assert s_design.shape == (len(targets), 3, 16)
    for j in range(N_PARAMETERS):
        plus, minus = np.zeros(16), np.zeros(16)
        plus[j], minus[j] = NUMERICAL_STEP[j], -NUMERICAL_STEP[j]
        numeric = (solve_error_for_parameters(targets, poses, plus) - solve_error_for_parameters(targets, poses, minus)) / (2 * NUMERICAL_STEP[j])
        scale = np.abs(numeric).max()
        assert scale > 0
        np.testing.assert_allclose(s_design[:, :, j], numeric, atol=2e-3 * scale, err_msg=PARAMETER_NAME(j))


def PARAMETER_NAME(j):
    from instruments.npl_model import PARAMETER_NAMES
    return PARAMETER_NAMES[j]


def test_consider_covariance_is_the_sensitivity_times_the_parameter_covariance():
    targets, poses, visible = scene()
    params = NPLErrorParameters(np.zeros(16), np.diag((np.maximum(HUGHES_RESIDUAL.standard_uncertainties(), 0)) ** 2))
    noise = design_covariances_for_poses(targets, poses, LaserTracker(), visible).point_covariance_m2
    total = design_covariances_for_poses(targets, poses, tracker_with(params), visible).point_covariance_m2
    s = design_parameter_sensitivity(targets, poses, tracker_with(params), visible)
    expected = np.einsum("tia,ab,tjb->tij", s, params.covariance, s)
    np.testing.assert_allclose(total - noise, expected, rtol=1e-9, atol=1e-22)
    assert np.all(np.linalg.eigvalsh(total - noise) >= -1e-22)  # a covariance term is positive semi-definite


def test_joint_covariance_diagonal_blocks_equal_the_marginal_covariances_and_cross_terms_are_dense():
    targets, poses, visible = scene()
    tracker = tracker_with(HUGHES_RESIDUAL)
    marginal = design_covariances_for_poses(targets, poses, tracker, visible).point_covariance_m2
    subset = [1, 4, 6]
    joint = design_joint_covariance(targets, poses, tracker, visible, subset)[0]
    for i, t in enumerate(subset):
        np.testing.assert_allclose(joint[3 * i:3 * i + 3, 3 * i:3 * i + 3], marginal[t], rtol=1e-9)
    noise_joint = design_joint_covariance(targets, poses, LaserTracker(), visible, subset)[0]
    cross = (joint - noise_joint)[0:3, 3:6]
    assert np.abs(cross).max() > 0.1 * np.abs((joint - noise_joint)[0:3, 0:3]).max()  # shared parameters correlate targets


# -- what two faces can and cannot cancel (physics, independent of the algebra) ---------------------------
def consider_variance(params, faces=None, trace=True):
    targets, poses, visible = scene()
    noise = design_covariances_for_poses(targets, poses, LaserTracker(), visible, faces=faces).point_covariance_m2
    total = design_covariances_for_poses(targets, poses, tracker_with(params), visible, faces=faces).point_covariance_m2
    return np.einsum("tii->t", total - noise)


def test_two_faces_cancel_collimation_error_but_not_a_range_offset():
    """Collimation error gamma moves the reported point in opposite directions
    on the two faces, so a two-face average has none (to second order); a
    range offset is the same on both, so averaging two faces leaves exactly
    the single-face effect."""
    gamma = only_parameter_uncertain("gamma_rad", 10 * ARCSEC_TO_RAD)
    lam = only_parameter_uncertain("lambda_m", 1e-5)
    both = [(FRONT, BACK)] * 3
    ratio_gamma = consider_variance(gamma, both) / consider_variance(gamma)
    ratio_lambda = consider_variance(lam, both) / consider_variance(lam)
    assert np.all(ratio_gamma < 1e-5)
    np.testing.assert_allclose(ratio_lambda, 1.0, rtol=1e-6)


def test_instrument_ids_set_which_stations_share_parameter_errors():
    targets, poses, visible = scene()
    tracker = tracker_with(only_parameter_uncertain("lambda_m", 1e-5))
    one = design_parameter_sensitivity(targets, poses, tracker, visible)
    two = design_parameter_sensitivity(targets, poses, tracker, visible, instrument_ids=[0, 1, 1])
    assert one.shape == (len(targets), 3, 16) and two.shape == (len(targets), 3, 32)
    np.testing.assert_allclose(two[:, :, :16] + two[:, :, 16:], one, atol=1e-12)  # the same shared error, split by tracker
    with pytest.raises(ValueError, match="one entry per station"):
        design_parameter_sensitivity(targets, poses, tracker, visible, instrument_ids=[0, 1])
    # independent trackers: the range-offset error of one tracker is not the other's
    shared = design_covariances_for_poses(targets, poses, tracker, visible).point_covariance_m2
    independent = design_covariances_for_poses(targets, poses, tracker, visible, instrument_ids=[0, 1, 2]).point_covariance_m2
    assert not np.allclose(shared, independent, rtol=1e-3, atol=0.0)
