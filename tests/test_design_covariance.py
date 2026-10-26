"""Tests for network.design: the fast analytic design covariance must match
network.solve's nonlinear fit (which it replaces for planning), and must
refuse -- not silently mis-report -- a network with a gauge freedom."""
import time

import numpy as np
import pytest

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from network.design import (
    design_covariances_for_poses,
    projected_uncertainty_m,
    shot_information_global,
)
from network.solve import Observation, exact_observations, solve_network
from planning.candidates import look_at_orientation


def _scene(n_targets=14, seed=0):
    rng = np.random.default_rng(seed)
    centre = np.array([1.0, 1.0, 1.2])
    targets = centre + rng.uniform(-0.7, 0.7, size=(n_targets, 3))
    station_positions = [
        np.array([-2.5, 0.5, 1.6]),
        np.array([3.5, 0.8, 2.0]),
        np.array([1.0, 4.0, 1.4]),
        np.array([0.5, -2.8, 2.2]),
    ]
    poses = [InstrumentPose(p, look_at_orientation(p, centre)) for p in station_positions]
    return targets, poses


def _reference_covariances(targets, poses, tracker, visible):
    obs = [o for o in exact_observations(targets, poses) if visible[o.station_index, o.target_index]]
    result = solve_network(obs, targets.copy(), poses, tracker, anchor_index=0)
    return np.array([result.target_point_covariance(i) for i in range(len(targets))])


def _max_relative_error(a, b):
    return max(np.linalg.norm(a[i] - b[i]) / np.linalg.norm(b[i]) for i in range(len(b)))


def test_shot_information_is_the_inverse_of_the_step1_covariance():
    tracker = LaserTracker()
    pose = InstrumentPose(np.array([0.0, 0.0, 1.0]), look_at_orientation(np.array([0.0, 0.0, 1.0]), np.array([3.0, 1.0, 1.0])))
    point = np.array([3.0, 1.2, 0.8])
    info = shot_information_global((point - pose.position_m)[None], pose.orientation[None], tracker)[0]
    expected = np.linalg.inv(tracker.covariance_global(point, pose))
    assert np.allclose(info, expected, rtol=1e-8)


def test_design_covariance_matches_nonlinear_solve_when_every_station_sees_every_target():
    tracker = LaserTracker()
    targets, poses = _scene()
    visible = np.ones((len(poses), len(targets)), dtype=bool)
    fast = design_covariances_for_poses(targets, poses, tracker, visible)
    assert fast.connected
    reference = _reference_covariances(targets, poses, tracker, visible)
    assert _max_relative_error(fast.point_covariance_m2, reference) < 1e-4


def test_design_covariance_matches_nonlinear_solve_under_partial_visibility():
    """Each target seen by a different subset of stations, but every
    station shares several targets with the others (a connected network)."""
    tracker = LaserTracker()
    targets, poses = _scene(n_targets=16, seed=3)
    rng = np.random.default_rng(1)
    visible = rng.random((len(poses), len(targets))) < 0.6
    visible[:, :6] = True  # shared tie targets: guarantees the network is connected
    unseen = ~visible.any(axis=0)
    visible[0, unseen] = True  # nothing unobserved
    fast = design_covariances_for_poses(targets, poses, tracker, visible)
    assert fast.connected and fast.observed.all()
    reference = _reference_covariances(targets, poses, tracker, visible)
    assert _max_relative_error(fast.point_covariance_m2, reference) < 1e-4


def test_station_tied_to_the_anchor_by_only_three_shared_targets_is_still_connected():
    tracker = LaserTracker()
    targets, poses = _scene(n_targets=14, seed=5)
    visible = np.zeros((len(poses), len(targets)), dtype=bool)
    visible[0, :] = True  # anchor sees everything
    visible[1, :10] = True
    visible[2, 4:] = True
    visible[3, 11:] = True  # only targets 11, 12, 13 in common with the anchor
    fast = design_covariances_for_poses(targets, poses, tracker, visible)
    assert fast.connected
    reference = _reference_covariances(targets, poses, tracker, visible)
    assert _max_relative_error(fast.point_covariance_m2, reference) < 1e-3


def test_network_with_a_gauge_freedom_is_reported_not_silently_solved():
    """Stations 2 and 3 see only targets nobody else sees: nothing ties
    them to the anchor, so their placement is a free rigid motion.
    solve_network's pseudo-inverse would hand back zero variance for that
    direction; the fast evaluator must say 'not connected' instead."""
    tracker = LaserTracker()
    targets, poses = _scene(n_targets=14, seed=2)
    visible = np.zeros((len(poses), len(targets)), dtype=bool)
    visible[0, :7] = True
    visible[1, :7] = True
    visible[2, 7:] = True
    visible[3, 7:] = True
    fast = design_covariances_for_poses(targets, poses, tracker, visible)
    assert not fast.connected
    assert np.isnan(fast.point_covariance_m2).all()


def test_unobserved_targets_are_flagged_and_do_not_affect_the_rest():
    tracker = LaserTracker()
    targets, poses = _scene()
    visible = np.ones((len(poses), len(targets)), dtype=bool)
    visible[:, 3] = False
    fast = design_covariances_for_poses(targets, poses, tracker, visible)
    assert not fast.observed[3] and np.isnan(fast.point_covariance_m2[3]).all()
    keep = np.arange(len(targets)) != 3
    reference = design_covariances_for_poses(targets[keep], poses, tracker, visible[:, keep])
    assert np.allclose(fast.point_covariance_m2[keep], reference.point_covariance_m2, rtol=1e-9)


def test_single_station_reduces_to_the_step1_covariance():
    tracker = LaserTracker()
    targets, poses = _scene()
    visible = np.ones((1, len(targets)), dtype=bool)
    fast = design_covariances_for_poses(targets, poses[:1], tracker, visible)
    for i in range(len(targets)):
        assert np.allclose(
            fast.point_covariance_m2[i], tracker.covariance_global(targets[i], poses[0]), rtol=1e-8
        )


def test_projected_uncertainty_adds_the_systematic_term_in_quadrature():
    cov = np.array([np.diag([4e-12, 1e-12, 9e-12])])
    normal = np.array([[0.0, 0.0, 1.0]])
    assert projected_uncertainty_m(cov, normal)[0] == pytest.approx(3e-6)
    assert projected_uncertainty_m(cov, normal, systematic_std_m=4e-6)[0] == pytest.approx(5e-6)


def test_five_hundred_targets_evaluate_in_well_under_a_second():
    """The point of the module: a 500-target plan is scored in milliseconds
    (the dense nonlinear route inverts a ~1500x1500 matrix per plan)."""
    tracker = LaserTracker()
    rng = np.random.default_rng(0)
    centre = np.array([1.0, 1.0, 1.2])
    targets = centre + rng.uniform(-0.9, 0.9, size=(500, 3))
    positions = [np.array(p) for p in [(-2.5, 0.5, 1.6), (3.5, 0.8, 2.0), (1.0, 4.0, 1.4), (0.5, -2.8, 2.2), (2.5, 3.0, 2.5)]]
    poses = [InstrumentPose(p, look_at_orientation(p, centre)) for p in positions]
    visible = np.ones((len(poses), 500), dtype=bool)
    design_covariances_for_poses(targets, poses, tracker, visible)  # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        result = design_covariances_for_poses(targets, poses, tracker, visible)
    per_call_s = (time.perf_counter() - t0) / 5
    assert result.connected
    assert per_call_s < 0.25
