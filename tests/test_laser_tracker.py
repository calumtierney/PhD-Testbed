"""Tests for instruments.laser_tracker.

The headline test, test_boresight_anisotropy_matches_hughes_reference, is
the gate for build step 1 (CLAUDE.md §5): a point at 2.5 m must show
~10:1 anisotropy, ~40 um transverse to the beam vs ~4 um along it, large
axis perpendicular to the beam, small axis along it.
"""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import ARCSEC_TO_RAD, LaserTracker, scene_point_covariances


def test_boresight_anisotropy_matches_hughes_reference():
    tracker = LaserTracker()  # calibrated defaults -- see module docstring
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([2.5, 0.0, 0.0])  # 2.5 m straight down the beam

    covariance = tracker.covariance_global(target, pose)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending order
    stds_um = np.sqrt(eigenvalues) * 1e6

    radial_std_um, *lateral_stds_um = stds_um
    assert radial_std_um == pytest.approx(4.0, rel=0.05)
    for lateral_std_um in lateral_stds_um:
        assert lateral_std_um == pytest.approx(40.0, rel=0.05)

    ratio = np.mean(lateral_stds_um) / radial_std_um
    assert 8.0 < ratio < 12.0

    # Large axis perpendicular to the beam, small axis along it.
    beam_direction = target / np.linalg.norm(target)
    small_axis = eigenvectors[:, 0]
    assert abs(np.dot(small_axis, beam_direction)) == pytest.approx(1.0, abs=1e-6)
    for axis in eigenvectors[:, 1:].T:
        assert abs(np.dot(axis, beam_direction)) < 1e-6


def test_anisotropy_direction_is_robust_to_literal_headline_defaults():
    """Sensitivity check, not a hard gate: even with Hughes et al.'s
    headline a-priori figures (sigma_d ~ 1.2 um, angular sigma ~ 0.5
    arcsec) rather than the calibrated defaults above, the ellipsoid is
    still flattened along the beam -- the *direction* of the anisotropy
    doesn't depend on the exact noise numbers, only its magnitude does.
    See docs/step1_laser_tracker_physics.md for the gap between these
    headline figures and the ~40/4 um a-posteriori numbers in CLAUDE.md §4.
    """
    tracker = LaserTracker(
        sigma_d_m=1.2e-6,
        sigma_theta_rad=0.5 * ARCSEC_TO_RAD,
        sigma_phi_rad=0.5 * ARCSEC_TO_RAD,
    )
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([2.5, 0.0, 0.0])

    eigenvalues, _ = np.linalg.eigh(tracker.covariance_global(target, pose))
    radial_variance, *lateral_variances = eigenvalues
    for lateral_variance in lateral_variances:
        assert lateral_variance > radial_variance


def test_off_axis_point_radial_axis_still_exact():
    """Away from boresight the numbers are messier, but the smallest
    eigenvalue must still equal sigma_d^2 exactly, with the beam direction
    as its eigenvector -- range noise doesn't care which way the tracker
    is pointed."""
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([1.3, 2.1, 0.6])

    eigenvalues, eigenvectors = np.linalg.eigh(tracker.covariance_global(target, pose))

    assert eigenvalues[0] == pytest.approx(tracker.sigma_d_m**2, rel=1e-9)
    beam_direction = target / np.linalg.norm(target)
    assert abs(np.dot(eigenvectors[:, 0], beam_direction)) == pytest.approx(1.0, abs=1e-9)


def test_rotated_pose_rotates_the_ellipsoid():
    """A tracker set up at a different orientation gives the same
    ellipsoid shape, just rotated -- the physics lives in the local frame;
    only the reporting frame changes."""
    tracker = LaserTracker()
    target_local = np.array([2.5, 0.0, 0.0])

    pose_identity = InstrumentPose(position_m=np.zeros(3))
    covariance_identity = tracker.covariance_global(target_local, pose_identity)

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90 deg about z
    pose_rotated = InstrumentPose(position_m=np.zeros(3), orientation=rotation)
    target_global = rotation @ target_local  # same point in the tracker's local frame
    covariance_rotated = tracker.covariance_global(target_global, pose_rotated)

    assert np.allclose(covariance_rotated, rotation @ covariance_identity @ rotation.T)


def test_scene_point_covariances_shape_and_values():
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    points = np.array([[2.5, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    scene = Scene(target_points_m=points, instrument_pose=pose)

    covariances = scene_point_covariances(scene, tracker)
    assert covariances.shape == (3, 3, 3)
    for i in range(3):
        assert np.allclose(covariances[i], tracker.covariance_global(points[i], pose))
        assert np.allclose(covariances[i], covariances[i].T)  # covariance is symmetric


def test_covariance_at_the_instrument_origin_raises():
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    with pytest.raises(ValueError):
        tracker.covariance_global(np.zeros(3), pose)
