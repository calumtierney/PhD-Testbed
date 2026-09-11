"""Tests for instruments.laser_tracker.

The headline test, test_boresight_anisotropy_matches_hughes_table4_prediction,
is the gate for build step 1 (CLAUDE.md §5, §4b): propagating the Hughes et
al. (2011) Table 4 sensor noise (§4a) through the spherical-to-Cartesian
Jacobian at 2.5 m must predict lateral sigma of about 5.9 um (azimuth) and
8.4 um (elevation), radial sigma of about 1.2 um, and an anisotropy ratio
of roughly 5:1 to 7:1 -- with the long axes of the ellipsoid perpendicular
to the beam. These are the model's *inputs* (§4a) and its predicted
*output* (§4b); per CLAUDE.md §4e, the inputs are not to be retuned to hit
a target -- the ratio is a prediction that falls out of them.
"""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker, reallocatable_fraction, scene_point_covariances


def test_boresight_anisotropy_matches_hughes_table4_prediction():
    tracker = LaserTracker()  # Hughes et al. 2011 Table 4 values, CLAUDE.md §4a
    pose = InstrumentPose(position_m=np.zeros(3))
    target = np.array([2.5, 0.0, 0.0])  # 2.5 m straight down the beam (boresight)

    covariance = tracker.covariance_global(target, pose)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending order
    stds_um = np.sqrt(eigenvalues) * 1e6
    # At boresight (theta = phi = 0) the eigenvalues decouple exactly onto
    # the beam axis (radial, sigma_d) and the two encoder axes (azimuth,
    # elevation) -- see docs/step1_laser_tracker_physics.md. Ascending
    # order is therefore radial, azimuth-lateral, elevation-lateral,
    # since sigma_d < d*sigma_theta < d*sigma_phi for these inputs.
    radial_std_um, azimuth_lateral_std_um, elevation_lateral_std_um = stds_um

    # Exact analytic predictions, worked from CLAUDE.md §4a independently
    # of the tracker under test: Table 4 reports azimuth/elevation noise as
    # a lateral standard deviation *at 1 m* (2.351 um, 3.365 um), which
    # scales linearly with range since it's an angular noise; sigma_d is a
    # fixed length noise and does not scale with range at all. §4b quotes
    # the 2.5 m values rounded to "~5.9", "~8.4", "~1.2" um.
    assert radial_std_um == pytest.approx(1.216, rel=1e-6)
    assert azimuth_lateral_std_um == pytest.approx(2.5 * 2.351, rel=1e-3)
    assert elevation_lateral_std_um == pytest.approx(2.5 * 3.365, rel=1e-3)

    # CLAUDE.md §4b gate: lateral between about 5 and 9 um, radial about
    # 1.2 um, ratio roughly 5:1 to 7:1.
    assert 5.0 <= azimuth_lateral_std_um <= 9.0
    assert 5.0 <= elevation_lateral_std_um <= 9.0
    assert radial_std_um == pytest.approx(1.2, rel=0.05)
    assert 4.5 <= azimuth_lateral_std_um / radial_std_um <= 7.5
    assert 4.5 <= elevation_lateral_std_um / radial_std_um <= 7.5

    # Large axes perpendicular to the beam, small axis along it.
    beam_direction = target / np.linalg.norm(target)
    small_axis = eigenvectors[:, 0]
    assert abs(np.dot(small_axis, beam_direction)) == pytest.approx(1.0, abs=1e-6)
    for axis in eigenvectors[:, 1:].T:
        assert abs(np.dot(axis, beam_direction)) < 1e-6


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


def test_systematic_std_defaults_to_zero_and_changes_nothing():
    """CLAUDE.md referee note M6's fix must not disturb any of steps
    1-5's existing, validated sensor-noise-only behaviour."""
    tracker_default = LaserTracker()
    tracker_explicit_zero = LaserTracker(systematic_std_m=0.0)
    point_local = np.array([2.5, 0.0, 0.0])
    assert np.array_equal(
        tracker_default.covariance_local(point_local), tracker_explicit_zero.covariance_local(point_local)
    )


def test_systematic_std_adds_isotropic_variance():
    point_local = np.array([2.5, 0.0, 0.0])
    baseline = LaserTracker()
    with_systematic = LaserTracker(systematic_std_m=10e-6)

    covariance_baseline = baseline.covariance_local(point_local)
    covariance_with_systematic = with_systematic.covariance_local(point_local)

    assert np.allclose(covariance_with_systematic, covariance_baseline + (10e-6**2) * np.eye(3))


def test_reallocatable_fraction_is_one_without_a_systematic_term():
    tracker = LaserTracker()
    point_local = np.array([2.5, 0.0, 0.0])
    assert reallocatable_fraction(tracker, point_local) == pytest.approx(1.0)


def test_reallocatable_fraction_drops_as_systematic_term_dominates():
    point_local = np.array([2.5, 0.0, 0.0])
    modest_systematic = LaserTracker(systematic_std_m=1e-6)
    dominant_systematic = LaserTracker(systematic_std_m=1e-3)  # 1 mm, swamps sensor noise

    fraction_modest = reallocatable_fraction(modest_systematic, point_local)
    fraction_dominant = reallocatable_fraction(dominant_systematic, point_local)

    assert 0.0 < fraction_dominant < fraction_modest < 1.0
    assert fraction_dominant == pytest.approx(0.0, abs=1e-3)
