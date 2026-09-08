"""Tests for geometry.pose and geometry.scene."""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from geometry.scene import Scene


def test_pose_to_local_identity_orientation():
    pose = InstrumentPose(position_m=np.array([1.0, 2.0, 3.0]))
    point_global = np.array([1.0, 2.0, 5.0])
    assert np.allclose(pose.to_local(point_global), [0.0, 0.0, 2.0])


def test_pose_to_local_rotated_orientation():
    # 90 degree rotation about z: local x-axis points along global y.
    orientation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    pose = InstrumentPose(position_m=np.zeros(3), orientation=orientation)
    point_global = np.array([0.0, 5.0, 0.0])  # 5 m along global y
    # ... which is 5 m along the tracker's local x-axis
    assert np.allclose(pose.to_local(point_global), [5.0, 0.0, 0.0])


def test_pose_covariance_to_global_round_trips_identity():
    pose = InstrumentPose(position_m=np.zeros(3))
    covariance_local = np.diag([1.0, 2.0, 3.0])
    assert np.allclose(pose.covariance_to_global(covariance_local), covariance_local)


def test_pose_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        InstrumentPose(position_m=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        InstrumentPose(position_m=np.zeros(3), orientation=np.eye(2))


def test_scene_holds_points_and_pose():
    pose = InstrumentPose(position_m=np.zeros(3))
    points = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    scene = Scene(target_points_m=points, instrument_pose=pose)
    assert scene.n_points == 3
    assert scene.instrument_pose is pose


def test_scene_accepts_single_point():
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=[1.0, 2.0, 3.0], instrument_pose=pose)
    assert scene.n_points == 1


def test_scene_rejects_wrong_shape():
    pose = InstrumentPose(position_m=np.zeros(3))
    with pytest.raises(ValueError):
        Scene(target_points_m=np.array([[1.0, 2.0]]), instrument_pose=pose)
