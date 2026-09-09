"""Tests for geometry.visibility.

test_target_behind_wall_is_occluded_wall_removed_is_visible is the gate
for build step 2 (CLAUDE.md §5): a target placed behind a wall is
reported unmeasurable; the same target with the wall removed is
measurable.
"""
import numpy as np
import pytest

from geometry.mesh import load_stl
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from geometry.visibility import (
    VisibilityLimits,
    VisibilityReason,
    check_target_visibility,
    scene_visibility,
)

GENEROUS_LIMITS = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.deg2rad(80.0))


def test_target_behind_wall_is_occluded_wall_removed_is_visible():
    wall = load_stl("scenes/wall.stl")  # a flat wall in the plane x = 1.5
    instrument_position = np.zeros(3)
    target_point = np.array([3.0, 0.0, 0.0])  # beyond the wall, beam straight along +x
    target_normal = np.array([-1.0, 0.0, 0.0])  # facing back towards the instrument

    behind_wall = check_target_visibility(
        instrument_position, target_point, GENEROUS_LIMITS,
        target_normal_m=target_normal, obstacles=wall,
    )
    assert behind_wall.reason is VisibilityReason.OCCLUDED
    assert not behind_wall.is_visible

    wall_removed = check_target_visibility(
        instrument_position, target_point, GENEROUS_LIMITS,
        target_normal_m=target_normal, obstacles=None,
    )
    assert wall_removed.reason is VisibilityReason.VISIBLE
    assert wall_removed.is_visible


def test_target_beyond_max_range_is_out_of_range():
    limits = VisibilityLimits(max_range_m=5.0, max_incidence_angle_rad=np.deg2rad(80.0))
    result = check_target_visibility(
        np.zeros(3), np.array([10.0, 0.0, 0.0]), limits,
        target_normal_m=np.array([-1.0, 0.0, 0.0]),
    )
    assert result.reason is VisibilityReason.OUT_OF_RANGE


def test_target_within_range_is_not_out_of_range():
    limits = VisibilityLimits(max_range_m=5.0, max_incidence_angle_rad=np.deg2rad(80.0))
    result = check_target_visibility(
        np.zeros(3), np.array([4.0, 0.0, 0.0]), limits,
        target_normal_m=np.array([-1.0, 0.0, 0.0]),
    )
    assert result.is_visible


def test_grazing_incidence_fails_head_on_passes():
    limits = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.deg2rad(45.0))
    instrument_position = np.zeros(3)
    target_point = np.array([1.0, 0.0, 0.0])

    # Surface normal pointing straight back at the instrument: beam is
    # perfectly perpendicular to the surface, incidence angle 0.
    head_on = check_target_visibility(
        instrument_position, target_point, limits, target_normal_m=np.array([-1.0, 0.0, 0.0])
    )
    assert head_on.is_visible

    # Surface normal perpendicular to the beam: the beam grazes the
    # surface at 90 degrees incidence, well past the 45 degree limit.
    grazing = check_target_visibility(
        instrument_position, target_point, limits, target_normal_m=np.array([0.0, 1.0, 0.0])
    )
    assert grazing.reason is VisibilityReason.INCIDENCE_TOO_STEEP


def test_incidence_check_is_skipped_without_a_normal():
    """No surface normal supplied -> incidence is unconstrained, not failed."""
    limits = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.deg2rad(1.0))
    result = check_target_visibility(
        np.zeros(3), np.array([1.0, 0.0, 0.0]), limits, target_normal_m=None
    )
    assert result.is_visible


def test_range_checked_before_expensive_occlusion():
    """An out-of-range target is flagged OUT_OF_RANGE even when it would
    also be occluded -- range is checked first (module docstring: cheap
    checks before the ray cast)."""
    wall = load_stl("scenes/wall.stl")
    limits = VisibilityLimits(max_range_m=2.0, max_incidence_angle_rad=np.deg2rad(80.0))
    result = check_target_visibility(
        np.zeros(3), np.array([3.0, 0.0, 0.0]), limits,
        target_normal_m=np.array([-1.0, 0.0, 0.0]), obstacles=wall,
    )
    assert result.reason is VisibilityReason.OUT_OF_RANGE


def test_scene_visibility_mixed_results():
    wall = load_stl("scenes/wall.stl")
    pose = InstrumentPose(position_m=np.zeros(3))
    points = np.array(
        [
            [3.0, 0.0, 0.0],   # behind the wall -> occluded
            [3.0, 5.0, 0.0],   # off to the side, misses the wall -> visible
            [200.0, 0.0, 0.0],  # way out of range
        ]
    )
    normals = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    scene = Scene(
        target_points_m=points,
        instrument_pose=pose,
        target_normals_m=normals,
        obstacles=wall,
    )

    results = scene_visibility(scene, GENEROUS_LIMITS)
    assert [r.reason for r in results] == [
        VisibilityReason.OCCLUDED,
        VisibilityReason.VISIBLE,
        VisibilityReason.OUT_OF_RANGE,
    ]


def test_scene_visibility_without_normals_skips_incidence_only():
    """A scene with no target_normals_m still runs range and occlusion
    checks -- only incidence is skipped."""
    wall = load_stl("scenes/wall.stl")
    pose = InstrumentPose(position_m=np.zeros(3))
    points = np.array([[3.0, 0.0, 0.0]])  # behind the wall
    scene = Scene(target_points_m=points, instrument_pose=pose, obstacles=wall)

    results = scene_visibility(scene, GENEROUS_LIMITS)
    assert results[0].reason is VisibilityReason.OCCLUDED
