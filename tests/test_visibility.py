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


def test_visibility_matrix_matches_the_scalar_check_on_random_scenes():
    """The batched version must agree with check_target_visibility on every
    station-target pair -- range, incidence and occlusion together."""
    from geometry.mesh import box_mesh
    from geometry.visibility import check_target_visibility, visibility_matrix

    rng = np.random.default_rng(7)
    mesh = box_mesh(np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 1.0]))
    limits = VisibilityLimits(max_range_m=6.0, max_incidence_angle_rad=np.deg2rad(75.0))
    stations = rng.uniform([-4, -4, 0.2], [4, 4, 3.0], size=(12, 3))
    targets = rng.uniform([-0.9, -0.9, 0.0], [0.9, 0.9, 1.2], size=(60, 3))
    normals = rng.normal(size=(60, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    batched = visibility_matrix(stations, targets, limits, target_normals_m=normals, obstacles=mesh)
    scalar = np.array([
        [check_target_visibility(s, t, limits, target_normal_m=n, obstacles=mesh).is_visible
         for t, n in zip(targets, normals)]
        for s in stations
    ])
    assert np.array_equal(batched, scalar)
    assert 0 < scalar.sum() < scalar.size  # the comparison is not vacuous


def test_visibility_matrix_without_obstacles_or_normals_is_just_range():
    from geometry.visibility import visibility_matrix

    limits = VisibilityLimits(max_range_m=5.0, max_incidence_angle_rad=np.pi / 2)
    stations = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    targets = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    result = visibility_matrix(stations, targets, limits)
    assert result.tolist() == [[True, True], [False, False]]


# -- elevation limits (the tracker's mechanical head travel) ---------------------
# Hughes et al. 2011 §8: the API T3's head tilts from -60 to +77 degrees. Elevation is
# measured above the instrument's own x-y plane, so these tests pick beams of known
# elevation and check the boundary, the orientation dependence, and that the batched
# version matches the scalar one.
from geometry.visibility import API_T3_MAX_ELEVATION_RAD, API_T3_MIN_ELEVATION_RAD  # noqa: E402

T3_LIMITS = VisibilityLimits(
    max_range_m=100.0, max_incidence_angle_rad=np.pi,
    min_elevation_rad=API_T3_MIN_ELEVATION_RAD, max_elevation_rad=API_T3_MAX_ELEVATION_RAD,
)


# A proper rotation whose local z axis (third column) is world +x and whose local x axis is
# world -z: the instrument lying on its side.
SIDEWAYS = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _beam_at_elevation(elevation_deg: float, range_m: float = 3.0, azimuth_deg: float = 30.0) -> np.ndarray:
    el, az = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
    return range_m * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def test_api_t3_elevation_constants_are_minus_60_and_plus_77_degrees():
    assert np.rad2deg(API_T3_MIN_ELEVATION_RAD) == pytest.approx(-60.0)
    assert np.rad2deg(API_T3_MAX_ELEVATION_RAD) == pytest.approx(77.0)


@pytest.mark.parametrize("elevation_deg, expected", [
    (0.0, VisibilityReason.VISIBLE),
    (56.5, VisibilityReason.VISIBLE),  # the steepest shot in the Hughes network test
    (76.9, VisibilityReason.VISIBLE),
    (77.1, VisibilityReason.ELEVATION_OUT_OF_RANGE),
    (89.0, VisibilityReason.ELEVATION_OUT_OF_RANGE),  # nearly straight up: the blind-spot region
    (-59.9, VisibilityReason.VISIBLE),
    (-60.1, VisibilityReason.ELEVATION_OUT_OF_RANGE),
    (-89.0, VisibilityReason.ELEVATION_OUT_OF_RANGE),
])
def test_elevation_limits_follow_the_mechanical_range_in_a_levelled_frame(elevation_deg, expected):
    result = check_target_visibility(np.zeros(3), _beam_at_elevation(elevation_deg), T3_LIMITS)
    assert result.reason is expected


def test_azimuth_never_limits_a_tracker():
    # The same elevation at every azimuth: head travel in azimuth is a full circle.
    for azimuth_deg in range(0, 360, 45):
        beam = _beam_at_elevation(50.0, azimuth_deg=azimuth_deg)
        assert check_target_visibility(np.zeros(3), beam, T3_LIMITS).is_visible


def test_no_elevation_limits_reproduces_the_old_behaviour_for_any_orientation():
    old = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.pi)
    assert not old.has_elevation_limits
    for elevation_deg in (-89.0, 0.0, 89.0):
        beam = _beam_at_elevation(elevation_deg)
        assert check_target_visibility(np.zeros(3), beam, old, instrument_orientation=SIDEWAYS).is_visible


def test_elevation_is_measured_in_the_instrument_frame_not_the_world_frame():
    # Lay the instrument on its side: its local z axis (the head's vertical axis) points along
    # world +x. A beam along world +z is then 90 degrees from that axis -- on the instrument's
    # own horizon -- and visible, while a beam along world +x is straight up in the instrument's
    # frame and is not.
    assert np.allclose(SIDEWAYS.T @ SIDEWAYS, np.eye(3)) and np.isclose(np.linalg.det(SIDEWAYS), 1.0)
    assert np.allclose(SIDEWAYS[:, 2], [1.0, 0.0, 0.0])
    along_world_z = np.array([0.0, 0.0, 3.0])
    along_world_x = np.array([3.0, 0.0, 0.0])

    levelled_z = check_target_visibility(np.zeros(3), along_world_z, T3_LIMITS)
    assert levelled_z.reason is VisibilityReason.ELEVATION_OUT_OF_RANGE  # straight up, levelled

    sideways_z = check_target_visibility(np.zeros(3), along_world_z, T3_LIMITS, instrument_orientation=SIDEWAYS)
    assert sideways_z.is_visible

    sideways_x = check_target_visibility(np.zeros(3), along_world_x, T3_LIMITS, instrument_orientation=SIDEWAYS)
    assert sideways_x.reason is VisibilityReason.ELEVATION_OUT_OF_RANGE


def test_one_sided_elevation_limit_constrains_only_that_side():
    only_max = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.pi, max_elevation_rad=np.deg2rad(45.0))
    assert check_target_visibility(np.zeros(3), _beam_at_elevation(-80.0), only_max).is_visible
    assert not check_target_visibility(np.zeros(3), _beam_at_elevation(50.0), only_max).is_visible


def test_target_at_the_instrument_has_no_defined_elevation_and_is_not_rejected_for_it():
    assert check_target_visibility(np.zeros(3), np.zeros(3), T3_LIMITS).is_visible


def test_elevation_limits_are_validated():
    with pytest.raises(ValueError, match="elevation"):
        VisibilityLimits(max_range_m=1.0, max_incidence_angle_rad=1.0, max_elevation_rad=2.0)
    with pytest.raises(ValueError, match="below"):
        VisibilityLimits(max_range_m=1.0, max_incidence_angle_rad=1.0, min_elevation_rad=0.5, max_elevation_rad=0.2)


def test_scene_visibility_uses_the_scenes_instrument_orientation():
    points = np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])  # straight up, and along +x
    levelled = Scene(target_points_m=points, instrument_pose=InstrumentPose(position_m=np.zeros(3)))
    sideways = Scene(
        target_points_m=points, instrument_pose=InstrumentPose(position_m=np.zeros(3), orientation=SIDEWAYS)
    )
    assert [r.is_visible for r in scene_visibility(levelled, T3_LIMITS)] == [False, True]
    assert [r.is_visible for r in scene_visibility(sideways, T3_LIMITS)] == [True, False]


def test_visibility_matrix_with_elevation_limits_matches_the_scalar_check():
    from geometry.visibility import visibility_matrix

    rng = np.random.default_rng(20241127)
    stations = rng.uniform(-3.0, 3.0, size=(12, 3))
    targets = rng.uniform(-3.0, 3.0, size=(40, 3))
    normals = rng.normal(size=(40, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # random proper rotations as station orientations, so the pin covers tilted frames
    orientations = np.empty((12, 3, 3))
    for i in range(12):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        orientations[i] = q if np.linalg.det(q) > 0 else q @ np.diag([1.0, 1.0, -1.0])
    limits = VisibilityLimits(
        max_range_m=6.0, max_incidence_angle_rad=np.deg2rad(75.0),
        min_elevation_rad=API_T3_MIN_ELEVATION_RAD, max_elevation_rad=API_T3_MAX_ELEVATION_RAD,
    )
    for orient in (orientations, None):
        batched = visibility_matrix(stations, targets, limits, normals, station_orientations=orient)
        scalar = np.array([
            [check_target_visibility(
                s, t, limits, target_normal_m=n,
                instrument_orientation=None if orient is None else orient[i]).is_visible
             for t, n in zip(targets, normals)]
            for i, s in enumerate(stations)
        ])
        assert np.array_equal(batched, scalar)
        no_elev = visibility_matrix(
            stations, targets, VisibilityLimits(6.0, np.deg2rad(75.0)), normals)
        # elevation limits only ever remove targets, and here remove some but not all
        assert np.all(batched <= no_elev) and batched.sum() < no_elev.sum() and batched.sum() > 0


def test_visibility_matrix_zero_range_pair_is_not_rejected_for_elevation():
    from geometry.visibility import visibility_matrix

    result = visibility_matrix(np.zeros((1, 3)), np.zeros((1, 3)), T3_LIMITS)
    assert result.tolist() == [[True]]
