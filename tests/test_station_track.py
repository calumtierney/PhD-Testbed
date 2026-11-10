"""Tests for geometry.station_track and the track candidates in planning.candidates."""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from geometry.station_track import SEGMENT_NAMES, StationTrack
from instruments.laser_tracker import LaserTracker
from planning.candidates import (
    TrackPose, levelled_orientation, look_at_orientation, track_candidate_poses, track_s_values,
)

LENGTH, WIDTH, STANDOFF = 15.0, 2.5, 3.0


@pytest.fixture
def track():
    return StationTrack(0.0, LENGTH, WIDTH, STANDOFF)


def footprint_distance(xy, track):
    """Distance from a plan-view point to the part's rectangular footprint."""
    half = track.part_width_m / 2.0
    dx = max(track.part_x_min_m - xy[0], 0.0, xy[0] - track.part_x_max_m)
    dy = max(track.part_centre_y_m - half - xy[1], 0.0, xy[1] - track.part_centre_y_m - half)
    return float(np.hypot(dx, dy))


def test_perimeter_is_two_lengths_plus_two_semicircles_in_the_symmetric_case(track):
    radius = WIDTH / 2 + STANDOFF
    assert track.loop_radius_m == pytest.approx(radius)
    assert track.perimeter_m == pytest.approx(2 * LENGTH + 2 * np.pi * radius)


def test_row_points_lie_at_the_expected_y_and_loop_points_at_radius_r_from_the_tip_centre(track):
    radius = track.loop_radius_m
    for s in (0.0, 5.0, 14.9):  # -y row
        assert track.position_xy_m(s)[1] == pytest.approx(-(WIDTH / 2 + STANDOFF))
        assert track.segment(s) == "-y row"
    s_plus_y = LENGTH + np.pi * radius
    for s in (s_plus_y + 0.1, s_plus_y + 7.0):  # +y row
        assert track.position_xy_m(s)[1] == pytest.approx(WIDTH / 2 + STANDOFF)
        assert track.segment(s) == "+y row"
    for s in np.linspace(LENGTH + 0.01, LENGTH + np.pi * radius - 0.01, 9):  # +x loop
        assert np.linalg.norm(track.position_xy_m(s) - np.array([LENGTH, 0.0])) == pytest.approx(radius)
        assert track.segment(s) == "+x loop"
    s_minus_x = 2 * LENGTH + np.pi * radius
    for s in np.linspace(s_minus_x + 0.01, track.perimeter_m - 0.01, 9):  # -x loop
        assert np.linalg.norm(track.position_xy_m(s) - np.array([0.0, 0.0])) == pytest.approx(radius)
        assert track.segment(s) == "-x loop"
    assert set(track.segment(s) for s in np.linspace(0, track.perimeter_m, 200, endpoint=False)) == set(SEGMENT_NAMES)


def test_s_and_s_plus_perimeter_are_the_same_point_and_the_path_is_closed_and_continuous(track):
    for s in (0.0, 3.3, 22.2, 41.0):
        assert np.allclose(track.position_xy_m(s), track.position_xy_m(s + track.perimeter_m))
    assert np.allclose(track.position_xy_m(0.0), track.position_xy_m(track.perimeter_m - 1e-9), atol=1e-6)
    s = np.linspace(0.0, track.perimeter_m, 4001)
    xy = np.array([track.position_xy_m(v) for v in s])
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert step.max() <= (s[1] - s[0]) * 1.0001  # no jump anywhere, including the four joins
    assert s[0] == 0.0 and np.allclose(xy[0], [0.0, -(WIDTH / 2 + STANDOFF)])  # s = 0: -x end of the -y row


def test_s_increases_anticlockwise_seen_from_above(track):
    xy = np.array([track.position_xy_m(s) for s in np.linspace(0.0, track.perimeter_m, 400, endpoint=False)])
    shoelace = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    assert shoelace > 0.0  # positive signed area = anticlockwise


def test_unequal_standoffs_keep_the_path_continuous_and_never_closer_than_the_smaller_standoff():
    track = StationTrack(0.0, LENGTH, WIDTH, 3.0, standoff_left_m=2.0, standoff_right_m=4.0)
    assert track.y_left_row_m == pytest.approx(WIDTH / 2 + 2.0)
    assert track.y_right_row_m == pytest.approx(-(WIDTH / 2 + 4.0))
    assert track.loop_radius_m == pytest.approx(WIDTH / 2 + 3.0)
    assert track.perimeter_m == pytest.approx(2 * LENGTH + 2 * np.pi * track.loop_radius_m)
    # the loops meet both rows exactly
    assert np.allclose(track.position_xy_m(LENGTH), [LENGTH, track.y_right_row_m])
    assert np.allclose(track.position_xy_m(LENGTH + np.pi * track.loop_radius_m), [LENGTH, track.y_left_row_m], atol=1e-9)
    s = np.linspace(0.0, track.perimeter_m, 3000, endpoint=False)
    clearance = min(footprint_distance(track.position_xy_m(v), track) for v in s)
    assert clearance >= 2.0 - 1e-6  # the smaller standoff
    # the left and right rows really are at their own standoffs
    assert footprint_distance(track.position_xy_m(5.0), track) == pytest.approx(4.0)
    assert footprint_distance(track.position_xy_m(LENGTH + np.pi * track.loop_radius_m + 5.0), track) == pytest.approx(2.0)


def test_candidates_are_about_spacing_apart_including_across_the_joins(track):
    poses = track_candidate_poses(track, 1.0, (1.0,))
    xy = np.array([p.position_m[:2] for p in poses])
    gaps = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)  # includes the wrap from the last back to s = 0
    assert len(poses) == int(round(track.perimeter_m / 1.0))
    assert gaps.max() <= track.perimeter_m / len(poses) * 1.0001  # chord never exceeds arc
    assert gaps.min() >= 0.97 * track.perimeter_m / len(poses)
    assert abs(track.perimeter_m / len(poses) - 1.0) < 0.02


def test_no_candidate_is_closer_to_the_footprint_than_the_standoff(track):
    for p in track_candidate_poses(track, 0.7, (1.0, 2.4)):
        assert footprint_distance(p.position_m[:2], track) >= STANDOFF - 1e-6


def test_candidates_are_ordered_by_s_then_height_and_carry_track_coordinates(track):
    poses = track_candidate_poses(track, 2.0, (1.0, 2.4))
    assert all(isinstance(p, TrackPose) and isinstance(p, InstrumentPose) for p in poses)
    keys = [(p.s_m, p.height_m) for p in poses]
    assert keys == sorted(keys)
    assert {p.height_m for p in poses} == {1.0, 2.4}
    for p in poses:
        assert p.position_m[2] == p.height_m
        assert np.allclose(p.position_m[:2], track.position_xy_m(p.s_m))
        assert p.segment == track.segment(p.s_m)


def test_blocked_intervals_are_respected_including_one_that_wraps_past_s_zero():
    perim = StationTrack(0.0, LENGTH, WIDTH, STANDOFF).perimeter_m
    track = StationTrack(0.0, LENGTH, WIDTH, STANDOFF,
                         blocked_intervals_m=((10.0, 14.0), (perim - 3.0, 2.0)))  # the second wraps past s = 0
    assert track.is_blocked(12.0) and track.is_blocked(10.0) and track.is_blocked(14.0)
    assert track.is_blocked(1.0) and track.is_blocked(0.0) and track.is_blocked(perim - 1.0)
    assert not track.is_blocked(5.0) and not track.is_blocked(20.0) and not track.is_blocked(perim - 5.0)
    poses = track_candidate_poses(track, 1.0, (1.0,))
    assert poses and all(not track.is_blocked(p.s_m) for p in poses)
    unblocked = track_candidate_poses(StationTrack(0.0, LENGTH, WIDTH, STANDOFF), 1.0, (1.0,))
    assert len(poses) < len(unblocked)
    s_values = {round(p.s_m, 6) for p in poses}
    assert not any(s_values & {round(v, 6) for v in track_s_values(track, 1.0) if track.is_blocked(v)})


def test_travel_distance_takes_the_shorter_way_round(track):
    assert track.travel_distance_m(1.0, 4.0) == pytest.approx(3.0)
    assert track.travel_distance_m(1.0, track.perimeter_m - 1.0) == pytest.approx(2.0)


def test_levelled_poses_have_local_z_equal_to_world_up_and_look_at_poses_are_tilted(track):
    for p in track_candidate_poses(track, 3.0, (1.0, 2.4)):
        assert np.allclose(p.orientation[:, 2], [0.0, 0.0, 1.0])
        assert np.allclose(p.orientation.T @ p.orientation, np.eye(3))
        assert np.linalg.det(p.orientation) == pytest.approx(1.0)
    tilted = track_candidate_poses(track, 3.0, (2.4,), orientation_mode="look_at")
    assert any(abs(p.orientation[2, 2] - 1.0) > 0.01 for p in tilted)  # aimed at 1.25 m from 2.4 m up: tilted
    with pytest.raises(ValueError):
        track_candidate_poses(track, 3.0, (1.0,), orientation_mode="nonsense")


def test_levelled_and_look_at_are_identical_when_the_aim_point_is_at_station_height(track):
    for p in track_candidate_poses(track, 4.0, (1.25,), aim_point_m=None, aim_height_m=1.25):
        aim = np.array([np.clip(p.position_m[0], 0.0, LENGTH), 0.0, 1.25])
        assert np.allclose(levelled_orientation(p.position_m, aim), look_at_orientation(p.position_m, aim))


def test_yaw_alone_changes_no_covariance_so_the_aim_point_cannot_change_a_result():
    """The azimuth encoder covers 360 degrees with the same noise at every
    angle: rotating a levelled station about the vertical leaves a target's
    global covariance unchanged."""
    tracker = LaserTracker()
    position, point = np.array([1.0, -2.0, 2.4]), np.array([6.0, 1.0, 1.4])

    def yaw(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    base = tracker.covariance_global(point, InstrumentPose(position_m=position, orientation=yaw(0.0)))
    for angle in np.linspace(0.0, 2 * np.pi, 9):
        other = tracker.covariance_global(point, InstrumentPose(position_m=position, orientation=yaw(angle)))
        assert np.allclose(other, base, rtol=1e-9, atol=0.0)


def test_tilting_the_frame_does_change_the_covariance():
    """Unlike yaw, a tilt swaps the roles of the azimuth and elevation
    encoders (different noise), so it changes the covariance: the reason the
    track defaults to levelled poses."""
    tracker = LaserTracker()
    position, point, aim = np.array([0.0, 0.0, 2.4]), np.array([6.0, 3.0, 1.2]), np.array([6.0, 3.0, -2.0])
    levelled = tracker.covariance_global(point, InstrumentPose(position_m=position, orientation=levelled_orientation(position, aim)))
    tilted = tracker.covariance_global(point, InstrumentPose(position_m=position, orientation=look_at_orientation(position, aim)))
    assert np.abs(levelled - tilted).max() / np.abs(levelled).max() > 1e-3


def test_tip_arc_lengths_are_the_loop_apexes(track):
    radius = track.loop_radius_m
    assert np.allclose(track.position_xy_m(track.tip_plus_x_s_m), [LENGTH + radius, 0.0])
    assert np.allclose(track.position_xy_m(track.tip_minus_x_s_m), [-radius, 0.0])
    assert track.segment(track.tip_plus_x_s_m) == "+x loop" and track.segment(track.tip_minus_x_s_m) == "-x loop"
