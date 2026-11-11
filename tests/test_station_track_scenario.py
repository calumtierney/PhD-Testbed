"""Physics gates for `planning.experiment.long_part_two_cluster_scenario`: the
long part on a station track. Inputs are fixed (CLAUDE.md §4a); these check
that the outputs behave as the physics says they must."""
import itertools

import numpy as np
import pytest

from geometry.pose import InstrumentPose
from geometry.visibility import VisibilityLimits, VisibilityReason, check_target_visibility
from planning.candidates import levelled_orientation, track_candidate_poses
from planning.experiment import (
    LONG_PART_GRID_HEIGHTS_M,
    iso_risk_station_counts,
    long_part_track,
    long_part_two_cluster_scenario,
    run_headline_experiment,
    with_marginal_prior,
)
from planning.objectives import (
    evaluate_plan,
    objective_a0_trace_covariance,
    station_visibility_matrix,
    visibility_deficit,
)
from planning.search import (
    INFEASIBLE_PENALTY,
    TrackStationCost,
    grid_search_station_placement,
    track_genetic_algorithm_station_placement,
)
from risk.jcgm106 import ProcessPrior

TRACK = long_part_track()
PLUS_Y_ROW_S = TRACK.row_length_m + np.pi * TRACK.loop_radius_m + 7.5  # middle of the +y row


# -- the scene itself --------------------------------------------------------
def test_scene_has_two_clusters_on_opposite_long_faces_offset_along_x_with_pm_y_normals():
    scenario = long_part_two_cluster_scenario()
    points, normals = scenario.target_points_m, scenario.target_normals_m
    names = [c.name for c in scenario.characteristics]
    assert names == [f"cluster{k}_profile_{i}" for k in (1, 2) for i in range(3)]
    cluster1, cluster2 = points[:3], points[3:6]
    assert np.allclose(cluster1[:, 1], 1.25) and np.allclose(cluster2[:, 1], -1.25)  # opposite long faces
    assert cluster1[:, 0].mean() < cluster2[:, 0].mean() - 3.0  # offset along the part
    for characteristic in scenario.characteristics:
        direction = characteristic.tolerance.surface_normal
        assert np.allclose(direction, [0.0, 1.0, 0.0] if characteristic.name.startswith("cluster1") else [0.0, -1.0, 0.0])
    assert scenario.assignments["cluster1_profile_0"].cluster.name == "high-capability"
    assert scenario.assignments["cluster2_profile_0"].cluster.name == "marginal"
    assert all(a.confidence == 0.9 for a in scenario.assignments.values())
    # ties: 5 per end face, 5 per long face, none of them characteristics
    assert len(points) == 6 + 10 + 10
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_every_target_lies_on_the_part_surface_and_ties_are_not_characteristics():
    scenario = long_part_two_cluster_scenario()
    points = scenario.target_points_m
    on_surface = (
        np.isclose(np.abs(points[:, 1]), 1.25) | np.isclose(points[:, 0], 0.0) | np.isclose(points[:, 0], 15.0)
    )
    assert on_surface.all()
    assert ((points[:, 2] >= 0.5) & (points[:, 2] <= 2.0)).all()  # between underside and top
    used = {i for c in scenario.characteristics for i in c.target_indices}
    assert used == set(range(6))


def test_anchor_sits_on_the_track_at_the_minus_x_tip_by_default_and_s_moves_it():
    default = long_part_two_cluster_scenario()
    assert np.allclose(default.anchor_pose.position_m[:2], TRACK.position_xy_m(TRACK.tip_minus_x_s_m))
    opposite = long_part_two_cluster_scenario(anchor_s_m=TRACK.tip_plus_x_s_m)
    assert np.allclose(opposite.anchor_pose.position_m[:2], TRACK.position_xy_m(TRACK.tip_plus_x_s_m))
    assert np.allclose(opposite.target_points_m, default.target_points_m)  # only the anchor moved


def test_marginal_prior_swap_works_on_the_new_scene():
    scenario = long_part_two_cluster_scenario()
    swapped = with_marginal_prior(scenario, ProcessPrior(mean_m=0.0, std_m=12.5e-6))
    assert swapped.assignments["cluster2_profile_1"].cluster.prior.std_m == 12.5e-6
    assert swapped.assignments["cluster1_profile_1"].cluster.name == "high-capability"


# -- gate: single-station uncertainty grows with range, faster across the beam than along it
def test_single_station_uncertainty_rises_with_range_faster_across_the_beam_than_along_it():
    scenario = long_part_two_cluster_scenario()
    target = scenario.target_points_m[1]  # a +y-face target at x = 4.0 m
    lateral, radial, ranges = [], [], []
    for x in (4.0, 8.0, 12.0, 15.0):  # a row station sliding away along the part
        position = np.array([x, TRACK.y_left_row_m, 1.25])
        pose = InstrumentPose(position_m=position, orientation=levelled_orientation(position, np.array([x, 0.0, 1.25])))
        covariance = scenario.tracker.covariance_global(target, pose)
        beam = target - position
        ranges.append(np.linalg.norm(beam))
        beam = beam / np.linalg.norm(beam)
        radial.append(np.sqrt(beam @ covariance @ beam))
        perpendicular = np.eye(3) - np.outer(beam, beam)
        lateral.append(np.sqrt(np.trace(perpendicular @ covariance @ perpendicular) / 2.0))
    ranges, lateral, radial = np.array(ranges), np.array(lateral), np.array(radial)
    assert np.all(np.diff(ranges) > 0)
    assert np.all(np.diff(lateral) > 0)  # grows with range
    assert lateral[-1] / lateral[0] > 0.8 * ranges[-1] / ranges[0]  # roughly in proportion (angle noise x range)
    assert radial[-1] / radial[0] < 1.05  # the distance sensor does not care about range
    assert lateral[-1] / lateral[0] > radial[-1] / radial[0]  # faster across the beam than along it
    assert np.all(lateral > 3.0 * radial)  # and always the larger


# -- gate: occlusion by the part itself ---------------------------------------
def test_a_station_on_the_plus_y_row_cannot_see_the_minus_y_face():
    scenario = long_part_two_cluster_scenario()
    position = np.array([*TRACK.position_xy_m(PLUS_Y_ROW_S), 1.0])
    far_face = [i for i in range(len(scenario.target_points_m)) if i in (3, 4, 5) or 21 <= i <= 25]
    near_face = [0, 1, 2, 16, 17, 18, 19, 20]
    visible = lambda i, limits: check_target_visibility(
        position, scenario.target_points_m[i], limits, scenario.target_normals_m[i], scenario.obstacles
    ).reason
    for i in far_face:
        # with the repo's incidence limit the far face is rejected first: it faces away from the tracker
        assert visible(i, scenario.visibility_limits) is not VisibilityReason.VISIBLE
        # take the incidence rule away and it is the part's own body that blocks the beam
        no_incidence = VisibilityLimits(max_range_m=30.0, max_incidence_angle_rad=np.pi)
        assert visible(i, no_incidence) is VisibilityReason.OCCLUDED
    assert sum(visible(i, scenario.visibility_limits) is VisibilityReason.VISIBLE for i in near_face) >= 5


def test_end_loop_stations_bridge_an_end_face_and_one_long_face_but_no_station_sees_both_long_faces():
    scenario = long_part_two_cluster_scenario()
    poses = track_candidate_poses(TRACK, 0.5, LONG_PART_GRID_HEIGHTS_M)
    visible = station_visibility_matrix(scenario, poses)[1:]
    plus_y = [0, 1, 2, *range(16, 21)]
    minus_y = [3, 4, 5, *range(21, 26)]
    end_faces = list(range(6, 16))
    loops = [i for i, p in enumerate(poses) if "loop" in p.segment]
    rows = [i for i, p in enumerate(poses) if "row" in p.segment]
    assert not (visible[:, plus_y].any(axis=1) & visible[:, minus_y].any(axis=1)).any()
    bridging = [i for i in loops if visible[i, end_faces].any() and visible[i, plus_y + minus_y].any()]
    assert bridging, "no end-loop station sees both an end face and a long face"
    assert any(visible[i, end_faces].sum() >= 3 and visible[i, plus_y].any() for i in loops)  # can tie to its end face
    assert any(visible[i, end_faces].sum() >= 3 and visible[i, minus_y].any() for i in loops)
    assert not visible[np.ix_(rows, end_faces)].any()  # a row station sees no end face (grazing incidence)


# -- gate: the network gets better with stations, then flattens ------------------
def test_mean_uncertainty_falls_with_station_count_and_flattens():
    scenario = long_part_two_cluster_scenario()
    poses = track_candidate_poses(TRACK, 2.0, LONG_PART_GRID_HEIGHTS_M)
    best = grid_search_station_placement(poses, 3, scenario, objective_a0_trace_covariance, prefilter_visibility=True).best
    assert best is not None
    chosen = list(best.station_poses)

    def mean_uncertainty_um(stations):
        evaluation = evaluate_plan(stations, scenario)
        solve = evaluation.network_solve
        return 1e6 * np.mean([np.sqrt(np.trace(solve.target_point_covariance(i))) for i in range(len(scenario.target_points_m))])

    means = [mean_uncertainty_um(chosen)]  # anchor + 3 stations
    for _ in range(4):  # greedy: add the single station that lowers the mean most
        scored = [(mean_uncertainty_um(chosen + [p]), k) for k, p in enumerate(poses)]
        value, k = min(scored)
        chosen.append(poses[k])
        means.append(value)
    assert all(b < a for a, b in zip(means, means[1:])), means  # falls with every added station
    gains = -np.diff(means)
    assert gains[-1] < 0.5 * gains[0], means  # and the last station buys far less than the first


# -- connectivity: what a plan needs, stated as tests --------------------------------
def test_two_added_stations_cannot_cover_and_tie_the_whole_part_but_three_can():
    scenario = long_part_two_cluster_scenario()
    poses = track_candidate_poses(TRACK, 2.0, LONG_PART_GRID_HEIGHTS_M)
    visible = station_visibility_matrix(scenario, poses)
    for n_added, expect_any in ((1, False), (2, False), (3, True)):
        passing = [
            c for c in itertools.combinations(range(len(poses)), n_added)
            if visibility_deficit(visible[[0] + [i + 1 for i in c]], 0) == 0.0
        ]
        assert bool(passing) is expect_any, (n_added, len(passing))
    assert all(
        any("loop" in poses[i].segment for i in c)
        for c in itertools.combinations(range(len(poses)), 3)
        if visibility_deficit(visible[[0] + [i + 1 for i in c]], 0) == 0.0
    ), "a feasible plan with no end-loop station"


def _few_candidates():
    """A handful of row and loop candidates (1 m spacing, both heights): enough
    that some three-station plans are feasible and most are not, few enough
    that an exhaustive search without the prefilter takes seconds."""
    wanted = {0.0, 5.0, 15.9, 16.9, 30.0, 44.8, 45.8, 54.7, 55.7}
    return [p for p in track_candidate_poses(TRACK, 1.0, LONG_PART_GRID_HEIGHTS_M) if round(p.s_m, 1) in wanted]


def test_prefilter_changes_no_grid_result_only_the_speed():
    scenario = long_part_two_cluster_scenario()
    poses = _few_candidates()
    plain = grid_search_station_placement(poses, 3, scenario, objective_a0_trace_covariance)
    fast = grid_search_station_placement(poses, 3, scenario, objective_a0_trace_covariance, prefilter_visibility=True)
    assert plain.n_infeasible == fast.n_infeasible and plain.best is not None
    assert 0 < len(plain.all_evaluations) < plain.n_infeasible
    assert len(plain.all_evaluations) == len(fast.all_evaluations)
    assert plain.best_objective_value == fast.best_objective_value
    assert [p.s_m for p in plain.best.station_poses] == [p.s_m for p in fast.best.station_poses]


def test_run_headline_experiment_accepts_track_candidates():
    scenario = long_part_two_cluster_scenario()
    poses = _few_candidates()
    result = run_headline_experiment(scenario, n_stations=3, candidate_poses=poses)
    assert result.search_a0.best is not None
    counts = iso_risk_station_counts(
        scenario, 1e9, {"A0": objective_a0_trace_covariance}, max_stations=3, candidate_poses=poses
    )
    assert counts == {"A0": 3}  # one and two stations are infeasible whatever the target


# -- the track GA ------------------------------------------------------------------------
def test_track_cost_takes_s_modulo_the_perimeter_and_blocked_s_is_infeasible():
    scenario = long_part_two_cluster_scenario()
    blocked = long_part_track(blocked_intervals_m=((10.0, 14.0),))
    aim = scenario.target_points_m.mean(axis=0)
    cost = TrackStationCost(scenario, objective_a0_trace_covariance, blocked, 1, aim)
    assert cost(np.array([12.0, 1.0])) == INFEASIBLE_PENALTY
    assert cost(np.array([12.0 + blocked.perimeter_m, 1.0])) == INFEASIBLE_PENALTY  # wraps to the same place
    poses_a = cost.station_poses(np.array([5.0, 1.0]))
    poses_b = cost.station_poses(np.array([5.0 + blocked.perimeter_m, 1.0]))
    assert np.allclose(poses_a[0].position_m, poses_b[0].position_m)
    assert poses_a[0].segment == "-y row" and poses_a[0].height_m == 1.0


def test_track_ga_seeded_with_the_grid_winner_is_never_worse_and_stays_on_the_track():
    """Plumbing check, in the spirit of the ring's `test_ga_search_is_at_least_
    as_good_as_a_coarse_grid_search`: a GA over (s, height) that starts from
    the exhaustive grid's own winner can only improve on it. That the *unseeded*
    GA finds the grid's optimum (the real cross-check, per objective) is a
    result of the experiment run, `viz/run_station_track_experiment.py`, which
    asserts it; it needs minutes, not a unit test's seconds. Three added
    stations, because one or two cannot make a feasible plan on this scene."""
    scenario = long_part_two_cluster_scenario()
    poses = track_candidate_poses(TRACK, 1.0, LONG_PART_GRID_HEIGHTS_M)
    grid = grid_search_station_placement(poses, 3, scenario, objective_a0_trace_covariance, prefilter_visibility=True)
    winner = [(p.s_m, p.height_m) for p in grid.best.station_poses]
    ga = track_genetic_algorithm_station_placement(
        scenario, TRACK, 3, objective_a0_trace_covariance, seed=0, maxiter=8, popsize=10, tol=1e-8, workers=1,
        seed_stations=[winner],
    )
    assert ga.best is not None
    assert ga.best_objective_value <= grid.best_objective_value * (1 + 1e-9)
    assert all(p.segment in ("-y row", "+x loop", "+y row", "-x loop") for p in ga.best.station_poses)
    assert all(0.8 <= p.height_m <= 3.0 for p in ga.best.station_poses)
    with pytest.raises(ValueError):
        track_genetic_algorithm_station_placement(scenario, TRACK, 3, objective_a0_trace_covariance,
                                                  orientation_mode="nonsense", maxiter=1)
