"""Tests for planning.candidates, planning.objectives, planning.search and
planning.experiment (CLAUDE.md §5, step 6 -- the headline experiment).

These check correctness/plumbing with small, fast candidate grids. The
actual scientific finding (do the four objectives differ, in which
direction, by how much) is run at full resolution separately and reported
in docs/step6_headline_experiment.md -- these tests are not where that
claim is made.
"""
import itertools
import pickle
from dataclasses import replace

import numpy as np
import pytest

from characteristics.characteristic import Characteristic
from characteristics.tolerances import ProfileTolerance
from geometry.mesh import TriangleMesh
from geometry.pose import InstrumentPose
from geometry.visibility import VisibilityLimits, check_target_visibility
from instruments.laser_tracker import LaserTracker
from network.solve import simulate_observations, solve_network
from planning.candidates import floor_grid_candidate_poses, hemisphere_candidate_poses, look_at_orientation
from planning.experiment import (
    default_two_cluster_scenario,
    iso_risk_station_counts,
    mixed_tolerance_two_cluster_scenario,
    run_headline_experiment,
    run_wing_experiment,
    wing_four_point_scenario,
)
from planning.objectives import (
    PlanningScenario,
    constrained_objective,
    evaluate_plan,
    make_weighted_uncertainty_objective,
    objective_a0_trace_covariance,
    objective_c_direct_risk,
    plan_infeasibility_deficit,
    risk_derived_weights,
    uniform_weights,
)
from planning.search import genetic_algorithm_station_placement, grid_search_station_placement
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior


def test_hemisphere_pose_at_zero_elevation_is_level_with_centre():
    centre = np.array([3.0, 0.0, 1.0])
    poses = hemisphere_candidate_poses(centre, radius_m=4.0, azimuths_deg=[0.0], elevations_deg=[0.0])
    assert poses[0].position_m[2] == pytest.approx(centre[2])  # same height
    assert np.linalg.norm(poses[0].position_m - centre) == pytest.approx(4.0)


def test_hemisphere_pose_at_90_elevation_is_directly_above_centre():
    centre = np.array([3.0, 0.0, 1.0])
    poses = hemisphere_candidate_poses(centre, radius_m=4.0, azimuths_deg=[0.0], elevations_deg=[90.0])
    position_m = poses[0].position_m
    assert position_m[2] == pytest.approx(centre[2] + 4.0)
    assert position_m[0] == pytest.approx(centre[0], abs=1e-9)
    assert position_m[1] == pytest.approx(centre[1], abs=1e-9)


def test_hemisphere_pose_boresight_points_at_centre():
    centre = np.array([3.0, 0.0, 1.0])
    poses = hemisphere_candidate_poses(centre, radius_m=4.0, azimuths_deg=[45.0], elevations_deg=[35.0])
    pose = poses[0]
    boresight = pose.orientation[:, 0]  # local +x, this codebase's boresight direction
    to_centre = (centre - pose.position_m)
    to_centre = to_centre / np.linalg.norm(to_centre)
    assert np.allclose(boresight, to_centre, atol=1e-9)


def test_hemisphere_candidate_count_is_azimuths_times_elevations():
    poses = hemisphere_candidate_poses(
        np.zeros(3), radius_m=4.0, azimuths_deg=[0, 90, 180], elevations_deg=[20, 40]
    )
    assert len(poses) == 6


def test_floor_grid_candidate_count_is_the_full_cartesian_product():
    poses = floor_grid_candidate_poses(
        aim_point_m=np.array([15.0, 0.0, 1.0]),
        x_positions_m=[0.0, 5.0, 10.0],
        y_positions_m=[-3.0, 3.0],
        station_heights_m=[1.0, 2.0],
    )
    assert len(poses) == 3 * 2 * 2


def test_floor_grid_candidates_are_boresighted_at_the_aim_point():
    aim_point_m = np.array([15.0, 2.0, 1.0])
    poses = floor_grid_candidate_poses(
        aim_point_m=aim_point_m, x_positions_m=[0.0], y_positions_m=[-3.0], station_heights_m=[1.5]
    )
    pose = poses[0]
    boresight = pose.orientation[:, 0]
    to_aim = aim_point_m - pose.position_m
    to_aim = to_aim / np.linalg.norm(to_aim)
    assert np.allclose(boresight, to_aim, atol=1e-9)


def test_floor_grid_exclusion_radius_drops_candidates_too_close_to_the_part():
    aim_point_m = np.array([0.0, 0.0, 1.0])
    poses = floor_grid_candidate_poses(
        aim_point_m=aim_point_m,
        x_positions_m=[0.0, 0.1, 5.0],
        y_positions_m=[0.0],
        station_heights_m=[1.0],
        exclusion_radius_m=1.0,
    )
    # (0, 0, 1) and (0.1, 0, 1) both sit within 1 m of the aim point and
    # must be dropped; only the (5, 0, 1) candidate should survive.
    assert len(poses) == 1
    assert poses[0].position_m[0] == pytest.approx(5.0)


def test_uniform_weights_are_all_one():
    scenario = default_two_cluster_scenario()
    weights = uniform_weights(scenario)
    assert np.allclose(weights, 1.0)
    assert len(weights) == len(scenario.characteristics)


def test_risk_derived_weights_favour_the_marginal_cluster():
    """The core premise the whole experiment rests on: a characteristic
    drawn from a high-capability process (near-zero risk sensitivity,
    step 5's gate) gets a smaller weight than one from a marginal
    process, at a shared, realistic reference uncertainty. Both clusters
    are now AS13006-legal (Cpk >= 1.33) -- see planning.experiment's
    module-level comment on why an earlier, less capable marginal
    cluster (Cpk ~ 0.56) overstated the effect -- so the contrast here is
    real but no longer an artificial near-lexicographic wipeout."""
    scenario = default_two_cluster_scenario()
    weights = risk_derived_weights(scenario, reference_uncertainty_m=8e-6)

    high_capability_weights = weights[:3]  # cluster1_profile_*
    marginal_weights = weights[3:]  # cluster2_profile_*

    assert np.allclose(high_capability_weights, high_capability_weights[0])  # identical within a cluster
    assert np.allclose(marginal_weights, marginal_weights[0])
    assert marginal_weights[0] > high_capability_weights[0]
    assert weights.mean() == pytest.approx(1.0)  # normalised


def test_evaluate_plan_design_covariance_matches_a_noisy_solve():
    """The shortcut evaluate_plan relies on (design covariance from exact
    observations at truth) should agree with a real noisy solve's
    covariance at the same geometry -- the empirical check
    planning.objectives' docstring refers to, pinned down as a test."""
    scenario = default_two_cluster_scenario()
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    evaluation = evaluate_plan(station_poses, scenario)

    all_poses = [scenario.anchor_pose] + station_poses
    rng = np.random.default_rng(0)
    noisy_observations = simulate_observations(scenario.target_points_m, all_poses, scenario.tracker, rng)
    noisy_result = solve_network(
        noisy_observations, scenario.target_points_m.copy(), all_poses, scenario.tracker, anchor_index=0
    )

    assert np.allclose(
        evaluation.network_solve.target_covariance_m2, noisy_result.target_covariance_m2, rtol=0.05
    )


def test_visibility_disabled_scenario_is_unaffected_by_the_new_fields():
    """`PlanningScenario.visibility_enabled` is False whenever a scenario is
    built the way every scenario before this field existed was built
    (`obstacles=None`, `target_normals_m=None`) -- the fast path
    `evaluate_plan` always took. This is the backward-compatibility
    contract the rest of the occlusion wiring depends on."""
    scenario = default_two_cluster_scenario()
    assert not scenario.visibility_enabled
    assert scenario.obstacles is None
    assert scenario.visibility_limits is None


def test_visibility_limits_out_of_range_disqualifies_the_whole_plan():
    """A `visibility_limits.max_range_m` tighter than the distance to
    every target (no mesh needed) leaves every station -- including the
    anchor -- seeing nothing. `evaluate_plan` must refuse to solve this
    rather than hand back a covariance computed from zero observations
    (see `_visible_observations`'s docstring on why `np.linalg.pinv`
    would otherwise silently report a *perfect*, not infinite,
    uncertainty for a target nothing actually measured)."""
    scenario = replace(
        default_two_cluster_scenario(),
        visibility_limits=VisibilityLimits(max_range_m=0.5, max_incidence_angle_rad=np.pi / 2),
    )
    station_poses = [InstrumentPose(position_m=np.array([0.0, 4.0, 3.0]))]
    assert evaluate_plan(station_poses, scenario) is None


def test_visibility_limits_permissive_matches_the_no_visibility_fast_path():
    """Switching the visibility layer on with a range/incidence envelope
    generous enough that nothing is ever out of range or occluded must
    reproduce the exact same result as leaving it off entirely -- the
    mechanism should only ever change an outcome when it actually binds."""
    baseline_scenario = default_two_cluster_scenario()
    permissive_scenario = replace(
        baseline_scenario,
        visibility_limits=VisibilityLimits(max_range_m=1000.0, max_incidence_angle_rad=np.pi / 2),
    )
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    baseline = evaluate_plan(station_poses, baseline_scenario)
    permissive = evaluate_plan(station_poses, permissive_scenario)
    assert permissive is not None
    assert np.allclose(
        baseline.network_solve.target_covariance_m2, permissive.network_solve.target_covariance_m2
    )


def _wall_occlusion_scenario(obstacles) -> PlanningScenario:
    """A minimal hand-placed scene for exercising mesh occlusion plumbing:
    4 targets, an anchor sitting on the far side of a wall from a
    candidate station so the geometry can put the wall in the path of
    exactly one station-target pair (candidate -> target 3) and no
    others -- see the calling tests for the worked-out ray/wall
    intersection each coordinate was chosen to produce."""
    target_points_m = np.array(
        [
            [3.0, 0.0, 1.0],
            [3.2, 0.3, 1.2],
            [2.8, -0.3, 1.1],  # not collinear with targets 0 and 1: three collinear shared
            #                    targets leave the candidate free to roll about their line
            [3.0, 0.0, 2.0],
        ]
    )
    surface_normal = np.array([0.0, 0.0, 1.0])
    characteristics = [
        Characteristic(
            name=f"pt{i}",
            target_indices=[i],
            tolerance=ProfileTolerance(zone_width_m=1e-4, surface_normal=surface_normal),
        )
        for i in range(4)
    ]
    cluster = FeatureCluster("cluster", ProcessPrior(mean_m=0.0, std_m=5e-6))
    assignments = {c.name: ClusterAssignment(c.name, cluster, confidence=0.9) for c in characteristics}
    decision_rules = {c.name: DecisionRule.from_zone_width(1e-4) for c in characteristics}
    return PlanningScenario(
        target_points_m=target_points_m,
        tracker=LaserTracker(),
        characteristics=characteristics,
        assignments=assignments,
        decision_rules=decision_rules,
        anchor_pose=InstrumentPose(position_m=np.array([6.0, 0.0, 2.0])),
        obstacles=obstacles,
        visibility_limits=VisibilityLimits(max_range_m=10.0, max_incidence_angle_rad=np.pi / 2),
    )


def test_mesh_obstacle_filters_one_station_target_pair_but_plan_stays_feasible():
    """A wall between the candidate station and target 3 only (the anchor,
    on the far side, has an unobstructed line to every target) must leave
    the plan feasible -- target 3 is still seen by the anchor, and the
    candidate still sees the other 3 targets (meeting
    `MIN_TARGETS_PER_NON_ANCHOR_STATION`) -- while making target 3's own
    uncertainty measurably worse than it would be with the wall absent,
    since it loses one of its two observations."""
    wall = TriangleMesh(
        vertices_m=np.array(
            [
                [1.5, -2.0, 2.3],
                [1.5, 2.0, 2.3],
                [1.5, 2.0, 2.7],
                [1.5, -2.0, 2.7],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    candidate = InstrumentPose(position_m=np.array([0.0, 0.0, 3.0]))

    occluded = evaluate_plan([candidate], _wall_occlusion_scenario(obstacles=wall))
    clear = evaluate_plan([candidate], _wall_occlusion_scenario(obstacles=None))

    assert occluded is not None
    target3_trace_occluded = np.trace(occluded.network_solve.target_point_covariance(3))
    target3_trace_clear = np.trace(clear.network_solve.target_point_covariance(3))
    assert target3_trace_occluded > target3_trace_clear
    # Targets 0-2's own uncertainty also moves (not held exactly fixed):
    # with only two stations total, losing one observation of target 3
    # leaves the candidate's own 6-DOF pose less well constrained, and
    # every target's recovered position shares that same station-pose
    # uncertainty in a joint network solve (CLAUDE.md §5 step 3 -- this
    # is genuine cross-target correlation, not a test artefact). The wall
    # not being in those targets' own sightlines means this is an
    # indirect, shared-pose effect rather than a direct one.


def test_grid_search_reports_n_infeasible_and_avoids_disqualified_candidates():
    """A candidate placed far enough away to see nothing at all must be
    disqualified outright -- not scored, not chosen -- and counted in
    `GridSearchResult.n_infeasible` (CLAUDE.md §8: a null/negative result,
    here "this candidate cannot be used", reported rather than silently
    dropped)."""
    scenario = replace(
        default_two_cluster_scenario(),
        visibility_limits=VisibilityLimits(max_range_m=10.0, max_incidence_angle_rad=np.pi / 2),
    )
    centre_m = scenario.target_points_m.mean(axis=0)
    near_candidate = hemisphere_candidate_poses(
        centre_m, radius_m=4.0, azimuths_deg=[0.0], elevations_deg=[0.0]
    )[0]
    far_candidate = InstrumentPose(position_m=np.array([1000.0, 0.0, 1.0]))

    result = grid_search_station_placement(
        [near_candidate, far_candidate], n_stations=1, scenario=scenario, objective_fn=objective_a0_trace_covariance
    )

    assert result.n_infeasible == 1
    assert len(result.all_evaluations) == 1
    assert result.best is not None
    assert np.allclose(result.best.station_poses[0].position_m, near_candidate.position_m)


def test_weighted_uncertainty_objective_matches_hand_computation():
    scenario = default_two_cluster_scenario()
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    weights = np.array([1.0, 2.0, 0.5, 1.0, 1.0, 1.0])
    evaluation = evaluate_plan(station_poses, scenario)
    objective = make_weighted_uncertainty_objective(weights)

    hand_computed = sum(
        w * evaluation.per_characteristic_uncertainty_m[c.name]
        for w, c in zip(weights, scenario.characteristics)
    )
    assert objective(evaluation, scenario) == pytest.approx(hand_computed)


def test_objective_a0_sums_full_covariance_trace_not_projected_uncertainty():
    """A0 (the field's actual criterion) must differ, in general, from the
    projected-uncertainty objectives -- it's blind to tolerance direction
    entirely (CLAUDE.md referee note M3)."""
    scenario = default_two_cluster_scenario()
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    evaluation = evaluate_plan(station_poses, scenario)

    a0_value = objective_a0_trace_covariance(evaluation, scenario)
    hand_computed = sum(
        np.trace(evaluation.network_solve.target_point_covariance(i)) for i in range(6)
    )
    assert a0_value == pytest.approx(hand_computed)

    a1_value = make_weighted_uncertainty_objective(uniform_weights(scenario))(evaluation, scenario)
    # A0 sums variances (trace of full 3x3 covariance); A1 sums
    # projected (1-direction) standard deviations -- different
    # quantities, not expected to coincide.
    assert a0_value != pytest.approx(a1_value)


def test_objective_c_matches_direct_hand_computed_risk():
    from risk.jcgm106 import evaluate_conformity_risk

    scenario = default_two_cluster_scenario()
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    evaluation = evaluate_plan(station_poses, scenario)

    hand_computed = 0.0
    for characteristic in scenario.characteristics:
        assignment = scenario.assignments[characteristic.name]
        decision_rule = scenario.decision_rules[characteristic.name]
        u = evaluation.per_characteristic_uncertainty_m[characteristic.name]
        r = evaluate_conformity_risk(assignment, u, decision_rule)
        hand_computed += r.probability_false_acceptance + r.probability_false_rejection

    assert objective_c_direct_risk(evaluation, scenario) == pytest.approx(hand_computed)


def test_constrained_objective_disqualifies_violating_candidates():
    scenario = default_two_cluster_scenario()
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    evaluation = evaluate_plan(station_poses, scenario)
    base_objective = make_weighted_uncertainty_objective(uniform_weights(scenario))
    unconstrained_value = base_objective(evaluation, scenario)

    max_uncertainty_m = max(evaluation.per_characteristic_uncertainty_m.values()) - 1e-9
    constrained = constrained_objective(base_objective, max_uncertainty_m)
    assert constrained(evaluation, scenario) == float("inf")

    generous = constrained_objective(base_objective, max_uncertainty_m * 100)
    assert generous(evaluation, scenario) == pytest.approx(unconstrained_value)

    # None applies no cap at all.
    assert constrained_objective(base_objective, None)(evaluation, scenario) == pytest.approx(unconstrained_value)


def test_grid_search_finds_the_true_minimum_by_brute_force_cross_check():
    scenario = default_two_cluster_scenario()
    centre = scenario.target_points_m.mean(axis=0)
    candidates = hemisphere_candidate_poses(centre, radius_m=4.0, azimuths_deg=[0, 90, 180, 270], elevations_deg=[30])
    objective = make_weighted_uncertainty_objective(uniform_weights(scenario))

    result = grid_search_station_placement(candidates, n_stations=2, scenario=scenario, objective_fn=objective)

    # Independently re-evaluate every combination and confirm nothing beats "best".
    for i, j in itertools.combinations(range(len(candidates)), 2):
        evaluation = evaluate_plan([candidates[i], candidates[j]], scenario)
        assert objective(evaluation, scenario) >= result.best_objective_value - 1e-15

    n_combinations = len(list(itertools.combinations(range(len(candidates)), 2)))
    assert len(result.all_evaluations) == n_combinations


def test_grid_search_rejects_more_stations_than_candidates():
    scenario = default_two_cluster_scenario()
    candidates = hemisphere_candidate_poses(np.zeros(3), 4.0, [0, 90], [30])
    objective = make_weighted_uncertainty_objective(uniform_weights(scenario))
    with pytest.raises(ValueError):
        grid_search_station_placement(candidates, n_stations=5, scenario=scenario, objective_fn=objective)


def test_grid_search_returns_none_when_every_candidate_is_disqualified():
    scenario = default_two_cluster_scenario()
    candidates = hemisphere_candidate_poses(np.zeros(3), 4.0, [0, 90], [30])
    objective = make_weighted_uncertainty_objective(uniform_weights(scenario))
    impossible = constrained_objective(objective, max_uncertainty_m=1e-12)  # nothing can satisfy this
    result = grid_search_station_placement(candidates, n_stations=2, scenario=scenario, objective_fn=impossible)
    assert result.best is None
    assert result.best_objective_value is None
    assert len(result.all_evaluations) > 0  # still evaluated everything


def test_headline_experiment_each_plan_wins_its_own_objective():
    """Optimizer-correctness invariant, true regardless of scene tuning:
    a plan found by minimising a given objective cannot be beaten, under
    that same objective, by a plan found under a different one -- each
    grid search found the actual best of its own candidate set for its
    own objective."""
    result = run_headline_experiment(
        azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30], n_stations=2
    )
    scenario = result.scenario

    objectives_and_plans = [
        (objective_a0_trace_covariance, result.plan_a0),
        (make_weighted_uncertainty_objective(uniform_weights(scenario)), result.plan_a1),
        (make_weighted_uncertainty_objective(result.weights_risk), result.plan_b),
        (objective_c_direct_risk, result.plan_c),
    ]

    for objective_fn, own_plan in objectives_and_plans:
        own_evaluation = evaluate_plan(own_plan.station_poses, scenario)
        own_value = objective_fn(own_evaluation, scenario)
        for _, other_plan in objectives_and_plans:
            other_evaluation = evaluate_plan(other_plan.station_poses, scenario)
            assert own_value <= objective_fn(other_evaluation, scenario) + 1e-9


def test_mixed_tolerance_scenario_can_separate_a0_from_a1():
    """`default_two_cluster_scenario`'s A0 (trace) and A1 (uniform-
    projected) land on the *identical* plan -- documented in
    docs/step6_headline_experiment.md as a property of that scene's
    symmetry (all six characteristics share one tolerance type, one
    projected direction), not a general equivalence, with "a scene with
    mixed tolerance types" named there as the natural next check.

    This is that check. `mixed_tolerance_two_cluster_scenario` gives each
    cluster one `PositionTolerance` (spherical -- A1 sees the whole
    covariance here, same as A0 always does), one `ProfileTolerance`
    (one direction, the surface normal), and one `ParallelismTolerance`
    (one direction, an external datum unrelated to the surface normal) --
    CLAUDE.md §3's own position-vs-parallelism example, built directly
    into one scene. A0 stays blind to all of this (it only ever sees each
    point's full trace); A1 does not (each tolerance type keeps only the
    slice of the covariance its own direction(s) constrain) -- so A0 and
    A1 now have a genuine reason to rank candidates differently, not just
    a hypothetical one.

    At this scene's full candidate resolution (the default 24-candidate,
    12 azimuth x 2 elevation grid `run_headline_experiment` uses) A0 and
    A1 still happen to land on the same plan -- not every mixed scene
    separates them, and that's reported honestly in docs/
    step6_headline_experiment.md alongside this result, not hidden. At a
    coarser candidate grid (fewer alternatives for a close-but-different
    ranking to lose to) they do separate, which is what this test checks:
    a concrete, reproducible scene and resolution where A0 != A1 is no
    longer just a theoretical possibility."""
    scenario = mixed_tolerance_two_cluster_scenario()
    result = run_headline_experiment(
        scenario, azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30], n_stations=2
    )

    def same_station_poses(plan_x, plan_y):
        return all(
            np.allclose(pose_x.position_m, pose_y.position_m)
            for pose_x, pose_y in zip(plan_x.station_poses, plan_y.station_poses)
        )

    assert not same_station_poses(result.plan_a0, result.plan_a1), (
        "expected mixed tolerance types to give A0 and A1 a genuine reason to "
        "disagree at this candidate resolution -- if this now fails, either "
        "the scene or the candidate grid changed in a way that restored the "
        "coincidence, which is itself worth noting in docs/"
        "step6_headline_experiment.md, not silently accepting"
    )
    # B and C can still legitimately agree or disagree independently of
    # A0/A1 -- not asserted either way here, since (per the module
    # docstring) whether the linearised weight (B) matches the direct
    # nonlinear objective (C) is itself the open empirical question the
    # ablation exists to answer, not something this test should presume.


def test_headline_experiment_reports_required_fields():
    """CLAUDE.md §5 step 6: report resulting station positions, summed
    characteristic uncertainty, global PFA, global PFR, and
    per-characteristic uncertainty, for every plan."""
    result = run_headline_experiment(azimuths_deg=[0, 120, 240], elevations_deg=[30], n_stations=2)
    for plan in (result.plan_a0, result.plan_a1, result.plan_b, result.plan_c):
        assert len(plan.station_poses) == 2
        assert plan.summed_characteristic_uncertainty_m > 0
        assert plan.global_pfa >= 0
        assert plan.global_pfr >= 0
        assert 0.0 <= plan.bounded_part_level_risk <= 1.0
        assert len(plan.per_characteristic_uncertainty_m) == len(result.scenario.characteristics)


def test_headline_experiment_respects_max_uncertainty_constraint():
    result_unconstrained = run_headline_experiment(
        azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30], n_stations=2
    )
    loosest_uncertainty_m = max(result_unconstrained.plan_b.per_characteristic_uncertainty_m.values())
    cap = loosest_uncertainty_m * 0.9  # tighter than what B picked unconstrained

    result_constrained = run_headline_experiment(
        azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30], n_stations=2, max_uncertainty_m=cap
    )
    assert max(result_constrained.plan_b.per_characteristic_uncertainty_m.values()) <= cap + 1e-9


def test_iso_risk_station_counts_reports_smallest_sufficient_station_count():
    scenario = default_two_cluster_scenario()
    objective_fns = {
        "A1": make_weighted_uncertainty_objective(uniform_weights(scenario)),
        "C": objective_c_direct_risk,
    }
    # A generous target every objective should clear at 1 station already,
    # given this scenario's process priors -- exercises the "found early"
    # path without a slow multi-station search.
    counts = iso_risk_station_counts(
        scenario, target_total_risk=10.0, objective_fns_by_name=objective_fns,
        azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30], max_stations=2,
    )
    assert counts["A1"] == 1
    assert counts["C"] == 1


def test_iso_risk_station_counts_returns_none_when_unreachable():
    scenario = default_two_cluster_scenario()
    objective_fns = {"A1": make_weighted_uncertainty_objective(uniform_weights(scenario))}
    counts = iso_risk_station_counts(
        scenario, target_total_risk=-1.0,  # impossible: risk can't be negative
        objective_fns_by_name=objective_fns,
        azimuths_deg=[0, 180], elevations_deg=[0], max_stations=1,
    )
    assert counts["A1"] is None


# Small, fast differential-evolution budgets throughout this section -- these
# tests check plumbing (does it run, does it respect visibility, is it at
# least competitive with grid search), not GA convergence quality, which
# docs/step6_headline_experiment.md's full-resolution runs are what actually
# demonstrates.
_FAST_DE_KWARGS = dict(seed=0, maxiter=15, popsize=8, tol=1e-4)


def test_ga_search_is_at_least_as_good_as_a_coarse_grid_search():
    """A continuous search over the same space a coarse hemisphere grid
    discretises should find a plan at least as good -- it can always fall
    back to (close to) the grid's own candidates, plus everywhere between
    them the grid never tried."""
    scenario = default_two_cluster_scenario()
    centre_m = scenario.target_points_m.mean(axis=0)
    candidates = hemisphere_candidate_poses(centre_m, radius_m=4.0, azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30])
    grid_result = grid_search_station_placement(candidates, 2, scenario, objective_a0_trace_covariance)

    ga_result = genetic_algorithm_station_placement(
        scenario, n_stations=2, objective_fn=objective_a0_trace_covariance,
        bounds_m=[(-2.0, 8.0), (-6.0, 6.0), (0.0, 6.0)], **_FAST_DE_KWARGS,
    )

    assert ga_result.best is not None
    assert ga_result.best_objective_value <= grid_result.best_objective_value * 1.05  # 5% slack for DE noise


def test_ga_search_disqualifies_out_of_range_positions():
    """Shrinking bounds_m so every reachable position is out of the
    tracker's visibility range must leave the GA unable to find a feasible
    plan at all -- best=None, not a silently-wrong solved covariance."""
    scenario = replace(
        default_two_cluster_scenario(),
        visibility_limits=VisibilityLimits(max_range_m=0.1, max_incidence_angle_rad=np.pi / 2),
    )
    result = genetic_algorithm_station_placement(
        scenario, n_stations=1, objective_fn=objective_a0_trace_covariance,
        bounds_m=[(-2.0, 8.0), (-6.0, 6.0), (0.0, 6.0)], **_FAST_DE_KWARGS,
    )
    assert result.best is None
    assert result.best_objective_value is None
    assert result.n_infeasible_evals == result.n_function_evals


def test_ga_search_seeded_population_finds_the_seeded_optimum():
    """Seeding with the true optimum (found by a prior, cheap grid search)
    should let the GA immediately report that same value even with a
    tiny budget that would otherwise have no realistic chance of finding
    it blind -- the direct test that seeding actually biases the search
    rather than being silently ignored."""
    scenario = default_two_cluster_scenario()
    centre_m = scenario.target_points_m.mean(axis=0)
    candidates = hemisphere_candidate_poses(centre_m, radius_m=4.0, azimuths_deg=[0, 90, 180, 270], elevations_deg=[0, 30])
    grid_result = grid_search_station_placement(candidates, 2, scenario, objective_a0_trace_covariance)
    seed_positions = [p.position_m for p in grid_result.best.station_poses]

    result = genetic_algorithm_station_placement(
        scenario, n_stations=2, objective_fn=objective_a0_trace_covariance,
        bounds_m=[(-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0)],  # huge box a blind search has little hope in
        seed=0, maxiter=1, popsize=4, tol=1e-2,
        seed_station_positions_m=[seed_positions],
    )
    assert result.best_objective_value <= grid_result.best_objective_value * 1.01


def test_ga_search_rejects_too_many_seeds():
    scenario = default_two_cluster_scenario()
    too_many_seeds = [[np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])] for _ in range(1000)]
    with pytest.raises(ValueError):
        genetic_algorithm_station_placement(
            scenario, n_stations=2, objective_fn=objective_a0_trace_covariance,
            bounds_m=[(-2.0, 8.0), (-6.0, 6.0), (0.0, 6.0)],
            popsize=4, maxiter=1, seed_station_positions_m=too_many_seeds,
        )


def test_weighted_and_constrained_objectives_survive_a_pickle_round_trip():
    """genetic_algorithm_station_placement's workers != 1 path sends the
    whole ObjectiveFn to a multiprocessing worker process via pickle --
    a plain closure cannot survive that (pickle has no way to name a
    function defined inside another function), which is exactly why
    make_weighted_uncertainty_objective and constrained_objective return
    plain callable classes instead. This test is the cheap, deterministic
    way to check that contract without actually spinning up a worker
    pool (test_ga_search_runs_in_parallel_and_reports_zero_infeasible_count
    below does that, once, as an integration check)."""
    scenario = default_two_cluster_scenario()
    weighted = make_weighted_uncertainty_objective(uniform_weights(scenario))
    capped = constrained_objective(weighted, max_uncertainty_m=5e-5)

    restored_weighted = pickle.loads(pickle.dumps(weighted))
    restored_capped = pickle.loads(pickle.dumps(capped))

    station_poses = [InstrumentPose(position_m=np.array([0.0, 4.0, 3.0]))]
    evaluation = evaluate_plan(station_poses, scenario)
    assert restored_weighted(evaluation, scenario) == pytest.approx(weighted(evaluation, scenario))
    assert restored_capped(evaluation, scenario) == pytest.approx(capped(evaluation, scenario))


def test_ga_search_runs_in_parallel_and_reports_zero_infeasible_count():
    """workers=2 must run to completion (the real integration check that
    the previous test's pickling guarantee is actually sufficient for
    scipy.optimize.differential_evolution's own multiprocessing path),
    and n_infeasible_evals must read 0 -- not a claim nothing was
    infeasible, but the documented, honest limitation that a live count
    cannot be kept accurate once evaluations are split across worker
    processes (GASearchResult.n_infeasible_evals's own docstring)."""
    scenario = default_two_cluster_scenario()
    result = genetic_algorithm_station_placement(
        scenario, n_stations=1, objective_fn=objective_a0_trace_covariance,
        bounds_m=[(-2.0, 8.0), (-6.0, 6.0), (0.0, 6.0)],
        seed=0, maxiter=3, popsize=4, tol=1e-2, workers=2,
    )
    assert result.best is not None
    assert result.n_infeasible_evals == 0


def test_ga_search_boresights_every_station_at_the_aim_point():
    """A direct check that the position-only encoding genetic_algorithm_
    station_placement searches over is turned into full poses the same
    way the candidate-list generators do -- boresighted at the aim point,
    via planning.candidates.look_at_orientation."""
    scenario = default_two_cluster_scenario()
    aim_point_m = np.array([3.0, 0.0, 1.0])
    result = genetic_algorithm_station_placement(
        scenario, n_stations=1, objective_fn=objective_a0_trace_covariance,
        bounds_m=[(-2.0, 8.0), (-6.0, 6.0), (0.0, 6.0)], aim_point_m=aim_point_m, **_FAST_DE_KWARGS,
    )
    pose = result.best.station_poses[0]
    expected_orientation = look_at_orientation(pose.position_m, aim_point_m)
    assert np.allclose(pose.orientation, expected_orientation)


def test_wing_scenario_has_four_characteristics_and_eight_targets():
    """4 characteristics (root/midspan/neartip/tip) plus 4 unassigned
    network tie points -- see wing_four_point_scenario's own docstring
    for why the tie points exist."""
    scenario = wing_four_point_scenario()
    assert len(scenario.characteristics) == 4
    assert scenario.target_points_m.shape[0] == 8
    assert scenario.visibility_enabled
    characteristic_indices = {c.target_indices[0] for c in scenario.characteristics}
    assert characteristic_indices == {0, 3, 6, 7}


def test_wing_scenario_support_tower_reduces_but_does_not_eliminate_visibility():
    """The support tower (include_support_tower=True, the default) must
    make a real, measurable difference to how many station-target pairs
    are visible across a representative candidate set -- not zero
    (CLAUDE.md §8: a null result would need to be reported honestly, but
    this is not meant to be one), and not so severe that some target
    becomes unseeable from every candidate (a modelling artefact, not a
    finding -- see the midspan-target placement comment in
    wing_four_point_scenario)."""
    scenario_towered = wing_four_point_scenario(include_support_tower=True)
    scenario_clear = wing_four_point_scenario(include_support_tower=False)
    aim_point_m = scenario_towered.target_points_m.mean(axis=0)
    candidates = floor_grid_candidate_poses(
        aim_point_m=aim_point_m,
        x_positions_m=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        y_positions_m=[-6.0, -3.0, 0.0, 3.0, 6.0],
        station_heights_m=[3.0, 5.0, 7.0],
        exclusion_radius_m=1.0,
    )

    def per_target_coverage(scenario):
        coverage = np.zeros(scenario.target_points_m.shape[0], dtype=int)
        for pose in candidates:
            for i in range(scenario.target_points_m.shape[0]):
                visible = check_target_visibility(
                    pose.position_m,
                    scenario.target_points_m[i],
                    scenario.visibility_limits,
                    target_normal_m=scenario.target_normals_m[i],
                    obstacles=scenario.obstacles,
                ).is_visible
                coverage[i] += int(visible)
        return coverage

    coverage_towered = per_target_coverage(scenario_towered)
    coverage_clear = per_target_coverage(scenario_clear)

    assert coverage_towered.sum() < coverage_clear.sum()  # the tower costs real visibility...
    assert np.all(coverage_towered > 0)  # ...but never eliminates a target entirely


def test_a_station_tied_to_the_network_by_one_shared_target_is_disqualified():
    """Regression for a real defect: the first wing-scenario experiment's
    reported plans (this is its objective-A1, no-tower result) all had a
    station seeing only targets {5, 6, 7}, sharing exactly one target (5)
    with the rest. One shared point cannot pin a station's pose -- it is
    free to rotate about it -- but the old rule only asked each station to
    see >= 3 targets of any kind, and solve_network's pinv assigned the
    unconstrained direction zero variance, so the plan scored as excellent.
    It must now be rejected, with a positive graded deficit."""
    scenario = wing_four_point_scenario(include_support_tower=False)
    aim_point_m = scenario.target_points_m.mean(axis=0)
    reported_stations_m = [
        [2.286908128143982, -0.15140443388573377, 5.379420159054965],
        [15.067035198174558, 3.05741684795127, 3.205144396893839],
        [26.518909860831247, 0.24392045457679304, 2.846858864831216],
    ]
    poses = [InstrumentPose(np.array(p), look_at_orientation(np.array(p), aim_point_m)) for p in reported_stations_m]
    assert evaluate_plan(poses, scenario) is None
    assert plan_infeasibility_deficit(poses, scenario) > 0.0


def test_run_wing_experiment_wires_four_objectives_with_cascaded_seeding():
    """Not a test of search quality -- docs/step6_headline_experiment.md's
    own validated run (60 generations, population 20, workers=-1) is
    where that claim is made and checked. This just confirms the four
    objectives, the cascaded seeding between them, and
    WingExperimentResult's reporting properties all wire together
    correctly, at a tiny budget fast enough for the test suite."""
    result = run_wing_experiment(
        n_stations=1, de_kwargs=dict(maxiter=1, popsize=4, tol=1e-1, workers=1),
    )
    assert result.scenario is not None
    for plan in (result.plan_a0, result.plan_a1, result.plan_b, result.plan_c):
        if plan is not None:
            assert isinstance(plan.summed_characteristic_uncertainty_m, float)


def test_wing_scenario_ga_search_finds_a_multi_station_plan():
    """End-to-end smoke test: occlusion (box_mesh), the floor-grid-style
    continuous search space, and the GA search all have to work together
    to plan this scene at all -- a grid search over a candidate set fine
    enough to matter here would be combinatorially infeasible (the
    reason genetic_algorithm_station_placement exists in the first
    place, per planning.search's module docstring)."""
    scenario = wing_four_point_scenario()
    result = genetic_algorithm_station_placement(
        scenario, n_stations=2, objective_fn=objective_a0_trace_covariance,
        bounds_m=[(-2.0, 34.0), (-8.0, 8.0), (1.0, 10.0)],
        seed=0, maxiter=6, popsize=6, tol=1e-3,
    )
    assert result.best is not None
    assert result.best_objective_value is not None and result.best_objective_value > 0.0
