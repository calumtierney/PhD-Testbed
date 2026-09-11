"""Tests for planning.candidates, planning.objectives, planning.search and
planning.experiment (CLAUDE.md §5, step 6 -- the headline experiment).

These check correctness/plumbing with small, fast candidate grids. The
actual scientific finding (do the four objectives differ, in which
direction, by how much) is run at full resolution separately and reported
in docs/step6_headline_experiment.md -- these tests are not where that
claim is made.
"""
import itertools

import numpy as np
import pytest

from geometry.pose import InstrumentPose
from network.solve import simulate_observations, solve_network
from planning.candidates import hemisphere_candidate_poses
from planning.experiment import default_two_cluster_scenario, iso_risk_station_counts, run_headline_experiment
from planning.objectives import (
    constrained_objective,
    evaluate_plan,
    make_weighted_uncertainty_objective,
    objective_a0_trace_covariance,
    objective_c_direct_risk,
    risk_derived_weights,
    uniform_weights,
)
from planning.search import grid_search_station_placement


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
