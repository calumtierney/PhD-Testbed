"""Tests for planning.candidates, planning.objectives, planning.search and
planning.experiment (CLAUDE.md §5, step 6 -- the headline experiment).

These check correctness/plumbing with small, fast candidate grids. The
actual scientific finding (does the risk-weighted plan differ from the
uniform plan, in which direction, by how much) is run at full resolution
separately and reported in docs/step6_headline_experiment.md -- these
tests are not where that claim is made.
"""
import numpy as np
import pytest

from geometry.pose import InstrumentPose
from network.solve import exact_observations, simulate_observations, solve_network
from planning.candidates import hemisphere_candidate_poses
from planning.experiment import default_two_cluster_scenario, run_headline_experiment
from planning.objectives import evaluate_plan, risk_derived_weights, uniform_weights
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


def test_risk_derived_weights_favour_the_marginal_cluster_by_orders_of_magnitude():
    """The core premise the whole experiment rests on: a characteristic
    drawn from a high-capability process (near-zero risk sensitivity,
    step 5's gate) gets a far smaller weight than one from a marginal
    process, at a shared, realistic reference uncertainty."""
    scenario = default_two_cluster_scenario()
    weights = risk_derived_weights(scenario, reference_uncertainty_m=8e-6)

    high_capability_weights = weights[:3]  # cluster1_flatness_*
    marginal_weights = weights[3:]  # cluster2_flatness_*

    assert np.allclose(high_capability_weights, high_capability_weights[0])  # identical within a cluster
    assert np.allclose(marginal_weights, marginal_weights[0])
    assert marginal_weights[0] > 1000 * high_capability_weights[0]
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
    weights = uniform_weights(scenario)
    evaluation = evaluate_plan(station_poses, scenario, weights)

    all_poses = [scenario.anchor_pose] + station_poses
    rng = np.random.default_rng(0)
    noisy_observations = simulate_observations(scenario.target_points_m, all_poses, scenario.tracker, rng)
    noisy_result = solve_network(
        noisy_observations, scenario.target_points_m.copy(), all_poses, scenario.tracker, anchor_index=0
    )

    assert np.allclose(
        evaluation.network_solve.target_covariance_m2, noisy_result.target_covariance_m2, rtol=0.05
    )


def test_evaluate_plan_weighted_objective_matches_hand_computation():
    scenario = default_two_cluster_scenario()
    station_poses = [
        InstrumentPose(position_m=np.array([0.0, 4.0, 3.0])),
        InstrumentPose(position_m=np.array([6.0, -2.0, 2.0])),
    ]
    weights = np.array([1.0, 2.0, 0.5, 1.0, 1.0, 1.0])
    evaluation = evaluate_plan(station_poses, scenario, weights)

    hand_computed = sum(
        w * evaluation.per_characteristic_uncertainty_m[c.name]
        for w, c in zip(weights, scenario.characteristics)
    )
    assert evaluation.weighted_objective == pytest.approx(hand_computed)


def test_grid_search_finds_the_true_minimum_by_brute_force_cross_check():
    scenario = default_two_cluster_scenario()
    centre = scenario.target_points_m.mean(axis=0)
    candidates = hemisphere_candidate_poses(centre, radius_m=4.0, azimuths_deg=[0, 90, 180, 270], elevations_deg=[30])
    weights = uniform_weights(scenario)

    result = grid_search_station_placement(candidates, n_stations=2, scenario=scenario, weights=weights)

    # Independently re-evaluate every combination and confirm nothing beats "best".
    import itertools

    for i, j in itertools.combinations(range(len(candidates)), 2):
        evaluation = evaluate_plan([candidates[i], candidates[j]], scenario, weights)
        assert evaluation.weighted_objective >= result.best.weighted_objective - 1e-15

    n_combinations = len(list(itertools.combinations(range(len(candidates)), 2)))
    assert len(result.all_evaluations) == n_combinations


def test_grid_search_rejects_more_stations_than_candidates():
    scenario = default_two_cluster_scenario()
    candidates = hemisphere_candidate_poses(np.zeros(3), 4.0, [0, 90], [30])
    with pytest.raises(ValueError):
        grid_search_station_placement(candidates, n_stations=5, scenario=scenario, weights=uniform_weights(scenario))


def test_headline_experiment_each_plan_wins_its_own_objective():
    """Optimizer-correctness invariant, true regardless of scene tuning:
    plan A (found by minimising the uniform-weighted objective) cannot be
    beaten by plan B under the uniform weights, and vice versa for the
    risk weights -- each grid search found the actual best of its own
    candidate set for its own objective."""
    result = run_headline_experiment(
        azimuths_deg=[0, 90, 180, 270], elevations_deg=[20, 50], n_stations=2
    )
    scenario = result.scenario
    weights_uniform = uniform_weights(scenario)

    plan_a_uniform_objective = sum(
        w * result.plan_a.per_characteristic_uncertainty_m[c.name]
        for w, c in zip(weights_uniform, scenario.characteristics)
    )
    plan_b_uniform_objective = sum(
        w * result.plan_b.per_characteristic_uncertainty_m[c.name]
        for w, c in zip(weights_uniform, scenario.characteristics)
    )
    assert plan_a_uniform_objective <= plan_b_uniform_objective + 1e-12

    plan_a_risk_objective = sum(
        w * result.plan_a.per_characteristic_uncertainty_m[c.name]
        for w, c in zip(result.weights_risk, scenario.characteristics)
    )
    plan_b_risk_objective = sum(
        w * result.plan_b.per_characteristic_uncertainty_m[c.name]
        for w, c in zip(result.weights_risk, scenario.characteristics)
    )
    assert plan_b_risk_objective <= plan_a_risk_objective + 1e-12


def test_headline_experiment_reports_required_fields():
    """CLAUDE.md §5 step 6: report resulting station positions, total
    inspection effort, global PFA, global PFR, and per-characteristic
    uncertainty, for both plans."""
    result = run_headline_experiment(azimuths_deg=[0, 120, 240], elevations_deg=[30], n_stations=2)
    for plan in (result.plan_a, result.plan_b):
        assert len(plan.station_poses) == 2
        assert plan.total_inspection_effort_m > 0
        assert plan.global_pfa >= 0
        assert plan.global_pfr >= 0
        assert len(plan.per_characteristic_uncertainty_m) == len(result.scenario.characteristics)
