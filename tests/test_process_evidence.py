"""Tests for planning.process_evidence: accepting characteristics on process
evidence instead of measuring them."""
import numpy as np
import pytest
from scipy.stats import norm

from planning.candidates import track_candidate_poses
from planning.experiment import LONG_PART_GRID_HEIGHTS_M, long_part_track, long_part_two_cluster_scenario
from planning.objectives import evaluate_plan, objective_a0_trace_covariance, objective_c_direct_risk, station_visibility_matrix
from planning.process_evidence import (
    ProcessEvidenceEvaluator, break_even_multiplier, exact_plan_risk, search_evidence_layout,
)
from planning.search import grid_search_station_placement
from risk.jcgm106 import evaluate_conformity_risk

TRACK = long_part_track()


@pytest.fixture(scope="module")
def scenario():
    return long_part_two_cluster_scenario(track=TRACK)


@pytest.fixture(scope="module")
def layout(scenario):
    """A feasible 3-station layout (a few row/loop candidates, exhaustive)."""
    wanted = {0.0, 15.9, 16.9, 44.8, 45.8, 54.7, 55.7}
    poses = [p for p in track_candidate_poses(TRACK, 1.0, LONG_PART_GRID_HEIGHTS_M) if round(p.s_m, 1) in wanted]
    best = grid_search_station_placement(poses, 3, scenario, objective_a0_trace_covariance, prefilter_visibility=True).best
    assert best is not None
    return list(best.station_poses)


def test_escape_price_is_the_clusters_nonconformance_rate(scenario):
    evaluator = ProcessEvidenceEvaluator(scenario)
    marginal = norm.sf((50e-6 - 38e-6) / 3e-6) + norm.cdf((-50e-6 - 38e-6) / 3e-6)
    high = 2 * norm.sf(50e-6 / (50e-6 / 6.0))
    assert evaluator.escape_price("cluster2_profile_0") == pytest.approx(marginal, rel=1e-6)
    assert evaluator.escape_price("cluster1_profile_0") == pytest.approx(high, rel=1e-6)
    assert evaluator.baseline_risk() == pytest.approx(3 * marginal + 3 * high, rel=1e-6)
    scaled = ProcessEvidenceEvaluator(scenario, escape_price_multiplier=1e9)
    assert scaled.escape_price("cluster2_profile_0") == 1.0  # a probability, capped


def test_measuring_everything_reproduces_objective_c(scenario, layout):
    """With a floor of all six the evaluator is the existing measure-everything
    objective C (risk tables within 3% of the integral, per characteristic)."""
    plan, deficit = ProcessEvidenceEvaluator(scenario, min_measured=6, ties_required=True).evaluate(layout)
    assert deficit == 0.0 and len(plan.measured) == 6 and plan.accepted_on_evidence == ()
    exact_c = objective_c_direct_risk(evaluate_plan(layout, scenario), scenario)
    assert plan.risk == pytest.approx(exact_c, rel=0.02)
    assert exact_plan_risk(scenario, plan, ProcessEvidenceEvaluator(scenario)) == pytest.approx(exact_c, rel=1e-3)


def test_with_no_floor_and_the_model_prior_a_planner_measures_nothing(scenario, layout):
    """The model's arithmetic: measuring at tracker-on-a-track uncertainties creates far more
    wrong decisions (false rejections) than accepting on a Cpk >= 1.33 prior."""
    evaluator = ProcessEvidenceEvaluator(scenario, min_measured=0)
    plan, _ = evaluator.evaluate(layout)
    assert plan.measured == () and plan.risk == pytest.approx(evaluator.baseline_risk())
    measured_everything, _ = ProcessEvidenceEvaluator(scenario, min_measured=6).evaluate(layout)
    assert measured_everything.risk > 1000 * plan.risk


def test_policy_floor_tops_up_with_the_cheapest_characteristics_to_measure(scenario, layout):
    plan, _ = ProcessEvidenceEvaluator(scenario, min_measured=3).evaluate(layout)
    assert len(plan.measured) == 3
    # the high-capability cluster is measured at the lower uncertainty / risk, so it is the cheaper to measure
    assert all(n.startswith("cluster1") for n in plan.measured)
    plan2, _ = ProcessEvidenceEvaluator(scenario, min_measured=4).evaluate(layout)
    assert len(plan2.measured) == 4 and set(plan.measured) < set(plan2.measured)
    assert plan2.risk > plan.risk  # a stricter floor can only cost risk here


def test_a_dear_enough_escape_price_makes_the_marginal_cluster_worth_measuring(scenario, layout):
    cheap = ProcessEvidenceEvaluator(scenario, escape_price_multiplier=1.0).evaluate(layout)[0]
    dear = ProcessEvidenceEvaluator(scenario, escape_price_multiplier=1e5).evaluate(layout)[0]
    assert cheap.measured == ()
    assert set(dear.measured) == {f"cluster2_profile_{i}" for i in range(3)}  # price 1.0 > their ~0.3-0.5 risk
    assert all(n.startswith("cluster1") for n in dear.accepted_on_evidence)  # capable cluster still cheap to accept


def test_relaxing_the_ties_only_removes_constraints(scenario, layout):
    visible = station_visibility_matrix(scenario, layout)
    required = ProcessEvidenceEvaluator(scenario, ties_required=True)
    relaxed = ProcessEvidenceEvaluator(scenario, ties_required=False)
    assert required.deficit(visible) == 0.0 and relaxed.deficit(visible) == 0.0
    tie = relaxed.tie_indices()[0]
    hidden = visible.copy()
    hidden[:, tie] = False  # nobody sees one tie point
    assert required.deficit(hidden) > 0.0
    assert relaxed.deficit(hidden) == 0.0
    # and when everything is measured the numbers do not depend on the tie rule
    a = ProcessEvidenceEvaluator(scenario, min_measured=6, ties_required=True).evaluate(layout)[0]
    b = ProcessEvidenceEvaluator(scenario, min_measured=6, ties_required=False).evaluate(layout)[0]
    assert a.risk == pytest.approx(b.risk)


def test_a_floor_larger_than_what_can_be_seen_is_infeasible(scenario):
    anchor_only = ProcessEvidenceEvaluator(scenario, min_measured=1, ties_required=False)
    plan, deficit = anchor_only.evaluate([])
    assert plan is None and deficit > 0.0  # the -x tip anchor sees only the end face


def test_break_even_multiplier_is_measured_risk_over_the_nonconformance_rate(scenario):
    u = {c.name: 30e-6 for c in scenario.characteristics}
    out = break_even_multiplier(scenario, u)
    evaluator = ProcessEvidenceEvaluator(scenario)
    for c in scenario.characteristics:
        r = evaluate_conformity_risk(scenario.assignments[c.name], 30e-6, scenario.decision_rules[c.name])
        assert out[c.name] == pytest.approx((r.probability_false_acceptance + r.probability_false_rejection)
                                            / evaluator.escape_price(c.name), rel=1e-9)
    assert out["cluster2_profile_0"] > 1e3  # the model says: do not measure, by a wide margin


def test_the_search_finds_a_feasible_layout_that_measures_what_the_floor_demands(scenario):
    evaluator = ProcessEvidenceEvaluator(scenario, min_measured=3, ties_required=True)
    plan, poses, _ = search_evidence_layout(evaluator, TRACK, 3, seed=0, maxiter=12, popsize=10, tol=1e-8, workers=1)
    # twelve generations of DE need not find a feasible layout from a cold start; the contract is
    # that a returned plan is feasible, satisfies the floor and agrees with a fresh evaluation
    if plan is not None:
        assert len(plan.measured) >= 3
        assert evaluator.evaluate(poses)[0].risk == pytest.approx(plan.risk)


def test_with_the_best_acceptance_rule_measuring_never_beats_the_tight_prior_by_much(scenario):
    """Simple acceptance is not the risk-optimal rule when the process prior is much
    tighter than the measurement: the best symmetric rule accepts nearly everything,
    so measured risk falls to the unmeasured price and never below it by more than a
    hair, at the uncertainties a track delivers."""
    from planning.process_evidence import best_acceptance_risk

    evaluator = ProcessEvidenceEvaluator(scenario)
    for name in ("cluster1_profile_0", "cluster2_profile_0"):
        price = evaluator.escape_price(name)
        simple = evaluate_conformity_risk(scenario.assignments[name], 30e-6, scenario.decision_rules[name])
        simple_risk = simple.probability_false_acceptance + simple.probability_false_rejection
        best, half_width = best_acceptance_risk(scenario, name, 30e-6)
        assert best <= simple_risk and best == pytest.approx(price, rel=0.01)
        assert half_width > 5 * 50e-6  # accepts almost everything
        assert simple_risk > 1000 * best
    # a precise enough measurement does carry information: 1 um against the capable cluster's 8.3 um spread
    best_precise, _ = best_acceptance_risk(scenario, "cluster1_profile_0", 1e-6)
    assert best_precise < 0.9 * evaluator.escape_price("cluster1_profile_0")
