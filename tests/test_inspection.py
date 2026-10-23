"""Tests for planning.inspection: the wing-box inspection experiment's
scenario, risk tables, layout evaluator and plan constraints.

The scenario build integrates ~280 risk values once (cached for the process),
so it is a module-scoped fixture. `GOOD_LAYOUT` is a fixed five-station layout
(found by `greedy_cover_layouts` on a 6000-point candidate pool, seed 0, then
written out at full precision so the tests do not depend on the pool or on
random numbers): it ties together and sees 444 of the 500 points -- every
point except the low, downward-facing ones -- so it is a valid plan R layout
but not a valid plan U layout."""
import pickle

import numpy as np
import pytest

from planning.inspection import (
    DEFAULT_TUR, Z540_3_MAX_PFA, CandidatePool, PlanSpec, RiskTable, _LayoutCost, _escape_probability,
    build_inspection_scenario, cluster_priors, evaluate_layout, greedy_cover_layouts, iso_risk_plan,
    nonconformance_rate, plan_violation, task_summary, tur_rule_plan, uncertainty_plan, knee_station_count,
)
from planning.search import INFEASIBLE_PENALTY
from risk.jcgm106 import ClusterAssignment, DecisionRule, evaluate_conformity_risk

GOOD_LAYOUT = np.array([
    [4.8343345217955545, 0.7879725476690798, 2.242344981343432],
    [-1.379912994525914, 2.80843670775711, 2.376551330137165],
    [2.068748014072086, 1.0174825412036572, 0.8417989804417512],
    [1.523643438570205, 2.5945041460234135, 0.8176741426734451],
    [1.6565075390548065, -0.9079865680086256, 0.824827576341964],
])
TOLERANCE_HALF_M = 50e-6


@pytest.fixture(scope="module")
def scenario():
    return build_inspection_scenario()


def test_nonconformance_rates_match_the_four_and_six_sigma_tails(scenario):
    """Marginal process: mean 38 um, std 3 um against +-50 um -> the limit is
    4 sigma away (one tail); high-capability: std 50/6 um centred -> 6 sigma
    (two tails)."""
    marginal, high = scenario.nonconformance
    assert marginal == pytest.approx(3.167e-5, rel=1e-3)
    assert high == pytest.approx(1.973e-9, rel=1e-3)


def test_priors_keep_their_capability_when_the_tolerance_changes():
    for tol in (25e-6, 50e-6, 250e-6):
        for style in ("offset", "centred"):
            marginal, high = cluster_priors(tol, style)
            for cluster, cpk in ((marginal, 4 / 3), (high, 2.0)):
                prior = cluster.prior
                assert (tol - abs(prior.mean_m)) / (3 * prior.std_m) == pytest.approx(cpk)
    with pytest.raises(ValueError):
        cluster_priors(50e-6, "nonsense")
    # the repo's established marginal prior: 38 um mean, 3 um std at +-50 um
    prior = cluster_priors(50e-6, "offset")[0].prior
    assert prior.mean_m == pytest.approx(38e-6) and prior.std_m == pytest.approx(3e-6)


def test_tur_rule_gives_the_documented_uncertainty_limit(scenario):
    """Z540.3 TUR = tolerance half-width / U95 with U = 2u: +-50 um at 4:1
    allows u = 6.25 um."""
    assert DEFAULT_TUR == 4.0 and Z540_3_MAX_PFA == 0.02
    assert scenario.uniform_uncertainty_limit_m(4.0) == pytest.approx(6.25e-6)
    assert tur_rule_plan(scenario).u_max_m == pytest.approx(6.25e-6)
    assert tur_rule_plan(scenario, 2.0).u_max_m == pytest.approx(12.5e-6)


def test_risk_table_reproduces_the_direct_integral(scenario):
    rule = DecisionRule.from_zone_width(2 * TOLERANCE_HALF_M)
    for table, cluster in zip(scenario.risk_tables, scenario.clusters):
        assignment = ClusterAssignment(cluster.name, cluster)
        for u_um in (1.5, 4.0, 6.25, 17.0):
            direct = evaluate_conformity_risk(assignment, u_um * 1e-6, rule)
            assert table.pfa(u_um * 1e-6) == pytest.approx(direct.probability_false_acceptance, rel=0.03)
            assert table.pfr(u_um * 1e-6) == pytest.approx(direct.probability_false_rejection, rel=0.03, abs=1e-12)


def test_false_accept_is_capped_by_the_process_and_false_reject_grows_with_uncertainty(scenario):
    """The structural fact the experiment rests on: PFA cannot exceed the
    process's own nonconformance rate (and is within ~3x of it at tracker-
    scale uncertainty), so it is nowhere near Z540.3's 2%; PFR is what grows."""
    u = np.geomspace(1e-6, 100e-6, 30)
    for table, cap in zip(scenario.risk_tables, scenario.nonconformance):
        assert np.all(table.pfa(u) <= cap * 1.0001)
        assert np.all(np.diff(table.pfr(u)) >= -1e-15)
    marginal = scenario.risk_tables[0]
    assert marginal.pfa(2e-6) > 0.3 * scenario.nonconformance[0]  # already a third of the ceiling at 2 um
    assert marginal.pfa(6.25e-6) > 0.4 * scenario.nonconformance[0]
    assert marginal.pfr(20e-6) > 100 * marginal.pfr(3e-6)
    assert max(t.pfa(100e-6) for t in scenario.risk_tables) < Z540_3_MAX_PFA / 100


def test_escape_probability_is_one_minus_the_product_of_survivals():
    pfa = np.array([0.1, 0.2, 0.0])
    assert _escape_probability(pfa) == pytest.approx(1 - 0.9 * 0.8)
    tiny = np.full(1000, 1e-9)
    assert _escape_probability(tiny) == pytest.approx(1e-6, rel=1e-3)


def test_good_layout_is_a_feasible_plan_r_but_not_a_feasible_plan_u(scenario):
    outcome = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)
    assert outcome.feasible and outcome.n_stations == 5
    assert outcome.n_measured == 444
    assert outcome.visible.shape == (5, 500)
    # every free station really is tied to the anchor by >= 3 shared points
    for s in range(1, 5):
        assert (outcome.visible[0] & outcome.visible[s]).sum() >= 3 or (outcome.visible[:s].any(axis=0) & outcome.visible[s]).sum() >= 3
    strict = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=False)
    assert not strict.feasible and strict.deficit == pytest.approx(56 / 500, abs=1e-6)


def test_unmeasured_points_are_priced_at_the_process_nonconformance_rate_and_never_rejected(scenario):
    outcome = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)
    c = scenario.assignment_confidence
    nc = np.array(scenario.nonconformance)
    label, other = scenario.point_cluster, 1 - scenario.point_cluster
    expected_skip = c * nc[label] + (1 - c) * nc[other]
    skipped = ~outcome.measured
    assert skipped.sum() == 56
    assert np.allclose(outcome.pfa[skipped], expected_skip[skipped])
    assert np.all(outcome.pfr[skipped] == 0.0)
    assert np.all(np.isnan(outcome.u_m[skipped])) and not np.any(np.isnan(outcome.u_m[outcome.measured]))
    assert outcome.e_fa == pytest.approx(outcome.pfa.sum())
    assert outcome.e_wrong == pytest.approx(outcome.pfa.sum() + outcome.pfr.sum())


def test_full_assignment_confidence_makes_skipping_a_high_capability_point_almost_free():
    exact = build_inspection_scenario(assignment_confidence=1.0)
    unsure = build_inspection_scenario(assignment_confidence=0.9)
    skipped_high = exact.point_cluster == 1
    none_measured = np.zeros(exact.n_points, dtype=bool)
    pfa_exact, _ = exact.point_risks(np.ones(exact.n_points), none_measured)
    pfa_unsure, _ = unsure.point_risks(np.ones(exact.n_points), none_measured)
    assert np.allclose(pfa_exact[skipped_high], exact.nonconformance[1])
    assert pfa_unsure[skipped_high][0] > 1000 * pfa_exact[skipped_high][0]  # 10% chance it is really marginal


def test_a_station_inside_the_part_or_fixture_is_not_a_plan(scenario):
    inside = GOOD_LAYOUT.copy()
    inside[3] = [0.75, 1.0, 1.2]  # in the wing box
    outcome = evaluate_layout(scenario, inside, allow_unmeasured=True)
    assert not outcome.feasible and outcome.deficit > 1.0


def test_a_station_that_sees_nothing_in_common_is_not_tied_in(scenario):
    stranded = np.vstack([GOOD_LAYOUT[:2], [60.0, 60.0, 2.0]])
    outcome = evaluate_layout(scenario, stranded, allow_unmeasured=True)
    assert not outcome.feasible and outcome.deficit >= 1.0


def test_more_stations_never_leave_more_points_unmeasured_or_raise_risk(scenario):
    outcomes = [evaluate_layout(scenario, GOOD_LAYOUT[:k], allow_unmeasured=True) for k in (1, 2, 3, 4, 5)]
    assert all(o.feasible for o in outcomes)
    measured = [o.n_measured for o in outcomes]
    assert measured == sorted(measured) and measured[0] == 345
    escapes = [o.p_escape for o in outcomes]
    assert escapes == sorted(escapes, reverse=True)


def test_task_summary_accounts_for_every_point(scenario):
    outcome = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)
    summary = task_summary(scenario, outcome)
    assert sum(d["n_points"] for d in summary.values()) == 500
    assert sum(d["n_measured"] for d in summary.values()) == outcome.n_measured
    assert sum(d["shots"] for d in summary.values()) == outcome.n_shots
    assert sum(d["e_fr"] for d in summary.values()) == pytest.approx(outcome.e_fr)
    assert {d["cluster"] for d in summary.values()} == {"marginal", "high-capability"}


def test_iso_risk_plan_is_what_the_uniform_rule_guarantees(scenario):
    spec = iso_risk_plan(scenario)
    efa, efr, p_escape = scenario.risk_at_uniform_uncertainty(6.25e-6)
    assert spec.max_p_escape == pytest.approx(p_escape) and spec.max_e_fr == pytest.approx(efr)
    assert spec.allow_unmeasured
    assert p_escape < Z540_3_MAX_PFA  # the consumer's-risk bound is nowhere near binding for the part


def test_plan_violation_is_zero_when_met_and_grows_as_the_limit_tightens(scenario):
    outcome = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)
    loose = iso_risk_plan(scenario)
    assert plan_violation(loose, outcome) == 0.0  # 5 stations: 3.2e-3 < 3.5e-3, 7.7 < 10.1
    tight = iso_risk_plan(scenario, slack=0.5)
    assert plan_violation(tight, outcome) > 0.0
    tighter = iso_risk_plan(scenario, slack=0.25)
    assert plan_violation(tighter, outcome) > plan_violation(tight, outcome)
    # the TUR rule needs every point under its uncertainty limit: the layout's worst point is ~12 um
    spec_u = PlanSpec("T", u_max_m=6.25e-6)
    assert plan_violation(spec_u, evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)) > 0.0
    # the uncertainty-driven plan has no threshold: any all-measured layout meets it, a skipped point does not
    assert plan_violation(uncertainty_plan(), outcome) == pytest.approx(56 / 500)
    with pytest.raises(ValueError):
        PlanSpec("X")


def test_layout_cost_separates_infeasible_from_feasible_and_pickles(scenario):
    cost = _LayoutCost(scenario, iso_risk_plan(scenario))
    feasible = cost(GOOD_LAYOUT.ravel())
    stranded = cost(np.vstack([GOOD_LAYOUT[:2], [60.0, 60.0, 2.0]]).ravel())
    assert feasible < INFEASIBLE_PENALTY <= stranded
    # constraint met -> only the small secondary term is paid
    assert feasible < 10.0
    clone = pickle.loads(pickle.dumps(cost))  # workers=-1 sends the cost to subprocesses
    assert clone(GOOD_LAYOUT.ravel()) == pytest.approx(feasible)


def test_greedy_cover_seeds_are_tied_in_and_cover_greedily():
    """Synthetic pool of four candidate stations over 30 points. The first
    pick is the best cover (the anchor); a later station is only eligible if
    it shares >= min_shared_points with what is already covered, even when a
    disconnected one would add more new points."""
    visible = np.zeros((4, 30), dtype=bool)
    visible[0, 0:10] = True    # shares nothing with station 1 at first
    visible[1, 10:25] = True   # 15 points: the biggest cover, so the anchor
    visible[2, 6:18] = True    # shares 8 with station 1's points, adds 6-9
    visible[3, 23:30] = True   # adds 25-29 but shares only 23, 24 with the cover: never eligible
    positions = np.arange(12, dtype=float).reshape(4, 3)
    pool = CandidatePool(positions, visible)
    layout = greedy_cover_layouts(pool, np.ones(30), n_layouts=1, max_stations=4, min_shared_points=3)[0]
    chosen = [int(np.where((positions == p).all(axis=1))[0][0]) for p in layout]
    assert chosen == [1, 2, 0]  # 0 becomes eligible once 2 has covered 6-9; 3 never does


def test_normal_directed_candidates_make_every_point_reachable(scenario):
    """A uniform candidate sample misses the narrow region that sees the
    underside of the nose (two points are seen from none of 6000 uniform
    candidates); candidates placed along each point's own normal do not."""
    from planning.inspection import build_candidate_pool

    uniform_only = build_candidate_pool(scenario, n_candidates=1500, seed=0, n_directed_per_point=0)
    with_directed = build_candidate_pool(scenario, n_candidates=1500, seed=0, n_directed_per_point=4)
    assert not uniform_only.visible.any(axis=0).all()
    assert with_directed.visible.any(axis=0).all()
    assert len(with_directed.positions_m) > len(uniform_only.positions_m)
    lo = np.array([b[0] for b in scenario.part.station_bounds_m])
    hi = np.array([b[1] for b in scenario.part.station_bounds_m])
    assert np.all(with_directed.positions_m >= lo) and np.all(with_directed.positions_m <= hi)
    inside = np.all((with_directed.positions_m > scenario.part.keep_out_min_m)
                    & (with_directed.positions_m < scenario.part.keep_out_max_m), axis=1)
    assert not inside.any()


def test_a_minimum_task_coverage_floor_turns_a_skipped_task_into_a_violation(scenario):
    """GOOD_LAYOUT leaves the lower flap skins unseen but measures every
    other point, so it satisfies the plain risk plan and fails a floor that
    insists on 100% of every task; a modest floor it still meets."""
    outcome = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)
    cov = outcome.task_coverage
    assert cov.shape == (6,) and np.all((cov >= 0) & (cov <= 1))
    assert cov[2] == 1.0 and cov[3] == 1.0  # T3 free-form surface and T4 top skin fully seen
    assert cov[4] < 1.0 and cov[5] < 1.0    # the flaps lose their lower skins
    base = iso_risk_plan(scenario)
    assert plan_violation(base, outcome) == 0.0
    assert plan_violation(iso_risk_plan(scenario, min_task_coverage=0.4), outcome) == 0.0
    strict = plan_violation(iso_risk_plan(scenario, min_task_coverage=1.0), outcome)
    assert strict == pytest.approx(float(np.sum(1.0 - cov)))
    assert plan_violation(iso_risk_plan(scenario, min_task_coverage=0.9), outcome) > 0.0


def test_knee_station_count_stops_when_the_next_station_gains_too_little():
    j = {4: 100.0, 5: 60.0, 6: 40.0, 7: 37.0, 8: 36.5}
    assert knee_station_count(j, 0.10) == 6       # 6 -> 7 gains 7.5% < 10%
    assert knee_station_count(j, 0.02) == 7       # 7 -> 8 gains 1.4% < 2%; 6 -> 7 gained 7.5% > 2%
    assert knee_station_count(j, 0.50) == 4       # 4 -> 5 already gains only 40% < 50%
    assert knee_station_count({3: 10.0, 4: 5.0}, 0.10) == 4  # never flattens: the largest K is returned


def test_backward_refinement_never_returns_a_worse_layout_and_keeps_the_counts(scenario):
    """Fed a tiny sweep, `refine_backward` keeps every station count, never
    raises a layout's cost, and returns layouts that are exactly evaluable."""
    from planning.inspection import KResult, refine_backward

    spec = iso_risk_plan(scenario)
    cost = _LayoutCost(scenario, spec)
    sweep = []
    for k in (3, 4, 5):
        positions = GOOD_LAYOUT[:k]
        outcome = evaluate_layout(scenario, positions, True)
        sweep.append(KResult(k, positions, outcome, plan_violation(spec, outcome), 0))
    refined = refine_backward(scenario, spec, sweep, de_kwargs=dict(maxiter=3, popsize=6, workers=1), seed=0)
    assert [r.n_stations for r in refined] == [3, 4, 5]
    for before, after in zip(sweep, refined):
        assert cost(after.positions_m.ravel()) <= cost(before.positions_m.ravel()) + 1e-12
        assert after.positions_m.shape == (after.n_stations, 3)


def test_trace_criterion_is_at_least_the_projected_variance_and_prices_a_plan_a_layout(scenario):
    """The covariance trace sums all three axes; the variance along one unit
    direction cannot exceed the largest eigenvalue and is at most the trace:
    so trace >= u^2 for every measured point (systematic term included on
    both sides), and plan A (trace) is scored like plan U (projected)."""
    from planning.inspection import uncertainty_plan

    outcome = evaluate_layout(scenario, GOOD_LAYOUT, allow_unmeasured=True)
    m = outcome.measured
    assert np.all(outcome.trace_m2[m] >= outcome.u_m[m] ** 2 - 1e-18)
    assert np.all(np.isnan(outcome.trace_m2[~m]))
    plan_a = uncertainty_plan(trace=True)
    assert plan_a.kind == "A" and uncertainty_plan().kind == "U"
    assert plan_violation(plan_a, outcome) == pytest.approx(56 / 500)
    # plan A must measure every point: GOOD_LAYOUT leaves 56 unseen, so the search sees it as
    # infeasible, graded by the unmeasured fraction
    cost_a = _LayoutCost(scenario, plan_a)(GOOD_LAYOUT.ravel())
    assert cost_a == pytest.approx(INFEASIBLE_PENALTY + 56 / 500)
