"""Tests for risk.decision: process knowledge from few builds, and decision
rules based on conformance probability."""
import numpy as np
import pytest
from scipy import stats

from risk.decision import (
    ProcessKnowledge, acceptance_interval, best_rule_risk, conformance_probability, decision_risk, unmeasured_risk,
)
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, evaluate_conformity_risk

TOL = 50e-6
MARGINAL = (38e-6, 3e-6)
HIGH = (0.0, 50e-6 / 6.0)


@pytest.mark.parametrize("mean,std", [MARGINAL, HIGH, (10e-6, 12.5e-6)])
@pytest.mark.parametrize("u_um", [1.0, 5.0, 13.0, 30.0])
def test_known_process_and_simple_acceptance_reproduce_jcgm106(mean, std, u_um):
    u = u_um * 1e-6
    reference = evaluate_conformity_risk(ClusterAssignment("x", FeatureCluster("c", ProcessPrior(mean, std))), u,
                                         DecisionRule.from_zone_width(2 * TOL))
    pfa, pfr = decision_risk(ProcessKnowledge.known(mean, std), -TOL, TOL, u)
    # jcgm106's adaptive quadrature runs at scipy's default absolute tolerance, so its own error is
    # ~0.05% where the risk is ~1e-9; this module agrees with a 4-million-point brute-force
    # integral to 1e-10 there (checked when it was written)
    assert pfa == pytest.approx(reference.probability_false_acceptance, rel=1e-3, abs=1e-15)
    assert pfr == pytest.approx(reference.probability_false_rejection, rel=1e-3, abs=1e-15)


def test_guard_banded_limits_reproduce_jcgm106_too():
    rule = DecisionRule(-TOL, TOL, -40e-6, 45e-6)
    reference = evaluate_conformity_risk(ClusterAssignment("x", FeatureCluster("c", ProcessPrior(*MARGINAL))), 6e-6, rule)
    pfa, pfr = decision_risk(ProcessKnowledge.known(*MARGINAL), -TOL, TOL, 6e-6, -40e-6, 45e-6)
    assert pfa == pytest.approx(reference.probability_false_acceptance, rel=1e-4)
    assert pfr == pytest.approx(reference.probability_false_rejection, rel=1e-4)


def test_predictive_distribution_from_builds_is_student_t_and_tends_to_normal():
    five = ProcessKnowledge.from_builds(*MARGINAL, n_builds=5)
    component = five.components[0]
    assert component.dof == 4 and component.scale_m == pytest.approx(3e-6 * np.sqrt(1.2))
    by_hand = stats.t(4, 38e-6, 3e-6 * np.sqrt(1.2)).sf(TOL) + stats.t(4, 38e-6, 3e-6 * np.sqrt(1.2)).cdf(-TOL)
    assert five.nonconformance(-TOL, TOL) == pytest.approx(by_hand)
    known = ProcessKnowledge.known(*MARGINAL).nonconformance(-TOL, TOL)
    rates = [ProcessKnowledge.from_builds(*MARGINAL, n_builds=n).nonconformance(-TOL, TOL) for n in (3, 5, 10, 30, 300, 30000)]
    assert all(a > b for a, b in zip(rates, rates[1:]))  # more builds, thinner tails
    assert rates[1] > 100 * known  # five builds: ~1%, against 0.003% if the process were known
    assert rates[-1] == pytest.approx(known, rel=0.02)
    assert ProcessKnowledge.from_builds(*MARGINAL, n_builds=None).components[0].dof is None
    with pytest.raises(ValueError):
        ProcessKnowledge.from_builds(*MARGINAL, n_builds=1)


def test_accepting_everything_costs_the_nonconformance_rate_and_no_false_rejects():
    prior = ProcessKnowledge.from_builds(*MARGINAL, n_builds=5)
    pfa, pfr = decision_risk(prior, -TOL, TOL, 20e-6, -np.inf, np.inf)
    assert pfr == 0.0 and pfa == pytest.approx(prior.nonconformance(-TOL, TOL))
    # very wide finite limits tend to the same thing
    wide_pfa, wide_pfr = decision_risk(prior, -TOL, TOL, 20e-6, -5e-3, 5e-3)
    assert wide_pfa == pytest.approx(pfa, rel=1e-3) and wide_pfr < 1e-12
    assert unmeasured_risk(prior, -TOL, TOL) == pytest.approx(pfa)


def test_pfa_plus_true_rejects_account_for_all_nonconforming_parts():
    """PFA can never exceed the nonconformance rate, whatever the rule or uncertainty."""
    prior = ProcessKnowledge.from_builds(*HIGH, n_builds=4)
    cap = prior.nonconformance(-TOL, TOL)
    for u in (2e-6, 10e-6, 40e-6):
        for limit in (30e-6, 50e-6, 90e-6):
            pfa, pfr = decision_risk(prior, -TOL, TOL, u, -limit, limit)
            assert 0.0 <= pfa <= cap * (1 + 1e-6) and 0.0 <= pfr <= 1.0 - cap


def test_conformance_probability_matches_the_conjugate_normal_posterior():
    mean, std, u, m = 20e-6, 10e-6, 8e-6, 46e-6
    post_var = 1.0 / (1.0 / std**2 + 1.0 / u**2)
    post_mean = post_var * (mean / std**2 + m / u**2)
    by_hand = stats.norm(post_mean, np.sqrt(post_var)).cdf(TOL) - stats.norm(post_mean, np.sqrt(post_var)).cdf(-TOL)
    assert conformance_probability(ProcessKnowledge.known(mean, std), -TOL, TOL, u, [m])[0] == pytest.approx(by_hand, rel=1e-6)


def test_the_half_probability_rule_minimises_total_risk_and_never_loses_to_not_measuring():
    for prior in (ProcessKnowledge.from_builds(*MARGINAL, n_builds=5), ProcessKnowledge.from_builds(*HIGH, n_builds=5),
                  ProcessKnowledge.known(30e-6, 12e-6)):
        baseline = unmeasured_risk(prior, -TOL, TOL)
        previous = 0.0
        for u in (1e-6, 3e-6, 8e-6, 20e-6, 60e-6):
            pfa, pfr, (lower, upper) = best_rule_risk(prior, -TOL, TOL, u)
            best = pfa + pfr
            assert best <= baseline * (1 + 1e-3)  # measuring cannot hurt under this rule
            assert best >= previous * (1 - 1e-3)  # and a better measurement cannot hurt either
            previous = best
            simple = sum(decision_risk(prior, -TOL, TOL, u))
            assert best <= simple * (1 + 1e-3)
            for scale in (0.9, 1.1):  # nudging the limits away from the rule's cannot improve on it
                lo = lower * scale if np.isfinite(lower) else lower
                hi = upper * scale if np.isfinite(upper) else upper
                assert sum(decision_risk(prior, -TOL, TOL, u, lo, hi)) >= best * (1 - 2e-3)


def test_a_tight_known_prior_makes_the_best_rule_accept_everything_but_few_builds_do_not():
    """Known Cpk 1.33 process, u = 30 um: the measurement says nothing and the best rule accepts all.
    The same m and s from five builds: the measurement is worth taking at small u."""
    u = 30e-6
    known = ProcessKnowledge.known(*MARGINAL)
    lower, upper = acceptance_interval(known, -TOL, TOL, u)
    assert lower < -20 * TOL and upper > 20 * TOL  # accepts any measured value that could plausibly occur
    assert sum(best_rule_risk(known, -TOL, TOL, u)[:2]) == pytest.approx(unmeasured_risk(known, -TOL, TOL), rel=1e-3)
    five = ProcessKnowledge.from_builds(*MARGINAL, n_builds=5)
    baseline = unmeasured_risk(five, -TOL, TOL)
    precise = sum(best_rule_risk(five, -TOL, TOL, 3e-6)[:2])
    coarse = sum(best_rule_risk(five, -TOL, TOL, 30e-6)[:2])
    assert precise < 0.7 * baseline  # a 3 um measurement removes a good part of the risk (1.09% -> 0.63%)
    assert precise < coarse <= baseline * (1 + 1e-3)  # a 30 um one removes none
    centred = ProcessKnowledge.from_builds(*HIGH, n_builds=5)
    assert sum(best_rule_risk(centred, -TOL, TOL, 3e-6)[:2]) < 0.25 * unmeasured_risk(centred, -TOL, TOL)
    # and under simple acceptance a coarse measurement is far worse than not measuring at all
    assert sum(decision_risk(five, -TOL, TOL, 30e-6)) > 20 * baseline


def test_a_stricter_conformance_probability_trades_false_accepts_for_false_rejects():
    prior = ProcessKnowledge.from_builds(*MARGINAL, n_builds=5)
    pfa_50, pfr_50, limits_50 = best_rule_risk(prior, -TOL, TOL, 5e-6, 0.5)
    pfa_95, pfr_95, limits_95 = best_rule_risk(prior, -TOL, TOL, 5e-6, 0.95)
    assert pfa_95 < pfa_50 and pfr_95 > pfr_50
    assert limits_95[1] < limits_50[1]  # the upper acceptance limit moves inside


def test_mixture_propagates_assignment_doubt():
    high, marginal = ProcessKnowledge.known(*HIGH), ProcessKnowledge.known(*MARGINAL)
    mixed = high.mixed_with(marginal, 0.9)
    expected = 0.9 * high.nonconformance(-TOL, TOL) + 0.1 * marginal.nonconformance(-TOL, TOL)
    assert mixed.nonconformance(-TOL, TOL) == pytest.approx(expected)
    assert len(mixed.components) == 2 and sum(c.weight for c in mixed.components) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ProcessKnowledge((mixed.components[0],))
