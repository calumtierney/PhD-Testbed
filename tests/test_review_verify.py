"""review.verify: independent recomputations of a review's numbers, and that they would notice a wrong one."""
import dataclasses

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from interchange.plan_file import plan_file_from_stations
from review.core import build_context, review_with_context
from review.demo import demo_plan
from review.scene import ReviewScene
from review.settings import ReviewSettings
from review.verify import (
    best_fit_rigid, check_review, monte_carlo_uncertainty, risk_closed_form, risk_monte_carlo, sample_process,
)
from risk.decision import PriorComponent, ProcessKnowledge, decision_risk

TOL = 50e-6
CASES = [  # (process mean, process std, measurement u), metres
    (0.0, 8.3e-6, 10e-6),          # the built-in cell's high-capability cluster
    (38e-6, 3e-6, 9.6e-6),         # its marginal cluster: Cpk 1.33 lost to offset
    (0.0, 12.5e-6, 10e-6),         # the same Cpk lost to spread
    (40e-6, 6e-6, 20e-6),
    (-25e-6, 5e-6, 4e-6),
]


@pytest.fixture(scope="module")
def factory():
    return ReviewScene.from_factory_cell()


@pytest.fixture(scope="module")
def noise(factory):
    context = build_context(factory, ReviewSettings("noise_only"))
    plan = demo_plan(factory, 4)
    return context, plan, review_with_context(context, plan)


@pytest.mark.parametrize("mean, std, u", CASES)
def test_the_closed_form_agrees_with_the_library_integral(mean, std, u):
    pfa, pfr = risk_closed_form(mean, std, -TOL, TOL, u)
    lib_pfa, lib_pfr = decision_risk(ProcessKnowledge.known(mean, std), -TOL, TOL, u)
    assert pfr == pytest.approx(lib_pfr, rel=1e-6, abs=1e-15)
    assert pfa == pytest.approx(lib_pfa, rel=1e-6, abs=1e-15)


def test_the_closed_form_agrees_with_a_bivariate_normal_and_with_guard_banded_limits():
    mean, std, u = 38e-6, 3e-6, 9.6e-6
    pfa, pfr = risk_closed_form(mean, std, -TOL, TOL, u)
    # (x, m) is bivariate normal: var x = std^2, var m = std^2 + u^2, cov = std^2
    joint = multivariate_normal(mean=[mean, mean], cov=[[std**2, std**2], [std**2, std**2 + u**2]])
    both_in = joint.cdf([TOL, TOL]) - joint.cdf([-TOL, TOL]) - joint.cdf([TOL, -TOL]) + joint.cdf([-TOL, -TOL])
    x_in = norm.cdf(TOL, mean, std) - norm.cdf(-TOL, mean, std)
    m_in = norm.cdf(TOL, mean, np.hypot(std, u)) - norm.cdf(-TOL, mean, np.hypot(std, u))
    assert pfr == pytest.approx(x_in - both_in, abs=2e-6)
    assert pfa == pytest.approx(m_in - both_in, abs=2e-6)
    banded = risk_closed_form(mean, std, -TOL, TOL, u, -45e-6, 45e-6)  # a guard band trades false accepts for false rejects
    assert banded[0] < pfa and banded[1] > pfr
    assert banded == pytest.approx(decision_risk(ProcessKnowledge.known(mean, std), -TOL, TOL, u, -45e-6, 45e-6), rel=1e-5)


def test_the_process_sampler_draws_the_distributions_it_says():
    rng = np.random.default_rng(0)
    normal = sample_process(ProcessKnowledge.known(5e-6, 2e-6), 400_000, rng)
    assert normal.mean() == pytest.approx(5e-6, abs=1e-8) and normal.std() == pytest.approx(2e-6, rel=5e-3)
    student = sample_process(ProcessKnowledge.from_builds(0.0, 4e-6, 6), 800_000, rng)  # scale 4 * sqrt(1 + 1/6), 5 dof
    scale = 4e-6 * np.sqrt(1 + 1 / 6)
    assert student.var() == pytest.approx(scale**2 * 5 / 3, rel=0.05)  # variance of a t with nu dof is scale^2 nu / (nu - 2)
    mixture = ProcessKnowledge((PriorComponent(0.3, -10e-6, 1e-6, None), PriorComponent(0.7, 20e-6, 1e-6, None)))
    draws = sample_process(mixture, 200_000, rng)
    assert np.mean(draws < 0) == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize("mean, std, u", CASES[:3])
def test_the_decision_step_monte_carlo_agrees_with_the_closed_form(mean, std, u):
    mc = risk_monte_carlo(ProcessKnowledge.known(mean, std), -TOL, TOL, u, 2_000_000, np.random.default_rng(1))
    pfa, pfr = risk_closed_form(mean, std, -TOL, TOL, u)
    assert abs(mc["pfr"] - pfr) <= 4 * mc["pfr_se"] + 1e-12
    assert abs(mc["pfa"] - pfa) <= 4 * mc["pfa_se"] + 1e-12


def test_best_fit_rigid_recovers_a_known_motion_and_never_returns_a_reflection():
    rng = np.random.default_rng(3)
    points = rng.uniform(-3, 3, (12, 3))
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    moved = points @ rotation.T + np.array([1.0, -2.0, 0.5])
    r, t = best_fit_rigid(points, moved)
    np.testing.assert_allclose(r, rotation, atol=1e-12)
    np.testing.assert_allclose(points @ r.T + t, moved, atol=1e-12)
    flat = np.column_stack([points[:, :2], np.zeros(len(points))])  # coplanar points: the case where a reflection could fit as well
    r2, _ = best_fit_rigid(flat, flat @ rotation.T)
    assert np.linalg.det(r2) == pytest.approx(1.0)


# -- the checks of a review --------------------------------------------------------------------------------
def test_a_correct_review_passes_every_decision_step_check(noise):
    context, plan, review = noise
    checks = check_review(context, plan, review, n_parts=400_000)
    assert len(checks) == 6 * 4  # six measured characteristics: closed form and Monte Carlo, PFA and PFR
    assert all(c.passed for c in checks), [c.name for c in checks if not c.passed]


def test_a_wrong_uncertainty_is_caught(noise):
    context, plan, review = noise
    wrong = dataclasses.replace(review, characteristics=[{**c, "u_um": c["u_um"] * 1.5} for c in review.characteristics])
    checks = check_review(context, plan, wrong, n_parts=200_000)
    assert any(not c.passed for c in checks)  # the reviewed risks no longer follow from the (inflated) u


def test_a_wrong_risk_is_caught(noise):
    context, plan, review = noise
    wrong = dataclasses.replace(review, characteristics=[{**c, "pfr": c["pfr"] * 1.2} for c in review.characteristics])
    failed = [c for c in check_review(context, plan, wrong, n_parts=200_000) if not c.passed]
    assert any("closed form" in c.name for c in failed) and any("Monte Carlo" in c.name for c in failed)


def test_an_infeasible_review_has_nothing_to_check_and_an_unmeasured_characteristic_is_skipped(factory):
    context = build_context(factory, ReviewSettings("noise_only"))
    lost = plan_file_from_stations("lost", "engineer", "t", "2025-11-27T00:00:00Z", "cell", [[0.0, -4.0, 1.5], [900.0, 900.0, 1.0]])
    assert check_review(context, lost, review_with_context(context, lost)) == []
    one = plan_file_from_stations("one", "engineer", "t", "2025-11-27T00:00:00Z", "cell", [list(demo_plan(factory, 4).stations[0].position_m)])
    review = review_with_context(context, one)
    names = {c.name.split(":")[0] for c in check_review(context, one, review, n_parts=50_000)}
    assert names == {c["name"] for c in review.characteristics if c["measured"]} and len(names) < 6


# -- the whole chain, Monte Carlo ----------------------------------------------------------------------------
@pytest.mark.parametrize("model, n_trials", [("noise_only", 90), ("npl_residual", 150)])
def test_the_whole_chain_monte_carlo_agrees_with_the_reviewed_uncertainty(factory, model, n_trials):
    """Simulated readings -> nonlinear network solve -> explicit best-fit datum frame -> deviation along the tolerance direction. A modest
    trial count (the standard error of each estimate is 1 / sqrt(2 (n - 1)), 7-8% here), judged at 4 standard errors; the full-size runs
    (3000 trials: within 2% for every characteristic, both models) are in docs/plan_review.md."""
    context = build_context(factory, ReviewSettings(model))
    plan = demo_plan(factory, 4)
    review = review_with_context(context, plan)
    run = monte_carlo_uncertainty(context, plan, n_trials, seed=2024)
    assert ("npl" in run["tracker_model"]) == (model == "npl_residual")
    for c, row in zip(review.characteristics, run["characteristics"]):
        u_mc = row["u_m"] * 1e6
        assert abs(u_mc - c["u_um"]) <= 4 * row["relative_se"] * c["u_um"], (c["name"], u_mc, c["u_um"])
        assert abs(row["mean_deviation_m"][0]) * 1e6 <= 4 * c["u_um"] / np.sqrt(n_trials)  # unbiased: the part frame removes any offset
        assert abs(u_mc / c["u_um"] - 2.0) > 0.3  # not vacuous: it would not have agreed with a value twice as large


def test_the_whole_chain_monte_carlo_sees_the_tracker_parameter_errors(factory):
    """With the NPL model the simulated spread is larger than sensor noise alone gives, by about what the review says."""
    plan = demo_plan(factory, 4)
    noise_ctx, npl_ctx = (build_context(factory, ReviewSettings(m)) for m in ("noise_only", "npl_residual"))
    review_noise, review_npl = review_with_context(noise_ctx, plan), review_with_context(npl_ctx, plan)
    assert all(b["u_um"] > a["u_um"] * 1.02 for a, b in zip(review_noise.characteristics, review_npl.characteristics))
    run = monte_carlo_uncertainty(npl_ctx, plan, 150, seed=7)
    mean_ratio = np.mean([row["u_m"] * 1e6 / c["u_um"] for c, row in zip(review_npl.characteristics, run["characteristics"])])
    assert 0.85 < mean_ratio < 1.15
