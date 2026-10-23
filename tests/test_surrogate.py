"""Tests for planning.surrogate: features, the grouped split (the leakage
guard CLAUDE.md §6 asks for), the model on a small simulated dataset, and
the assisted search's contract that whatever it returns was scored exactly."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from planning.inspection import (
    CandidatePool, build_inspection_scenario, evaluate_layout, iso_risk_plan, plan_violation,
)
from planning.surrogate import (
    LayoutSurrogate, SurrogateCost, batch_features, generate_layout_records, grouped_split, layout_features,
    predicted_violation, region_holdout_mask, regression_report, rescore, surrogate_assisted_search,
)
from test_inspection import GOOD_LAYOUT


@pytest.fixture(scope="module")
def scenario():
    return build_inspection_scenario()


@pytest.fixture(scope="module")
def pool(scenario):
    from planning.inspection import build_candidate_pool
    return build_candidate_pool(scenario, n_candidates=1200, seed=0)


@pytest.fixture(scope="module")
def records(scenario, pool):
    return generate_layout_records(scenario, pool, k_values=(3, 4), n_random_per_k=40, n_greedy_bases_per_k=10,
                                   jitters_per_base=6, search_bases=[GOOD_LAYOUT[:3], GOOD_LAYOUT[:4]], seed=1)


def test_features_do_not_depend_on_station_order_and_pad_to_k_max():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 5.0, 1.0], [4.0, 4.0, 2.0]])
    b = a[[2, 0, 1]]
    fa, fb = layout_features(a, 6), layout_features(b, 6)
    assert np.array_equal(fa, fb) and fa.shape == (6 * 3 + 1,)
    assert fa[-1] == 3 and np.all(fa[9:18] == 0.0)
    assert batch_features([a, b], 6).shape == (2, 19)


def test_grouped_split_never_puts_a_group_on_two_sides():
    groups = np.repeat(np.arange(40), 13)  # 40 base layouts with 13 near-duplicates each
    train, val, test = grouped_split(groups, seed=3)
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert (train | val | test).all()
    for mask_a, mask_b in ((train, val), (train, test), (val, test)):
        assert set(groups[mask_a]).isdisjoint(set(groups[mask_b]))
    assert 0.5 < train.mean() < 0.7 and 0.1 < test.mean() < 0.3


def test_generated_records_cover_the_three_families_and_keep_groups_whole(records):
    families = {r.family for r in records}
    assert families == {"random", "greedy-jitter", "search-jitter"}
    by_group = {}
    for r in records:
        by_group.setdefault(r.group, set()).add(r.family)
    assert all(len(f) == 1 for f in by_group.values())  # a group never mixes families
    assert max(len(set(map(tuple, np.round(r.positions_m, 6).tolist()))) for r in records) <= 4
    assert any(r.feasible for r in records) and any(not r.feasible for r in records)


def test_region_holdout_flags_layouts_with_a_station_in_the_box(records):
    lo, hi = np.array([3.0, -3.0, 0.0]), np.array([6.0, 6.0, 4.0])
    mask = region_holdout_mask(records, lo, hi)
    for r, flagged in zip(records, mask):
        inside = np.any(np.all((r.positions_m > lo) & (r.positions_m < hi), axis=1))
        assert flagged == inside


def test_rescore_reproduces_the_exact_evaluator_from_stored_records(scenario, records):
    feasible = [r for r in records if r.feasible][:5]
    targets = rescore(scenario, feasible)
    for r, p_esc, e_fr, cov in zip(feasible, targets["log10_p_escape"], targets["log10_e_fr"], targets["coverage"]):
        outcome = evaluate_layout(scenario, r.positions_m, allow_unmeasured=True)
        assert 10 ** p_esc == pytest.approx(outcome.p_escape, rel=1e-9)
        assert 10 ** p_esc == pytest.approx(outcome.p_escape)
        assert cov == pytest.approx(outcome.n_measured / 500)
        assert max(10 ** e_fr, 1e-12) == pytest.approx(max(outcome.e_fr, 1e-12), rel=1e-9)


def test_surrogate_learns_coverage_and_risk_on_held_out_groups(scenario, records):
    """Split by group, fit on train (early-stopping on validation), score on
    the untouched test groups. Thresholds are deliberately loose -- a few
    hundred records -- and guard against a broken pipeline (shuffled
    targets, a mis-ordered feature column) rather than assert an accuracy;
    the real held-out scores are produced by the experiment on the full
    dataset and reported as measured."""
    groups = np.array([r.group for r in records])
    train, val, test = grouped_split(groups, seed=0)
    pick = lambda m: [r for r, keep in zip(records, m) if keep]
    kmax = 6
    x = {name: batch_features([r.positions_m for r in pick(m)], kmax) for name, m in (("train", train), ("val", val), ("test", test))}
    y = {name: rescore(scenario, pick(m)) for name, m in (("train", train), ("val", val), ("test", test))}
    model = LayoutSurrogate(k_max=kmax, max_iter=150).fit(x["train"], y["train"], x["val"], y["val"])
    pred = model.predict(x["test"])
    report = regression_report(y["test"], pred)
    assert report["coverage"]["spearman"] > 0.6
    assert report["log10_p_escape"]["spearman"] > 0.5
    assert 0.0 <= pred["p_feasible"].min() and pred["p_feasible"].max() <= 1.0
    accuracy = np.mean((pred["p_feasible"] > 0.5) == y["test"]["feasible"])
    majority = max(y["test"]["feasible"].mean(), 1.0 - y["test"]["feasible"].mean())
    assert accuracy >= majority - 0.1  # a few dozen test records: no worse than a constant guess, give or take noise


def test_predicted_violation_matches_plan_violation_arithmetic(scenario):
    spec = iso_risk_plan(scenario)
    outcome = evaluate_layout(scenario, GOOD_LAYOUT[:2], allow_unmeasured=True)  # violates the limit
    prediction = dict(log10_p_escape=np.array([np.log10(outcome.p_escape)]), log10_e_fr=np.array([np.log10(outcome.e_fr)]))
    assert predicted_violation(spec, prediction)[0] == pytest.approx(plan_violation(spec, outcome))
    assert plan_violation(spec, outcome) > 0.0


def test_assisted_search_returns_only_exactly_scored_layouts(scenario, records):
    groups = np.array([r.group for r in records])
    train, val, _ = grouped_split(groups, seed=0)
    pick = lambda m: [r for r, keep in zip(records, m) if keep]
    kmax = 6
    model = LayoutSurrogate(k_max=kmax, max_iter=80).fit(
        batch_features([r.positions_m for r in pick(train)], kmax), rescore(scenario, pick(train)),
        batch_features([r.positions_m for r in pick(val)], kmax), rescore(scenario, pick(val)))
    spec = iso_risk_plan(scenario)
    cost = SurrogateCost(model, spec, 3)
    batch = np.random.default_rng(0).uniform(-1, 4, size=(9, 5))  # (ndim, S) like scipy's vectorized DE
    assert cost(np.random.default_rng(0).uniform(-1, 4, size=(9, 7))).shape == (7,)
    result = surrogate_assisted_search(scenario, spec, model, 3, seed_layouts=[GOOD_LAYOUT[:3]], n_verify=8,
                                       de_kwargs=dict(maxiter=6, popsize=6), seed=0)
    exact = evaluate_layout(scenario, result.best.positions_m, allow_unmeasured=True)
    assert result.best.outcome.p_escape == pytest.approx(exact.p_escape) or not exact.feasible
    assert result.best.violation == pytest.approx(plan_violation(spec, exact)) or not exact.feasible
    assert result.n_exact_evals <= 8 and 0 <= result.n_shortlist_satisfied <= result.n_exact_evals
    assert result.n_surrogate_evals >= 6 * 3 * 3  # at least the initial population was scored
    # several rounds: more exact evaluations, a model updated (not mutated), best still exactly scored
    before = model.predict(batch_features([GOOD_LAYOUT[:3]], kmax))["log10_p_escape"].copy()
    again = surrogate_assisted_search(scenario, spec, model, 3, seed_layouts=[GOOD_LAYOUT[:3]], n_verify=8,
                                      de_kwargs=dict(maxiter=4, popsize=6), seed=0, rounds=2)
    assert again.rounds == 2 and again.n_exact_evals <= 16
    assert np.array_equal(before, model.predict(batch_features([GOOD_LAYOUT[:3]], kmax))["log10_p_escape"])
    exact2 = evaluate_layout(scenario, again.best.positions_m, allow_unmeasured=True)
    assert again.best.violation == pytest.approx(plan_violation(spec, exact2)) or not exact2.feasible


def test_study_splits_are_disjoint_and_the_region_is_never_trained_on(scenario, pool, records):
    """run_surrogate_study on a small dataset: train/val/test groups are
    disjoint, no training layout has a station in the held-out region (the
    study itself asserts this on the records), and the reports contain every
    target."""
    from planning.surrogate import run_surrogate_study

    spec = iso_risk_plan(scenario)
    region = ((2.5, 2.0, 0.0), (5.0, 5.5, 3.0))
    study = run_surrogate_study(
        scenario, pool, spec, k_values=(2, 3), held_out_region=region,
        generate_kwargs=dict(n_random_per_k=40, n_greedy_bases_per_k=8, jitters_per_base=5), seed=2)
    assert set(study.train_groups).isdisjoint(study.val_groups)
    assert set(study.train_groups).isdisjoint(study.test_groups)
    assert set(study.val_groups).isdisjoint(study.test_groups)
    assert study.n_test_region > 0 and study.n_train > study.n_test_groups
    for report in (study.report_groups, study.report_region):
        assert set(report) == {"coverage", "log10_p_escape", "log10_e_fr"}
    assert 0.0 <= study.feasibility_accuracy_groups <= 1.0
    assert study.exact_ms_per_layout > 0 and study.surrogate_ms_per_layout_batch > 0
    assert set(study.boundary_groups) == {"n", "n_satisfying", "accuracy", "precision", "recall"}


def test_regression_report_does_not_let_clamped_zeros_flatter_the_fit():
    """A handful of layouts with *no* false rejections sit at the log-risk
    clamp, far from the rest; a model that gets those roughly right scores a
    high R^2 on everything while knowing nothing about the bulk. The report
    carries both."""
    rng = np.random.default_rng(0)
    bulk = rng.normal(1.0, 0.5, 200)                      # log10 E_FR of ordinary layouts
    truth = np.concatenate([bulk, np.full(10, -12.0)])    # plus ten clamped zeros
    pred = np.concatenate([rng.normal(1.0, 0.5, 200), np.full(10, -11.0)])  # bulk predicted by noise only
    feasible = np.ones(210, bool)
    zeros = np.zeros(210)
    t = dict(coverage=zeros, log10_p_escape=zeros, log10_e_fr=truth, feasible=feasible)
    p = dict(coverage=zeros, log10_p_escape=zeros, log10_e_fr=pred)
    report = regression_report(t, p)["log10_e_fr"]
    assert report["r2"] > 0.9                    # flattered by the ten clamped points
    assert report["r2_bulk"] < 0.1               # the ordinary layouts are not predicted at all
    assert report["n"] == 210 and report["n_bulk"] == 200
