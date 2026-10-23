"""A learned surrogate of the digital twin's plan evaluator, and the checks
that say whether it can be trusted.

**What is being learned, and why.** `planning.inspection.evaluate_layout` is
the digital twin: station layout -> visibility -> covariance -> per-point
uncertainty -> JCGM 106 risk. It is already fast (~5 ms for 500 points; CLAUDE.md
§3 asked for the block-diagonal structure to be exploited before reaching for
a surrogate, and it is). A surrogate is therefore not a way out of an
intractable calculation. It is a way to *screen* many more candidate plans per
second -- the "simulate plans before executing them" loop -- and the thing
that will later accept a real measurement dataset alongside simulated ones.
Its speed-up is reported against this fast baseline, never against a slow one.

**What it predicts.** Four numbers per layout: the fraction of points
measured, the (log) escape probability, the (log) expected false rejections,
and whether the layout is a rigid network at all. These are exactly the
quantities plan R's constraint is written in, so a surrogate prediction can
be dropped into `planning.inspection.PlanSpec` arithmetic unchanged.

**Validation discipline (CLAUDE.md §6: "the split must be visible in the
code").** The danger here is specific. Layouts are generated in families --
a base layout and many small jitters of it -- and jitters of one base layout
are near-duplicates: they share coverage and uncertainty almost exactly. A
random split over *layouts* would put a base layout's jitters on both sides
and the held-out score would measure memorisation. `grouped_split` therefore
splits by *group* (the base layout), with the three sets disjoint in groups,
and `tests/test_surrogate.py` asserts that disjointness. A second, harder test
holds out a whole spatial region (`region_holdout_mask`): no training layout
has a station there, so the score measures extrapolation, not interpolation.
The hyper-parameters are fixed in code before the test set is touched; the
validation set is used only for early stopping inside the fit.

**Using it without being fooled by it.** Near a constraint boundary --
exactly where the plan decision is made -- a regression surrogate's error is
comparable to the margin. `surrogate_assisted_search` therefore uses it only
to rank and shortlist candidates; every plan it returns has been re-scored by
the exact evaluator, and the exact result is what is reported.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from planning.inspection import (
    InspectionScenario, LayoutOutcome, PlanSpec, evaluate_layout, plan_violation,
)

TARGET_NAMES = ("coverage", "log10_p_escape", "log10_e_fr")
FEATURE_STATION_FIELDS = 3  # x, y, z


def layout_features(positions_m: np.ndarray, k_max: int) -> np.ndarray:
    """Feature vector for one layout (K, 3): the stations sorted by x (so
    the representation does not depend on the order they were listed in --
    the outcome nearly doesn't either; only the choice of anchor/datum
    does), flattened, zero-padded to `k_max` stations, with K itself and the
    number of padded slots appended. A tree model needs no scaling."""
    positions = np.asarray(positions_m, dtype=float).reshape(-1, 3)
    k = len(positions)
    ordered = positions[np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))]
    padded = np.zeros((k_max, 3))
    padded[:k] = ordered
    return np.concatenate([padded.ravel(), [k]])


def batch_features(layouts: Sequence[np.ndarray], k_max: int) -> np.ndarray:
    return np.array([layout_features(l, k_max) for l in layouts])


@dataclass(frozen=True)
class LayoutRecord:
    """One simulated layout: the inputs, and everything about the exact
    outcome needed to re-derive risk under *any* scenario variant (tolerance,
    prior, confidence change the risk but not the geometry, so the stored
    per-point uncertainty and measured mask are reusable)."""

    positions_m: np.ndarray
    group: int  # id of the base layout this one was generated from
    family: str  # "random" | "greedy-jitter" | "search-jitter"
    feasible: bool
    measured: np.ndarray  # (N,) bool, all False when infeasible
    u_m: np.ndarray  # (N,) NaN where unmeasured / infeasible


def record_from_outcome(positions_m: np.ndarray, group: int, family: str, outcome: LayoutOutcome) -> LayoutRecord:
    return LayoutRecord(np.asarray(positions_m, float), group, family, outcome.feasible, outcome.measured, outcome.u_m)


def rescore(scenario: InspectionScenario, records: Sequence[LayoutRecord]) -> Dict[str, np.ndarray]:
    """Targets for `scenario` from stored records, with no geometry
    recomputed. Infeasible records get NaN targets and are excluded from
    the regressors (they train the feasibility classifier)."""
    from planning.inspection import _escape_probability

    n = len(records)
    out = dict(coverage=np.full(n, np.nan), log10_p_escape=np.full(n, np.nan), log10_e_fr=np.full(n, np.nan),
               feasible=np.array([r.feasible for r in records]))
    for i, r in enumerate(records):
        if not r.feasible:
            continue
        pfa, pfr = scenario.point_risks(np.nan_to_num(r.u_m, nan=1e-6), r.measured)
        out["coverage"][i] = r.measured.mean()
        out["log10_p_escape"][i] = np.log10(_escape_probability(pfa))
        out["log10_e_fr"][i] = np.log10(max(pfr.sum(), 1e-12))
    return out


def grouped_split(groups: np.ndarray, fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, validation, test) boolean masks over records, splitting by
    *group* so no group appears in two sets. See the module docstring."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(round(fractions[0] * len(unique)))
    n_val = int(round(fractions[1] * len(unique)))
    train_g, val_g, test_g = unique[:n_train], unique[n_train:n_train + n_val], unique[n_train + n_val:]
    return np.isin(groups, train_g), np.isin(groups, val_g), np.isin(groups, test_g)


def region_holdout_mask(records: Sequence[LayoutRecord], lo_m: np.ndarray, hi_m: np.ndarray) -> np.ndarray:
    """True for records with at least one station inside the box [lo, hi]:
    the region held out of training for the extrapolation test."""
    lo, hi = np.asarray(lo_m, float), np.asarray(hi_m, float)
    return np.array([bool(np.any(np.all((r.positions_m > lo) & (r.positions_m < hi), axis=1))) for r in records])


@dataclass
class LayoutSurrogate:
    """Gradient-boosted trees: one regressor per target (fitted on feasible
    layouts only) and a classifier for feasibility.

    Hyper-parameters are fixed here, not tuned on the test set. Early
    stopping uses a validation fraction carved from the *training* records
    by group (sklearn's own `validation_fraction` would split by record and
    leak jitters across the fold), so `fit` takes the validation records
    explicitly.
    """

    k_max: int
    max_iter: int = 400
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    random_state: int = 0

    def _regressor(self):
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
            early_stopping=False, random_state=self.random_state)

    def fit(self, train_x: np.ndarray, train_targets: Dict[str, np.ndarray],
            val_x: Optional[np.ndarray] = None, val_targets: Optional[Dict[str, np.ndarray]] = None) -> "LayoutSurrogate":
        feasible = train_targets["feasible"]
        self.classifier_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
            early_stopping=False, random_state=self.random_state).fit(train_x, feasible)
        self.regressors_ = {}
        for name in TARGET_NAMES:
            model = self._regressor()
            model.fit(train_x[feasible], train_targets[name][feasible])
            if val_x is not None:
                # choose the number of boosting rounds on the validation records
                # (staged predictions: no refitting), then keep that many
                v_feasible = val_targets["feasible"]
                errors = [np.mean((p - val_targets[name][v_feasible]) ** 2)
                          for p in model.staged_predict(val_x[v_feasible])]
                best = int(np.argmin(errors)) + 1
                model = HistGradientBoostingRegressor(
                    max_iter=best, learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
                    early_stopping=False, random_state=self.random_state).fit(train_x[feasible], train_targets[name][feasible])
            self.regressors_[name] = model
        return self

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        out = {name: model.predict(x) for name, model in self.regressors_.items()}
        out["p_feasible"] = self.classifier_.predict_proba(x)[:, 1]
        return out


def predicted_violation(spec: PlanSpec, prediction: Dict[str, np.ndarray]) -> np.ndarray:
    """Plan R's `plan_violation` computed from surrogate output (arrays)."""
    p_escape = 10.0 ** prediction["log10_p_escape"]
    e_fr = 10.0 ** prediction["log10_e_fr"]
    return (np.maximum(0.0, p_escape / spec.max_p_escape - 1.0) + np.maximum(0.0, e_fr / spec.max_e_fr - 1.0))


def regression_report(truth: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Held-out scores per target on the feasible records: R^2, MAE, Spearman
    rank correlation (what a shortlist depends on)."""
    feasible = truth["feasible"]
    report = {}
    for name in TARGET_NAMES:
        y, p = truth[name][feasible], prediction[name][feasible]
        ss_res, ss_tot = float(np.sum((y - p) ** 2)), float(np.sum((y - y.mean()) ** 2))
        report[name] = dict(
            r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            mae=float(np.mean(np.abs(y - p))),
            spearman=float(spearmanr(y, p).statistic),
            n=int(feasible.sum()),
        )
    return report


# --------------------------------------------------------------------------
# Generating training data
# --------------------------------------------------------------------------

def _clip_to_bounds(positions_m: np.ndarray, bounds_m) -> np.ndarray:
    lo = np.array([b[0] for b in bounds_m])
    hi = np.array([b[1] for b in bounds_m])
    return np.clip(positions_m, lo, hi)


def generate_layout_records(
    scenario: InspectionScenario,
    pool,
    k_values: Sequence[int] = (2, 3, 4, 5, 6),
    n_random_per_k: int = 400,
    n_greedy_bases_per_k: int = 60,
    jitters_per_base: int = 12,
    jitter_scales_m: Sequence[float] = (0.15, 0.5, 1.2),
    search_bases: Sequence[np.ndarray] = (),
    seed: int = 0,
) -> List[LayoutRecord]:
    """Simulate a population of layouts with the exact evaluator.

    Three families, because the surrogate will be used on layouts an
    optimiser proposes -- mostly neighbourhoods of good layouts -- and a
    purely random sample (almost all of it not a rigid network) would not
    train it where it is needed:

      random         uniform in the station box; each its own group.
      greedy-jitter  a base from `greedy_cover_layouts` under randomly
                     drawn point weights, plus `jitters_per_base` Gaussian
                     perturbations of it (scales cycle through
                     `jitter_scales_m`); the base and its jitters share one
                     group.
      search-jitter  the same, around the caller's `search_bases`
                     (layouts a real search returned).

    `group` is what `grouped_split` splits on -- see the module docstring
    for why that, not the record, is the unit.
    """
    from planning.inspection import greedy_cover_layouts

    rng = np.random.default_rng(seed)
    bounds = scenario.part.station_bounds_m
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    records: List[LayoutRecord] = []
    next_group = 0

    def add(positions, group, family):
        outcome = evaluate_layout(scenario, positions, allow_unmeasured=True)
        records.append(record_from_outcome(positions, group, family, outcome))

    def add_family(base, family):
        nonlocal next_group
        group = next_group
        next_group += 1
        add(base, group, family)
        for j in range(jitters_per_base):
            scale = jitter_scales_m[j % len(jitter_scales_m)]
            add(_clip_to_bounds(base + rng.normal(0.0, scale, size=base.shape), bounds), group, family)

    for k in k_values:
        for _ in range(n_random_per_k):
            add(rng.uniform(lo, hi, size=(k, 3)), next_group, "random")
            next_group += 1
        for _ in range(n_greedy_bases_per_k):
            weights = rng.uniform(0.0, 1.0, scenario.n_points) ** 2 + 1e-6
            layout = greedy_cover_layouts(pool, weights, 1, k, seed=int(rng.integers(1 << 31)), top_choices=6)[0]
            if len(layout) < k:  # the cover ran out of connected stations to add: fill up at random
                layout = np.vstack([layout, rng.uniform(lo, hi, size=(k - len(layout), 3))])
            add_family(layout, "greedy-jitter")
        for base in search_bases:
            if len(base) == k:
                add_family(np.asarray(base, float), "search-jitter")
    return records


# --------------------------------------------------------------------------
# Surrogate-assisted station search
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateCost:
    """Plan R's cost, computed from surrogate predictions for a whole DE
    generation at once. The same shape as `planning.inspection._LayoutCost`
    (a large weight on the violation, a small secondary term, a barrier for
    layouts predicted not to be a rigid network) so only the *evaluator*
    differs between the exact and the assisted search."""

    surrogate: "LayoutSurrogate"
    spec: PlanSpec
    n_stations: int

    def __call__(self, flat_positions: np.ndarray) -> np.ndarray:
        from planning.inspection import VIOLATION_WEIGHT
        from planning.search import INFEASIBLE_PENALTY

        x = np.atleast_2d(flat_positions.T if flat_positions.ndim == 2 else flat_positions)
        layouts = x.reshape(len(x), self.n_stations, 3)
        prediction = self.surrogate.predict(batch_features(layouts, self.surrogate.k_max))
        violation = predicted_violation(self.spec, prediction)
        secondary = 10.0 ** prediction["log10_p_escape"] / self.spec.max_p_escape \
            + 10.0 ** prediction["log10_e_fr"] / self.spec.max_e_fr
        cost = VIOLATION_WEIGHT * violation + secondary
        p_ok = prediction["p_feasible"]
        return np.where(p_ok < 0.5, INFEASIBLE_PENALTY + (1.0 - p_ok), cost)


@dataclass(frozen=True)
class AssistedResult:
    """One station count searched with the surrogate, then verified exactly.

    `best` is the exact evaluator's verdict on the best shortlisted layout
    (a `KResult`, so it is directly comparable to the plain search's);
    `n_shortlist_satisfied` how many of the `n_verified` exactly-scored
    shortlist layouts really met the constraint -- the surrogate's
    precision at the decision boundary."""

    n_stations: int
    best: "KResult"
    n_surrogate_evals: int
    n_exact_evals: int
    n_verified: int
    n_shortlist_satisfied: int
    seconds_search: float
    seconds_verify: float


def surrogate_assisted_search(
    scenario: InspectionScenario,
    spec: PlanSpec,
    surrogate: LayoutSurrogate,
    n_stations: int,
    seed_layouts: Sequence[np.ndarray] = (),
    n_verify: int = 60,
    de_kwargs: Optional[dict] = None,
    seed: int = 0,
) -> AssistedResult:
    """Search `n_stations` with the surrogate as the objective (same
    differential-evolution driver and settings as the exact search --
    `planning.search.run_station_search`, here with `vectorized=True` so a
    whole generation is one batch prediction), take the `n_verify` best
    distinct layouts of the final population, score *those* with the exact
    evaluator, and return the best by the exact cost."""
    import time
    from planning.inspection import KResult, _LayoutCost
    from planning.search import run_station_search

    kwargs = dict(maxiter=60, popsize=12, tol=1e-8)
    kwargs.update(de_kwargs or {})
    kwargs.pop("workers", None)  # vectorized batches replace worker processes
    seeds = [np.asarray(l)[:n_stations] for l in seed_layouts if len(l) >= n_stations] or None
    t0 = time.time()
    result, _ = run_station_search(SurrogateCost(surrogate, spec, n_stations), n_stations,
                                   scenario.part.station_bounds_m, seed, seeds,
                                   vectorized=True, updating="deferred", **kwargs)
    t_search = time.time() - t0

    t0 = time.time()
    order = np.argsort(result.population_energies)
    exact_cost = _LayoutCost(scenario, spec)
    shortlist, seen = [], set()
    for i in order:
        key = tuple(np.round(result.population[i], 3))
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(result.population[i].reshape(-1, 3))
        if len(shortlist) >= n_verify:
            break
    scored = []
    for positions in shortlist:
        outcome = evaluate_layout(scenario, positions, spec.allow_unmeasured)
        scored.append((exact_cost(positions.ravel()), positions, outcome))
    scored.sort(key=lambda row: row[0])
    cost, positions, outcome = scored[0]
    best = KResult(n_stations, positions, outcome, plan_violation(spec, outcome), len(shortlist))
    n_ok = sum(1 for _, _, o in scored if o.feasible and plan_violation(spec, o) <= 1e-12)
    return AssistedResult(n_stations, best, int(result.nfev), len(shortlist), len(shortlist), n_ok,
                          t_search, time.time() - t0)


# --------------------------------------------------------------------------
# The study: train, test on held-out groups and a held-out region, time it
# --------------------------------------------------------------------------

@dataclass
class SurrogateStudy:
    """Everything the figures and the write-up need, with the split
    recorded so it can be audited: record counts per set, the group ids in
    each (disjoint -- asserted below), the region held out of training, and
    every score measured on data the model never saw."""

    model: LayoutSurrogate
    n_records: int
    n_feasible: int
    n_train: int
    n_val: int
    n_test_groups: int
    n_test_region: int
    train_groups: np.ndarray
    val_groups: np.ndarray
    test_groups: np.ndarray
    report_groups: Dict[str, Dict[str, float]]
    report_region: Dict[str, Dict[str, float]]
    feasibility_accuracy_groups: float
    feasibility_accuracy_region: float
    boundary_groups: Dict[str, float]
    truth_groups: Dict[str, np.ndarray]
    prediction_groups: Dict[str, np.ndarray]
    truth_region: Dict[str, np.ndarray]
    prediction_region: Dict[str, np.ndarray]
    seconds_generate: float
    seconds_train: float
    exact_ms_per_layout: float
    surrogate_ms_per_layout_batch: float


def boundary_metrics(spec: PlanSpec, truth: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray]) -> Dict[str, float]:
    """How well the surrogate answers the question a plan actually asks --
    does this layout meet the risk limits? -- on held-out feasible layouts:
    accuracy, and precision/recall of "satisfies". Scored against the exact
    evaluator's own verdict (the exact log-risks, thresholded the same way)."""
    feasible = truth["feasible"]
    p_true = 10.0 ** truth["log10_p_escape"][feasible]
    e_true = 10.0 ** truth["log10_e_fr"][feasible]
    ok_true = (p_true <= spec.max_p_escape) & (e_true <= spec.max_e_fr)
    ok_pred = predicted_violation(spec, {k: v[feasible] for k, v in prediction.items() if k in TARGET_NAMES}) <= 1e-12
    tp = int(np.sum(ok_true & ok_pred))
    return dict(
        n=int(feasible.sum()), n_satisfying=int(ok_true.sum()), accuracy=float(np.mean(ok_true == ok_pred)),
        precision=tp / max(int(ok_pred.sum()), 1), recall=tp / max(int(ok_true.sum()), 1),
    )


def run_surrogate_study(
    scenario: InspectionScenario,
    pool,
    spec: PlanSpec,
    k_values: Sequence[int] = (1, 2, 3, 4, 5, 6),
    search_bases: Sequence[np.ndarray] = (),
    held_out_region: Tuple[Sequence[float], Sequence[float]] = ((2.5, 2.0, 0.0), (5.0, 5.5, 3.0)),
    generate_kwargs: Optional[dict] = None,
    seed: int = 0,
) -> SurrogateStudy:
    """Generate records, split, train, and score the surrogate.

    The split, in order: (1) groups with a station in `held_out_region`
    are removed from training/validation altogether and form the
    *extrapolation* test set; (2) the rest are split 60/20/20 by group into
    train / validation / test (the *interpolation* test set). The model is
    fitted on train with its boosting rounds chosen on validation; nothing
    else touches test. The default region is the far corner of the station
    box (x 2.5-5 m, y 2-5.5 m): stations there see the part at long range
    from one side, a regime the training layouts mostly lack.
    """
    import time

    t0 = time.time()
    records = generate_layout_records(scenario, pool, k_values=k_values, search_bases=search_bases, seed=seed,
                                      **(generate_kwargs or {}))
    seconds_generate = time.time() - t0
    exact_ms = 1000.0 * seconds_generate / len(records)

    k_max = max(k_values) + 2
    groups = np.array([r.group for r in records])
    # A group is held out as a whole if *any* of its layouts has a station in
    # the region: a jitter that wandered out of the box is still a
    # near-duplicate of its base layout, and must not train the model that is
    # then tested on the base.
    in_region = np.isin(groups, np.unique(groups[region_holdout_mask(records, *held_out_region)]))
    train_m, val_m, test_m = grouped_split(np.where(in_region, -1, groups), seed=seed)
    train_m &= ~in_region
    val_m &= ~in_region
    test_m &= ~in_region
    assert not (train_m & val_m).any() and not (train_m & test_m).any() and not (val_m & test_m).any()
    for mask in (train_m, val_m, test_m):  # the region is out of all three, not only out of training
        assert not region_holdout_mask([r for r, keep in zip(records, mask) if keep], *held_out_region).any()
    train_g, val_g, test_g = (np.unique(groups[m]) for m in (train_m, val_m, test_m))
    assert set(train_g).isdisjoint(val_g) and set(train_g).isdisjoint(test_g) and set(val_g).isdisjoint(test_g)

    def part(mask):
        chosen = [r for r, keep in zip(records, mask) if keep]
        return batch_features([r.positions_m for r in chosen], k_max), rescore(scenario, chosen)

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te), (x_re, y_re) = part(train_m), part(val_m), part(test_m), part(in_region)
    t0 = time.time()
    model = LayoutSurrogate(k_max=k_max).fit(x_tr, y_tr, x_va, y_va)
    seconds_train = time.time() - t0

    pred_te, pred_re = model.predict(x_te), model.predict(x_re)
    t0 = time.time()
    batch = np.tile(x_te, (max(1, 2000 // max(len(x_te), 1)), 1))
    model.predict(batch)
    ms_per_layout = 1000.0 * (time.time() - t0) / len(batch)

    return SurrogateStudy(
        model=model, n_records=len(records), n_feasible=int(sum(r.feasible for r in records)),
        n_train=int(train_m.sum()), n_val=int(val_m.sum()), n_test_groups=int(test_m.sum()), n_test_region=int(in_region.sum()),
        train_groups=train_g, val_groups=val_g, test_groups=test_g,
        report_groups=regression_report(y_te, pred_te), report_region=regression_report(y_re, pred_re),
        feasibility_accuracy_groups=float(np.mean((pred_te["p_feasible"] > 0.5) == y_te["feasible"])),
        feasibility_accuracy_region=float(np.mean((pred_re["p_feasible"] > 0.5) == y_re["feasible"])),
        boundary_groups=boundary_metrics(spec, y_te, pred_te),
        truth_groups=y_te, prediction_groups=pred_te, truth_region=y_re, prediction_region=pred_re,
        seconds_generate=seconds_generate, seconds_train=seconds_train,
        exact_ms_per_layout=exact_ms, surrogate_ms_per_layout_batch=ms_per_layout,
    )
