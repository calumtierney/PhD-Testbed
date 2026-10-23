"""The wing-box inspection experiment: plan U (uncertainty-driven) vs plan R
(risk-driven), searched identically.

`planning.inspection` defines the scenario, the layout evaluator and the
plans' constraints; this module runs them. The experiment varies the plan's
*objective* and nothing else: every plan is searched by the same
differential-evolution driver (`planning.search.run_station_search`), over
the same bounds, with the same hyper-parameters and seeded the same two ways
(coverage-greedy covers, and a forward-greedy layout built with that plan's
own cost) -- so any difference in the number of stations is attributable to
what the plan was asked for (CLAUDE.md §3: the optimiser is held constant).

  U       the conventional, A-optimal planner: every point measured, layouts
          chosen to minimise mean variance, the number of stations taken
          from the knee of that criterion's station-count curve (the point
          where one more station stops paying -- `knee_station_count`).
          Blind to the tolerance and to which features are hard to hold.
  R-match plan R held to the risk plan U's own layout *actually delivers*
          (P_escape and E_FR no worse), unmeasured points allowed and priced.
          The strictest comparison: same or lower risk than the
          conventional plan really achieves, with fewer stations.
  R       plan R held to what the TUR 4:1 rule *guarantees* (the risk if every
          point sat exactly at the TUR limit) -- an externally anchored limit
          that does not depend on plan U.
  R-all   (optional) the same limit, every point still measured -- isolates
          what the risk allocation alone buys, before any point is skipped.
  A       (optional) the same as U but minimising the raw covariance trace --
          what existing tools minimise, with no tolerance direction at all
          (U, which projects onto the surface normal, is the stronger
          baseline). R-match-A is plan R held to A's realised risk.
  T       (optional) the hard TUR rule itself: every point at u <= tolerance/
          8. Reported because it is the criterion a standards-minded reader
          expects; at the tolerances here it is usually not achievable at all
          (a handful of grazing, single-station points defeat it), which is
          itself the finding.

Each plan's layout is then re-scored under the others' metrics
(`cross_evaluate`), so a plan that is cheaper by meeting a weaker guarantee
cannot hide: the table says what each one's layout delivers on both
yardsticks.
"""
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from planning.inspection import (
    CandidatePool, InspectionScenario, KResult, LayoutOutcome, PlanSpec, build_candidate_pool, evaluate_layout,
    greedy_cost_layout, greedy_cover_layouts, iso_risk_plan, knee_station_count, plan_violation, refine_backward,
    search_station_counts, task_summary, tur_rule_plan, uncertainty_plan,
)

DEFAULT_DE = dict(maxiter=60, popsize=12, tol=1e-8, workers=-1)


@dataclass(frozen=True)
class PlanRun:
    """One plan's K-sweep.

    `sweep` has one `KResult` per station count tried, ascending. `answer`
    is `chosen` when the plan picks its own count (plan U, from the knee of
    its curve), otherwise the first count whose best layout met the plan's
    constraint, or None if none did within the sweep -- reported as "no
    layout found up to K", never as "impossible" (a GA finds a layout, not a
    proof)."""

    name: str
    spec: PlanSpec
    sweep: List[KResult]
    seconds: float
    chosen: Optional[KResult] = None

    @property
    def answer(self) -> Optional[KResult]:
        if self.chosen is not None:
            return self.chosen
        for r in self.sweep:
            if r.satisfied:
                return r
        return None


@dataclass(frozen=True)
class InspectionExperimentResult:
    scenario: InspectionScenario
    runs: Dict[str, PlanRun]
    seed_layouts: List[np.ndarray]
    pool_size: int
    timing_s: Dict[str, float]
    u_curve: Dict[int, float]  # plan U's criterion (mean variance, um^2) at each station count
    knee_by_gain: Dict[float, int]  # plan U's station count under several "stops paying" thresholds


def seed_layouts_for(scenario: InspectionScenario, pool: CandidatePool, n_each: int = 4, max_stations: int = 14,
                     seed: int = 0) -> List[np.ndarray]:
    """Coverage-greedy seeds shared by every plan: greedy covers weighted two
    ways (all points equally; by each point's escape risk if left unseen),
    each built twice, requiring a new station to share at least 6 points
    with the cover so far (a well-tied network) and at least 3 (the bare
    minimum for a rigid one). The looser tie is what lets a seed reach the
    few points -- the underside of the nose and the lower flap skin -- that
    only a station standing off the part's end can see; without it no seed
    covers all 500 points and a search that must measure every point never
    finds a feasible layout (a first version of this experiment failed
    exactly so). Seeds only: they start the population, the plans' own
    objectives decide what survives."""
    ones = np.ones(scenario.n_points)
    skip_risk, _ = scenario.point_risks(np.full(scenario.n_points, 1e-6), np.zeros(scenario.n_points, dtype=bool))
    risk = skip_risk / skip_risk.max()
    seeds: List[np.ndarray] = []
    for offset, (weights, min_shared) in enumerate(((ones, 6), (ones, 3), (risk, 6), (risk, 3))):
        seeds += greedy_cover_layouts(pool, weights, n_each, max_stations, seed=seed + offset,
                                      min_shared_points=min_shared)
    return seeds


def rescore_run(scenario: InspectionScenario, run: PlanRun) -> PlanRun:
    """The same layouts as `run`, re-evaluated under `scenario` (a different
    tolerance, prior, ... -- the geometry and uncertainties are unchanged,
    the risk numbers are not). For plans whose search did not depend on what
    differs between the scenarios (plan U)."""
    sweep = []
    for r in run.sweep:
        outcome = evaluate_layout(scenario, r.positions_m, run.spec.allow_unmeasured)
        sweep.append(KResult(r.n_stations, r.positions_m, outcome, plan_violation(run.spec, outcome), r.n_function_evals))
    chosen = None
    if run.chosen is not None:
        chosen = next(r for r in sweep if r.n_stations == run.chosen.n_stations)
    return PlanRun(run.name, run.spec, sweep, run.seconds, chosen)


def run_plan(scenario: InspectionScenario, name: str, spec: PlanSpec, k_values: Sequence[int],
             seed_layouts: Sequence[np.ndarray], de_kwargs: Optional[dict] = None, seed: int = 0,
             extension_positions_m: Optional[np.ndarray] = None) -> PlanRun:
    t0 = time.time()
    sweep = search_station_counts(scenario, spec, k_values, de_kwargs=de_kwargs or DEFAULT_DE, seed=seed,
                                  seed_layouts=seed_layouts, extension_positions_m=extension_positions_m)
    return PlanRun(name, spec, sweep, time.time() - t0)


def run_inspection_experiment(
    scenario: InspectionScenario,
    tur: float = 4.0,
    k_values_u: Sequence[int] = tuple(range(3, 15)),
    k_values_r: Sequence[int] = tuple(range(1, 11)),
    n_candidates: int = 20000,
    de_kwargs: Optional[dict] = None,
    seed: int = 0,
    risk_slack: float = 1.0,
    min_task_coverage: float = 0.0,
    knee_gain: float = 0.10,
    include_r_all: bool = False,
    include_tur_rule: bool = False,
    include_a_trace: bool = False,
    backward_pass: bool = True,
    pool: Optional[CandidatePool] = None,
    u_run: Optional[PlanRun] = None,
    verbose: bool = False,
) -> InspectionExperimentResult:
    """Run plans U, R-match, R (and optionally R-all and T) on `scenario`
    with identical search settings.

    Plan U is swept over `k_values_u` (all of them -- it never "meets" a
    constraint, it just gets better), its count taken at the knee of the
    mean-variance curve (`knee_gain`: the smallest K where the next station
    lowers mean variance by less than that fraction). Plan R's limits are
    scaled by `risk_slack` (1.0 = no worse than the reference) and floored by
    `min_task_coverage` (see `planning.inspection.PlanSpec`).

    Plan U's cost -- mean variance of the points' uncertainty -- involves
    neither the tolerance nor the process priors, so its layouts are the
    same in every scenario variant; pass an earlier result's `runs["U"]` as
    `u_run` and they are re-scored under this scenario's risk model instead
    of searched again (the search is most of the run time). `backward_pass`
    adds `planning.inspection.refine_backward` to every sweep.
    """
    timing: Dict[str, float] = {}
    t0 = time.time()
    pool = pool if pool is not None else build_candidate_pool(scenario, n_candidates, seed)
    coverage_seeds = seed_layouts_for(scenario, pool, seed=seed)
    timing["candidate_pool_and_cover_seeds"] = time.time() - t0

    runs: Dict[str, PlanRun] = {}

    def run(name: str, spec: PlanSpec, k_values, stop_at_first_satisfied: bool = True) -> PlanRun:
        t1 = time.time()
        own_greedy = greedy_cost_layout(scenario, spec, pool, max(k_values), seed=seed)
        seeds = coverage_seeds + [own_greedy]
        sweep = search_station_counts(scenario, spec, k_values, de_kwargs=de_kwargs or DEFAULT_DE, seed=seed,
                                      stop_at_first_satisfied=stop_at_first_satisfied, seed_layouts=seeds,
                                      extension_positions_m=pool.positions_m)
        if backward_pass and len(sweep) > 1 and sweep[-1].outcome.feasible:
            sweep = refine_backward(scenario, spec, sweep, de_kwargs=de_kwargs, seed=seed)
        runs[name] = PlanRun(name, spec, sweep, time.time() - t1)
        timing[name] = runs[name].seconds
        if verbose:
            r = runs[name].answer
            print(f"  plan {name}: " + (f"{r.n_stations} stations" if r else "no layout found")
                  + f" in {runs[name].seconds:.0f}s", flush=True)
        return runs[name]

    if u_run is None:
        run_u = run("U", uncertainty_plan(), k_values_u, stop_at_first_satisfied=False)
    else:
        run_u = rescore_run(scenario, u_run)
        runs["U"] = run_u
    u_curve = {r.n_stations: float(np.mean(r.outcome.u_m ** 2)) * 1e12 for r in run_u.sweep if r.outcome.feasible}
    knee_by_gain = {g: knee_station_count(u_curve, g) for g in (0.20, 0.10, 0.05, 0.02)} if len(u_curve) > 1 else {}
    chosen = None
    if len(u_curve) > 1:
        k_star = knee_station_count(u_curve, knee_gain)
        chosen = next(r for r in run_u.sweep if r.n_stations == k_star)
        runs["U"] = replace(run_u, chosen=chosen)

    if chosen is not None:
        # The strictest comparison: plan R held to the risk plan U's own layout
        # actually delivers, not a ceiling it merely stays under.
        run("R-match", PlanSpec("R", max_p_escape=chosen.outcome.p_escape * risk_slack,
                                max_e_fr=chosen.outcome.e_fr * risk_slack, allow_unmeasured=True,
                                min_task_coverage=min_task_coverage), k_values_r)
    run("R", iso_risk_plan(scenario, tur, allow_unmeasured=True, slack=risk_slack, min_task_coverage=min_task_coverage),
        k_values_r)
    if include_a_trace:
        # The raw A-optimal criterion (mean covariance trace), for fidelity to
        # what existing tools minimise; plan U above is the stronger baseline,
        # since projecting onto the normal gives it the tolerance direction.
        run_a = run("A", uncertainty_plan(trace=True), k_values_u, stop_at_first_satisfied=False)
        a_curve = {r.n_stations: float(np.mean(r.outcome.trace_m2)) * 1e12 for r in run_a.sweep if r.outcome.feasible}
        if len(a_curve) > 1:
            a_star = next(r for r in run_a.sweep if r.n_stations == knee_station_count(a_curve, knee_gain))
            runs["A"] = replace(run_a, chosen=a_star)
            run("R-match-A", PlanSpec("R", max_p_escape=a_star.outcome.p_escape * risk_slack,
                                      max_e_fr=a_star.outcome.e_fr * risk_slack, allow_unmeasured=True,
                                      min_task_coverage=min_task_coverage), k_values_r)
    if include_r_all:
        run("R-all", iso_risk_plan(scenario, tur, allow_unmeasured=False, slack=risk_slack), tuple(k_values_u))
    if include_tur_rule:
        run("T", tur_rule_plan(scenario, tur), tuple(k_values_u))
    return InspectionExperimentResult(scenario, runs, coverage_seeds, len(pool.positions_m), timing, u_curve,
                                      knee_by_gain)


def cross_evaluate(scenario: InspectionScenario, positions_m: np.ndarray, tur: float = 4.0) -> Dict[str, float]:
    """What a layout delivers on both yardsticks, whatever plan it was
    built for: coverage, the uniform rule's uncertainty test, and the risk
    metrics."""
    outcome = evaluate_layout(scenario, positions_m, allow_unmeasured=True)
    if not outcome.feasible:
        return dict(feasible=False)
    u_limit = scenario.uniform_uncertainty_limit_m(tur)
    measured = outcome.measured
    return dict(
        feasible=True, n_stations=outcome.n_stations, n_measured=outcome.n_measured,
        n_unmeasured=int((~measured).sum()),
        n_over_tur_limit=int((outcome.u_m[measured] > u_limit).sum()),
        mean_u_um=float(np.mean(outcome.u_m[measured]) * 1e6), max_u_um=float(np.max(outcome.u_m[measured]) * 1e6),
        n_shots=outcome.n_shots, e_fa=outcome.e_fa, e_fr=outcome.e_fr, p_escape=outcome.p_escape,
        max_point_pfa=float(outcome.pfa.max()),
    )
