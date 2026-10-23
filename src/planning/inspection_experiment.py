"""The wing-box inspection experiment: plan U (uniform uncertainty rule) vs
plan R (risk-driven), searched identically.

`planning.inspection` defines the scenario, the layout evaluator and the two
plans' constraints; this module runs them. The experiment varies the plan's
*constraint* and nothing else: both plans are searched by the same
differential-evolution driver (`planning.search.run_station_search`), over
the same bounds, with the same hyper-parameters and the same set of seed
layouts -- so any difference in the number of stations is attributable to
what the plan was asked to guarantee (CLAUDE.md §3: the optimiser is held
constant).

Three plans are compared, because two effects are tangled in "risk-driven":

  U      every point measured, every point at u <= the TUR limit.
  R-all  risk limits as plan R, but every point must still be measured --
         isolates what the *risk allocation* (a marginal cluster held
         tight, a high-capability cluster allowed to be loose) buys alone.
  R      risk limits as plan R, unmeasured points allowed and priced --
         adds "do not measure what does not matter".

Each is then re-scored under the others' metrics (`cross_evaluate`), so a
plan that is cheaper by meeting a weaker guarantee cannot hide: the table
says what each one's layout delivers on both yardsticks.
"""
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from planning.inspection import (
    CandidatePool, InspectionScenario, KResult, LayoutOutcome, PlanSpec, build_candidate_pool, evaluate_layout,
    greedy_cover_layouts, iso_risk_plan, plan_violation, search_station_counts, task_summary, uniform_rule_plan,
)

DEFAULT_DE = dict(maxiter=60, popsize=12, tol=1e-8, workers=-1)


@dataclass(frozen=True)
class PlanRun:
    """One plan's K-sweep.

    `sweep` has one `KResult` per station count tried, ascending. `answer`
    is the first count whose best layout met the plan's constraint, or None
    if none did within the sweep -- reported as "no layout found up to K",
    never as "impossible" (a GA finds a layout, not a proof)."""

    name: str
    spec: PlanSpec
    sweep: List[KResult]
    seconds: float

    @property
    def answer(self) -> Optional[KResult]:
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


def seed_layouts_for(scenario: InspectionScenario, pool: CandidatePool, n_each: int = 6, max_stations: int = 10,
                     seed: int = 0) -> List[np.ndarray]:
    """The seed set shared by every plan: greedy covers weighted two ways
    (all points equally -- the cover plan U needs; and by each point's
    escape risk if left unseen -- the cover plan R values). Seeds only: they
    start the population, the plans' own constraints decide what survives."""
    ones = np.ones(scenario.n_points)
    skip_risk, _ = scenario.point_risks(np.full(scenario.n_points, 1e-6), np.zeros(scenario.n_points, dtype=bool))
    return (greedy_cover_layouts(pool, ones, n_each, max_stations, seed=seed)
            + greedy_cover_layouts(pool, skip_risk / skip_risk.max(), n_each, max_stations, seed=seed + 1))


def run_plan(scenario: InspectionScenario, name: str, spec: PlanSpec, k_values: Sequence[int],
             seed_layouts: Sequence[np.ndarray], de_kwargs: Optional[dict] = None, seed: int = 0) -> PlanRun:
    t0 = time.time()
    sweep = search_station_counts(scenario, spec, k_values, de_kwargs=de_kwargs or DEFAULT_DE, seed=seed,
                                  seed_layouts=seed_layouts)
    return PlanRun(name, spec, sweep, time.time() - t0)


def run_inspection_experiment(
    scenario: InspectionScenario,
    tur: float = 4.0,
    k_values: Sequence[int] = tuple(range(1, 11)),
    n_candidates: int = 20000,
    de_kwargs: Optional[dict] = None,
    seed: int = 0,
    risk_slack: float = 1.0,
    pool: Optional[CandidatePool] = None,
    verbose: bool = False,
) -> InspectionExperimentResult:
    """Run plans U, R-all and R on `scenario` with identical search
    settings. `risk_slack` scales plan R's two risk limits (1.0 = no worse
    than the TUR rule's own guarantee)."""
    timing: Dict[str, float] = {}
    t0 = time.time()
    pool = pool if pool is not None else build_candidate_pool(scenario, n_candidates, seed)
    seeds = seed_layouts_for(scenario, pool, seed=seed)
    timing["candidate_pool_and_seeds"] = time.time() - t0

    specs = {
        "U": uniform_rule_plan(scenario, tur),
        "R-all": iso_risk_plan(scenario, tur, allow_unmeasured=False, slack=risk_slack),
        "R": iso_risk_plan(scenario, tur, allow_unmeasured=True, slack=risk_slack),
    }
    runs = {}
    for name, spec in specs.items():
        runs[name] = run_plan(scenario, name, spec, k_values, seeds, de_kwargs, seed)
        timing[name] = runs[name].seconds
        if verbose:
            r = runs[name].answer
            print(f"  plan {name}: " + (f"{r.n_stations} stations" if r else "no layout found")
                  + f" in {runs[name].seconds:.0f}s", flush=True)
    return InspectionExperimentResult(scenario, runs, seeds, len(pool.positions_m), timing)


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
