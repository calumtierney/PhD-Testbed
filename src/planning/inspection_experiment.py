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

Four plans are compared, because two effects are tangled in "risk-driven"
and the yardstick for "no riskier" has two defensible readings:

  U      every point measured, every point at u <= the TUR limit.
  R-match risk limits = the risk plan U's own layout actually delivers
         (strictest: same or lower risk than the conventional plan really
         achieves), unmeasured points allowed and priced.
  R-all  risk limits = what the TUR rule *guarantees* (risk if every point
         sat exactly at the TUR limit), every point must still be measured --
         isolates what the *risk allocation* (a marginal cluster held
         tight, a high-capability cluster allowed to be loose) buys alone.
  R      the same TUR-guarantee limits, unmeasured points allowed and priced --
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


def seed_layouts_for(scenario: InspectionScenario, pool: CandidatePool, n_each: int = 4, max_stations: int = 14,
                     seed: int = 0) -> List[np.ndarray]:
    """The seed set shared by every plan: greedy covers weighted two ways
    (all points equally -- the cover plan U needs; and by each point's
    escape risk if left unseen -- the cover plan R values), each built twice,
    requiring a new station to share at least 6 points with the cover so far
    (a well-tied network) and at least 3 (the bare minimum for a rigid one).
    The looser tie is what lets a seed reach the few points -- the underside
    of the nose and the lower flap skin -- that only a station standing off
    the part's end can see; without it no seed covers all 500 points and a
    search that must measure every point never finds a feasible layout (a
    first version of this experiment failed exactly so). Seeds only: they
    start the population, the plans' own constraints decide what survives."""
    ones = np.ones(scenario.n_points)
    skip_risk, _ = scenario.point_risks(np.full(scenario.n_points, 1e-6), np.zeros(scenario.n_points, dtype=bool))
    risk = skip_risk / skip_risk.max()
    seeds: List[np.ndarray] = []
    for offset, (weights, min_shared) in enumerate(((ones, 6), (ones, 3), (risk, 6), (risk, 3))):
        seeds += greedy_cover_layouts(pool, weights, n_each, max_stations, seed=seed + offset,
                                      min_shared_points=min_shared)
    return seeds


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
    k_values: Sequence[int] = tuple(range(1, 11)),
    n_candidates: int = 20000,
    de_kwargs: Optional[dict] = None,
    seed: int = 0,
    risk_slack: float = 1.0,
    min_task_coverage: float = 0.0,
    include_r_all: bool = True,
    pool: Optional[CandidatePool] = None,
    verbose: bool = False,
) -> InspectionExperimentResult:
    """Run plans U, R-match, R-all and R on `scenario` with identical search
    settings. `risk_slack` scales plan R's two risk limits (1.0 = no worse
    than the TUR rule's own guarantee). `min_task_coverage` is plan R's
    sampling-policy floor (see `planning.inspection.PlanSpec`);
    `include_r_all=False` skips the every-point-measured risk plan, which
    sensitivity sweeps do not need."""
    timing: Dict[str, float] = {}
    t0 = time.time()
    pool = pool if pool is not None else build_candidate_pool(scenario, n_candidates, seed)
    seeds = seed_layouts_for(scenario, pool, seed=seed)
    timing["candidate_pool_and_seeds"] = time.time() - t0

    specs = {
        "U": uniform_rule_plan(scenario, tur),
        "R-all": iso_risk_plan(scenario, tur, allow_unmeasured=False, slack=risk_slack),
        "R": iso_risk_plan(scenario, tur, allow_unmeasured=True, slack=risk_slack, min_task_coverage=min_task_coverage),
    }
    runs = {}

    def run(name, spec):
        runs[name] = run_plan(scenario, name, spec, k_values, seeds, de_kwargs, seed, pool.positions_m)
        timing[name] = runs[name].seconds
        if verbose:
            r = runs[name].answer
            print(f"  plan {name}: " + (f"{r.n_stations} stations" if r else "no layout found")
                  + f" in {runs[name].seconds:.0f}s", flush=True)

    run("U", specs["U"])
    answer_u = runs["U"].answer
    if answer_u is not None:
        # The strictest comparison: plan R held to the risk plan U's own
        # layout *actually delivers* (not the ceiling the TUR rule merely
        # guarantees). U's layouts usually beat that ceiling, so matching
        # their realised risk with fewer stations leaves no room to say
        # plan R is the less safe plan.
        specs["R-match"] = PlanSpec("R", max_p_escape=answer_u.outcome.p_escape * risk_slack,
                                    max_e_fr=answer_u.outcome.e_fr * risk_slack, allow_unmeasured=True,
                                    min_task_coverage=min_task_coverage)
        run("R-match", specs["R-match"])
    if include_r_all:
        run("R-all", specs["R-all"])
    run("R", specs["R"])
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
