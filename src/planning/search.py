"""Station-placement search (CLAUDE.md §5, step 6): grid search, and a
standard GA for when grid search's cost stops being tractable.

CLAUDE.md §3: "The optimiser is deliberately conventional and held
constant. The experiment varies the objective, not the search method.
Use grid search or a standard GA. Do not invent an optimiser."

**Grid search** (`grid_search_station_placement`) -- exhaustive evaluation
of every combination of `n_stations` distinct poses drawn from a candidate
set (typically `planning.candidates.hemisphere_candidate_poses`). Only
tractable because `n_stations` is small and the candidate set is modest:
the number of combinations is `C(n_candidates, n_stations)`, and every one
costs one `objectives.evaluate_plan` call plus one `objectives.ObjectiveFn`
call. This was CLAUDE.md §5 step 6's original scene's whole budget (24
candidates, 2 stations, C(24, 2) = 276 combinations) -- small enough that
a GA was flagged as an unneeded fallback (docs/step6_headline_experiment.md).

**GA** (`genetic_algorithm_station_placement`) -- needed the moment either
factor above grows past grid search's reach: a finer or larger-area
candidate grid (`planning.candidates.floor_grid_candidate_poses`, for a
scene where distance to the part is itself part of what a placement
should vary, not just angle at one fixed standoff) multiplies
`n_candidates`, and `C(n_candidates, n_stations)` grows combinatorially
in both factors at once -- a few hundred floor-grid candidates with 3+
stations easily reaches millions of combinations, each paying objective
C's `scipy.integrate.quad` cost. Uses `scipy.optimize.differential_evolution`
-- scipy's own standard, broadly-used evolutionary/genetic-family global
optimiser (no new dependency, CLAUDE.md §6) -- searching directly over
each station's continuous (x, y, z) position rather than discretising
first and then combinatorially choosing among the discretisation. This
also removes the double-discretisation problem a *finer* grid search
would otherwise have (candidate spacing *and* which combination of
candidates), at the cost of losing grid search's exhaustiveness guarantee
-- a GA finds a good plan, not provably the best one in the search space,
same caveat any GA carries.
"""
import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from geometry.pose import InstrumentPose
from planning.candidates import look_at_orientation
from planning.objectives import ObjectiveFn, PlanEvaluation, PlanningScenario, evaluate_plan


@dataclass(frozen=True)
class GridSearchResult:
    """The outcome of an exhaustive grid search.

    Attributes
    ----------
    best : the lowest-objective-value PlanEvaluation found, or None if
        every candidate was disqualified (e.g. by a `constrained_objective`
        cap that nothing in the candidate set satisfies, or -- once a
        scenario's visibility layer is switched on -- every combination
        left some target unseen by any station).
    best_objective_value : the objective value `best` achieved (for
        reporting/comparison across searches); None alongside `best=None`.
    all_evaluations : every *feasible* candidate combination's
        PlanEvaluation, in the order evaluated -- kept for
        reporting/plotting (e.g. a distribution of achieved objective
        values across the candidate set), not just the single winner.
        Combinations `objectives.evaluate_plan` returned `None` for (see
        `n_infeasible`) are not included here -- there is no evaluation to
        report.
    n_infeasible : how many of the `C(n_candidates, n_stations)`
        combinations `evaluate_plan` disqualified outright (visibility
        left some target or non-anchor station unobservable), before an
        objective value was ever computed for them. 0 for every scenario
        built without a visibility layer (CLAUDE.md §8: a null result --
        here, "visibility never binds" -- is reported as prominently as a
        positive one, not silently omitted).
    """

    best: Optional[PlanEvaluation]
    best_objective_value: Optional[float]
    all_evaluations: List[PlanEvaluation]
    n_infeasible: int = 0


def grid_search_station_placement(
    candidate_poses: List[InstrumentPose],
    n_stations: int,
    scenario: PlanningScenario,
    objective_fn: ObjectiveFn,
) -> GridSearchResult:
    """Exhaustively score every `n_stations`-combination of
    `candidate_poses` under `objective_fn`, returning the best and every
    evaluation.

    Combinations, not permutations: which candidate pose is "station 1"
    vs "station 2" doesn't affect a plan's quality (the network solve is
    symmetric in station order beyond the fixed anchor -- CLAUDE.md §5
    step 3), so there's no reason to pay for evaluating both orderings.

    `objective_fn` is any `objectives.ObjectiveFn` -- e.g.
    `objectives.make_weighted_uncertainty_objective(objectives.uniform_weights(scenario))`
    for A1, the same with `risk_derived_weights(...)` for B,
    `objectives.objective_a0_trace_covariance` for A0, or
    `objectives.objective_c_direct_risk` for C -- optionally wrapped in
    `objectives.constrained_objective` for a hard per-characteristic
    uncertainty cap.
    """
    if n_stations > len(candidate_poses):
        raise ValueError(
            f"n_stations ({n_stations}) exceeds the number of candidate poses ({len(candidate_poses)})"
        )

    all_evaluations = []
    scored = []
    n_infeasible = 0
    for combo_indices in itertools.combinations(range(len(candidate_poses)), n_stations):
        station_poses = [candidate_poses[i] for i in combo_indices]
        evaluation = evaluate_plan(station_poses, scenario)
        if evaluation is None:
            n_infeasible += 1
            continue
        all_evaluations.append(evaluation)
        value = objective_fn(evaluation, scenario)
        if value < float("inf"):
            scored.append((value, evaluation))

    if not scored:
        return GridSearchResult(
            best=None, best_objective_value=None, all_evaluations=all_evaluations, n_infeasible=n_infeasible
        )

    best_value, best_evaluation = min(scored, key=lambda pair: pair[0])
    return GridSearchResult(
        best=best_evaluation,
        best_objective_value=best_value,
        all_evaluations=all_evaluations,
        n_infeasible=n_infeasible,
    )


# A value no real objective in this codebase ever reaches: projected
# characteristic uncertainty is metres (~1e-5 scale), a trace of
# covariance similarly small, and total risk (objective C) is bounded by
# the number of characteristics (a handful, each PFA+PFR <= ~1) -- so
# 1e6 is always worse than any feasible candidate, while still being a
# finite number `scipy.optimize.differential_evolution` can treat as an
# ordinary (bad) objective value. This is deliberately not float("inf"):
# DE's internal operations (recombination, convergence-tolerance checks)
# are arithmetic on the objective values themselves, and propagate inf/nan
# rather than comparing past them the way `min()` over a Python list does.
INFEASIBLE_PENALTY = 1e6


@dataclass(frozen=True)
class GASearchResult:
    """The outcome of a differential-evolution station-placement search --
    the GA-search analogue of `GridSearchResult`.

    Attributes
    ----------
    best : the best PlanEvaluation the search found, or None if even the
        search's own best-found position vector was infeasible (e.g. a
        `bounds_m` box inside which every arrangement leaves some target
        occluded from every candidate).
    best_objective_value : the objective value `best` achieved; None
        alongside `best=None`.
    n_function_evals, n_generations, converged : scipy's own
        `differential_evolution` diagnostics (`nfev`, `nit`, `success`) --
        report these alongside the result, the same way `NetworkSolveResult`
        reports `n_function_evals`/`success` for the network solve's own
        optimiser (CLAUDE.md §6: a search result is a claim about an
        optimisation having actually converged, not just a number).
    n_infeasible_evals : how many of the search's own candidate
        evaluations hit `objectives.evaluate_plan`'s disqualification path
        (visibility leaving some target or station unobservable) and were
        scored at `INFEASIBLE_PENALTY` instead of their real objective
        value -- CLAUDE.md §8's "report a null/negative result
        prominently" applied to the GA path the same way
        `GridSearchResult.n_infeasible` applies it to the grid path.
    """

    best: Optional[PlanEvaluation]
    best_objective_value: Optional[float]
    n_function_evals: int
    n_generations: int
    converged: bool
    n_infeasible_evals: int


def genetic_algorithm_station_placement(
    scenario: PlanningScenario,
    n_stations: int,
    objective_fn: ObjectiveFn,
    bounds_m: Sequence[Tuple[float, float]],
    aim_point_m: Optional[np.ndarray] = None,
    seed: Optional[int] = 0,
    **differential_evolution_kwargs,
) -> GASearchResult:
    """Search `n_stations` continuous (x, y, z) station positions with
    `scipy.optimize.differential_evolution`, instead of exhaustively
    combining a pre-built candidate list (see the module docstring for
    when this is the one to reach for instead of grid search).

    Every station is boresighted at `aim_point_m` (default: the scenario's
    target centroid -- same convention `planning.candidates.
    hemisphere_candidate_poses` uses) via `planning.candidates.
    look_at_orientation`, so the search is genuinely 3 DOF per station
    (position only), not the full 6 -- the same dimensionality reduction
    the candidate-list approach makes, just without discretising first.
    For a scene spread over tens of metres where one station's best
    aim point is clearly not the whole scene's centroid, pass a different
    `aim_point_m` explicitly; this function does not (yet) let different
    *stations* aim at different points -- CLAUDE.md §10's "flag anything
    that would need a bigger design change before building it" applies
    here: that would need per-station aim points as a further search
    dimension, not built in this first cut.

    `bounds_m` is `[(x_min, x_max), (y_min, y_max), (z_min, z_max)]`,
    applied identically to every station (the same box, searched
    independently per station -- stations are free to land anywhere in
    it, including near each other; nothing here forces them apart). Scale
    it to the part: CLAUDE.md §1's 2-30 m domain means a compact scene's
    box might be a few metres on a side, a 30 m wing's considerably larger.

    A candidate position that `objectives.evaluate_plan` disqualifies
    (visibility leaves some target or station unobservable) scores
    `INFEASIBLE_PENALTY` rather than being excluded from the search space
    outright -- DE needs an ordinary numeric objective value everywhere in
    its box to operate at all, unlike grid search's hard skip. This also
    means DE can report a best position that is itself infeasible (if
    *every* position in `bounds_m` is), surfaced as `best=None`.

    `**differential_evolution_kwargs` passes straight through to
    `scipy.optimize.differential_evolution` (e.g. `popsize`, `maxiter`,
    `tol`, `workers` for parallel evaluation) -- tune these against
    `objective_fn`'s own cost per call (objective C's risk integration is
    roughly two orders of magnitude slower per call than A0/A1/B, per
    `planning.experiment`'s own runtime note) rather than this function
    hard-coding a one-size-fits-all budget.
    """
    if aim_point_m is None:
        aim_point_m = scenario.target_points_m.mean(axis=0)
    aim_point_m = np.asarray(aim_point_m, dtype=float)
    bounds_m = list(bounds_m)
    if len(bounds_m) != 3:
        raise ValueError(f"bounds_m must have exactly 3 (min, max) pairs (x, y, z), got {len(bounds_m)}")

    full_bounds = bounds_m * n_stations
    infeasible_evals = 0

    def _station_poses(flat_params: np.ndarray) -> List[InstrumentPose]:
        positions_m = np.asarray(flat_params).reshape(n_stations, 3)
        return [
            InstrumentPose(position_m=position_m, orientation=look_at_orientation(position_m, aim_point_m))
            for position_m in positions_m
        ]

    def cost(flat_params: np.ndarray) -> float:
        nonlocal infeasible_evals
        evaluation = evaluate_plan(_station_poses(flat_params), scenario)
        if evaluation is None:
            infeasible_evals += 1
            return INFEASIBLE_PENALTY
        value = objective_fn(evaluation, scenario)
        return value if value < float("inf") else INFEASIBLE_PENALTY

    de_result = differential_evolution(cost, full_bounds, seed=seed, **differential_evolution_kwargs)

    best_evaluation = evaluate_plan(_station_poses(de_result.x), scenario)
    best_value = (
        None if best_evaluation is None else objective_fn(best_evaluation, scenario)
    )
    if best_value is not None and best_value >= INFEASIBLE_PENALTY:
        best_evaluation, best_value = None, None

    return GASearchResult(
        best=best_evaluation,
        best_objective_value=best_value,
        n_function_evals=de_result.nfev,
        n_generations=de_result.nit,
        converged=bool(de_result.success),
        n_infeasible_evals=infeasible_evals,
    )
