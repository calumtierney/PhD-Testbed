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
from geometry.station_track import StationTrack
from planning.candidates import TrackPose, levelled_orientation, look_at_orientation
from planning.objectives import (
    ObjectiveFn,
    PlanEvaluation,
    PlanningScenario,
    evaluate_plan,
    plan_infeasibility_deficit,
    station_visibility_matrix,
    visibility_deficit,
)


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
    prefilter_visibility: bool = False,
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

    `prefilter_visibility=True` is a speed-up that changes no result: each
    candidate's visibility row is computed once, and a combination that
    already fails the cheap necessary conditions (a target nobody sees, a
    station seeing too few targets, a station not tied to the anchor --
    `objectives.visibility_deficit`) is counted as infeasible without being
    sent to `evaluate_plan`, which would ray-cast every station-target pair
    again just to reach the same verdict. Everything that passes goes through
    `evaluate_plan` exactly as before (it still applies the full rank test),
    so the evaluations, the best plan and `n_infeasible` are identical
    (tests/test_station_track_scenario.py checks this). It matters when
    nearly every combination is infeasible, as on a long part where only
    stations on the end loops can bridge the two sides.
    """
    if n_stations > len(candidate_poses):
        raise ValueError(
            f"n_stations ({n_stations}) exceeds the number of candidate poses ({len(candidate_poses)})"
        )

    all_evaluations = []
    scored = []
    n_infeasible = 0
    visibility_rows = (
        station_visibility_matrix(scenario, candidate_poses)
        if prefilter_visibility and scenario.visibility_enabled else None
    )
    for combo_indices in itertools.combinations(range(len(candidate_poses)), n_stations):
        if visibility_rows is not None:
            rows = [0] + [i + 1 for i in combo_indices]
            if visibility_deficit(visibility_rows[rows], 0) > 0.0:
                n_infeasible += 1
                continue
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
class _StationPlacementCost:
    """Picklable (module-level, no closure) per-candidate cost evaluator
    for `genetic_algorithm_station_placement`.

    Why this has to be a plain class rather than the nested function an
    earlier revision used: `scipy.optimize.differential_evolution`'s
    `workers` parameter (anything other than the default `1`) hands the
    cost function to a `multiprocessing.Pool`, which sends it to worker
    processes via the standard library's `pickle` -- and `pickle` cannot
    serialise a function defined inside another function (a closure),
    only a module-level name. `objective_fn` itself has to be picklable
    too for the same reason, which is why `objectives.
    make_weighted_uncertainty_objective`/`constrained_objective` return
    plain callable classes rather than closures as well.

    Deliberately stateless: `multiprocessing` gives each worker its own
    pickled copy, so a live "how many infeasible so far" counter could
    not be kept accurate across worker processes even if this held one
    -- see `genetic_algorithm_station_placement`'s own handling of
    `n_infeasible_evals` for the serial (`workers=1`, the default) case,
    where a counter *is* kept, precisely because it doesn't cross a
    process boundary there.
    """

    scenario: PlanningScenario
    objective_fn: ObjectiveFn
    aim_point_m: np.ndarray
    n_stations: int

    def station_poses(self, flat_params: np.ndarray) -> List[InstrumentPose]:
        positions_m = np.asarray(flat_params).reshape(self.n_stations, 3)
        return [
            InstrumentPose(position_m=position_m, orientation=look_at_orientation(position_m, self.aim_point_m))
            for position_m in positions_m
        ]

    def __call__(self, flat_params: np.ndarray) -> float:
        poses = self.station_poses(flat_params)
        evaluation = evaluate_plan(poses, self.scenario)
        if evaluation is None:
            # INFEASIBLE_PENALTY plus a small graded deficit (unobserved
            # targets, stations not tied to the anchor): DE only compares
            # values, so this slopes the infeasible region towards
            # feasibility instead of leaving it one flat plateau -- needed
            # once connectivity is enforced and most random layouts fail.
            return INFEASIBLE_PENALTY + plan_infeasibility_deficit(poses, self.scenario)
        value = self.objective_fn(evaluation, self.scenario)
        return value if value < float("inf") else INFEASIBLE_PENALTY


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
        `GridSearchResult.n_infeasible` applies it to the grid path. Only
        tracked when the search ran serially (`workers=1`, the default);
        0 whenever `workers != 1` was passed, since a live count cannot
        survive being split across `multiprocessing` worker processes
        (see `genetic_algorithm_station_placement`'s own docstring on
        `workers`) -- this is a reporting gap in the parallel path, not a
        claim that nothing was infeasible.
    """

    best: Optional[PlanEvaluation]
    best_objective_value: Optional[float]
    n_function_evals: int
    n_generations: int
    converged: bool
    n_infeasible_evals: int


def run_station_search(
    cost,
    n_stations: int,
    bounds_m: Sequence[Tuple[float, float]],
    seed: Optional[int] = 0,
    seed_station_positions_m: Optional[Sequence[Sequence[np.ndarray]]] = None,
    dims_per_station: int = 3,
    **differential_evolution_kwargs,
):
    """The one differential-evolution driver every station-placement
    search in this codebase goes through -- `genetic_algorithm_station_
    placement` (step 6) and `planning.inspection` (the wing-box inspection
    experiment) -- so the optimiser is held constant across experiments and
    only the objective varies (CLAUDE.md §3).

    `cost` maps a flat vector of `n_stations` stations to a number to
    minimise, and must be a picklable callable when `workers != 1` is passed
    (see `_StationPlacementCost`). Each station has `dims_per_station`
    searched numbers -- 3 by default, an (x, y, z) position; 2 for a station
    on a track, `(s_m, height_m)` (`TrackStationCost`) -- and `bounds_m` holds
    that many (min, max) pairs, applied to every station. Seeding, parallel
    evaluation and the infeasibility counter behave exactly as documented on
    `genetic_algorithm_station_placement`; a seed row is `n_stations *
    dims_per_station` numbers.

    Returns `(scipy OptimizeResult, n_infeasible_evals)`; the count is 0
    when `workers != 1`, where a live count cannot cross process
    boundaries.
    """
    bounds_m = list(bounds_m)
    if len(bounds_m) != dims_per_station:
        raise ValueError(
            f"bounds_m must have exactly {dims_per_station} (min, max) pairs (one per searched number "
            f"per station), got {len(bounds_m)}"
        )
    full_bounds = bounds_m * n_stations
    ndim = dims_per_station * n_stations

    if seed_station_positions_m is not None:
        if "init" in differential_evolution_kwargs:
            raise ValueError("pass either seed_station_positions_m or init=..., not both")
        requested_popsize = differential_evolution_kwargs.get("popsize", 15)
        total_population = requested_popsize * ndim
        seed_rows = [np.asarray(positions, dtype=float).reshape(-1) for positions in seed_station_positions_m]
        for row in seed_rows:
            if row.shape != (ndim,):
                raise ValueError(
                    f"each seed_station_positions_m entry must contain exactly n_stations={n_stations} "
                    f"stations of {dims_per_station} numbers ({ndim} values total), got shape {row.shape}"
                )
        if len(seed_rows) > total_population:
            raise ValueError(
                f"{len(seed_rows)} seed_station_positions_m entries given, but the population "
                f"size (popsize * n_stations * {dims_per_station} = {total_population}) can only hold that many -- "
                "raise popsize or pass fewer seeds"
            )
        rng = np.random.default_rng(seed)
        low = np.array([b[0] for b in full_bounds])
        high = np.array([b[1] for b in full_bounds])
        random_rows = rng.uniform(low, high, size=(total_population - len(seed_rows), ndim))
        differential_evolution_kwargs["init"] = np.vstack([np.array(seed_rows), random_rows])

    workers = differential_evolution_kwargs.get("workers", 1)
    if workers != 1 or differential_evolution_kwargs.get("vectorized", False):
        # (vectorized=True hands `cost` a whole generation as an (ndim, S)
        # array -- how `planning.surrogate` scores a population in one batch
        # -- which the per-call infeasible counter below cannot wrap)
        # scipy requires (and silently forces, with a warning, if not told)
        # updating="deferred" whenever workers != 1 -- a whole generation
        # evaluated at once is what makes handing it to a worker pool
        # meaningful. Setting it explicitly avoids the warning.
        differential_evolution_kwargs.setdefault("updating", "deferred")
        return differential_evolution(cost, full_bounds, seed=seed, **differential_evolution_kwargs), 0

    infeasible_evals = 0

    def counted_cost(flat_params: np.ndarray) -> float:
        nonlocal infeasible_evals
        value = cost(flat_params)
        if value >= INFEASIBLE_PENALTY:
            infeasible_evals += 1
        return value

    result = differential_evolution(counted_cost, full_bounds, seed=seed, **differential_evolution_kwargs)
    return result, infeasible_evals


def genetic_algorithm_station_placement(
    scenario: PlanningScenario,
    n_stations: int,
    objective_fn: ObjectiveFn,
    bounds_m: Sequence[Tuple[float, float]],
    aim_point_m: Optional[np.ndarray] = None,
    seed: Optional[int] = 0,
    seed_station_positions_m: Optional[Sequence[Sequence[np.ndarray]]] = None,
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
    `tol`) -- tune these against `objective_fn`'s own cost per call
    (objective C's risk integration is roughly two orders of magnitude
    slower per call than A0/A1/B, per `planning.experiment`'s own runtime
    note) rather than this function hard-coding a one-size-fits-all
    budget.

    **Parallel evaluation (`workers`).** Pass `workers=N` (or `-1` for
    `os.cpu_count()`) to evaluate a generation's population across
    `multiprocessing` worker processes instead of one at a time --
    `_StationPlacementCost` above exists specifically so this works
    (a closure could not survive being pickled to a worker). Measured on
    this codebase's own wing scenario (4 cores): `workers=-1` gave a
    ~3.5x speedup over `workers=1`, but *only* after pinning every BLAS
    library to one thread per process first (`OMP_NUM_THREADS=1`,
    `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
    as environment variables, set before the Python process starts --
    numpy/scipy's linear algebra reads these once at import time, not
    per call). Without pinning them, each worker process's own
    multi-threaded BLAS calls oversubscribe the machine's cores against
    each other and the parallel run can come out *slower* than serial --
    confirmed directly, not a theoretical caveat. `n_infeasible_evals` is
    only tracked in the serial (`workers=1`, the default) case -- see
    that field's own docstring for why a live count can't survive
    `workers != 1`.

    **Seeding the initial population (`seed_station_positions_m`).**
    Confirmed directly on this codebase's own wing scenario: even a
    34000-evaluation budget left one objective's own search stuck in a
    clearly worse region than another objective's search found, judged
    by the stuck objective's *own* metric (`objectives.
    make_weighted_uncertainty_objective`'s risk-derived weights can span
    three orders of magnitude across characteristics -- `risk_derived_
    weights`'s own docstring -- which makes for a landscape DE can
    struggle to explore blind over a box spanning tens of metres). Pass
    known-good station arrangements here -- e.g. another objective's own
    `result.best.station_poses` from a search over the *same*
    `bounds_m`/`aim_point_m` -- as `[[pos0, pos1, ...], ...]`, one inner
    list per seed arrangement, each exactly `n_stations` positions long.
    Each is flattened into one member of DE's initial population
    (`differential_evolution`'s own `init=` array argument); the rest of
    the population is filled out by a Latin hypercube sample over
    `bounds_m`, exactly as when no seeds are given, so DE still explores
    freely from there -- a seed is a starting point to refine from, not a
    constraint on where the search can end up. Raises `ValueError` if
    more seeds are given than `popsize * n_stations * 3` (the population
    size `differential_evolution` would otherwise use), rather than
    silently dropping some.
    """
    if aim_point_m is None:
        aim_point_m = scenario.target_points_m.mean(axis=0)
    aim_point_m = np.asarray(aim_point_m, dtype=float)
    bounds_m = list(bounds_m)
    if len(bounds_m) != 3:
        raise ValueError(f"bounds_m must have exactly 3 (min, max) pairs (x, y, z), got {len(bounds_m)}")

    cost = _StationPlacementCost(scenario, objective_fn, aim_point_m, n_stations)
    de_result, infeasible_evals = run_station_search(
        cost, n_stations, bounds_m, seed, seed_station_positions_m, **differential_evolution_kwargs
    )

    best_evaluation = evaluate_plan(cost.station_poses(de_result.x), scenario)
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


# -- station track ---------------------------------------------------------
@dataclass(frozen=True)
class TrackStationCost:
    """Picklable cost for a differential-evolution search over stations that
    stand on a `geometry.station_track.StationTrack`: two searched numbers
    per station, `(s_m, height_m)`, instead of an (x, y, z) position.

    `s_m` is taken modulo the track's perimeter, so any value is a valid point
    on the closed path. A station whose `s_m` falls in one of the track's
    blocked intervals is infeasible: the whole plan scores `INFEASIBLE_
    PENALTY`, exactly as an unobservable plan does (and `evaluate_plan`'s own
    visibility disqualification still applies on top).

    Orientation is `levelled` by default (a real tracker is levelled; yaw
    cannot change any covariance, tests/test_station_track.py) or `look_at`.
    A plain frozen dataclass with no closures, so `workers=-1` can pickle it.
    """

    scenario: PlanningScenario
    objective_fn: ObjectiveFn
    track: StationTrack
    n_stations: int
    aim_point_m: np.ndarray
    orientation_mode: str = "levelled"

    def station_poses(self, flat_params: np.ndarray) -> List[TrackPose]:
        make_orientation = levelled_orientation if self.orientation_mode == "levelled" else look_at_orientation
        poses = []
        for s_raw, height in np.asarray(flat_params, dtype=float).reshape(self.n_stations, 2):
            s = float(s_raw) % self.track.perimeter_m
            x, y = self.track.position_xy_m(s)
            position = np.array([x, y, float(height)])
            poses.append(
                TrackPose(
                    position_m=position, orientation=make_orientation(position, self.aim_point_m),
                    s_m=s, height_m=float(height), segment=self.track.segment(s),
                )
            )
        return poses

    def __call__(self, flat_params: np.ndarray) -> float:
        poses = self.station_poses(flat_params)
        if any(self.track.is_blocked(p.s_m) for p in poses):
            return INFEASIBLE_PENALTY
        if self.scenario.visibility_enabled:
            # The same graded deficit `_StationPlacementCost` uses, but from one
            # visibility computation: nearly every random layout on a long part
            # is infeasible, so the ray casting is most of the cost.
            deficit = visibility_deficit(station_visibility_matrix(self.scenario, poses), 0)
            if deficit > 0.0:
                return INFEASIBLE_PENALTY + deficit
        evaluation = evaluate_plan(poses, self.scenario)
        if evaluation is None:
            return INFEASIBLE_PENALTY + plan_infeasibility_deficit(poses, self.scenario)
        value = self.objective_fn(evaluation, self.scenario)
        return value if value < float("inf") else INFEASIBLE_PENALTY


def track_genetic_algorithm_station_placement(
    scenario: PlanningScenario,
    track: StationTrack,
    n_stations: int,
    objective_fn: ObjectiveFn,
    height_bounds_m: Tuple[float, float] = (0.8, 3.0),
    aim_point_m: Optional[np.ndarray] = None,
    orientation_mode: str = "levelled",
    seed: Optional[int] = 0,
    seed_stations: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    **differential_evolution_kwargs,
) -> GASearchResult:
    """The GA over `(s_m, height_m)` per station on a `StationTrack`: the same
    `run_station_search` driver, objective handling, `INFEASIBLE_PENALTY` and
    seeding as `genetic_algorithm_station_placement`, with two searched numbers
    per station instead of three.

    `s_m` ranges over [0, perimeter_m] and `height_m` over `height_bounds_m`
    (default 0.8-3.0 m, a placeholder range, TODO(source): the real stands).
    Blocked intervals are infeasible, not excluded from the box.

    **s = 0 is a boundary of the search box.** `differential_evolution` clips
    its trial vectors at the bounds and has no notion that the path is
    closed, so the track is *not* periodic as DE sees it: a good region that
    straddles the s = 0 join (the -x end of the -y row) is reachable but sits
    at both edges of the box, and mutations between a point at s ~ 0.5 and one
    at s ~ perimeter - 0.5 make a long jump rather than a short one. The cost
    itself does take s modulo the perimeter, so a station at s = perimeter
    scores identically to s = 0; nothing is lost in the answer, only some
    search efficiency near that one place. Choose the origin of s accordingly
    when laying out a real cell.

    `seed_stations` is a list of arrangements, each `[(s_m, height_m), ...]`
    of exactly `n_stations` entries (e.g. another objective's winning plan,
    from `[(p.s_m, p.height_m) for p in result.best.station_poses]`).
    `aim_point_m` None aims at the scene's target centroid; it changes
    nothing when levelled (yaw invariance) and sets the tilt for "look_at".
    """
    if aim_point_m is None:
        aim_point_m = scenario.target_points_m.mean(axis=0)
    aim_point_m = np.asarray(aim_point_m, dtype=float)
    if orientation_mode not in ("levelled", "look_at"):
        raise ValueError(f"orientation_mode must be 'levelled' or 'look_at', got {orientation_mode!r}")

    cost = TrackStationCost(scenario, objective_fn, track, n_stations, aim_point_m, orientation_mode)
    bounds = [(0.0, float(track.perimeter_m)), (float(height_bounds_m[0]), float(height_bounds_m[1]))]
    de_result, infeasible_evals = run_station_search(
        cost, n_stations, bounds, seed, seed_stations, dims_per_station=2, **differential_evolution_kwargs
    )

    best_evaluation = evaluate_plan(cost.station_poses(de_result.x), scenario)
    best_value = None if best_evaluation is None else objective_fn(best_evaluation, scenario)
    if any(track.is_blocked(p.s_m) for p in cost.station_poses(de_result.x)) or (
        best_value is not None and best_value >= INFEASIBLE_PENALTY
    ):
        best_evaluation, best_value = None, None

    return GASearchResult(
        best=best_evaluation,
        best_objective_value=best_value,
        n_function_evals=de_result.nfev,
        n_generations=de_result.nit,
        converged=bool(de_result.success),
        n_infeasible_evals=infeasible_evals,
    )
