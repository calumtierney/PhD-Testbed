"""The long-part-on-a-station-track experiment: the same four objectives as the
headline experiment (A0 trace of the 3D covariance, A1 uniform projected
uncertainty, B risk-derived weights at 8 um, C direct expected risk), searched
identically on `planning.experiment.long_part_two_cluster_scenario`.

Nothing here changes what an objective means or how the optimiser works
(CLAUDE.md §3: the optimiser is held constant and only the objective varies).
This module only runs the searches and reports their winners in track
coordinates, `(s_m, height_m, segment)`:

  - `grid_all_objectives`: one exhaustive pass over every combination of track
    candidates that scores each feasible plan under all four objectives. The
    result for each objective is identical to running
    `planning.search.grid_search_station_placement` once per objective (the
    plan is solved once instead of four times); it is the ground truth the GA
    is checked against.
  - `run_track_ga`: the differential-evolution search over `(s_m, height_m)`
    per station (`planning.search.track_genetic_algorithm_station_placement`),
    one run per objective, same settings and seed for every objective.
  - `summarise_plan`, `cross_score`: report a plan as the headline reports
    (summed uncertainty, PFA, PFR, summed risk) and score every objective's
    winner under every objective.
  - `time_evaluations`: the cost of one plan evaluation and of one objective
    call, measured on feasible plans of the scene, used to choose the grid
    spacing before the real runs.
"""
import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose
from geometry.station_track import StationTrack
from planning.experiment import _plan_report, total_expected_risk
from planning.objectives import (
    ObjectiveFn,
    PlanEvaluation,
    PlanningScenario,
    evaluate_plan,
    make_weighted_uncertainty_objective,
    objective_a0_trace_covariance,
    objective_c_direct_risk,
    risk_derived_weights,
    station_visibility_matrix,
    uniform_weights,
    visibility_deficit,
)
from planning.search import GASearchResult, track_genetic_algorithm_station_placement

OBJECTIVE_NAMES = ("A0", "A1", "B", "C")
UM = 1e6


def track_objectives(scenario: PlanningScenario, reference_uncertainty_m: float = 8e-6) -> Dict[str, ObjectiveFn]:
    """The four objectives, built exactly as `run_headline_experiment` builds
    them (B's weights are the risk slopes at `reference_uncertainty_m`)."""
    return {
        "A0": objective_a0_trace_covariance,
        "A1": make_weighted_uncertainty_objective(uniform_weights(scenario)),
        "B": make_weighted_uncertainty_objective(risk_derived_weights(scenario, reference_uncertainty_m)),
        "C": objective_c_direct_risk,
    }


@dataclass(frozen=True)
class GridAllResult:
    """Exhaustive search over `n_stations`-combinations of candidates, every
    feasible plan scored under every objective.

    `best[name]` is the lowest-`name` plan; `best_value[name]` its value;
    `n_combinations`, `n_feasible` count what was searched (everything else is
    infeasible: some target unseen, or a station not tied to the anchor).
    """

    best: Dict[str, Optional[PlanEvaluation]]
    best_value: Dict[str, Optional[float]]
    n_combinations: int
    n_feasible: int
    seconds: float


def grid_all_objectives(
    candidate_poses: Sequence[InstrumentPose],
    n_stations: int,
    scenario: PlanningScenario,
    objectives: Dict[str, ObjectiveFn],
) -> GridAllResult:
    """Exhaustive search, all objectives in one pass (see the module
    docstring). Combinations failing the cheap necessary conditions are
    dropped using each candidate's precomputed visibility row
    (`planning.objectives.visibility_deficit`); everything else goes through
    `evaluate_plan` as in `grid_search_station_placement`."""
    start = time.time()
    visible = station_visibility_matrix(scenario, candidate_poses)
    best: Dict[str, Optional[PlanEvaluation]] = {name: None for name in objectives}
    best_value: Dict[str, Optional[float]] = {name: None for name in objectives}
    n_combinations = n_feasible = 0
    for combo in itertools.combinations(range(len(candidate_poses)), n_stations):
        n_combinations += 1
        if visibility_deficit(visible[[0] + [i + 1 for i in combo]], 0) > 0.0:
            continue
        evaluation = evaluate_plan([candidate_poses[i] for i in combo], scenario)
        if evaluation is None:
            continue
        n_feasible += 1
        for name, objective_fn in objectives.items():
            value = objective_fn(evaluation, scenario)
            if value < float("inf") and (best_value[name] is None or value < best_value[name]):
                best[name], best_value[name] = evaluation, value
    return GridAllResult(best, best_value, n_combinations, n_feasible, time.time() - start)


def run_track_ga(
    scenario: PlanningScenario,
    track: StationTrack,
    n_stations: int,
    objectives: Dict[str, ObjectiveFn],
    height_bounds_m: Tuple[float, float],
    de_kwargs: Dict,
    seed_stations: Optional[Dict[str, Sequence[Sequence[Tuple[float, float]]]]] = None,
) -> Dict[str, GASearchResult]:
    """One track GA per objective, identical settings and seed for each.
    `seed_stations[name]`, if given, seeds that objective's initial population."""
    results = {}
    for name, objective_fn in objectives.items():
        seeds = None if seed_stations is None else seed_stations.get(name)
        results[name] = track_genetic_algorithm_station_placement(
            scenario, track, n_stations, objective_fn, height_bounds_m=height_bounds_m,
            seed_stations=seeds, **de_kwargs,
        )
    return results


def stations_of(evaluation: PlanEvaluation) -> List[Tuple[float, float, str]]:
    """A plan's added stations as `(s_m, height_m, segment)`, in s order."""
    return sorted((float(p.s_m), float(p.height_m), p.segment) for p in evaluation.station_poses)


def summarise_plan(evaluation: PlanEvaluation, scenario: PlanningScenario) -> Dict:
    """What the headline reports for a plan: stations, summed projected
    uncertainty (um), summed PFA, summed PFR and their sum (the expected number
    of wrong decisions), plus each characteristic's own uncertainty (um)."""
    report = _plan_report(evaluation, scenario)
    return {
        "stations": [list(s) for s in stations_of(evaluation)],
        "sum_u_um": report.summed_characteristic_uncertainty_m * UM,
        "pfa": report.global_pfa,
        "pfr": report.global_pfr,
        "risk": report.global_pfa + report.global_pfr,
        "per_characteristic_u_um": {k: v * UM for k, v in report.per_characteristic_uncertainty_m.items()},
    }


def cross_score(
    winners: Dict[str, PlanEvaluation], scenario: PlanningScenario, objectives: Dict[str, ObjectiveFn]
) -> Dict[str, Dict[str, float]]:
    """`table[winner][objective]`: the winner plan of one objective scored
    under every objective. Column minimum on the diagonal means each search
    found the best plan for its own objective among the plans compared."""
    return {
        w: {o: float(objective_fn(evaluation, scenario)) for o, objective_fn in objectives.items()}
        for w, evaluation in winners.items()
    }


def time_evaluations(
    scenario: PlanningScenario,
    candidate_poses: Sequence[InstrumentPose],
    objectives: Dict[str, ObjectiveFn],
    n_stations: int = 3,
    n_plans: int = 200,
    seed: int = 0,
) -> Dict:
    """Milliseconds per feasible plan evaluation (`evaluate_plan`, which
    includes the visibility layer and the network solve) and per objective
    call, over `n_plans` feasible plans drawn at random from `candidate_poses`.
    Run before choosing the grid spacing (the budget rule: how long does the
    expensive objective take on this scene?)."""
    visible = station_visibility_matrix(scenario, candidate_poses)
    rng = np.random.default_rng(seed)
    plans: List[Tuple[int, ...]] = []
    while len(plans) < n_plans:
        combo = tuple(sorted(rng.choice(len(candidate_poses), n_stations, replace=False)))
        if visibility_deficit(visible[[0] + [i + 1 for i in combo]], 0) == 0.0:
            plans.append(combo)  # (random draws rarely pass; a few thousand tries are cheap)
    start = time.time()
    evaluations = [evaluate_plan([candidate_poses[i] for i in combo], scenario) for combo in plans]
    evaluation_ms = 1e3 * (time.time() - start) / len(plans)
    evaluations = [e for e in evaluations if e is not None]
    objective_ms = {}
    for name, objective_fn in objectives.items():
        start = time.time()
        for evaluation in evaluations:
            objective_fn(evaluation, scenario)
        objective_ms[name] = 1e3 * (time.time() - start) / len(evaluations)
    return {"n_plans": len(evaluations), "evaluate_plan_ms": evaluation_ms, "objective_ms": objective_ms}
