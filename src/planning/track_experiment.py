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


# -- variable station count: plans that choose how many stations to use --------------
# The fixed-count comparison above cannot say anything about measurement time:
# every plan gets the same number of stations. The functions below support the
# variable-count comparison of `viz/run_station_track_station_count.py`, which
# mirrors the wing-box experiment (`planning.inspection`, docs/wingbox_experiment.md):
#   A1-knee / A0-knee  minimise uncertainty; how many stations is read off the
#                      knee of the criterion's station-count curve (a trace never
#                      says when to stop);
#   T                  the TUR 4:1 rule, every characteristic's u <= tolerance/8;
#   R                  the fewest stations whose risk-optimal layout is no riskier
#                      than a reference (the A1 plan's own realised risk, or what
#                      the TUR rule guarantees).
def with_tolerance(scenario: PlanningScenario, tolerance_half_m: float) -> PlanningScenario:
    """The same scene at a different tolerance (+-`tolerance_half_m` on every
    characteristic), with the two clusters' priors scaled with it so Cpk keeps
    its meaning (`planning.inspection.cluster_priors`, "offset" style: at +-50 um
    this is exactly the repo's marginal mean 38 um / std 3 um and high-capability
    std 8.33 um). Geometry, stations and uncertainties are untouched; only the
    risk side changes."""
    from dataclasses import replace

    from characteristics.characteristic import Characteristic
    from characteristics.tolerances import ProfileTolerance
    from planning.inspection import cluster_priors
    from risk.jcgm106 import ClusterAssignment, DecisionRule

    marginal, high = cluster_priors(tolerance_half_m, "offset")
    by_name = {"marginal": marginal, "high-capability": high}
    characteristics, assignments, rules = [], {}, {}
    for c in scenario.characteristics:
        characteristics.append(Characteristic(
            name=c.name, target_indices=list(c.target_indices),
            tolerance=ProfileTolerance(zone_width_m=2.0 * tolerance_half_m, surface_normal=c.tolerance.surface_normal),
        ))
        old = scenario.assignments[c.name]
        assignments[c.name] = ClusterAssignment(c.name, by_name[old.cluster.name], confidence=old.confidence)
        rules[c.name] = DecisionRule.from_zone_width(2.0 * tolerance_half_m)
    return replace(scenario, characteristics=characteristics, assignments=assignments, decision_rules=rules)


@dataclass(frozen=True)
class TabulatedRiskObjective:
    """Objective C (summed PFA + PFR) read from per-characteristic tables of
    risk against uncertainty instead of integrating each time: risk depends on
    a characteristic only through its u, so ~140 integrations up front replace
    ~4 per characteristic per plan, which makes C as cheap as A0. The same
    log-log interpolation as `planning.inspection.RiskTable`; checked against
    `objective_c_direct_risk` in tests. Used only *inside* searches; every
    reported risk is recomputed exactly. Picklable (no closures)."""

    names: Tuple[str, ...]
    log_u_m: np.ndarray
    log_risk: np.ndarray  # (n_characteristics, n_grid)

    @classmethod
    def build(cls, scenario: PlanningScenario, n_points: int = 140, u_min_m: float = 0.2e-6, u_max_m: float = 1e-3):
        from risk.jcgm106 import evaluate_conformity_risk

        grid = np.geomspace(u_min_m, u_max_m, n_points)
        rows = []
        for c in scenario.characteristics:
            assignment, rule = scenario.assignments[c.name], scenario.decision_rules[c.name]
            risks = [evaluate_conformity_risk(assignment, u, rule) for u in grid]
            rows.append([max(r.probability_false_acceptance + r.probability_false_rejection, 1e-30) for r in risks])
        return cls(tuple(c.name for c in scenario.characteristics), np.log(grid), np.log(np.array(rows)))

    def risk_at(self, u_by_name: Dict[str, float]) -> float:
        return float(sum(
            np.exp(np.interp(np.log(u_by_name[name]), self.log_u_m, self.log_risk[i]))
            for i, name in enumerate(self.names)
        ))

    def __call__(self, evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
        return self.risk_at(evaluation.per_characteristic_uncertainty_m)


def objective_max_projected_uncertainty(evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
    """The TUR plan's search criterion: the worst characteristic's projected
    uncertainty (m). A layout meets a TUR k:1 rule iff this is <= the limit, so
    minimising it at each station count finds the fewest stations that can."""
    return float(max(evaluation.per_characteristic_uncertainty_m.values()))


def risk_at_uniform_uncertainty(scenario: PlanningScenario, u_m: float) -> float:
    """Summed PFA + PFR if every characteristic were measured at exactly `u_m`
    -- what a uniform uncertainty rule (e.g. TUR 4:1) *guarantees* at most."""
    from risk.jcgm106 import evaluate_conformity_risk

    total = 0.0
    for c in scenario.characteristics:
        r = evaluate_conformity_risk(scenario.assignments[c.name], u_m, scenario.decision_rules[c.name])
        total += r.probability_false_acceptance + r.probability_false_rejection
    return total
