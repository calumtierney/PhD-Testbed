"""Grid search over station placement (CLAUDE.md §5, step 6).

CLAUDE.md §3: "The optimiser is deliberately conventional and held
constant. The experiment varies the objective, not the search method.
Use grid search or a standard GA. Do not invent an optimiser." This
module is the grid search -- exhaustive evaluation of every combination
of `n_stations` distinct poses drawn from a candidate set (typically
`planning.candidates.hemisphere_candidate_poses`). A GA is CLAUDE.md's
documented fallback "if grid is too slow"; it isn't needed here (see
docs/step6_headline_experiment.md for the actual runtime at the
resolution step 6 uses) so it isn't built -- CLAUDE.md §9, don't add
machinery a step's own gate doesn't need.

This is only tractable because `n_stations` is small (the whole point of
`candidates.py`'s reduction to a 2-DOF-per-station search space): the
number of combinations is `C(n_candidates, n_stations)`, and every one of
them costs one `objectives.evaluate_plan` call (one network solve, cheap
because it's the noiseless/design-covariance shortcut) plus one call to
whichever `objectives.ObjectiveFn` is being searched under -- A0, A1, B or
C (`objectives.py`'s module docstring), or any of those wrapped in
`objectives.constrained_objective` for a hard uncertainty cap.
"""
import itertools
from dataclasses import dataclass
from typing import List, Optional

from geometry.pose import InstrumentPose
from planning.objectives import ObjectiveFn, PlanEvaluation, PlanningScenario, evaluate_plan


@dataclass(frozen=True)
class GridSearchResult:
    """The outcome of an exhaustive grid search.

    Attributes
    ----------
    best : the lowest-objective-value PlanEvaluation found, or None if
        every candidate was disqualified (e.g. by a `constrained_objective`
        cap that nothing in the candidate set satisfies).
    best_objective_value : the objective value `best` achieved (for
        reporting/comparison across searches); None alongside `best=None`.
    all_evaluations : every candidate combination's PlanEvaluation, in the
        order evaluated -- kept for reporting/plotting (e.g. a
        distribution of achieved objective values across the candidate
        set), not just the single winner.
    """

    best: Optional[PlanEvaluation]
    best_objective_value: Optional[float]
    all_evaluations: List[PlanEvaluation]


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
    for combo_indices in itertools.combinations(range(len(candidate_poses)), n_stations):
        station_poses = [candidate_poses[i] for i in combo_indices]
        evaluation = evaluate_plan(station_poses, scenario)
        all_evaluations.append(evaluation)
        value = objective_fn(evaluation, scenario)
        if value < float("inf"):
            scored.append((value, evaluation))

    if not scored:
        return GridSearchResult(best=None, best_objective_value=None, all_evaluations=all_evaluations)

    best_value, best_evaluation = min(scored, key=lambda pair: pair[0])
    return GridSearchResult(
        best=best_evaluation, best_objective_value=best_value, all_evaluations=all_evaluations
    )
