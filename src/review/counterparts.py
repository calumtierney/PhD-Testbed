"""Algorithm counterparts of a plan: what the testbed's own search finds for the same job.

Given a plan (an engineer's, say), `algorithm_counterparts` runs the existing search
(`planning.cell.search_cell_layout`: differential evolution over arc length and height per
station -- the one optimiser every other track search in this repo uses, with the same
settings and seed for every objective; CLAUDE.md §3 holds the optimiser constant so that the
objective is the only thing that varies) for

  (a) the same number of stations as the plan, under each objective:
      A0 (summed covariance trace: what the field minimises), A1 (summed uncertainty along the
      tolerances), TUR (the worst test uncertainty ratio) and risk (expected wrong decisions);
  (b) the fewest stations whose expected wrong decisions are no worse than the plan's (iso-risk),
      searched for 1, 2, ... up to the plan's own count under the risk objective;
  (c) the lowest-risk plan at one station fewer than the plan.

**The given plan is never re-optimised or altered.** It is reviewed as it is.

**What the search measures.** The algorithm plans always measure every characteristic (the
search scorer requires it, as `plan_qif_cell` does) and stand on the scene's station track within
the scene's station-height bounds; a plan being reviewed is not limited to either. Every
counterpart is then reviewed by the *same* context (scorer and risk model) as the given plan, so
the comparison is like with like. The search finds a good plan, not a provably best one: a
counterpart that does not beat the given plan is a result, and is shown as one.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from interchange.plan_file import PlanFile, plan_file_from_stations
from planning.cell import (
    PartFrameScorer, RiskObjective, TraceObjective, WeightedUncertaintyObjective, layout_poses, search_cell_layout,
)
from planning.qif_cell import WorstRatioObjective
from review.core import PlanReview, ReviewContext, build_context, review_with_context
from review.scene import ReviewScene
from review.settings import ReviewSettings

OBJECTIVES = ("A0", "A1", "TUR", "risk")
ISO_RISK_RELATIVE_TOLERANCE = 1e-9

Progress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class Counterpart:
    """One algorithm plan and its review. `kind`: "same_count", "iso_risk" or "one_fewer"."""

    kind: str
    label: str
    objective: str
    n_stations: int
    plan: PlanFile
    review: PlanReview


@dataclass(frozen=True)
class CounterpartResult:
    given: PlanReview
    counterparts: Tuple[Counterpart, ...]
    not_found: Tuple[str, ...] = ()           # what was searched for and not found, and why
    search: Dict[str, Any] = field(default_factory=dict)

    def reviews(self) -> List[PlanReview]:
        """The given plan first, then every counterpart: ready for `review.core.compare_plans`."""
        return [self.given] + [c.review for c in self.counterparts]


def _objectives(context: ReviewContext) -> Dict[str, Any]:
    limits = context.scene.tolerance_limits_m()
    half = 0.5 * (limits[:, 1] - limits[:, 0])
    return {"A0": TraceObjective(), "A1": WeightedUncertaintyObjective(np.ones(len(half))),
            "TUR": WorstRatioObjective(half), "risk": RiskObjective(context.risk_model)}


def algorithm_counterparts(scene: ReviewScene, plan: PlanFile, settings: ReviewSettings, context: Optional[ReviewContext] = None,
                           progress: Optional[Progress] = None, objectives: Sequence[str] = OBJECTIVES) -> CounterpartResult:
    """Review `plan` and search for its algorithm counterparts (see the module docstring).

    `progress(done, total, label)` is called as each search finishes (total is an upper bound: the iso-risk
    search stops at the first station count that is good enough). Slow: each search is a full differential
    evolution; with the default settings expect on the order of a minute or more per search for a large scene.
    """
    unknown = [o for o in objectives if o not in OBJECTIVES]
    if unknown:
        raise ValueError(f"unknown objectives {unknown}; choose from {list(OBJECTIVES)}")
    context = build_context(scene, settings) if context is None else context
    given = review_with_context(context, plan)
    k = plan.n_stations
    search_scorer = PartFrameScorer(context.scenario, tuple(scene.datum_indices), True)
    fns = _objectives(context)
    cache: Dict[Tuple[str, int], Any] = {}
    done = 0
    total = len(objectives) + k + 1

    def run(objective: str, n: int):
        nonlocal done
        key = (objective, n)
        if key not in cache:
            cache[key] = search_cell_layout(search_scorer, fns[objective], scene.track, n, scene.station_height_bounds_m,
                                            seed=settings.seed, **settings.de_kwargs())
            done += 1
            if progress is not None:
                progress(done, total, f"{objective}, {n} station(s)")
        return cache[key]

    def counterpart(kind: str, objective: str, n: int, result) -> Optional[Counterpart]:
        if result.value is None:
            return None
        poses = layout_poses(result.stations, context.scenario, scene.track)
        source = (f"review.algorithm_counterparts ({kind}): objective {objective}, {n} stations, differential evolution "
                  f"maxiter {settings.de_maxiter} popsize {settings.de_popsize} seed {settings.seed}")
        created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        algorithm_plan = plan_file_from_stations(
            f"{objective}, {n} stations" + (" (iso-risk)" if kind == "iso_risk" else ""), "algorithm", source, created, "cell",
            [p.position_m for p in poses], track_coordinates=[(s[0], s[1]) for s in result.stations])
        return Counterpart(kind, f"{objective}, {n} stations", objective, n, algorithm_plan, review_with_context(context, algorithm_plan))

    found: List[Counterpart] = []
    not_found: List[str] = []
    for objective in objectives:                                           # (a)
        c = counterpart("same_count", objective, k, run(objective, k))
        if c is None:
            not_found.append(f"{objective} at {k} stations: no feasible layout found on the track")
        else:
            found.append(c)
    if True:                                                               # (b) and (c) always use the risk objective
        if given.feasible:                                                 # (b)
            target = given.totals["expected_wrong_decisions"]
            iso = None
            for n in range(1, k + 1):
                c = counterpart("iso_risk", "risk", n, run("risk", n))
                if c is not None and c.review.feasible and c.review.totals["expected_wrong_decisions"] <= target * (1.0 + ISO_RISK_RELATIVE_TOLERANCE):
                    iso = c
                    break
            if iso is None:
                not_found.append(f"iso-risk: no plan of up to {k} stations reaches the given plan's expected wrong decisions ({target:.4g})")
            else:
                found.append(iso)
        else:
            not_found.append("iso-risk: the given plan is infeasible, so it has no risk to match")
        if k - 1 >= 1:                                                     # (c)
            c = counterpart("one_fewer", "risk", k - 1, run("risk", k - 1))
            if c is None:
                not_found.append(f"risk at {k - 1} stations: no feasible layout found on the track")
            else:
                found.append(c)
        else:
            not_found.append("one station fewer: the plan has a single station")
    if progress is not None:
        progress(total, total, "done")
    return CounterpartResult(given, tuple(found), tuple(not_found), {
        "optimiser": "differential evolution over (arc length, height) per station", "de": settings.de_kwargs(), "seed": settings.seed,
        "search_scorer_requires_all_characteristics": True})
