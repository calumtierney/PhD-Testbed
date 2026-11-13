"""Accepting a characteristic on process evidence instead of measuring it.

Every other plan in this repo measures every characteristic and asks where to
stand. This module lets a plan *choose which characteristics to measure*; the
rest are accepted on what is already known about the process that made them.
That is the lever the wing-box experiment (`docs/wingbox_experiment.md`) found
was where a risk-based plan saves stations, and the one a fixed-count,
measure-everything comparison cannot express.

**What an unmeasured characteristic costs.** Accepting it unseen means
accepting every part: no false rejections, and a false accept whenever the true
value is outside the tolerance, which is the process's own nonconformance rate
(`planning.inspection.nonconformance_rate`: P(|true| > tolerance) under the
cluster prior). It carries no measurement uncertainty at all, only the prior.
A measured characteristic carries the usual PFA + PFR at its projected
uncertainty. Total risk is the sum over characteristics, so a plan is compared
with a measure-everything plan in the same units (expected wrong decisions).

**What the model says, and why that needs care.** At the uncertainties a long
part on a track delivers (30-60 um against +-50 um), measuring a capable
characteristic creates ~0.1-0.4 expected wrong decisions (false rejections),
whereas accepting it on its prior costs 1e-9 (Cpk 2.0) to 3e-5 (Cpk 1.33). So
an unconstrained risk-minimiser measures nothing. That is the model's honest
arithmetic *given its priors*, and the priors are placeholders with no data
behind them (CLAUDE.md §3: process priors are elicited per cluster, and cluster
assignment carries its own uncertainty). Two controls therefore exist, and
every result is reported against them rather than at a hidden default:

  `min_measured`   a sampling *policy* floor: at least this many
                   characteristics must be measured whatever the risk says
                   (as `PlanSpec.min_task_coverage` in the wing-box code);
  `escape_price_multiplier`  the unmeasured price is the model's nonconformance
                   rate times this (capped at 1): how much worse than the
                   elicited prior the real process would have to be for
                   measuring to pay. `break_even_multiplier` reports it
                   directly.

**Which targets a plan must observe.** A measured characteristic's target must
be seen. Tie points are optional unless `ties_required` (the track scene's own
rule: every target, ties included, must be observed). With `ties_required` the
station floor is 3 whatever is measured, because the ties on both long faces and
both end faces must all be seen; with the ties relaxed a plan may use only the
ties it needs to tie its stations into one network, and the floor can fall.
Either way every station must share >= 3 targets with a chain leading to the
anchor (the rigid-network rule the whole repo enforces).

Layout evaluation uses the fast design covariance (`network.design`, pinned to
the full network solve by `tests/test_design_covariance.py`) and risk tables
(`planning.inspection.RiskTable`, within 3% of the integral). Reported numbers
for chosen plans can be recomputed exactly with `exact_plan_risk`.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose
from geometry.station_track import StationTrack
from network.design import count_untied_stations, design_covariances_for_poses, projected_uncertainty_m
from planning.inspection import RiskTable, nonconformance_rate
from planning.objectives import MIN_TARGETS_PER_NON_ANCHOR_STATION, PlanningScenario, station_visibility_matrix
from planning.search import INFEASIBLE_PENALTY, TrackStationCost, run_station_search
from risk.jcgm106 import evaluate_conformity_risk


@dataclass(frozen=True)
class EvidencePlan:
    """One layout, scored with the best choice of which characteristics to measure.

    `measured` / `accepted_on_evidence` are characteristic names;
    `risk_measured`, `risk_unmeasured` and their sum `risk` are expected wrong
    decisions (PFA + PFR for measured, the escape price for unmeasured);
    `u_m` is each *measurable* characteristic's projected uncertainty (NaN if its
    target is not observed)."""

    measured: Tuple[str, ...]
    accepted_on_evidence: Tuple[str, ...]
    risk_measured: float
    risk_unmeasured: float
    u_m: Dict[str, float]

    @property
    def risk(self) -> float:
        return self.risk_measured + self.risk_unmeasured


@dataclass(frozen=True)
class ProcessEvidenceEvaluator:
    """Scores layouts for a scenario where characteristics may go unmeasured.

    Picklable (a frozen dataclass holding the scenario and numpy tables), so a
    differential-evolution search can use `workers=-1`."""

    scenario: PlanningScenario
    min_measured: int = 0
    ties_required: bool = True
    escape_price_multiplier: float = 1.0
    tolerance_half_m: float = 50e-6

    # -- the ingredients --------------------------------------------------------
    def _tables(self) -> List[RiskTable]:
        # built on demand and cached on the instance (frozen dataclass: via object.__setattr__)
        cached = self.__dict__.get("_table_cache")
        if cached is None:
            cached = []
            for c in self.scenario.characteristics:
                assignment, rule = self.scenario.assignments[c.name], self.scenario.decision_rules[c.name]
                cached.append(RiskTable.build(assignment.cluster, rule, u_max_m=1e-3))
            object.__setattr__(self, "_table_cache", cached)
        return cached

    def escape_price(self, name: str) -> float:
        """What accepting characteristic `name` unseen costs in expected wrong
        decisions: the cluster's nonconformance rate x the price multiplier,
        capped at 1."""
        c = next(c for c in self.scenario.characteristics if c.name == name)
        half = c.tolerance.zone_width_m / 2.0
        rate = nonconformance_rate(self.scenario.assignments[name].cluster.prior, half)
        return float(min(1.0, rate * self.escape_price_multiplier))

    def tie_indices(self) -> List[int]:
        used = {i for c in self.scenario.characteristics for i in c.target_indices}
        return [i for i in range(len(self.scenario.target_points_m)) if i not in used]

    def _required_mask(self, n_targets: int) -> np.ndarray:
        required = np.zeros(n_targets, dtype=bool)
        if self.ties_required:
            required[self.tie_indices()] = True
        return required

    # -- feasibility ----------------------------------------------------------------
    def deficit(self, visible: np.ndarray) -> float:
        """0.0 if the visibility pattern can support a plan: every required
        target seen, every station seeing >= 3 targets and tied to the anchor by
        >= 3 shared targets, and at least `min_measured` characteristics
        measurable. Otherwise a graded positive number."""
        required = self._required_mask(visible.shape[1])
        n_unobserved = int((required & ~visible.any(axis=0)).sum())
        counts = visible.sum(axis=1)
        n_thin = sum(1 for i in range(1, visible.shape[0]) if counts[i] < MIN_TARGETS_PER_NON_ANCHOR_STATION)
        n_untied = count_untied_stations(visible, 0, MIN_TARGETS_PER_NON_ANCHOR_STATION)
        seen = visible.any(axis=0)
        n_measurable = sum(bool(seen[c.target_indices].all()) for c in self.scenario.characteristics)
        return float(n_unobserved + max(n_thin, n_untied) + max(0, self.min_measured - n_measurable))

    # -- scoring ------------------------------------------------------------------------
    def evaluate(self, station_poses: Sequence[InstrumentPose]) -> Tuple[Optional[EvidencePlan], float]:
        """`(plan, 0.0)` for a feasible layout, `(None, deficit)` otherwise."""
        scenario = self.scenario
        all_poses = [scenario.anchor_pose] + list(station_poses)
        visible = station_visibility_matrix(scenario, station_poses)
        deficit = self.deficit(visible)
        if deficit > 0.0:
            return None, deficit
        design = design_covariances_for_poses(scenario.target_points_m, all_poses, scenario.tracker, visible, 0)
        if not design.connected:
            return None, 1.0
        tables = self._tables()
        u_by_name, r_measured, p_unmeasured = {}, {}, {}
        for i, c in enumerate(scenario.characteristics):
            index = c.target_indices[0]
            p_unmeasured[c.name] = self.escape_price(c.name)
            if not design.observed[index]:
                u_by_name[c.name] = float("nan")
                continue
            u = float(projected_uncertainty_m(
                design.point_covariance_m2[index:index + 1], np.asarray(c.tolerance.surface_normal, float)[None, :],
                scenario.tracker.systematic_std_m)[0])
            u_by_name[c.name] = u
            r_measured[c.name] = float(tables[i].pfa(u) + tables[i].pfr(u))
        # measure every measurable characteristic that lowers risk, then top up to the floor
        # with the ones that hurt least (the cheapest compliance with the policy floor)
        gain = {n: p_unmeasured[n] - r for n, r in r_measured.items()}  # > 0: measuring pays
        measured = [n for n, g in gain.items() if g > 0.0]
        for n in sorted((n for n in gain if n not in measured), key=lambda n: -gain[n]):
            if len(measured) >= self.min_measured:
                break
            measured.append(n)
        names = [c.name for c in scenario.characteristics]
        unmeasured = [n for n in names if n not in measured]
        return EvidencePlan(
            measured=tuple(n for n in names if n in measured), accepted_on_evidence=tuple(unmeasured),
            risk_measured=float(sum(r_measured[n] for n in measured)),
            risk_unmeasured=float(sum(p_unmeasured[n] for n in unmeasured)), u_m=u_by_name,
        ), 0.0

    def baseline_risk(self) -> float:
        """No station at all: every characteristic accepted on evidence."""
        return float(sum(self.escape_price(c.name) for c in self.scenario.characteristics))


def break_even_multiplier(scenario: PlanningScenario, u_m: Dict[str, float], tolerance_half_m: float = 50e-6) -> Dict[str, float]:
    """Per characteristic, how many times the model's nonconformance rate the
    real process's would have to be before measuring it at uncertainty `u_m`
    beats accepting it unseen: (PFA + PFR at u) / nonconformance rate. Above 1
    means "the model says do not measure"; the size is the margin."""
    out = {}
    for c in scenario.characteristics:
        r = evaluate_conformity_risk(scenario.assignments[c.name], u_m[c.name], scenario.decision_rules[c.name])
        rate = nonconformance_rate(scenario.assignments[c.name].cluster.prior, c.tolerance.zone_width_m / 2.0)
        out[c.name] = float((r.probability_false_acceptance + r.probability_false_rejection) / rate)
    return out


# -- searching layouts on a station track -------------------------------------------------
@dataclass(frozen=True)
class EvidenceTrackCost:
    """Picklable cost for `run_station_search`: total risk (measured + accepted
    on evidence) of `n_stations` track stations given as `(s_m, height_m)` pairs.
    Blocked s and infeasible layouts score `INFEASIBLE_PENALTY` plus a graded
    deficit, as everywhere else."""

    evaluator: ProcessEvidenceEvaluator
    track: StationTrack
    n_stations: int
    aim_point_m: np.ndarray

    def poses(self, flat_params: np.ndarray):
        return TrackStationCost(self.evaluator.scenario, None, self.track, self.n_stations, self.aim_point_m).station_poses(flat_params)

    def __call__(self, flat_params: np.ndarray) -> float:
        poses = self.poses(flat_params)
        if any(self.track.is_blocked(p.s_m) for p in poses):
            return INFEASIBLE_PENALTY
        plan, deficit = self.evaluator.evaluate(poses)
        return INFEASIBLE_PENALTY + deficit if plan is None else plan.risk


def search_evidence_layout(
    evaluator: ProcessEvidenceEvaluator, track: StationTrack, n_stations: int,
    height_bounds_m: Tuple[float, float] = (0.8, 3.0), seed_stations=None, seed: int = 0, **de_kwargs,
):
    """Differential evolution over `(s_m, height_m)` per station, the same driver
    and settings style as every other station search here. Returns
    `(EvidencePlan or None, station poses, scipy result)`."""
    aim = evaluator.scenario.target_points_m.mean(axis=0)
    evaluator._tables()  # build the risk tables once, here, so worker processes receive them pickled
    cost = EvidenceTrackCost(evaluator, track, n_stations, aim)
    bounds = [(0.0, float(track.perimeter_m)), (float(height_bounds_m[0]), float(height_bounds_m[1]))]
    result, _ = run_station_search(cost, n_stations, bounds, seed, seed_stations, dims_per_station=2, **de_kwargs)
    poses = cost.poses(result.x)
    plan, _ = evaluator.evaluate(poses) if not any(track.is_blocked(p.s_m) for p in poses) else (None, 1.0)
    return plan, poses, result


def exact_plan_risk(scenario: PlanningScenario, plan: EvidencePlan, evaluator: ProcessEvidenceEvaluator) -> float:
    """Recompute a plan's total risk with the exact integral (no tables)."""
    total = 0.0
    for c in scenario.characteristics:
        if c.name in plan.measured:
            r = evaluate_conformity_risk(scenario.assignments[c.name], plan.u_m[c.name], scenario.decision_rules[c.name])
            total += r.probability_false_acceptance + r.probability_false_rejection
        else:
            total += evaluator.escape_price(c.name)
    return float(total)


def best_acceptance_risk(scenario: PlanningScenario, name: str, u_m: float, n_limits: int = 40) -> Tuple[float, float]:
    """`(risk, acceptance half-width)`: the smallest PFA + PFR over symmetric
    acceptance limits for characteristic `name` measured at uncertainty `u_m`,
    searched from the tolerance (simple acceptance) out to many times the
    tolerance (accepting nearly everything). Every other number in this repo
    uses simple acceptance (acceptance limits = tolerance limits), which is *not*
    the risk-optimal rule when the process prior is much tighter than the
    measurement: the measurement then carries almost no information, and the
    best rule accepts nearly everything. At the widest limit the risk tends to
    the unmeasured price, so this never exceeds `escape_price` (to within the
    search's resolution): measuring can only help, and the question is by how much."""
    from risk.jcgm106 import DecisionRule

    c = next(c for c in scenario.characteristics if c.name == name)
    half = c.tolerance.zone_width_m / 2.0
    best = (float("inf"), half)
    for a in half * np.geomspace(1.0, 40.0, n_limits):
        rule = DecisionRule(-half, half, -a, a)
        r = evaluate_conformity_risk(scenario.assignments[name], u_m, rule)
        total = r.probability_false_acceptance + r.probability_false_rejection
        if total < best[0]:
            best = (float(total), float(a))
    return best
