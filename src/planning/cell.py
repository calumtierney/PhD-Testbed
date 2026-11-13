"""The factory cell: the long part on its station track, with a reference network.

`planning.experiment.long_part_two_cluster_scenario` put every tie point on the
part itself and required every one to be observed. The station-track experiment
showed what that costs: only stations on the end loops can tie the two sides
together, every sight line to a characteristic is 10-15 m long, and no plan has
fewer than three added stations (docs/station_track_experiment.md).

Real assembly cells are not tied together through the part. They carry a
**reference network** of nests fixed to the cell structure (in aircraft
assembly, the "enhanced reference system", ERS), which every tracker station
observes and which a bundle adjustment uses to bring all stations into one
frame. This module adds that network to the same part, track and
characteristics, and provides a fast scorer for layouts in it:

  * `factory_cell_scenario` -- the long-part scene plus ERS nests on pillars
    along both sides of the cell and on its end walls, facing inwards. Their
    number and positions are placeholders (TODO(source): the real cell).
    They are ordinary unknown targets in the network solve (a bundle
    adjustment of all stations and nests together), not points of known
    position: that is the conservative reading, and needs no calibration
    uncertainty to be invented.
  * `CellLayoutScorer` -- visibility, the rigid-network test and the design
    covariance for one layout, returning each characteristic's projected
    uncertainty. A plan must tie every station to the anchor through >= 3
    shared targets (any targets: nests, ties or characteristics); it need not
    observe every nest. Characteristics must all be observed unless
    `require_all_characteristics` is False, in which case an unobserved one
    is simply not measured.
  * `RiskCurve` / `CellRiskModel` -- total decision risk as a function of each
    characteristic's uncertainty, for a chosen process knowledge (known, or
    from n builds: `risk.decision`) and decision rule (simple acceptance, or
    JCGM 106's conformance-probability rule), tabulated once so a search can
    evaluate it in microseconds.
  * objectives A0, A1, B and risk, and `search_cell_layout`, the same
    differential-evolution driver over `(s_m, height_m)` per station as every
    other track search.
  * `measurement_time_minutes` -- station count and shot count converted to
    time with the one published timing available (Hughes et al. 2011: 5
    stations, 126 observations, 55 minutes), read two ways.
"""
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose
from geometry.station_track import StationTrack
from geometry.visibility import visibility_matrix
from network.design import (
    _skew, count_untied_stations, design_covariances_for_poses, design_joint_covariance, projected_uncertainty_m,
)
from planning.experiment import long_part_track, long_part_two_cluster_scenario
from planning.objectives import MIN_TARGETS_PER_NON_ANCHOR_STATION, PlanningScenario, risk_derived_weights
from planning.search import INFEASIBLE_PENALTY, TrackStationCost, run_station_search
from risk.decision import ProcessKnowledge, best_rule_risk, decision_risk, unmeasured_risk

# -- the reference network: every number a placeholder, TODO(source): the real cell ------
CELL_ERS_SIDE_Y_M = 6.5                       # pillars 2.25 m behind each row of the track
CELL_ERS_SIDE_X_M = (-2.5, 2.5, 7.5, 12.5, 17.5)  # pillar positions along the cell
CELL_ERS_HEIGHTS_M = (1.0, 2.8)               # two nests per pillar: low, and above the part's top (2.0 m)
CELL_ERS_END_X_BEYOND_PART_M = 7.5            # end walls this far beyond each end of the part
CELL_ERS_END_Y_M = (-2.5, 2.5)                # nest columns on each end wall

# Hughes et al. 2011 (CLAUDE.md §4d): a 15-target, 5-station network, 126 observations, 55 minutes.
HUGHES_STATIONS, HUGHES_OBSERVATIONS, HUGHES_MINUTES = 5, 126, 55.0


def ers_nests(part_length_m: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """`(points, normals)` of the reference nests: on pillars along both sides
    (normals pointing across the cell) and on the two end walls (normals
    pointing along it), at `CELL_ERS_HEIGHTS_M`."""
    points, normals = [], []
    for side in (+1.0, -1.0):
        for x in CELL_ERS_SIDE_X_M:
            for z in CELL_ERS_HEIGHTS_M:
                points.append([x, side * CELL_ERS_SIDE_Y_M, z])
                normals.append([0.0, -side, 0.0])
    for x, nx in ((-CELL_ERS_END_X_BEYOND_PART_M, 1.0), (part_length_m + CELL_ERS_END_X_BEYOND_PART_M, -1.0)):
        for y in CELL_ERS_END_Y_M:
            for z in CELL_ERS_HEIGHTS_M:
                points.append([x, y, z])
                normals.append([nx, 0.0, 0.0])
    return np.array(points), np.array(normals)


def factory_cell_scenario(anchor_s_m: Optional[float] = None, track: Optional[StationTrack] = None,
                          with_reference_network: bool = True, **scene_kwargs) -> PlanningScenario:
    """`long_part_two_cluster_scenario` (same part, six characteristics, part
    tie points, anchor and visibility limits) plus the reference nests of
    `ers_nests`, appended after the existing targets so every existing index
    keeps its meaning. `with_reference_network=False` returns the original
    scene unchanged (for like-for-like comparisons through the same scorer)."""
    if track is None:
        track = long_part_track()
    base = long_part_two_cluster_scenario(anchor_s_m=anchor_s_m, track=track, **scene_kwargs)
    if not with_reference_network:
        return base
    points, normals = ers_nests(track.part_x_max_m - track.part_x_min_m)
    return replace(
        base,
        target_points_m=np.vstack([base.target_points_m, points]),
        target_normals_m=np.vstack([base.target_normals_m, normals]),
    )


# -- scoring a layout ----------------------------------------------------------------------
@dataclass(frozen=True)
class LayoutScore:
    """What a layout delivers. `u_m[i]` is characteristic i's projected standard
    uncertainty (NaN if its target is not observed), `trace_m2[i]` the trace
    of its target's covariance; `n_shots` counts station-target observations
    (anchor included), `n_stations` the added stations."""

    feasible: bool
    deficit: float
    u_m: np.ndarray
    trace_m2: np.ndarray
    n_shots: int
    n_stations: int


@dataclass(frozen=True)
class CellLayoutScorer:
    scenario: PlanningScenario
    require_all_characteristics: bool = True

    def visible(self, station_poses: Sequence[InstrumentPose]) -> np.ndarray:
        """(1 + n_stations, n_targets) visibility, anchor first: the vectorised
        form of the check `planning.objectives` applies pair by pair."""
        s = self.scenario
        positions = np.array([s.anchor_pose.position_m] + [p.position_m for p in station_poses])
        return visibility_matrix(positions, s.target_points_m, s.visibility_limits, s.target_normals_m, s.obstacles, 0.0)

    def score(self, station_poses: Sequence[InstrumentPose]) -> LayoutScore:
        s = self.scenario
        n_char = len(s.characteristics)
        nan = np.full(n_char, np.nan)
        visible = self.visible(station_poses)
        counts = visible.sum(axis=1)
        n_thin = sum(1 for i in range(1, visible.shape[0]) if counts[i] < MIN_TARGETS_PER_NON_ANCHOR_STATION)
        n_untied = count_untied_stations(visible, 0, MIN_TARGETS_PER_NON_ANCHOR_STATION)
        seen = visible.any(axis=0)
        char_targets = np.array([c.target_indices[0] for c in s.characteristics])
        n_missing = int((~seen[char_targets]).sum()) if self.require_all_characteristics else 0
        deficit = float(max(n_thin, n_untied) + n_missing)
        if deficit > 0.0:
            return LayoutScore(False, deficit, nan, nan, int(visible.sum()), len(station_poses))
        all_poses = [s.anchor_pose] + list(station_poses)
        design = design_covariances_for_poses(s.target_points_m, all_poses, s.tracker, visible, 0)
        if not design.connected:
            return LayoutScore(False, 1.0, nan, nan, int(visible.sum()), len(station_poses))
        directions = np.array([np.asarray(c.tolerance.surface_normal, dtype=float) for c in s.characteristics])
        covariance = design.point_covariance_m2[char_targets]
        u = projected_uncertainty_m(covariance, directions, s.tracker.systematic_std_m)
        trace = np.einsum("tii->t", covariance) + 3.0 * s.tracker.systematic_std_m ** 2
        return LayoutScore(True, 0.0, u, trace, int(visible.sum()), len(station_poses))


def part_datum_indices(scenario: PlanningScenario, n_part_targets: int = 26) -> List[int]:
    """The part's own tie points (targets on the part that are not toleranced
    characteristics): the datum features that define the part frame. The
    reference nests on the cell structure (indices >= `n_part_targets`) are not
    part of it."""
    used = {i for c in scenario.characteristics for i in c.target_indices}
    return [i for i in range(min(n_part_targets, len(scenario.target_points_m))) if i not in used]


@dataclass(frozen=True)
class DatumCellScorer:
    """Scores a layout by each characteristic's uncertainty **relative to the
    part's datum frame**, with every station planned.

    `CellLayoutScorer` reports uncertainty in the frame of a fixed anchor station
    (the convention of the earlier experiments): a characteristic far from the
    anchor inherits the anchor's long-range angular error, however well the
    station in front of it measures. But a deviation is toleranced against the
    part's datums, not against a tracker. Here the part frame is the rigid best
    fit to the *datum targets* the plan observes (`part_datum_indices`: the
    part's own tie points), and a characteristic's uncertainty is that of its
    position relative to that fitted frame (`network.design.
    datum_referenced_functional`, the geodetic S-transformation). The result
    does not depend on which station the network happens to be solved in, so
    no station is special: the scenario's anchor is not used, and
    `n_stations` in the score is the *total* number of stations.

    A layout is feasible if every station is tied to the first by >= 3 shared
    targets, at least three non-collinear datum targets are observed, and (when
    `require_all_characteristics`) every characteristic is observed.
    """

    scenario: PlanningScenario
    require_all_characteristics: bool = True
    n_part_targets: int = 26

    def visible(self, station_poses: Sequence[InstrumentPose]) -> np.ndarray:
        s = self.scenario
        positions = np.array([p.position_m for p in station_poses])
        return visibility_matrix(positions, s.target_points_m, s.visibility_limits, s.target_normals_m, s.obstacles, 0.0)

    def score(self, station_poses: Sequence[InstrumentPose]) -> LayoutScore:
        s = self.scenario
        n_char = len(s.characteristics)
        nan = np.full(n_char, np.nan)
        if len(station_poses) == 0:
            return LayoutScore(False, 1.0, nan, nan, 0, 0)
        visible = self.visible(station_poses)
        n_shots = int(visible.sum())
        counts = visible.sum(axis=1)
        n_thin = sum(1 for i in range(1, visible.shape[0]) if counts[i] < MIN_TARGETS_PER_NON_ANCHOR_STATION)
        n_untied = count_untied_stations(visible, 0, MIN_TARGETS_PER_NON_ANCHOR_STATION)
        seen = visible.any(axis=0)
        char_targets = [c.target_indices[0] for c in s.characteristics]
        n_missing = int(sum(not seen[t] for t in char_targets)) if self.require_all_characteristics else 0
        datum = [i for i in part_datum_indices(s, self.n_part_targets) if seen[i]]
        deficit = float(max(n_thin, n_untied) + n_missing + max(0, 3 - len(datum)))
        if deficit > 0.0:
            return LayoutScore(False, deficit, nan, nan, n_shots, len(station_poses))
        measured = [t for t in char_targets if seen[t]]
        subset = measured + datum
        joint, _, connected = design_joint_covariance(s.target_points_m, list(station_poses), s.tracker, visible, subset, 0)
        if not connected:
            return LayoutScore(False, 1.0, nan, nan, n_shots, len(station_poses))
        points = s.target_points_m
        centre = points[datum].mean(axis=0)
        g_rows = lambda x: np.concatenate([np.eye(3), -_skew((x - centre)[None, :])[0]], axis=1)
        g = np.vstack([g_rows(points[i]) for i in datum])
        normal = g.T @ g
        collinear = bool(np.any(np.diag(normal) <= 1e-12))
        if not collinear:
            scale = 1.0 / np.sqrt(np.diag(normal))
            collinear = np.linalg.eigvalsh(normal * scale[:, None] * scale[None, :]).min() < 1e-9
        if collinear:  # the observed datum targets cannot fix six degrees of freedom
            return LayoutScore(False, 1.0, nan, nan, n_shots, len(station_poses))
        fit = np.linalg.solve(normal, g.T)  # datum displacements -> rigid motion of the fitted frame
        n_m = len(measured)
        d = slice(3 * n_m, 3 * len(subset))
        cov_dd = joint[d, d]
        u, trace = nan.copy(), nan.copy()
        sys2 = s.tracker.systematic_std_m ** 2
        for k, c in enumerate(s.characteristics):
            t = c.target_indices[0]
            if not seen[t]:
                continue
            j = measured.index(t)
            p = slice(3 * j, 3 * j + 3)
            transfer = g_rows(points[t]) @ fit
            cross = joint[p, d] @ transfer.T
            cov = joint[p, p] - cross - cross.T + transfer @ cov_dd @ transfer.T
            n = np.asarray(c.tolerance.surface_normal, dtype=float)
            u[k] = np.sqrt(max(float(n @ cov @ n), 0.0) + sys2)
            trace[k] = float(np.trace(cov)) + 3.0 * sys2
        return LayoutScore(True, 0.0, u, trace, n_shots, len(station_poses))


# -- risk as a function of uncertainty ------------------------------------------------------
@dataclass(frozen=True)
class RiskCurve:
    """Total risk (PFA + PFR) of one characteristic against its uncertainty,
    tabulated on a log grid and read by log-log interpolation; `unmeasured` is
    the risk with no measurement at all (accept or reject on process knowledge),
    returned for a NaN uncertainty."""

    log_u_m: np.ndarray
    log_risk: np.ndarray
    unmeasured: float

    def __call__(self, u_m: float) -> float:
        if not np.isfinite(u_m):
            return self.unmeasured
        return float(np.exp(np.interp(np.log(u_m), self.log_u_m, self.log_risk)))


def build_risk_curve(prior: ProcessKnowledge, tolerance_lower_m: float, tolerance_upper_m: float, rule: str = "simple",
                     min_conformance_probability: float = 0.5, n_points: int = 60,
                     u_min_m: float = 0.3e-6, u_max_m: float = 1e-3) -> RiskCurve:
    """`rule` "simple": acceptance limits = tolerance limits. "best": accept when
    the conformance probability is at least `min_conformance_probability`."""
    if rule not in ("simple", "best"):
        raise ValueError(f"rule must be 'simple' or 'best', got {rule!r}")
    grid = np.geomspace(u_min_m, u_max_m, n_points)
    risks = []
    for u in grid:
        if rule == "simple":
            pfa, pfr = decision_risk(prior, tolerance_lower_m, tolerance_upper_m, u)
        else:
            pfa, pfr, _ = best_rule_risk(prior, tolerance_lower_m, tolerance_upper_m, u, min_conformance_probability)
        risks.append(max(pfa + pfr, 1e-300))
    return RiskCurve(np.log(grid), np.log(np.array(risks)), unmeasured_risk(prior, tolerance_lower_m, tolerance_upper_m))


def process_knowledge_for(scenario: PlanningScenario, name: str, n_builds: Optional[int] = None,
                          propagate_assignment_doubt: bool = False) -> ProcessKnowledge:
    """Process knowledge for characteristic `name`: its cluster's elicited mean
    and standard deviation read as the statistics of `n_builds` earlier parts
    (None: known exactly). With `propagate_assignment_doubt`, the cluster
    assignment's confidence c becomes a mixture: c x this cluster + (1 - c) x
    the scenario's other clusters, equally weighted."""
    own = scenario.assignments[name]
    knowledge = ProcessKnowledge.from_builds(own.cluster.prior.mean_m, own.cluster.prior.std_m, n_builds)
    if not propagate_assignment_doubt or own.confidence >= 1.0:
        return knowledge
    others = {a.cluster.name: a.cluster.prior for a in scenario.assignments.values() if a.cluster.name != own.cluster.name}
    if not others:
        return knowledge
    from risk.decision import PriorComponent

    parts = [ProcessKnowledge.from_builds(p.mean_m, p.std_m, n_builds) for p in others.values()]
    alternative = ProcessKnowledge(tuple(
        PriorComponent(c.weight / len(parts), c.mean_m, c.scale_m, c.dof) for part in parts for c in part.components))
    return knowledge.mixed_with(alternative, own.confidence)


@dataclass(frozen=True)
class CellRiskModel:
    """Risk curves for every characteristic of a scenario under one statement of
    process knowledge and one decision rule. `total(u)` is expected wrong
    decisions; an unobserved characteristic (NaN) costs its unmeasured risk."""

    names: Tuple[str, ...]
    curves: Tuple[RiskCurve, ...]
    label: str = ""

    @classmethod
    def build(cls, scenario: PlanningScenario, n_builds: Optional[int] = None, rule: str = "simple",
              min_conformance_probability: float = 0.5, propagate_assignment_doubt: bool = False) -> "CellRiskModel":
        curves, cache = [], {}
        for c in scenario.characteristics:
            a, r = scenario.assignments[c.name], scenario.decision_rules[c.name]
            key = (a.cluster.name, r.tolerance_lower_m, r.tolerance_upper_m)
            if key not in cache:
                prior = process_knowledge_for(scenario, c.name, n_builds, propagate_assignment_doubt)
                cache[key] = build_risk_curve(prior, r.tolerance_lower_m, r.tolerance_upper_m, rule, min_conformance_probability)
            curves.append(cache[key])
        label = f"{'known' if n_builds is None else f'{n_builds} builds'}, {rule} rule"
        return cls(tuple(c.name for c in scenario.characteristics), tuple(curves), label)

    def per_characteristic(self, u_m: np.ndarray) -> np.ndarray:
        return np.array([curve(u) for curve, u in zip(self.curves, u_m)])

    def total(self, u_m: np.ndarray) -> float:
        return float(self.per_characteristic(u_m).sum())

    def unmeasured_total(self) -> float:
        return float(sum(c.unmeasured for c in self.curves))


# -- objectives (each maps a feasible LayoutScore to a number to minimise) -----------------
@dataclass(frozen=True)
class TraceObjective:
    """A0: summed trace of the characteristic targets' covariance (m^2)."""

    def __call__(self, score: LayoutScore) -> float:
        return float(np.sum(score.trace_m2)) if np.all(np.isfinite(score.trace_m2)) else float("inf")


@dataclass(frozen=True)
class WeightedUncertaintyObjective:
    """A1 (weights all 1) and B (risk-derived weights): sum of w * u (m)."""

    weights: np.ndarray

    def __call__(self, score: LayoutScore) -> float:
        return float(np.sum(self.weights * score.u_m)) if np.all(np.isfinite(score.u_m)) else float("inf")


@dataclass(frozen=True)
class MaxUncertaintyObjective:
    """The TUR plan's criterion: the worst characteristic's uncertainty (m)."""

    def __call__(self, score: LayoutScore) -> float:
        return float(np.max(score.u_m)) if np.all(np.isfinite(score.u_m)) else float("inf")


@dataclass(frozen=True)
class RiskObjective:
    """Total decision risk under a `CellRiskModel` (expected wrong decisions)."""

    model: CellRiskModel

    def __call__(self, score: LayoutScore) -> float:
        return self.model.total(score.u_m)


def standard_objectives(scenario: PlanningScenario, reference_uncertainty_m: float = 8e-6) -> Dict[str, object]:
    """A0, A1, B and C exactly as the headline experiment defines them (B's
    weights at 8 um; C = simple acceptance, process known)."""
    n = len(scenario.characteristics)
    return {
        "A0": TraceObjective(),
        "A1": WeightedUncertaintyObjective(np.ones(n)),
        "B": WeightedUncertaintyObjective(np.asarray(risk_derived_weights(scenario, reference_uncertainty_m))),
        "C": RiskObjective(CellRiskModel.build(scenario, None, "simple")),
    }


# -- search --------------------------------------------------------------------------------
@dataclass(frozen=True)
class CellCost:
    """Picklable cost for `run_station_search`: `objective(score)` of the layout
    given as `(s_m, height_m)` pairs; infeasible layouts score the usual penalty
    plus a graded deficit."""

    scorer: CellLayoutScorer
    objective: object
    track: StationTrack
    n_stations: int
    aim_point_m: np.ndarray

    def poses(self, flat_params: np.ndarray):
        return TrackStationCost(self.scorer.scenario, None, self.track, self.n_stations, self.aim_point_m).station_poses(flat_params)

    def __call__(self, flat_params: np.ndarray) -> float:
        poses = self.poses(flat_params)
        if any(self.track.is_blocked(p.s_m) for p in poses):
            return INFEASIBLE_PENALTY
        score = self.scorer.score(poses)
        if not score.feasible:
            return INFEASIBLE_PENALTY + score.deficit
        value = self.objective(score)
        return value if np.isfinite(value) else INFEASIBLE_PENALTY


@dataclass(frozen=True)
class CellSearchResult:
    stations: Tuple[Tuple[float, float, str], ...]  # (s_m, height_m, segment), in s order
    value: Optional[float]
    score: Optional[LayoutScore]
    n_function_evals: int


def layout_poses(stations: Sequence[Sequence], scenario: PlanningScenario, track: StationTrack):
    """Station poses (levelled) for `(s_m, height_m, ...)` tuples."""
    flat = np.array([[s[0], s[1]] for s in stations], dtype=float).reshape(-1)
    aim = scenario.target_points_m.mean(axis=0)
    return TrackStationCost(scenario, None, track, len(stations), aim).station_poses(flat)


def search_cell_layout(scorer: CellLayoutScorer, objective, track: StationTrack, n_stations: int,
                       height_bounds_m: Tuple[float, float] = (0.8, 3.0), seed_stations=None, seed: int = 0,
                       **de_kwargs) -> CellSearchResult:
    """Differential evolution over `(s_m, height_m)` per station. `seed_stations`
    is a list of layouts, each `[(s_m, height_m), ...]`, placed in the initial
    population."""
    aim = scorer.scenario.target_points_m.mean(axis=0)
    cost = CellCost(scorer, objective, track, n_stations, aim)
    bounds = [(0.0, float(track.perimeter_m)), (float(height_bounds_m[0]), float(height_bounds_m[1]))]
    seeds = None if not seed_stations else [[(s[0], s[1]) for s in layout] for layout in seed_stations]
    result, _ = run_station_search(cost, n_stations, bounds, seed, seeds, dims_per_station=2, **de_kwargs)
    poses = cost.poses(result.x)
    score = scorer.score(poses)
    if not score.feasible or any(track.is_blocked(p.s_m) for p in poses):
        return CellSearchResult((), None, None, int(result.nfev))
    stations = tuple(sorted((float(p.s_m), float(p.height_m), p.segment) for p in poses))
    return CellSearchResult(stations, float(objective(score)), score, int(result.nfev))


# -- time ----------------------------------------------------------------------------------
def measurement_time_minutes(n_stations_total: int, n_shots: int) -> Dict[str, float]:
    """Two single-parameter readings of the one published timing (Hughes et al.
    2011: 5 stations, 126 observations, 55 min): time proportional to the number
    of stations (11 min each: set-up dominates), or to the number of
    observations (26 s each: pointing dominates). The truth lies between; both
    are reported, neither is a measured time for this cell."""
    return {
        "per_station_model_min": HUGHES_MINUTES / HUGHES_STATIONS * n_stations_total,
        "per_observation_model_min": HUGHES_MINUTES / HUGHES_OBSERVATIONS * n_shots,
    }
