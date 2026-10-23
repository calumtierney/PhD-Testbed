"""Inspection-plan experiment on the wing box: how few laser-tracker stations
does a ~500-point inspection need, and does letting the *risk* set the
requirement need fewer than the conventional uncertainty rule?

This is deliberately separate from `planning.experiment` (CLAUDE.md §5 step
6, four objectives on small scenes). Here there are exactly two plans, built
on the same chain -- station layout -> visibility -> covariance -> projected
uncertainty -> JCGM 106 risk -- but on a 500-point part and scored by
*station count*, the cost the digital twin exists to cut:

  Plan U  (uncertainty-driven)  the fewest stations such that EVERY point is
          measured and meets one uniform, process-blind uncertainty target:
          the conventional test-uncertainty-ratio rule (TUR >= 4:1 by
          default, ANSI/NCSL Z540.3-2006's own fallback criterion).
  Plan R  (risk-driven)         the fewest stations such that the part's
          decision risk stays within what the conventional rule would
          itself guarantee -- using each cluster's process capability, so a
          marginal cluster is held tight, a high-capability cluster may be
          loose, and a point the layout cannot see is simply left unmeasured
          *and priced at its process's own nonconformance risk*, not
          forbidden.

**Why coverage, not uncertainty, is what the stations are spent on.** A
greedy cover of this part needs ~6 stations to see all 500 points however
loose the uncertainty target (the first station sees 345; the rest add 56,
41, 33, 23, 2 -- the long tail is the downward-facing nose and flap
surfaces, visible only from low, nearby positions). A plan that must measure
every point therefore cannot go below that floor. The only way a risk-driven
plan can need fewer stations is to *not measure* some points -- which is
exactly "measure what matters, not what doesn't" -- and the digital twin's
job is to say how many it is safe to leave, and what that costs.

**The risk metrics, and why they are these.** Under simple acceptance a
point's global false-accept probability can never exceed its process's own
nonconformance rate (3.2e-5 for the marginal cluster, 2e-9 for the high-
capability one) and, for any laser-tracker-scale uncertainty (>= ~2 um), is
already within a factor of ~2 of that ceiling: the consumer's risk is set by
the process, not the plan, and sits ~600x below Z540.3's 2% per-point bound
for every plan. What a plan changes is (a) which points are measured at all
-- an unmeasured point carries its process's full nonconformance rate as
escape risk -- and (b) the false-*rejection* probability of the points that
are, which grows steeply with uncertainty for a marginal process. Plan R is
therefore held to two limits, both computed over all points:

  P_escape  the probability that at least one point is falsely accepted
            (the safety metric); and
  E_FR      the expected number of falsely rejected points per part (the
            producer's / yield metric).

**What "acceptable" is.** No published acceptable false-accept limit for
safety-critical aerospace inspection was found (the sourced anchors are
Z540.3's 2% PFA ceiling and its TUR >= 4:1 fallback, neither aerospace-
specific). So the headline limit is not an invented number: plan R is held
to *what plan U's rule guarantees* -- the risk if every point sat exactly at
the TUR limit -- and an absolute-limit sweep shows how the answer moves with
it. A programme or customer limit, if one exists, drops into `PlanSpec`
unchanged.

**Cluster-assignment confidence is part of the risk, not a footnote.**
CLAUDE.md §3: assignment to a process cluster carries its own uncertainty.
With confidence c in a point's label, its process is the labelled cluster
with probability c and the other tier otherwise, so an unmeasured
"high-capability" point carries c * 2e-9 + (1 - c) * 3.2e-5 of escape risk,
not 2e-9. At c = 1 skipping the high-capability features is nearly free; at
the repo's 0.9 it is not -- the number the whole saving hinges on.

**What is searched, and how.** Station positions (x, y, z), by the same
seeded differential-evolution driver as step 6 (`planning.search.
run_station_search`) -- the optimiser is held constant; only the cost
function differs between plans. Each plan's cost is a large weight on the
constraint violation plus a secondary term, so among layouts that meet the
constraint the better one is kept. A search finds *a* layout, not a proof
that none with fewer stations exists: "no feasible layout found at K
stations" is reported as exactly that.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from geometry.visibility import VisibilityLimits, visibility_matrix
from geometry.wing_box import TASK_NAMES, WingBoxPart, build_wing_box_part
from instruments.laser_tracker import LaserTracker
from network.design import count_untied_stations, design_point_covariances, projected_uncertainty_m
from planning.candidates import look_at_orientation
from planning.search import INFEASIBLE_PENALTY, run_station_search
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, evaluate_conformity_risk

# Sourced anchors (see the module docstring): Z540.3-2006 section 5.3 caps the
# probability of false accept at 2%, and allows TUR >= 4:1 only where PFA
# cannot be estimated. Its TUR is the tolerance half-width over the 95%
# expanded uncertainty U = 2u.
Z540_3_MAX_PFA = 0.02
DEFAULT_TUR = 4.0
EXPANDED_UNCERTAINTY_K = 2.0

# Weight of the constraint violation against the secondary term in a plan's
# cost: large enough that any violating layout costs more than any feasible one.
VIOLATION_WEIGHT = 100.0

# Placeholder assignment of the six tasks to process-capability tiers. No
# source exists for which wing-box features are harder to hold; compound-
# curvature leading-edge work is taken as the marginal cluster and the flat
# skin and flap as high capability. TODO(source): elicit per-cluster priors
# (CLAUDE.md §3).
DEFAULT_MARGINAL_TASKS = (1, 2, 3)
MARGINAL_CPK = 4.0 / 3.0  # AS13006 floor / GKN standard tier (planning.experiment)
HIGH_CAPABILITY_CPK = 2.0  # GKN critical / zero-defect tier (planning.experiment)
DEFAULT_ASSIGNMENT_CONFIDENCE = 0.9  # the repo's own value (planning.experiment's scenarios)


@dataclass(frozen=True)
class RiskTable:
    """PFA(u) and PFR(u) for one cluster, tabulated once on a log grid of
    uncertainties and read back by log-log interpolation.

    `evaluate_conformity_risk` integrates numerically (several
    `scipy.integrate.quad` calls, ~30 ms) -- fine for a handful of
    characteristics, ruinous inside a search that scores ~500 points tens of
    thousands of times. Risk depends on the point only through u, so one
    table per cluster replaces every call; `tests/test_inspection.py` pins
    it to the direct integral.
    """

    log_u_m: np.ndarray
    log_pfa: np.ndarray
    log_pfr: np.ndarray

    FLOOR = 1e-30  # risk below this is not physically meaningful, and log(0) is

    @classmethod
    def build(cls, cluster: FeatureCluster, rule: DecisionRule, n_points: int = 140,
              u_min_m: float = 0.2e-6, u_max_m: float = 400e-6) -> "RiskTable":
        grid = np.geomspace(u_min_m, u_max_m, n_points)
        assignment = ClusterAssignment(cluster.name, cluster)
        results = [evaluate_conformity_risk(assignment, u, rule) for u in grid]
        pfa = np.array([r.probability_false_acceptance for r in results])
        pfr = np.array([r.probability_false_rejection for r in results])
        return cls(np.log(grid), np.log(np.maximum(pfa, cls.FLOOR)), np.log(np.maximum(pfr, cls.FLOOR)))

    def pfa(self, u_m: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(np.asarray(u_m, float)), self.log_u_m, self.log_pfa))

    def pfr(self, u_m: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(np.asarray(u_m, float)), self.log_u_m, self.log_pfr))


@lru_cache(maxsize=32)
def _cached_risk_table(mean_m: float, std_m: float, tolerance_half_m: float) -> RiskTable:
    """One table per (process prior, tolerance), reused across scenarios --
    building one costs ~140 numerical integrations."""
    rule = DecisionRule.from_zone_width(2.0 * tolerance_half_m)
    return RiskTable.build(FeatureCluster("cached", ProcessPrior(mean_m, std_m)), rule)


def cluster_priors(tolerance_half_m: float, prior_style: str) -> Tuple[FeatureCluster, FeatureCluster]:
    """(marginal, high-capability) clusters, scaled to the tolerance so that
    Cpk means the same thing at every tolerance.

    prior_style "offset" reproduces this repo's established clusters
    (`planning.experiment`: marginal = mean 38 um, std 3 um against +-50 um,
    i.e. biased towards one limit with a tight spread; at other tolerances
    both scale with the half-width). "centred" is the neutral alternative
    -- the process sits on nominal and Cpk comes entirely from its spread
    (std = half-width / (3 Cpk)). Both have Cpk 4/3 (marginal) and 2
    (high-capability); they differ in how the margin is made up, which
    changes the false-rejection numbers enormously. Neither has a source for
    these wing-box features; both are reported.
    """
    if prior_style == "offset":
        std = 0.06 * tolerance_half_m
        marginal = ProcessPrior(mean_m=tolerance_half_m - 3.0 * MARGINAL_CPK * std, std_m=std)
    elif prior_style == "centred":
        marginal = ProcessPrior(mean_m=0.0, std_m=tolerance_half_m / (3.0 * MARGINAL_CPK))
    else:
        raise ValueError(f"prior_style must be 'offset' or 'centred', got {prior_style!r}")
    high = ProcessPrior(mean_m=0.0, std_m=tolerance_half_m / (3.0 * HIGH_CAPABILITY_CPK))
    return FeatureCluster("marginal", marginal), FeatureCluster("high-capability", high)


def nonconformance_rate(prior: ProcessPrior, tolerance_half_m: float) -> float:
    """P(true value outside +-tolerance): the ceiling on a point's PFA, and
    the escape risk of a point that is never measured."""
    return float(norm.sf((tolerance_half_m - prior.mean_m) / prior.std_m)
                 + norm.cdf((-tolerance_half_m - prior.mean_m) / prior.std_m))


@dataclass(frozen=True)
class InspectionScenario:
    """The wing box plus everything needed to score a station layout."""

    part: WingBoxPart
    tracker: LaserTracker
    limits: VisibilityLimits
    tolerance_half_m: float
    prior_style: str
    assignment_confidence: float
    clusters: Tuple[FeatureCluster, ...]  # (marginal, high-capability)
    point_cluster: np.ndarray  # (N,) 0 = marginal, 1 = high-capability (the label)
    risk_tables: Tuple[RiskTable, ...]
    nonconformance: Tuple[float, ...]  # per cluster
    aim_point_m: np.ndarray
    occlusion_offset_m: float = 1e-3

    @property
    def n_points(self) -> int:
        return self.part.n_points

    def uniform_uncertainty_limit_m(self, tur: float = DEFAULT_TUR) -> float:
        """Standard uncertainty a TUR of `tur`:1 allows: Z540.3's TUR is the
        tolerance half-width over the 95% expanded uncertainty U = 2u."""
        return self.tolerance_half_m / (tur * EXPANDED_UNCERTAINTY_K)

    def point_risks(self, u_m: np.ndarray, measured: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(pfa, pfr) per point. A measured point's risk is read from its
        cluster's table at its uncertainty u, mixed over the assignment
        confidence c (labelled tier with probability c, the other tier
        otherwise). An unmeasured point is never rejected (pfr = 0) and
        carries its process's full nonconformance rate as escape risk, with
        the same mixture."""
        c = self.assignment_confidence
        n = self.n_points
        idx = np.arange(n)
        label, other = self.point_cluster, 1 - self.point_cluster
        u = np.where(measured, u_m, 1e-6)
        pfa_t = np.stack([t.pfa(u) for t in self.risk_tables])
        pfr_t = np.stack([t.pfr(u) for t in self.risk_tables])
        pfa = c * pfa_t[label, idx] + (1 - c) * pfa_t[other, idx]
        pfr = c * pfr_t[label, idx] + (1 - c) * pfr_t[other, idx]
        nc = np.array(self.nonconformance)
        skip = c * nc[label] + (1 - c) * nc[other]
        return np.where(measured, pfa, skip), np.where(measured, pfr, 0.0)

    def risk_at_uniform_uncertainty(self, u_m: float) -> Tuple[float, float, float]:
        """(E_FA, E_FR, P_escape) if every point were measured at exactly
        `u_m`: what a uniform-uncertainty rule guarantees as its ceiling."""
        pfa, pfr = self.point_risks(np.full(self.n_points, u_m), np.ones(self.n_points, dtype=bool))
        return float(pfa.sum()), float(pfr.sum()), _escape_probability(pfa)


def _escape_probability(pfa: np.ndarray) -> float:
    """P(at least one point falsely accepted) = 1 - prod(1 - pfa), assuming
    independent points -- an upper-bound-style summary (errors that share a
    station-pose error are correlated), as `planning.experiment.PlanReport.
    bounded_part_level_risk`."""
    return float(-np.expm1(np.log1p(-np.minimum(pfa, 1.0 - 1e-16)).sum()))


def build_inspection_scenario(
    tolerance_half_m: float = 50e-6,
    prior_style: str = "offset",
    assignment_confidence: float = DEFAULT_ASSIGNMENT_CONFIDENCE,
    marginal_tasks: Sequence[int] = DEFAULT_MARGINAL_TASKS,
    part: Optional[WingBoxPart] = None,
    tracker: Optional[LaserTracker] = None,
    max_range_m: float = 10.0,
    max_incidence_deg: float = 80.0,
) -> InspectionScenario:
    """The wing-box inspection scenario.

    `tolerance_half_m` defaults to the +-50 um reference scale this repo
    already uses (CLAUDE.md §4d, from Francis et al. 2016); no wing-box
    tolerance is available, so `docs/wingbox_experiment.md` reports the
    result across several. `max_range_m` (10 m, generous for a 2 m part)
    and the 80 deg incidence limit (the convention `geometry.visibility`'s
    tests use) are not literature values. The part's `station_bounds_m`
    lower height (0.8 m) is likewise an assumption (a tripod, not a floor
    stand).
    """
    part = part if part is not None else build_wing_box_part()
    marginal, high = cluster_priors(tolerance_half_m, prior_style)
    point_cluster = np.where(np.isin(part.task_ids, list(marginal_tasks)), 0, 1)
    return InspectionScenario(
        part=part,
        tracker=tracker if tracker is not None else LaserTracker(),
        limits=VisibilityLimits(max_range_m=max_range_m, max_incidence_angle_rad=np.deg2rad(max_incidence_deg)),
        tolerance_half_m=tolerance_half_m,
        prior_style=prior_style,
        assignment_confidence=assignment_confidence,
        clusters=(marginal, high),
        point_cluster=point_cluster,
        risk_tables=tuple(_cached_risk_table(c.prior.mean_m, c.prior.std_m, tolerance_half_m) for c in (marginal, high)),
        nonconformance=tuple(nonconformance_rate(c.prior, tolerance_half_m) for c in (marginal, high)),
        aim_point_m=np.array([0.75, 1.0, 1.2]),
    )


@dataclass(frozen=True)
class LayoutOutcome:
    """The result of scoring one station layout.

    When `feasible` is False the layout is not a usable plan (a station
    inside the part, a station not tied to the anchor, or -- unless
    unmeasured points are allowed -- a point nobody sees) and the arrays are
    NaN; `deficit` is a graded measure of how far it is from feasible.
    Otherwise: `measured` marks the points some station sees (all visible
    points are shot); `u_m` is each measured point's standard uncertainty
    along its own surface normal (a profile tolerance), NaN for unmeasured;
    `pfa`/`pfr` per-point false-accept / false-reject probabilities (an
    unmeasured point: its process's nonconformance rate, and 0);
    `e_fa`, `e_fr`, `e_wrong` the expected numbers of false accepts, false
    rejects and wrong decisions per part; `p_escape` the probability at
    least one point is falsely accepted; `n_shots` the station-point
    sightlines used; `task_coverage` the fraction of each of the six tasks'
    points that is measured (what a minimum-sampling policy constrains).
    """

    feasible: bool
    deficit: float
    n_stations: int
    measured: np.ndarray
    u_m: np.ndarray
    pfa: np.ndarray
    pfr: np.ndarray
    e_fa: float
    e_fr: float
    e_wrong: float
    p_escape: float
    n_shots: int
    visible: np.ndarray
    task_coverage: Optional[np.ndarray] = None  # (6,) fraction of each task's points measured, tasks 1..6
    trace_m2: Optional[np.ndarray] = None  # (N,) trace of each point's 3x3 covariance (m^2); NaN if unmeasured

    @property
    def n_measured(self) -> int:
        return int(self.measured.sum())


def _keep_out_violation(scenario: InspectionScenario, positions_m: np.ndarray) -> float:
    """0 if no station is inside the part/fixture keep-out box; else, per
    offending station, 1 plus its penetration depth in metres."""
    lo, hi = scenario.part.keep_out_min_m, scenario.part.keep_out_max_m
    inside = np.all((positions_m > lo) & (positions_m < hi), axis=1)
    if not inside.any():
        return 0.0
    depth = np.minimum(positions_m[inside] - lo, hi - positions_m[inside]).min(axis=1)
    return float(np.sum(1.0 + depth))


def _infeasible(scenario: InspectionScenario, k: int, deficit: float, visible: np.ndarray) -> LayoutOutcome:
    n = scenario.n_points
    nan = np.full(n, np.nan)
    return LayoutOutcome(False, deficit, k, np.zeros(n, dtype=bool), nan, nan, nan,
                         np.nan, np.nan, np.nan, np.nan, int(visible.sum()), visible)


def evaluate_layout(
    scenario: InspectionScenario, station_positions_m: np.ndarray, allow_unmeasured: bool = False
) -> LayoutOutcome:
    """Score a station layout (K, 3); station 0 is the anchor (the
    network's datum -- see `network.design`). Every station is boresighted
    at the part's centre, as `planning.candidates` does, and every visible
    point is shot.

    `allow_unmeasured=False` (plan U) makes a point nobody sees infeasible.
    `True` (plan R) leaves it unmeasured and prices it (see `point_risks`).
    Either way every station must be tied to the anchor by shared measured
    points, or the layout is not a rigid network (`network.design`)."""
    positions = np.asarray(station_positions_m, dtype=float).reshape(-1, 3)
    part = scenario.part
    n, k = part.n_points, len(positions)
    keep_out = _keep_out_violation(scenario, positions)
    if keep_out > 0.0:  # a station inside the part: no geometry worth computing
        return _infeasible(scenario, k, keep_out + 1.0, np.zeros((k, n), dtype=bool))
    orientations = np.array([look_at_orientation(p, scenario.aim_point_m) for p in positions])
    visible = visibility_matrix(
        positions, part.points_m, scenario.limits, part.normals_m, part.occluder, scenario.occlusion_offset_m
    )
    measured = visible.any(axis=0)
    n_unmeasured = int((~measured).sum())
    n_untied = count_untied_stations(visible, 0)
    deficit = n_untied + (0.0 if allow_unmeasured else n_unmeasured / n)
    if deficit > 0.0:
        return _infeasible(scenario, k, deficit, visible)

    design = design_point_covariances(part.points_m, positions, orientations, scenario.tracker, visible)
    if not design.connected:
        return _infeasible(scenario, k, 1.0, visible)
    u = projected_uncertainty_m(design.point_covariance_m2, part.normals_m, scenario.tracker.systematic_std_m)
    pfa, pfr = scenario.point_risks(u, measured)
    task_coverage = np.array([measured[part.task_ids == t].mean() for t in sorted(TASK_NAMES)])
    trace = np.einsum("tii->t", design.point_covariance_m2) + 3.0 * scenario.tracker.systematic_std_m ** 2
    return LayoutOutcome(
        True, 0.0, k, measured, u, pfa, pfr,
        float(pfa.sum()), float(pfr.sum()), float(pfa.sum() + pfr.sum()), _escape_probability(pfa),
        int(visible.sum()), visible, task_coverage, trace,
    )


@dataclass(frozen=True)
class PlanSpec:
    """A plan's constraint.

    kind "U" (uncertainty-driven, the A-optimal planner's view): every
    point measured, the layout chosen to minimise the mean variance of the
    points' uncertainty along their surface normals (the trace criterion of
    CLAUDE.md §1, per point). It carries *no threshold* -- minimising a
    trace never says when to stop -- so how many stations to use is a
    separate rule applied to its station-count curve (`knee_station_count`).
    Blind to tolerance and to process capability by construction.
    kind "A" (the raw A-optimal criterion): the same, but minimising the mean
    *trace* of each point's full 3x3 covariance -- what existing tools do,
    with no knowledge of which direction the tolerance constrains.
    kind "T" (the TUR rule): every point measured, every point's
    uncertainty <= `u_max_m` -- Z540.3's own fallback criterion.
    kind "R": P_escape <= `max_p_escape` and E_FR <= `max_e_fr`, over all
    points; `allow_unmeasured` False forces every point to be measured (so
    only uncertainty, not coverage, is relaxed). `min_task_coverage` is a
    sampling *policy* floor, not a risk quantity: every one of the six tasks
    must have at least this fraction of its points measured, whatever the
    risk arithmetic says (0 = no floor, 1 = everything measured). It exists
    because a risk calculation resting on placeholder process priors should
    not be allowed to switch a whole inspection task off; no source gives
    the right floor, so it is swept, never defaulted to a hidden value.
    """

    kind: str
    u_max_m: float = np.inf
    max_p_escape: float = np.inf
    max_e_fr: float = np.inf
    allow_unmeasured: bool = False
    min_task_coverage: float = 0.0

    def __post_init__(self):
        if self.kind not in ("U", "A", "T", "R"):
            raise ValueError(f"kind must be 'U', 'A', 'T' or 'R', got {self.kind!r}")


def uncertainty_plan(trace: bool = False) -> PlanSpec:
    """Plan U: minimise mean variance along each point's normal, every point
    measured (see `PlanSpec`); `trace=True` minimises the mean covariance
    trace instead (the raw A-optimal criterion, kind "A")."""
    return PlanSpec("A" if trace else "U")


def tur_rule_plan(scenario: InspectionScenario, tur: float = DEFAULT_TUR) -> PlanSpec:
    """The conventional TUR rule: every point measured, every point's
    uncertainty at most tolerance / (tur * 2)."""
    return PlanSpec("T", u_max_m=scenario.uniform_uncertainty_limit_m(tur))


def iso_risk_plan(scenario: InspectionScenario, tur: float = DEFAULT_TUR, allow_unmeasured: bool = True,
                  slack: float = 1.0, min_task_coverage: float = 0.0) -> PlanSpec:
    """Plan R held to what the uniform TUR rule guarantees: the escape
    probability and expected false rejections if every point were measured
    at exactly the TUR limit (times `slack`, 1.0 = strictly no worse)."""
    _, e_fr, p_escape = scenario.risk_at_uniform_uncertainty(scenario.uniform_uncertainty_limit_m(tur))
    return PlanSpec("R", max_p_escape=p_escape * slack, max_e_fr=e_fr * slack, allow_unmeasured=allow_unmeasured,
                    min_task_coverage=min_task_coverage)


def plan_violation(spec: PlanSpec, outcome: LayoutOutcome) -> float:
    """0.0 when the layout meets the plan's constraint, else a positive,
    dimensionless excess (for U: the unmeasured fraction, there being no
    threshold; for T: mean over all points of the fractional
    overshoot of the uncertainty limit, an unmeasured point counting as 1;
    for R: summed fractional overshoot of the two risk limits)."""
    if not outcome.feasible:
        return float("inf")
    if spec.kind in ("U", "A"):
        return 0.0 if outcome.measured.all() else float(1.0 - outcome.measured.mean())
    if spec.kind == "T":
        # an unmeasured point is a full violation (it has no uncertainty to
        # compare); measured points count their fractional overshoot
        overshoot = np.where(outcome.measured, np.maximum(0.0, np.nan_to_num(outcome.u_m) / spec.u_max_m - 1.0), 1.0)
        return float(np.mean(overshoot))
    violation = max(0.0, outcome.p_escape / spec.max_p_escape - 1.0) + max(0.0, outcome.e_fr / spec.max_e_fr - 1.0)
    if spec.min_task_coverage > 0.0:
        shortfall = np.maximum(0.0, spec.min_task_coverage - outcome.task_coverage)
        violation += float(shortfall.sum() / spec.min_task_coverage)
    return float(violation)


@dataclass(frozen=True)
class _LayoutCost:
    """Picklable cost for `planning.search.run_station_search` (a module-
    level class, not a closure, so `workers != 1` can send it to worker
    processes -- see `planning.search._StationPlacementCost`)."""

    scenario: InspectionScenario
    spec: PlanSpec

    def __call__(self, flat_positions: np.ndarray) -> float:
        outcome = evaluate_layout(self.scenario, np.asarray(flat_positions).reshape(-1, 3), self.spec.allow_unmeasured)
        if not outcome.feasible:
            return INFEASIBLE_PENALTY + outcome.deficit
        violation = plan_violation(self.spec, outcome)
        if self.spec.kind == "A":
            secondary = float(np.mean(np.nan_to_num(outcome.trace_m2))) / 1e-12  # mean covariance trace, um^2
        elif self.spec.kind == "U":
            secondary = float(np.mean(np.nan_to_num(outcome.u_m) ** 2)) / 1e-12  # mean variance, um^2
        elif self.spec.kind == "T":
            secondary = float(np.mean(outcome.u_m)) / self.spec.u_max_m
        else:
            secondary = outcome.p_escape / self.spec.max_p_escape + outcome.e_fr / self.spec.max_e_fr
        return VIOLATION_WEIGHT * violation + secondary


@dataclass(frozen=True)
class KResult:
    """The best layout a search found at one station count."""

    n_stations: int
    positions_m: np.ndarray
    outcome: LayoutOutcome
    violation: float
    n_function_evals: int

    @property
    def satisfied(self) -> bool:
        return self.outcome.feasible and self.violation <= 1e-12


@dataclass(frozen=True)
class CandidatePool:
    """A large random sample of allowed station positions with each one's
    visibility of every point -- used only to *seed* the search with
    layouts that already cover the part (see `greedy_cover_layouts`). The
    objective, constraints and optimiser are untouched by it."""

    positions_m: np.ndarray  # (M, 3)
    visible: np.ndarray  # (M, N) bool


def build_candidate_pool(scenario: InspectionScenario, n_candidates: int = 20000, seed: int = 0,
                         n_directed_per_point: int = 4) -> CandidatePool:
    """`n_candidates` uniform random allowed positions, plus
    `n_directed_per_point` per inspection point placed 0.5-4 m out along
    that point's own surface normal (with 0.3 m of scatter, clipped to the
    station box). The directed ones exist because a uniform sample almost
    never lands in the narrow region from which a downward-facing point --
    the underside of the nose, the lower flap skin -- can be seen at all
    (the synthetic part's two lowest nose points are seen from none of
    6000 uniform candidates), which would make "every point measured"
    look impossible when it is only unsampled."""
    rng = np.random.default_rng(seed)
    part = scenario.part
    lo = np.array([b[0] for b in part.station_bounds_m])
    hi = np.array([b[1] for b in part.station_bounds_m])
    pos = rng.uniform(lo, hi, size=(n_candidates, 3))
    if n_directed_per_point > 0:
        idx = np.repeat(np.arange(part.n_points), n_directed_per_point)
        distance = rng.uniform(0.5, 4.0, size=(len(idx), 1))
        directed = part.points_m[idx] + part.normals_m[idx] * distance + rng.normal(0.0, 0.3, size=(len(idx), 3))
        pos = np.vstack([pos, np.clip(directed, lo, hi)])
    pos = pos[~np.all((pos > part.keep_out_min_m) & (pos < part.keep_out_max_m), axis=1)]
    vis = np.vstack([
        visibility_matrix(pos[i:i + 2000], part.points_m, scenario.limits, part.normals_m, part.occluder,
                          scenario.occlusion_offset_m)
        for i in range(0, len(pos), 2000)
    ])
    return CandidatePool(pos, vis)


def greedy_cover_layouts(
    pool: CandidatePool, weights: np.ndarray, n_layouts: int, max_stations: int, seed: int = 0, top_choices: int = 4,
    min_shared_points: int = 6,
) -> List[np.ndarray]:
    """Station sequences that cover the points with positive `weights`
    greedily (each next station adds the most uncovered weight), the first
    fully greedy and the rest randomised among the `top_choices` best at
    each step for diversity. Seeds only -- a layout's prefix of length K is
    a starting point the search then refines against the real constraint.
    Weight points by what a plan must cover: all ones for plan U, the
    nonconformance-weighted escape benefit for plan R.

    A station is only a candidate if it sees at least `min_shared_points` of
    the points already covered: a station that shares fewer than three with
    the rest cannot be tied into the network (`network.design`), so a pure
    cover -- which prizes *new* points -- would otherwise hand the search
    layouts that cover the part but are not a usable plan. The first
    station is exempt (it is the anchor, the network's datum)."""
    rng = np.random.default_rng(seed)
    layouts = []
    for layout_index in range(n_layouts):
        covered = np.zeros(pool.visible.shape[1], dtype=bool)
        chosen = []
        for _ in range(max_stations):
            gain = (pool.visible & ~covered[None, :]).astype(float) @ weights
            if chosen:
                gain = np.where((pool.visible & covered[None, :]).sum(axis=1) >= min_shared_points, gain, 0.0)
            if gain.max() <= 0:
                break
            order = np.argsort(-gain)[:top_choices if layout_index else 1]
            pick = int(rng.choice(order))
            chosen.append(pick)
            covered |= pool.visible[pick]
        if chosen:
            layouts.append(pool.positions_m[chosen])
    return layouts


def knee_station_count(j_by_k: Dict[int, float], min_gain: float = 0.10) -> int:
    """The station count where adding one more stops paying: the smallest K
    such that the next count (K+1) lowers the plan's criterion `j` by less
    than a fraction `min_gain` of its value at K. The standard
    diminishing-returns reading of an uncertainty-versus-stations curve
    (CLAUDE.md §5 step 3's own gate: uncertainty falls with stations and
    flattens). `min_gain` is a judgement, not a sourced number, so the
    experiment reports the answer across several. If the curve never
    flattens within the counts given, the largest K is returned."""
    ks = sorted(j_by_k)
    for k, k_next in zip(ks, ks[1:]):
        if j_by_k[k] <= 0 or (j_by_k[k] - j_by_k[k_next]) / j_by_k[k] < min_gain:
            return k
    return ks[-1]


def greedy_cost_layout(
    scenario: InspectionScenario,
    spec: PlanSpec,
    pool: CandidatePool,
    max_stations: int,
    first_station: Optional[np.ndarray] = None,
    n_trials: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Forward-greedy layout built with the plan's *own* cost: starting from
    `first_station` (default: the pool candidate that sees the most points),
    add, one at a time, the station that lowers `_LayoutCost` most, from
    `n_trials` candidates per step -- the 100 that see the most points in
    need of a better view, the 100 that see the most points not yet
    measured, and random ones from the pool.

    A coverage-only greedy cover (`greedy_cover_layouts`) cannot see what a
    plan actually asks for: it will happily add a station that sees new
    points but is tied to the network by three of them, which ruins every
    point it sees (the underside of the nose and lower flap skin are exactly
    this). Scoring each addition with the real evaluator picks stations that
    are both useful and well tied. Used only to seed the search (the
    optimiser and objective are unchanged), identically for every plan."""
    rng = np.random.default_rng(seed)
    cost = _LayoutCost(scenario, spec)
    if first_station is None:
        first_station = pool.positions_m[int(np.argmax(pool.visible.sum(axis=1)))]
    layout = np.asarray(first_station, float).reshape(1, 3)
    while len(layout) < max_stations:
        outcome = evaluate_layout(scenario, layout, True)
        covered = outcome.visible.any(axis=0)
        # Weight the points that most need another view: unmeasured ones, and
        # measured ones in proportion to how poor their uncertainty is. A
        # candidate that sees many of those is worth trying, even when it adds
        # no *new* point -- a second, better-placed look at a badly tied
        # point is exactly what lowers its uncertainty.
        if outcome.feasible:
            u_ref = spec.u_max_m if np.isfinite(spec.u_max_m) else float(np.nanmedian(outcome.u_m))
            need = np.where(covered, np.nan_to_num(outcome.u_m, nan=0.0) / u_ref, 5.0)
        else:
            need = np.where(covered, 0.0, 5.0)
        by_need = np.argsort(-(pool.visible.astype(float) @ need))[:100]
        by_new = np.argsort(-(pool.visible & ~covered[None, :]).sum(axis=1))[:100]
        random = rng.choice(len(pool.positions_m), size=max(n_trials - 200, 0), replace=False)
        trial = np.unique(np.concatenate([by_need, by_new, random]))
        best_cost, best_i = min((cost(np.vstack([layout, pool.positions_m[i]]).ravel()), i) for i in trial)
        layout = np.vstack([layout, pool.positions_m[best_i]])
    return layout


def search_station_counts(
    scenario: InspectionScenario,
    spec: PlanSpec,
    k_values: Sequence[int],
    de_kwargs: Optional[dict] = None,
    seed: int = 0,
    stop_at_first_satisfied: bool = True,
    seed_layouts: Optional[Sequence[np.ndarray]] = None,
    extension_positions_m: Optional[np.ndarray] = None,
    n_extension_trials: int = 300,
) -> List[KResult]:
    """Search each station count in `k_values` (ascending) for the best
    layout under `spec`. Each count is seeded with the prefixes of
    `seed_layouts` of that length (e.g. `greedy_cover_layouts`) and with the
    best layout of the previous count plus one more station -- the best of
    `n_extension_trials` positions drawn from `extension_positions_m` (e.g. a
    `CandidatePool`'s), scored by the plan's own cost, since a station
    dropped at random almost never ties into the network and would hand the
    search an infeasible seed. With
    `stop_at_first_satisfied` the sweep ends at the first count whose best
    layout meets the constraint -- the answer to "how few stations"; the
    K-curve before it (the best violation reached at each smaller count) is
    returned too, since how far short those fall is informative."""
    kwargs = dict(maxiter=60, popsize=12, tol=1e-8, workers=-1)
    kwargs.update(de_kwargs or {})
    bounds = scenario.part.station_bounds_m
    rng = np.random.default_rng(seed)
    results: List[KResult] = []
    previous: Optional[np.ndarray] = None
    for k in k_values:
        seeds = [np.asarray(l)[:k] for l in (seed_layouts or []) if len(l) >= k]
        if previous is not None:
            if extension_positions_m is not None and len(extension_positions_m):
                trial = rng.choice(len(extension_positions_m), size=min(n_extension_trials, len(extension_positions_m)),
                                   replace=False)
                cost = _LayoutCost(scenario, spec)
                options = [np.vstack([previous, extension_positions_m[i]]) for i in trial]
                seeds.append(min(options, key=lambda layout: cost(layout.ravel())))
            else:
                extra = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
                seeds.append(np.vstack([previous, extra]))
        de_result, _ = run_station_search(_LayoutCost(scenario, spec), k, bounds, seed, seeds or None, **kwargs)
        positions = np.asarray(de_result.x).reshape(-1, 3)
        outcome = evaluate_layout(scenario, positions, spec.allow_unmeasured)
        result = KResult(k, positions, outcome, plan_violation(spec, outcome), int(de_result.nfev))
        results.append(result)
        previous = positions
        if stop_at_first_satisfied and result.satisfied:
            break
    return results


def refine_backward(
    scenario: InspectionScenario,
    spec: PlanSpec,
    results: Sequence[KResult],
    de_kwargs: Optional[dict] = None,
    seed: int = 0,
) -> List[KResult]:
    """Backward-elimination pass over a finished sweep. A forward sweep seeds
    each count from the one below, so a good layout that needs a *specific*
    new station (a second, well-placed look at a badly tied group of points)
    is found only when the search happens to stumble on it -- and the curve
    of cost against stations then has steps that are search luck, not
    physics. Here, from the largest count down, every layout of K stations
    is also seeded with the best layout obtained by dropping one station
    from the (K+1)-station answer, and searched again; the better of the
    old and new K-station layout is kept. A K-station layout can never be
    worse than the best (K+1)-station one minus its least useful station, so
    this removes the artefact. Applied to every plan alike."""
    kwargs = dict(maxiter=40, popsize=12, tol=1e-8, workers=-1)
    kwargs.update(de_kwargs or {})
    cost = _LayoutCost(scenario, spec)
    by_k = {r.n_stations: r for r in results}
    for k in sorted(by_k, reverse=True)[1:]:
        if k + 1 not in by_k:
            continue
        bigger = by_k[k + 1].positions_m
        dropped = [np.delete(bigger, i, axis=0) for i in range(len(bigger))]
        seeds = [min(dropped, key=lambda layout: cost(layout.ravel())), by_k[k].positions_m]
        de_result, _ = run_station_search(cost, k, scenario.part.station_bounds_m, seed, seeds, **kwargs)
        positions = np.asarray(de_result.x).reshape(-1, 3)
        outcome = evaluate_layout(scenario, positions, spec.allow_unmeasured)
        candidate = KResult(k, positions, outcome, plan_violation(spec, outcome),
                            by_k[k].n_function_evals + int(de_result.nfev))
        if cost(positions.ravel()) < cost(by_k[k].positions_m.ravel()):
            by_k[k] = candidate
        else:
            by_k[k] = replace(by_k[k], n_function_evals=candidate.n_function_evals)
    return [by_k[k] for k in sorted(by_k)]


def task_summary(scenario: InspectionScenario, outcome: LayoutOutcome) -> Dict[str, Dict[str, float]]:
    """Per-task points, measured points, uncertainty and risk of a
    feasible layout."""
    summary = {}
    for task_id, name in TASK_NAMES.items():
        mask = scenario.part.task_mask(task_id)
        cluster = scenario.clusters[int(scenario.point_cluster[mask][0])].name
        measured = mask & outcome.measured
        has_u = bool(measured.any())
        summary[name] = dict(
            n_points=int(mask.sum()), cluster=cluster, n_measured=int(measured.sum()),
            mean_u_um=float(outcome.u_m[measured].mean() * 1e6) if has_u else float("nan"),
            max_u_um=float(outcome.u_m[measured].max() * 1e6) if has_u else float("nan"),
            e_fr=float(outcome.pfr[mask].sum()), e_fa=float(outcome.pfa[mask].sum()),
            shots=int(outcome.visible[:, mask].sum()),
        )
    return summary
