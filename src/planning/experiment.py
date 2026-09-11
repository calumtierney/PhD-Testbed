"""The headline experiment (CLAUDE.md §5, step 6): build a scene with both
a high-capability and a marginal feature cluster, and compare four station-
placement objectives over the *same* grid search, scene and computational
budget:

  A0. trace of the full 3D point covariance (the field's actual criterion)
  A1. uniform weights on projected characteristic uncertainty ("Plan A")
  B.  risk-derived weights, linearised at a fixed reference point ("Plan B")
  C.  direct nonlinear expected-risk minimisation, no linearisation

See `planning.objectives`'s module docstring for what each of these is and
why all four are needed (a supervisory review of an earlier draft's
"Plan A vs Plan B" framing pointed out that neither A1 nor B is literally
the field's own criterion -- A0 is; and that a linearised weight (B) has
no principled reason to be preferred over C when a single evaluation
costs ~10ms). See docs/step6_headline_experiment.md for the full write-up
of scene, results and what changed between draft and this revision.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from characteristics.characteristic import Characteristic
from characteristics.tolerances import ProfileTolerance
from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from planning.candidates import hemisphere_candidate_poses
from planning.objectives import (
    ObjectiveFn,
    PlanEvaluation,
    PlanningScenario,
    constrained_objective,
    make_weighted_uncertainty_objective,
    objective_a0_trace_covariance,
    objective_c_direct_risk,
    risk_derived_weights,
    uniform_weights,
)
from planning.search import GridSearchResult, grid_search_station_placement
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, RiskResult, evaluate_conformity_risk

# The two feature clusters. Process parameters chosen so BOTH are
# aerospace-plausible per SAE AS13006 ("a minimum capability index Cpk of
# 1.33"), not just the high-capability one -- an earlier draft's marginal
# cluster (Cpk ~ 0.56, ~4.6% baseline nonconformance) was pointed out in
# supervisory review as unrealistically incapable for a key characteristic
# in production: at that Cpk, the risk shown is mostly "this process makes
# bad parts regardless of how well they're measured", not "uncertainty
# matters more here because the process sits close to its limit" -- the
# actual point this experiment exists to make. Both clusters now clear the
# AS13006 floor; "marginal" means *closer to its limit*, at a still-legal
# capability, not *out of control*.
HIGH_CAPABILITY_CLUSTER = FeatureCluster("high-capability", ProcessPrior(mean_m=0.0, std_m=5e-6))
# Cpu = (50 - 38) / (3*3) = 1.333 -- exactly the AS13006 minimum.
MARGINAL_CLUSTER = FeatureCluster("marginal", ProcessPrior(mean_m=38e-6, std_m=3e-6))
TOLERANCE_ZONE_WIDTH_M = 1e-4  # +-50 um, CLAUDE.md §4d's own reference scale

# Candidate stations stand on a hemisphere around the target centroid
# (planning.candidates), but not the full hemisphere: floor- or
# tripod-mounted laser trackers cannot hover far above a part. With a
# 4 m standoff radius, elevations from level with the target to +20 deg
# (a station ~1.4 m above target height, a tripod on a low platform) keep
# every candidate within a station height a real tripod could reach for
# targets around head height -- not the 0-90 deg full-hemisphere range an
# earlier draft used, which permitted candidates directly overhead (a
# floor tracker cannot get there). Two elevation levels, not more: each
# grid search pays for a full C(n_candidates, n_stations) sweep, and
# objective C's per-candidate cost (direct nonlinear risk, several
# scipy.integrate.quad calls per characteristic) makes the *number* of
# candidates the dominant runtime cost -- see
# docs/step6_headline_experiment.md's "Runtime" note.
DEFAULT_AZIMUTHS_DEG = tuple(range(0, 360, 30))
DEFAULT_ELEVATIONS_DEG = (0.0, 20.0)


def default_two_cluster_scenario(
    anchor_position_m: np.ndarray = None, tracker: Optional[LaserTracker] = None
) -> PlanningScenario:
    """The scene this experiment runs on: two spatially separated groups
    of three targets each, all profile characteristics (see
    `characteristics.tolerances`'s naming note -- this is a point's normal
    deviation, not true multi-point GD&T flatness) against the same
    nominal surface normal, one group drawn from a high-capability
    process, the other from a marginal-but-still-AS13006-legal one.

    The spatial separation (cluster 1 at y ~ +2, cluster 2 at y ~ -2)
    matters physically, not just for bookkeeping: it means different
    station placements genuinely serve the two clusters differently
    (mainly via range -- a station azimuthally closer to one cluster is
    closer, and therefore more precise, for that cluster and worse for
    the other), so there is a real placement trade-off for an optimiser
    to make between them. Without that separation there would be nothing
    for the objectives to disagree about.

    `tracker` defaults to a pure-sensor-noise `LaserTracker()`
    (`systematic_std_m=0`); pass one with a nonzero `systematic_std_m` to
    see how an irreducible-by-placement uncertainty floor changes the
    comparison (CLAUDE.md referee note M6; `docs/step6_headline_experiment.md`'s
    "Uncertainty budget" section runs this).
    """
    cluster1_points_m = np.array(
        [[3.0, 2.0, 1.0], [3.2, 2.3, 1.2], [2.8, 1.8, 0.8]]
    )  # high-capability
    cluster2_points_m = np.array(
        [[3.0, -2.0, 1.0], [3.2, -2.3, 1.2], [2.8, -1.8, 0.8]]
    )  # marginal
    target_points_m = np.vstack([cluster1_points_m, cluster2_points_m])

    surface_normal = np.array([0.0, 0.0, 1.0])
    characteristics = [
        Characteristic(
            name=f"cluster1_profile_{i}",
            target_indices=[i],
            tolerance=ProfileTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=surface_normal),
        )
        for i in range(3)
    ] + [
        Characteristic(
            name=f"cluster2_profile_{i}",
            target_indices=[3 + i],
            tolerance=ProfileTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=surface_normal),
        )
        for i in range(3)
    ]

    assignments = {}
    for characteristic in characteristics[:3]:
        assignments[characteristic.name] = ClusterAssignment(
            characteristic.name, HIGH_CAPABILITY_CLUSTER, confidence=0.9
        )
    for characteristic in characteristics[3:]:
        assignments[characteristic.name] = ClusterAssignment(characteristic.name, MARGINAL_CLUSTER, confidence=0.9)

    decision_rules = {
        characteristic.name: DecisionRule.from_zone_width(TOLERANCE_ZONE_WIDTH_M)
        for characteristic in characteristics
    }

    if anchor_position_m is None:
        # A fixed reference station, well clear of the scene -- see the
        # module docstring's note that the anchor is a placed, not
        # searched-over, station (CLAUDE.md §5 step 3's frame-of-reference
        # constraint; docs/step6_headline_experiment.md's "datum
        # dependence" section on why this choice is a real modelling
        # decision, not an arbitrary one).
        anchor_position_m = np.array([0.0, 0.0, 1.0])
    anchor_pose = InstrumentPose(position_m=np.asarray(anchor_position_m, dtype=float))

    return PlanningScenario(
        target_points_m=target_points_m,
        tracker=tracker if tracker is not None else LaserTracker(),
        characteristics=characteristics,
        assignments=assignments,
        decision_rules=decision_rules,
        anchor_pose=anchor_pose,
    )


@dataclass(frozen=True)
class PlanReport:
    """Everything CLAUDE.md §5 step 6 asks to be reported for one plan:
    resulting station positions, summed characteristic uncertainty, global
    PFA, global PFR, and per-characteristic uncertainty."""

    station_poses: List[InstrumentPose]
    per_characteristic_uncertainty_m: Dict[str, float]
    risk_by_characteristic: Dict[str, RiskResult]

    @property
    def summed_characteristic_uncertainty_m(self) -> float:
        """The plain sum of characteristic uncertainties -- what A0/A1
        minimise. Named for exactly what it is: a sum of *uncertainty*,
        not a measure of inspection *effort* or *time* (station count and
        target set are identical across every plan compared here, so
        neither effort nor time actually varies between them -- this
        number is not a proxy for either)."""
        return float(sum(self.per_characteristic_uncertainty_m.values()))

    @property
    def global_pfa(self) -> float:
        """Expected number of falsely-accepted characteristics across the
        whole plan -- sum of each characteristic's own PFA. Additive
        because expectation is additive regardless of correlation between
        characteristics' errors; not a claim that this equals "the
        probability some characteristic is falsely accepted" (which,
        with correlated station-pose uncertainty behind several
        characteristics, would need the joint distribution, not just each
        marginal)."""
        return float(sum(r.probability_false_acceptance for r in self.risk_by_characteristic.values()))

    @property
    def global_pfr(self) -> float:
        """Expected number of falsely-rejected characteristics -- see
        `global_pfa`'s docstring for the same additive-expectation caveat."""
        return float(sum(r.probability_false_rejection for r in self.risk_by_characteristic.values()))

    @property
    def bounded_part_level_risk(self) -> float:
        """`1 - prod(1 - r_c)` over every characteristic's own total risk
        `r_c` (its PFA + PFR) -- a genuine probability (bounded in
        [0, 1]), unlike `global_pfa + global_pfr` summed across
        characteristics, which is an expected *count* of errors and can
        exceed 1 for a large enough characteristic set. Reported
        alongside, not instead of, `global_pfa`/`global_pfr`: this
        statistic assumes each characteristic's accept/reject error is
        independent of the others', which is not exactly true here (every
        characteristic in a plan shares the same stations, so their
        errors share the same station-pose uncertainty and are therefore
        correlated) -- treat it as an upper-bound-style summary, not an
        exact joint part-level risk. An exact version needs the joint
        distribution over all characteristics' decisions, not built here.
        """
        survival = 1.0
        for r in self.risk_by_characteristic.values():
            per_characteristic_risk = r.probability_false_acceptance + r.probability_false_rejection
            survival *= 1.0 - min(per_characteristic_risk, 1.0)
        return 1.0 - survival


def _plan_report(evaluation: Optional[PlanEvaluation], scenario: PlanningScenario) -> Optional[PlanReport]:
    if evaluation is None:
        return None
    risk_by_characteristic = {}
    for characteristic in scenario.characteristics:
        assignment = scenario.assignments[characteristic.name]
        decision_rule = scenario.decision_rules[characteristic.name]
        uncertainty_m = evaluation.per_characteristic_uncertainty_m[characteristic.name]
        risk_by_characteristic[characteristic.name] = evaluate_conformity_risk(
            assignment, uncertainty_m, decision_rule
        )
    return PlanReport(
        station_poses=evaluation.station_poses,
        per_characteristic_uncertainty_m=evaluation.per_characteristic_uncertainty_m,
        risk_by_characteristic=risk_by_characteristic,
    )


@dataclass(frozen=True)
class HeadlineExperimentResult:
    scenario: PlanningScenario
    weights_risk: np.ndarray
    search_a0: GridSearchResult
    search_a1: GridSearchResult
    search_b: GridSearchResult
    search_c: GridSearchResult

    @property
    def plan_a0(self) -> Optional[PlanReport]:
        """A0: trace of the full 3D point covariance -- the field's actual criterion."""
        return _plan_report(self.search_a0.best, self.scenario)

    @property
    def plan_a1(self) -> Optional[PlanReport]:
        """A1: uniform weights on projected characteristic uncertainty ("Plan A")."""
        return _plan_report(self.search_a1.best, self.scenario)

    @property
    def plan_b(self) -> Optional[PlanReport]:
        """B: risk-derived weights, linearised at a fixed reference point ("Plan B")."""
        return _plan_report(self.search_b.best, self.scenario)

    @property
    def plan_c(self) -> Optional[PlanReport]:
        """C: direct nonlinear expected-risk minimisation, no linearisation."""
        return _plan_report(self.search_c.best, self.scenario)


def run_headline_experiment(
    scenario: PlanningScenario = None,
    n_stations: int = 2,
    radius_m: float = 4.0,
    azimuths_deg: Sequence[float] = DEFAULT_AZIMUTHS_DEG,
    elevations_deg: Sequence[float] = DEFAULT_ELEVATIONS_DEG,
    reference_uncertainty_m: float = 8e-6,
    max_uncertainty_m: Optional[float] = None,
) -> HeadlineExperimentResult:
    """Run the step 6 comparison: same scenario, same candidate set, same
    grid search, same computational budget -- objectives A0, A1, B and C
    (see the module docstring), the only difference between them being
    what each one scores a candidate plan by.

    `max_uncertainty_m`, if given, disqualifies any candidate where *any*
    characteristic's uncertainty would exceed it (`objectives.
    constrained_objective` -- guards against B's linearised weight letting
    a low-weight characteristic degrade without bound; CLAUDE.md referee
    note M9). Applied identically to all four objectives for a fair
    comparison; `None` (the default) applies no cap.

    Defaults reproduce the experiment reported in
    docs/step6_headline_experiment.md: 12 azimuths x 2 elevations = 24
    candidate poses (a floor/tripod-reachable envelope -- see
    `DEFAULT_ELEVATIONS_DEG`), 2 searched stations (plus the scenario's
    fixed anchor), C(24, 2) = 276 combinations evaluated per objective.
    Objective C's per-candidate cost (several `scipy.integrate.quad`
    calls per characteristic) makes it the slowest of the four by roughly
    two orders of magnitude per evaluation; this resolution keeps a full
    4-objective run in well under a minute rather than searching a finer
    grid A0/A1/B could each individually afford much more cheaply.
    """
    if scenario is None:
        scenario = default_two_cluster_scenario()

    centre_m = scenario.target_points_m.mean(axis=0)
    candidate_poses = hemisphere_candidate_poses(centre_m, radius_m, azimuths_deg, elevations_deg)

    weights_uniform = uniform_weights(scenario)
    weights_risk = risk_derived_weights(scenario, reference_uncertainty_m)

    objective_a0: ObjectiveFn = constrained_objective(objective_a0_trace_covariance, max_uncertainty_m)
    objective_a1: ObjectiveFn = constrained_objective(
        make_weighted_uncertainty_objective(weights_uniform), max_uncertainty_m
    )
    objective_b: ObjectiveFn = constrained_objective(
        make_weighted_uncertainty_objective(weights_risk), max_uncertainty_m
    )
    objective_c: ObjectiveFn = constrained_objective(objective_c_direct_risk, max_uncertainty_m)

    search_a0 = grid_search_station_placement(candidate_poses, n_stations, scenario, objective_a0)
    search_a1 = grid_search_station_placement(candidate_poses, n_stations, scenario, objective_a1)
    search_b = grid_search_station_placement(candidate_poses, n_stations, scenario, objective_b)
    search_c = grid_search_station_placement(candidate_poses, n_stations, scenario, objective_c)

    return HeadlineExperimentResult(
        scenario=scenario,
        weights_risk=weights_risk,
        search_a0=search_a0,
        search_a1=search_a1,
        search_b=search_b,
        search_c=search_c,
    )


def total_expected_risk(evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
    """A plan's total expected risk (sum of PFA + PFR over every
    characteristic), independent of which objective produced it -- the
    common yardstick `iso_risk_station_counts` below compares every
    objective against. Identical to `objective_c_direct_risk`; exposed
    under this name because using it as a *reporting* metric, after a
    plan has already been chosen by some other objective, is a different
    use from using it as the thing being searched over."""
    return objective_c_direct_risk(evaluation, scenario)


def iso_risk_station_counts(
    scenario: PlanningScenario,
    target_total_risk: float,
    objective_fns_by_name: Dict[str, ObjectiveFn],
    radius_m: float = 4.0,
    azimuths_deg: Sequence[float] = DEFAULT_AZIMUTHS_DEG,
    elevations_deg: Sequence[float] = DEFAULT_ELEVATIONS_DEG,
    max_stations: int = 4,
) -> Dict[str, Optional[int]]:
    """For each named objective, the smallest number of (non-anchor)
    stations, from 1 up to `max_stations`, whose best plan under that
    objective achieves `total_expected_risk <= target_total_risk`.

    This is the "iso-risk" comparison a supervisory review asked for
    (CLAUDE.md referee notes M11/M14): rather than compare objectives at
    one fixed station count (as `run_headline_experiment` does) and note
    that risk differs by some percentage, ask the industrially meaningful
    question directly -- how many stations does each objective actually
    need to reach a given risk level? An objective that reaches the
    target with fewer stations is doing more with the same instrument
    budget; `None` for an objective means it didn't reach the target
    within `max_stations`.

    Every objective is searched over the *same* growing candidate set (one
    grid search per station count, per objective) -- no shortcuts specific
    to any one objective, so the comparison stays fair.
    """
    centre_m = scenario.target_points_m.mean(axis=0)
    candidate_poses = hemisphere_candidate_poses(centre_m, radius_m, azimuths_deg, elevations_deg)

    results: Dict[str, Optional[int]] = {name: None for name in objective_fns_by_name}
    for n_stations in range(1, max_stations + 1):
        for name, objective_fn in objective_fns_by_name.items():
            if results[name] is not None:
                continue
            search = grid_search_station_placement(candidate_poses, n_stations, scenario, objective_fn)
            if search.best is None:
                continue
            if total_expected_risk(search.best, scenario) <= target_total_risk:
                results[name] = n_stations
    return results
