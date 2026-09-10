"""The headline experiment (CLAUDE.md §5, step 6): build a scene with both
a high-capability and a marginal feature cluster, run uniform-weight and
risk-derived-weight station placement over the *same* grid search, and
report whether -- and how -- the two resulting plans differ.

See docs/step6_headline_experiment.md for the full write-up: why this
scene (two spatially separated clusters, one comfortably capable, one
sitting near its tolerance limit -- reusing step 5's exact validated
numbers), what "total inspection effort" and "global PFA/PFR" mean here,
and the actual result.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from characteristics.characteristic import Characteristic
from characteristics.tolerances import FlatnessTolerance
from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from planning.candidates import hemisphere_candidate_poses
from planning.objectives import PlanEvaluation, PlanningScenario, risk_derived_weights, uniform_weights
from planning.search import GridSearchResult, grid_search_station_placement
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, RiskResult, evaluate_conformity_risk

# The two feature clusters, reusing step 5's exact validated scenario
# (docs/step5_risk_layer_physics.md) so this experiment's risk numbers
# are already known to be sane, not newly-invented constants.
HIGH_CAPABILITY_CLUSTER = FeatureCluster("high-capability", ProcessPrior(mean_m=0.0, std_m=5e-6))
MARGINAL_CLUSTER = FeatureCluster("marginal", ProcessPrior(mean_m=40e-6, std_m=6e-6))
TOLERANCE_ZONE_WIDTH_M = 1e-4  # +-50 um, CLAUDE.md §4d's own reference scale


def default_two_cluster_scenario(anchor_position_m: np.ndarray = None) -> PlanningScenario:
    """The scene this experiment runs on: two spatially separated groups
    of three targets each, all flatness characteristics against the same
    nominal surface normal (z), one group drawn from a high-capability
    process, the other from a marginal one.

    The spatial separation (cluster 1 at y ~ +2, cluster 2 at y ~ -2)
    matters physically, not just for bookkeeping: it means different
    station placements genuinely serve the two clusters differently
    (mainly via range -- a station azimuthally closer to one cluster is
    closer, and therefore more precise, for that cluster and worse for
    the other), so there is a real placement trade-off for an optimiser
    to make between them. Without that separation there would be nothing
    for the two objectives to disagree about.
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
            name=f"cluster1_flatness_{i}",
            target_indices=[i],
            tolerance=FlatnessTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=surface_normal),
        )
        for i in range(3)
    ] + [
        Characteristic(
            name=f"cluster2_flatness_{i}",
            target_indices=[3 + i],
            tolerance=FlatnessTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=surface_normal),
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
        # constraint).
        anchor_position_m = np.array([0.0, 0.0, 1.0])
    anchor_pose = InstrumentPose(position_m=np.asarray(anchor_position_m, dtype=float))

    return PlanningScenario(
        target_points_m=target_points_m,
        tracker=LaserTracker(),
        characteristics=characteristics,
        assignments=assignments,
        decision_rules=decision_rules,
        anchor_pose=anchor_pose,
    )


@dataclass(frozen=True)
class PlanReport:
    """Everything CLAUDE.md §5 step 6 asks to be reported for one plan:
    resulting station positions, total inspection effort, global PFA,
    global PFR, and per-characteristic uncertainty."""

    station_poses: List[InstrumentPose]
    per_characteristic_uncertainty_m: Dict[str, float]
    risk_by_characteristic: Dict[str, RiskResult]

    @property
    def total_inspection_effort_m(self) -> float:
        """The traditional (unweighted) A-optimal metric: the plain sum
        of characteristic uncertainties, regardless of which objective
        produced this plan. Reported for both plans so they're
        comparable on the metric existing tools actually optimise, even
        for the plan (B) that wasn't chosen to minimise it."""
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


def _plan_report(evaluation: PlanEvaluation, scenario: PlanningScenario) -> PlanReport:
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
    search_uniform: GridSearchResult
    search_risk: GridSearchResult

    @property
    def plan_a(self) -> PlanReport:
        """Objective A: uniform weights (A-optimal)."""
        return _plan_report(self.search_uniform.best, self.scenario)

    @property
    def plan_b(self) -> PlanReport:
        """Objective B: risk-derived weights."""
        return _plan_report(self.search_risk.best, self.scenario)


def run_headline_experiment(
    scenario: PlanningScenario = None,
    n_stations: int = 2,
    radius_m: float = 4.0,
    azimuths_deg: Sequence[float] = tuple(range(0, 360, 30)),
    elevations_deg: Sequence[float] = (20.0, 40.0, 60.0),
    reference_uncertainty_m: float = 8e-6,
) -> HeadlineExperimentResult:
    """Run the step 6 comparison: same scenario, same candidate set, same
    grid search, same computational budget -- objective A (uniform
    weights) against objective B (risk-derived weights), the only
    difference between them.

    Defaults reproduce the experiment reported in
    docs/step6_headline_experiment.md: 12 azimuths x 3 elevations = 36
    candidate poses, 2 searched stations (plus the scenario's fixed
    anchor), C(36, 2) = 630 combinations evaluated per objective.
    """
    if scenario is None:
        scenario = default_two_cluster_scenario()

    centre_m = scenario.target_points_m.mean(axis=0)
    candidate_poses = hemisphere_candidate_poses(centre_m, radius_m, azimuths_deg, elevations_deg)

    weights_uniform = uniform_weights(scenario)
    weights_risk = risk_derived_weights(scenario, reference_uncertainty_m)

    search_uniform = grid_search_station_placement(candidate_poses, n_stations, scenario, weights_uniform)
    search_risk = grid_search_station_placement(candidate_poses, n_stations, scenario, weights_risk)

    return HeadlineExperimentResult(
        scenario=scenario,
        weights_risk=weights_risk,
        search_uniform=search_uniform,
        search_risk=search_risk,
    )
