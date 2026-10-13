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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from characteristics.characteristic import Characteristic
from characteristics.datum import Datum
from characteristics.tolerances import ParallelismTolerance, PositionTolerance, ProfileTolerance
from geometry.mesh import box_mesh
from geometry.pose import InstrumentPose
from geometry.visibility import VisibilityLimits
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
from planning.search import (
    GASearchResult,
    GridSearchResult,
    genetic_algorithm_station_placement,
    grid_search_station_placement,
)
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, RiskResult, evaluate_conformity_risk

# The two feature clusters, calibrated to GKN Aerospace's own two
# published capability tiers (CLAUDE.md §1: GKN is this project's named
# industrial partner) rather than an arbitrary pair of numbers: a
# minimum Cpk of 1.33 for standard manufacturing and quality
# requirements, and Cpk 2.0 (their "zero-defect" standard) for critical
# or high-performance programme requirements. An earlier revision set
# MARGINAL_CLUSTER at the 1.33 floor already (for the reason below) but
# left HIGH_CAPABILITY_CLUSTER at an unsourced std (5 um, Cpk ~ 3.33) --
# "comfortably above the floor" with no citation for *how far* above.
# CLAUDE.md §6 ("never fabricate a reference number") applies here just
# as much as to a physics constant: 2.0 is GKN's own real target for
# exactly the kind of critical characteristic this cluster represents,
# so it replaces the unsourced 3.33 rather than sitting alongside it.
#
# Marginal cluster: an earlier draft used mean 40 um, std 6 um (Cpk ~
# 0.56, ~4.6% baseline nonconformance) -- pointed out in supervisory
# review as unrealistically incapable for a key characteristic in
# production: at that Cpk, the risk shown is mostly "this process makes
# bad parts regardless of how well they're measured", not "uncertainty
# matters more here because the process sits close to its limit" -- the
# actual point this experiment exists to make. Fixed to sit exactly on
# GKN's own standard-process floor instead: Cpu = (50-38)/(3*3) = 1.333.
# "Marginal" means *closer to its limit, at a still-legal capability*,
# not *out of control*.
HIGH_CAPABILITY_CLUSTER = FeatureCluster("high-capability", ProcessPrior(mean_m=0.0, std_m=50e-6 / (3 * 2.0)))
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


def mixed_tolerance_two_cluster_scenario(
    anchor_position_m: np.ndarray = None, tracker: Optional[LaserTracker] = None
) -> PlanningScenario:
    """The same two clusters, same six target points, same process
    priors as `default_two_cluster_scenario` -- but each cluster now
    carries one of each tolerance type (CLAUDE.md §3's own example: "A
    hole-position tolerance and a parallelism tolerance on the same
    feature... carry different risk because they project the error
    ellipsoid differently") instead of six identical profile callouts.

    This is the check `docs/step6_headline_experiment.md`'s "What step 6
    still deliberately doesn't do" flagged as the natural next one: A0
    and A1 (and, separately, B and C) landed on the *identical* plan in
    `default_two_cluster_scenario` -- documented there as a property of
    that scene's symmetry (every characteristic shares one projected
    direction, one tolerance type), not a general equivalence. Breaking
    that symmetry is the direct test of whether they actually can
    diverge, per CLAUDE.md §4b/§4e's own insistence that a prediction is
    only meaningful if there's a way for it to come out wrong.

    Per cluster, the three points get:

      - point 0: `PositionTolerance` (spherical) -- constrains all three
        directions at once, so A1's projected uncertainty for this point
        sees the *whole* covariance, same as A0's raw trace does. This
        point is where A0 and A1 should agree most closely.
      - point 1: `ProfileTolerance`, surface normal = z (vertical) -- the
        single direction every characteristic used in the symmetric
        scene; kept as a baseline reference inside the mixed scene too.
      - point 2: `ParallelismTolerance`, datum normal = x (horizontal,
        along the nominal line of sight to the scene) -- a direction
        that has nothing to do with the point's own surface, and nothing
        to do with point 1's z-normal either, per `ParallelismTolerance`'s
        own docstring on why its direction is externally supplied, not
        derived from the measured surface.

    A1 can now legitimately diverge from A0 (it sees three different
    slices of each point's covariance, weighted equally, where A0 always
    sees the whole thing); B and C get the chance to diverge too, since
    a mixed scene's candidates no longer have to trade off identically
    between two interchangeable-by-symmetry clusters.
    """
    cluster1_points_m = np.array(
        [[3.0, 2.0, 1.0], [3.2, 2.3, 1.2], [2.8, 1.8, 0.8]]
    )  # high-capability
    cluster2_points_m = np.array(
        [[3.0, -2.0, 1.0], [3.2, -2.3, 1.2], [2.8, -1.8, 0.8]]
    )  # marginal
    target_points_m = np.vstack([cluster1_points_m, cluster2_points_m])

    surface_normal = np.array([0.0, 0.0, 1.0])  # z: same as the symmetric scene's profile points
    datum = Datum(name="D (mounting face)", normal=np.array([1.0, 0.0, 0.0]))  # x: unrelated to z

    def _tolerances_for_cluster(point_offset: int) -> List[Characteristic]:
        return [
            Characteristic(
                name=f"cluster{point_offset // 3 + 1}_position",
                target_indices=[point_offset],
                tolerance=PositionTolerance(zone_diameter_m=TOLERANCE_ZONE_WIDTH_M),
            ),
            Characteristic(
                name=f"cluster{point_offset // 3 + 1}_profile",
                target_indices=[point_offset + 1],
                tolerance=ProfileTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=surface_normal),
            ),
            Characteristic(
                name=f"cluster{point_offset // 3 + 1}_parallelism",
                target_indices=[point_offset + 2],
                tolerance=ParallelismTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M),
                datum=datum,
            ),
        ]

    characteristics = _tolerances_for_cluster(0) + _tolerances_for_cluster(3)

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


# A support structure under mid-span -- CLAUDE.md §1's own factory-floor
# framing: a 30 m wing in assembly sits in a cradle/jig and shares the
# hall with crane-rail support columns and other fixed structure, not
# loose on an open floor, and that structure is exactly the kind of thing
# that can block a tracker's line of sight. Representative dimensions,
# not drawn from a real jig drawing (none is available to this testbed) --
# flagged here, not silently assumed realistic. Taller (top at z=2.5 m)
# than every target point (all at z <= 2.0 m, `wing_four_point_scenario`'s
# own target_points_m) on purpose: every target needs a surface-normal
# -compliant station *above* it to have acceptable incidence against a
# top-facing surface (CLAUDE.md §3's "risk depends on direction" cuts
# both ways -- an instrument's own visibility is direction-dependent
# too), and a tower capped below every target's own height could never
# occlude a station that already clears that bar. Taller than the
# targets means it can still stand between an elevated, incidence-
# compliant station on one side of mid-span and a target on the other.
WING_SUPPORT_TOWER_MIN_CORNER_M = np.array([13.0, -1.5, 0.0])
WING_SUPPORT_TOWER_MAX_CORNER_M = np.array([17.0, 1.5, 2.5])

# CLAUDE.md §1: "measuring aerospace structures from 2 m to 30 m" -- used
# directly as this scenario's visibility range cap, rather than invented.
WING_MAX_RANGE_M = 30.0
# 80 deg is not a literature figure (CLAUDE.md §4 has none for incidence
# limits); it is the generous-but-finite convention this codebase's own
# visibility tests already use throughout (tests/test_visibility.py's
# GENEROUS_LIMITS, tests/test_visualization.py) -- reused here rather than
# inventing a second unsourced number alongside it.
WING_MAX_INCIDENCE_ANGLE_RAD = np.deg2rad(80.0)


def wing_four_point_scenario(
    anchor_position_m: np.ndarray = None,
    tracker: Optional[LaserTracker] = None,
    include_support_tower: bool = True,
) -> PlanningScenario:
    """A single ~30 m aircraft wing, four measurement points spread along
    its span, with a factory-floor obstacle in the way -- the scene the
    user asked for to make the headline experiment closer to a real
    measurement task: CLAUDE.md §1's own upper range bound (30 m) made
    literal, rather than the original two-cluster scene's few-metres-
    across compact geometry.

    Four points, root to tip, each its own characteristic (no clusters of
    three identical features this time -- a real wing inspection plan
    cares about specific named fittings, not a batch of interchangeable
    points), all on the wing's upper surface (z roughly 1.7-2.0 m --
    above `WING_SUPPORT_TOWER_MAX_CORNER_M`'s own top, see that
    constant's comment for why):

      - index 0, "root_profile" -- x=2 m, near the root/fuselage joint.
      - index 3, "midspan_profile" -- x=15 m, a spar attachment.
      - index 6, "neartip_profile" -- x=26 m, a rib close to the tip.
      - index 7, "tip_profile" -- x=29 m, the tip fitting itself.

    Four more points (indices 1, 2, 4, 5) sit between them, spread along
    the same surface, as pure network *tie points* -- not characteristics
    in their own right, no tolerance, no process-capability assignment,
    just extra shared observations a station can use to pin its own 6-DOF
    pose (CLAUDE.md §7's own glossary: "Network solve... returns a
    covariance" over "all station poses and target coordinates together" --
    nothing in that definition requires every target solved for to be a
    toleranced characteristic). A real tracker network over a 30 m part
    always includes common reference nests like these, beyond whatever
    specific features are being measured, precisely so stations spread
    along the span don't each need direct sight of every single
    characteristic -- CLAUDE.md §5 step 3's `MIN_TARGETS_PER_NON_ANCHOR_STATION`
    threshold (`planning.objectives`) otherwise has nothing nearby for a
    station covering just the tip, say, to see 3 of. Without them, a
    station near one end of the wing cannot be well-posed on its own
    (the 4 characteristics alone are too sparse and too far apart for
    any one station to see 3 of, except from very far back) -- this is
    an engineering necessity of the scene, not a free design choice to
    skip.

    Root and mid-span are assigned the high-capability process prior,
    near-tip and tip the marginal one (the same two GKN-sourced priors
    `default_two_cluster_scenario` uses -- CLAUDE.md §6: reusing a
    sourced number for a new "for the sake of argument" scene rather than
    inventing a third). This is a representative choice, not a claim
    about which real wing fittings are harder to hold to tolerance -- the
    experimental point is having two different priors across a spread-out
    part, as before, not this particular pairing.

    `include_support_tower=True` (the default) adds a block occluder
    (`WING_SUPPORT_TOWER_MIN_CORNER_M`/`_MAX_CORNER_M`, `geometry.mesh.
    box_mesh`) representing a wing-holding jig's support structure under
    mid-span -- see that constant's own comment for what it is and isn't
    based on. `False` gives the identical scene with the visibility layer
    still switched on (range and incidence limits still apply) but
    nothing to occlude -- the direct A/B comparison
    `docs/step6_headline_experiment.md`'s wing-scenario section reports.

    Visibility limits use CLAUDE.md §1's own stated LVM range (30 m) and
    this codebase's existing generous-incidence test convention (80 deg,
    not a literature number -- see `WING_MAX_INCIDENCE_ANGLE_RAD`).

    The default anchor sits elevated near the root (CLAUDE.md §5 step 3's
    fixed frame-of-reference station) and sees a local subset of points
    well (root plus three nearby tie points) -- it is not required to see
    the whole span on its own (nothing in `evaluate_plan`'s feasibility
    check asks the anchor specifically to; only that *some* station,
    anchor or not, sees each target -- see `objectives.
    MIN_TARGETS_PER_NON_ANCHOR_STATION`'s docstring). A single station
    covering all 30 m under a finite incidence limit is not achievable
    here regardless of height (checked directly: no position this
    module's own test suite tried sees every point at once) -- a real
    multi-station plan is exactly what a spread-out part actually needs,
    not an artefact of this scene being under-designed.
    """
    target_points_m = np.array(
        [
            [2.0, 0.0, 1.8],  # 0: root_profile
            [8.0, 0.2, 1.8],  # 1: tie point
            [11.0, -0.2, 1.9],  # 2: tie point
            [15.0, 3.0, 2.0],  # 3: midspan_profile -- offset clear of the
            #                   tower footprint (y in [-1.5, 1.5]): directly
            #                   above the tower it would be unseeable from
            #                   almost any direction at once, which would be
            #                   a modelling artefact (a real fitting is not
            #                   placed where nothing can ever see it), not a
            #                   finding -- CLAUDE.md §4e's own point that a
            #                   prediction has to have a way to come out
            #                   *interesting*, not merely impossible.
            [19.0, -0.2, 1.9],  # 4: tie point
            [22.0, 0.2, 1.8],  # 5: tie point
            [26.0, 0.2, 1.7],  # 6: neartip_profile
            [29.0, 0.1, 1.7],  # 7: tip_profile
        ]
    )
    surface_normal = np.array([0.0, 0.0, 1.0])
    target_normals_m = np.tile(surface_normal, (target_points_m.shape[0], 1))

    characteristic_specs = [
        ("root_profile", 0, HIGH_CAPABILITY_CLUSTER),
        ("midspan_profile", 3, HIGH_CAPABILITY_CLUSTER),
        ("neartip_profile", 6, MARGINAL_CLUSTER),
        ("tip_profile", 7, MARGINAL_CLUSTER),
    ]
    characteristics = [
        Characteristic(
            name=name,
            target_indices=[index],
            tolerance=ProfileTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=surface_normal),
        )
        for name, index, _cluster in characteristic_specs
    ]
    assignments = {
        name: ClusterAssignment(name, cluster, confidence=0.9) for name, _index, cluster in characteristic_specs
    }
    decision_rules = {
        name: DecisionRule.from_zone_width(TOLERANCE_ZONE_WIDTH_M) for name, _index, _cluster in characteristic_specs
    }

    if anchor_position_m is None:
        anchor_position_m = np.array([5.0, -4.0, 5.0])
    anchor_pose = InstrumentPose(position_m=np.asarray(anchor_position_m, dtype=float))

    obstacles = (
        box_mesh(WING_SUPPORT_TOWER_MIN_CORNER_M, WING_SUPPORT_TOWER_MAX_CORNER_M)
        if include_support_tower
        else None
    )

    return PlanningScenario(
        target_points_m=target_points_m,
        tracker=tracker if tracker is not None else LaserTracker(),
        characteristics=characteristics,
        assignments=assignments,
        decision_rules=decision_rules,
        anchor_pose=anchor_pose,
        target_normals_m=target_normals_m,
        obstacles=obstacles,
        visibility_limits=VisibilityLimits(
            max_range_m=WING_MAX_RANGE_M, max_incidence_angle_rad=WING_MAX_INCIDENCE_ANGLE_RAD
        ),
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


# Validated directly on this scenario (docs/step6_headline_experiment.md's
# "The wing scenario" section has the full story): grid search's
# C(n_candidates, n_stations) cost is intractable once candidates come
# from a floor-spanning grid rather than a compact ring, so this scene
# is searched with the GA instead (CLAUDE.md §3's named fallback). A
# plain, unseeded GA run was demonstrably unreliable here even at a ~6x
# bigger budget -- cross-checking one objective's winning stations
# against another's own formula caught misses of up to ~4x. Seeding each
# objective's search with every previously-run objective's winning
# stations (run_wing_experiment's own cascade below) fixed this with a
# far smaller budget than throwing more raw evaluations at the problem
# did. `workers=-1` is the default here (not in `run_headline_experiment`
# above, which doesn't need it) because this scene's own cost is large
# enough that serial evaluation would take on the order of an hour per
# scenario -- see genetic_algorithm_station_placement's own docstring for
# why pinning every BLAS library to one thread per process first
# (`OMP_NUM_THREADS=1` etc.) matters for that to actually help rather
# than hurt.
WING_DEFAULT_BOUNDS_M: Tuple[Tuple[float, float], ...] = ((-2.0, 34.0), (-8.0, 8.0), (1.0, 10.0))
WING_DEFAULT_DE_KWARGS = dict(seed=0, maxiter=60, popsize=20, tol=1e-6, workers=-1, updating="deferred")


@dataclass(frozen=True)
class WingExperimentResult:
    """The GA-search analogue of `HeadlineExperimentResult`, for
    `wing_four_point_scenario` -- same four objectives, same reporting
    (`PlanReport` via `plan_a0`/`plan_a1`/`plan_b`/`plan_c`), backed by
    `GASearchResult` instead of `GridSearchResult`."""

    scenario: PlanningScenario
    search_a0: GASearchResult
    search_a1: GASearchResult
    search_b: GASearchResult
    search_c: GASearchResult

    @property
    def plan_a0(self) -> Optional[PlanReport]:
        return _plan_report(self.search_a0.best, self.scenario)

    @property
    def plan_a1(self) -> Optional[PlanReport]:
        return _plan_report(self.search_a1.best, self.scenario)

    @property
    def plan_b(self) -> Optional[PlanReport]:
        return _plan_report(self.search_b.best, self.scenario)

    @property
    def plan_c(self) -> Optional[PlanReport]:
        return _plan_report(self.search_c.best, self.scenario)


def run_wing_experiment(
    include_support_tower: bool = True,
    n_stations: int = 3,
    bounds_m: Sequence[Tuple[float, float]] = WING_DEFAULT_BOUNDS_M,
    reference_uncertainty_m: float = 8e-6,
    de_kwargs: Optional[Dict] = None,
) -> WingExperimentResult:
    """Run the A0/A1/B/C comparison on `wing_four_point_scenario` via the
    GA search, cascading each objective's initial population seed from
    every objective already run (A0 first, from a blind random initial
    population; A1 seeded with A0's result; B seeded with A0's and A1's;
    C seeded with all three) -- see the module-level comment above and
    docs/step6_headline_experiment.md's "The wing scenario" section for
    why this is necessary and how it was validated, not just assumed to
    help.

    `de_kwargs`, if given, overrides `WING_DEFAULT_DE_KWARGS` entries
    (e.g. to trade budget for runtime -- this scene's own cost per
    evaluation makes that tradeoff much more consequential than for
    `run_headline_experiment`'s scene).
    """
    scenario = wing_four_point_scenario(include_support_tower=include_support_tower)
    weights_uniform = uniform_weights(scenario)
    weights_risk = risk_derived_weights(scenario, reference_uncertainty_m)

    objectives = [
        ("A0", objective_a0_trace_covariance),
        ("A1", make_weighted_uncertainty_objective(weights_uniform)),
        ("B", make_weighted_uncertainty_objective(weights_risk)),
        ("C", objective_c_direct_risk),
    ]
    kwargs = dict(WING_DEFAULT_DE_KWARGS)
    if de_kwargs:
        kwargs.update(de_kwargs)

    seed_positions: List[List[np.ndarray]] = []
    searches = {}
    for name, objective_fn in objectives:
        search = genetic_algorithm_station_placement(
            scenario,
            n_stations=n_stations,
            objective_fn=objective_fn,
            bounds_m=bounds_m,
            seed_station_positions_m=list(seed_positions) if seed_positions else None,
            **kwargs,
        )
        searches[name] = search
        if search.best is not None:
            seed_positions.append([pose.position_m for pose in search.best.station_poses])

    return WingExperimentResult(
        scenario=scenario,
        search_a0=searches["A0"],
        search_a1=searches["A1"],
        search_b=searches["B"],
        search_c=searches["C"],
    )
