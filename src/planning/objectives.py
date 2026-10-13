"""The objectives compared in step 6, the headline experiment (CLAUDE.md
§5, step 6), and the ablation a supervisory review of the resulting paper
required (see docs/step6_headline_experiment.md's "Ablation" section for
the full reasoning; summarised here):

  A0. **Trace of the full 3D point covariance** -- the field's actual
      A-optimal criterion (a sum of *variances*): Schmitt et al. 2016
      literally minimise this trace; Cai 2013 and Wang, Forbes &
      Maropoulos 2014 minimise closely related sums of point standard
      uncertainties. Ignores tolerance direction entirely.
  A1. **Uniform weights on projected characteristic uncertainty** --
      what this codebase calls "Plan A". This is *already* a
      goal-oriented criterion (it uses step 4's tolerance-direction
      projection), just an unweighted one -- not literally A0, though it
      behaves similarly when, as here, every characteristic happens to
      share one projected direction and one tolerance.
  B.  **Risk-derived weights** on the same projected uncertainties --
      "Plan B": each characteristic's weight is the *linearised*
      sensitivity of its own expected decision risk to its own
      uncertainty, computed once before the search (a fixed reference
      point), not recomputed per candidate.
  C.  **Direct nonlinear risk minimisation** -- no linearised weight
      proxy at all: score each candidate by its actual, achieved global
      risk, recomputed exactly for that candidate's own uncertainty.

All four share *exactly* the same `evaluate_plan` (one network solve, one
step 4 projection per characteristic) -- CLAUDE.md §1's "replacing the
objective function, not the optimiser" is implemented literally: one
evaluation, four interchangeable scoring functions (`ObjectiveFn` below),
picked as a plain argument to `planning.search.grid_search_station_placement`.

**Why B exists at all, next to C.** With the exact-observations shortcut
(`evaluate_plan`'s docstring) a single evaluation costs ~10 ms, so C is
computationally free to use directly -- there is no *runtime* reason to
prefer B's linearised weight. B's reason to exist is a *deployment* one:
CLAUDE.md's own "weighting substitution" framing is that a precomputed
weight can be dropped into an *existing* weighted-sum planning tool
(e.g. the Wang-Forbes-Maropoulos optimiser) without that tool ever
needing to know about JCGM 106 risk at all -- only ever having to accept
a per-characteristic weight, exactly as it already does for uniform
weights. C needs the full risk model inside the search loop; B needs it
only once, beforehand. Both are reported so a reader can see whether the
linearisation costs anything in this scene (docs/step6_headline_experiment.md).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from characteristics.characteristic import Characteristic, evaluate_characteristic
from geometry.mesh import TriangleMesh
from geometry.pose import InstrumentPose
from geometry.visibility import VisibilityLimits, check_target_visibility
from instruments.laser_tracker import LaserTracker
from network.solve import NetworkSolveResult, Observation, exact_observations, solve_network
from risk.jcgm106 import ClusterAssignment, DecisionRule, evaluate_conformity_risk


@dataclass(frozen=True)
class PlanningScenario:
    """Everything a station-placement search needs, other than the
    station poses themselves.

    Attributes
    ----------
    target_points_m : (N, 3) ndarray -- every target in the scene.
    tracker : the instrument model (shared by every station -- CLAUDE.md
        §5 step 3's simplification, unchanged here).
    characteristics : the characteristics the plan is judged on.
    assignments : characteristic name -> ClusterAssignment (step 5).
    decision_rules : characteristic name -> DecisionRule (step 5).
    anchor_pose : the fixed frame-of-reference station (CLAUDE.md §5 step
        3) -- included in every candidate plan as an already-placed,
        not-search-over station (e.g. a station whose position is fixed
        by some other constraint); pass an empty list of *additional*
        stations to search over just the anchor alone.
    target_normals_m : (N, 3) ndarray or None -- outward unit surface
        normal at each target, for the visibility layer's incidence-angle
        check (`geometry.visibility`, step 2). None (the default) means
        every scenario built before this field existed keeps running the
        occlusion-free, incidence-free fast path below exactly as before.
    obstacles : `geometry.mesh.TriangleMesh` or None -- blocking geometry
        for the visibility layer's occlusion check (step 2): a wall, a
        tooling fixture, another part of the structure, anything a beam
        can be stopped by on the factory floor. None means nothing can
        occlude anything, same as an empty scene.
    visibility_limits : `geometry.visibility.VisibilityLimits` or None --
        the instrument's max range and max incidence angle, and the
        single switch for whether the visibility layer runs at all
        (`visibility_enabled` below). Required (non-None) the moment
        `obstacles` or `target_normals_m` is set -- `__post_init__`
        enforces this, since those fields would otherwise be silently
        ignored. `None` with `obstacles=None` and `target_normals_m=None`
        (the default) skips the whole visibility layer: every station
        sees every target, exactly as step 6 worked before occlusion was
        wired in here.
    """

    target_points_m: np.ndarray
    tracker: LaserTracker
    characteristics: List[Characteristic]
    assignments: Dict[str, ClusterAssignment]
    decision_rules: Dict[str, DecisionRule]
    anchor_pose: InstrumentPose
    target_normals_m: Optional[np.ndarray] = None
    obstacles: Optional[TriangleMesh] = None
    visibility_limits: Optional[VisibilityLimits] = None

    def __post_init__(self):
        if self.visibility_limits is None and (self.obstacles is not None or self.target_normals_m is not None):
            raise ValueError(
                "PlanningScenario sets obstacles or target_normals_m but no visibility_limits -- "
                "without a max_range_m/max_incidence_angle_rad envelope those fields would be "
                "silently ignored (evaluate_plan only invokes the visibility layer when "
                "visibility_limits is set); pass visibility_limits explicitly"
            )

    @property
    def visibility_enabled(self) -> bool:
        """Whether the visibility layer (step 2: range, incidence,
        occlusion) is switched on for this scenario -- the single
        condition `evaluate_plan` checks to decide between the
        full-visibility fast path (every station sees every target, no
        `geometry.visibility` call at all) and the per-pair filtered
        path. `visibility_limits` alone is sufficient to turn it on (a
        bare range/incidence envelope, no mesh, is still a real
        constraint -- see `test_visibility_limits_out_of_range_disqualifies_the_whole_plan`);
        `__post_init__` above guarantees `obstacles`/`target_normals_m`
        never appear without it."""
        return self.visibility_limits is not None


def uniform_weights(scenario: PlanningScenario) -> np.ndarray:
    """Objective A1's weights: 1 for every characteristic."""
    return np.ones(len(scenario.characteristics))


def risk_derived_weights(
    scenario: PlanningScenario, reference_uncertainty_m: float, epsilon_m: float = 0.5e-6
) -> np.ndarray:
    """Objective B's weights: the local sensitivity of each
    characteristic's global risk to its own uncertainty, evaluated at a
    shared `reference_uncertainty_m` and normalised to a mean of 1.

    A central finite difference (`epsilon_m` either side of
    `reference_uncertainty_m`) estimates `d(PFA + PFR)/d(uncertainty)`
    using `risk.jcgm106.evaluate_conformity_risk` directly -- no new risk
    machinery, just the existing step 5 model differentiated numerically.

    `reference_uncertainty_m` is a single, shared value across every
    characteristic -- a representative single-station measurement
    precision (CLAUDE.md §4b's own ~5-9 um figures are a reasonable
    choice) -- rather than each characteristic's own value from whichever
    candidate plan is currently being scored. Using the candidate's own
    achieved uncertainty would make the weights change *during* the
    search (a moving target: the objective a candidate is scored against
    would depend on that candidate's own result), which would make A1 and
    B no longer comparable as "the same optimiser, different objective"
    (CLAUDE.md §3) -- the weights have to be fixed before the search
    starts, exactly as CLAUDE.md §5 step 6 describes them ("compute those
    weights from the risk layer", as a preprocessing step, not inside the
    search loop). Objective C below sidesteps this entirely by not
    linearising at all.

    Normalising to mean 1 puts B on the same total scale as A1 (which
    implicitly weights everything at 1): the comparison this way isolates
    *how the same total weight gets redistributed* between
    characteristics, rather than conflating that with an arbitrary
    overall scale difference between the two objectives.
    """
    slopes = []
    for characteristic in scenario.characteristics:
        assignment = scenario.assignments[characteristic.name]
        decision_rule = scenario.decision_rules[characteristic.name]
        risk_plus = evaluate_conformity_risk(assignment, reference_uncertainty_m + epsilon_m, decision_rule)
        risk_minus = evaluate_conformity_risk(assignment, reference_uncertainty_m - epsilon_m, decision_rule)
        total_plus = risk_plus.probability_false_acceptance + risk_plus.probability_false_rejection
        total_minus = risk_minus.probability_false_acceptance + risk_minus.probability_false_rejection
        slopes.append((total_plus - total_minus) / (2.0 * epsilon_m))

    slopes = np.array(slopes)
    mean_slope = slopes.mean()
    if mean_slope <= 0.0:
        raise ValueError(
            "risk sensitivity is zero or negative for every characteristic at this "
            "reference uncertainty -- weights cannot be normalised; try a larger "
            "reference_uncertainty_m closer to where these processes actually carry risk"
        )
    return slopes / mean_slope


@dataclass(frozen=True)
class PlanEvaluation:
    """The raw outcome of evaluating one candidate set of station poses --
    not yet reduced to a single objective value. Every `ObjectiveFn` below
    is computed FROM this, so every objective sees exactly the same
    physics (CLAUDE.md §1: "replacing the objective function, not the
    optimiser").

    Attributes
    ----------
    station_poses : the additional (non-anchor) stations that were scored.
    per_characteristic_uncertainty_m : characteristic name -> projected
        (tolerance-direction) uncertainty -- step 4's output. Independent
        of which objective produced this plan; CLAUDE.md §5 step 6 asks
        for this to be reported regardless.
    network_solve : the full NetworkSolveResult this was computed from
        (design covariance -- see `evaluate_plan`'s docstring), giving
        access to each target's full 3x3 covariance for objectives that
        need the raw coordinate covariance rather than a projection
        (`objective_a0_trace_covariance` below).
    """

    station_poses: List[InstrumentPose]
    per_characteristic_uncertainty_m: Dict[str, float]
    network_solve: NetworkSolveResult


# A non-anchor station's pose is 6 free parameters (position + orientation);
# each target it observes contributes 3 residuals (d, theta, phi). 3 targets
# (9 residuals) is the minimum that doesn't leave the pose under-determined
# in raw residual count, with margin for the fact that 2 targets can still
# leave some combination of rotation/translation poorly constrained for
# some geometries (e.g. two targets nearly along the same bearing). This is
# an engineering floor, not a rigorously derived bound -- see
# `_visible_observations`'s docstring for what it guards against.
MIN_TARGETS_PER_NON_ANCHOR_STATION = 3


def _visible_observations(
    target_points_m: np.ndarray, all_poses: List[InstrumentPose], scenario: PlanningScenario, anchor_index: int
) -> Tuple[List[Observation], bool]:
    """Every station-target observation this plan can actually take, after
    the visibility layer (step 2: range, incidence, occlusion), plus
    whether the result is even solvable.

    Without this filter, `network.solve.solve_network` would happily
    accept an observation list where some target has zero observations,
    or some non-anchor station has too few to pin its own pose -- its
    normal matrix would then carry an all-zero row/column for the
    unconstrained parameters, and `np.linalg.pinv` silently assigns that
    direction *zero* variance (the minimum-norm convention for a singular
    system), not the infinite variance "this was never actually measured"
    physically means. An occluded target would then look spuriously
    *perfect* instead of unmeasured -- exactly backwards. `feasible=False`
    tells `evaluate_plan` to refuse the solve entirely rather than return
    a number that looks fine and means the opposite.
    """
    observations = exact_observations(target_points_m, all_poses)
    if not scenario.visibility_enabled:
        return observations, True

    visible = []
    n_targets = target_points_m.shape[0]
    observed_count = np.zeros(n_targets, dtype=int)
    station_count = np.zeros(len(all_poses), dtype=int)
    for obs in observations:
        pose = all_poses[obs.station_index]
        normal_m = (
            None if scenario.target_normals_m is None else scenario.target_normals_m[obs.target_index]
        )
        result = check_target_visibility(
            pose.position_m,
            target_points_m[obs.target_index],
            scenario.visibility_limits,
            target_normal_m=normal_m,
            obstacles=scenario.obstacles,
        )
        if result.is_visible:
            visible.append(obs)
            observed_count[obs.target_index] += 1
            station_count[obs.station_index] += 1

    every_target_seen = bool(np.all(observed_count > 0))
    every_non_anchor_station_well_posed = all(
        station_count[i] >= MIN_TARGETS_PER_NON_ANCHOR_STATION
        for i in range(len(all_poses))
        if i != anchor_index
    )
    return visible, every_target_seen and every_non_anchor_station_well_posed


def evaluate_plan(
    station_poses: List[InstrumentPose], scenario: PlanningScenario
) -> Optional[PlanEvaluation]:
    """Compute the raw ingredients (network solve, per-characteristic
    uncertainty) for one candidate set of station poses -- deliberately
    *not* reduced to a single score here; pass the result to whichever
    `ObjectiveFn` the search is using.

    Solves the network from *exact* (noiseless) observations at the true
    scene geometry (`network.solve.exact_observations`) rather than
    simulating random noise and fitting it: for this linear-Gaussian
    measurement model, the resulting covariance is -- to numerical
    precision -- identical to what a real noisy solve's covariance would
    be (docs/step6_headline_experiment.md has the direct empirical
    check), while being both deterministic (every candidate compared on
    equal footing, no RNG state to manage across a large search) and far
    cheaper (the solve converges in a single evaluation, since the
    initial guess already *is* the exact answer).

    Returns `None`, instead of a `PlanEvaluation`, when `scenario.
    visibility_enabled` and this particular set of station poses leaves
    some target unseen by every station, or some non-anchor station
    unable to pin its own pose (`_visible_observations` above) -- a plan
    the search must disqualify, not score. Every caller that drives a
    search over many candidate combinations (`planning.search`) already
    has to handle "this candidate doesn't qualify" (`objectives.
    constrained_objective`'s `+inf` convention); `None` here is the same
    idea one layer earlier, before an objective value can even be computed.
    """
    all_poses = [scenario.anchor_pose] + list(station_poses)
    observations, feasible = _visible_observations(scenario.target_points_m, all_poses, scenario, anchor_index=0)
    if not feasible:
        return None

    network_solve = solve_network(
        observations, scenario.target_points_m.copy(), all_poses, scenario.tracker, anchor_index=0
    )

    per_characteristic_uncertainty_m = {
        characteristic.name: evaluate_characteristic(characteristic, network_solve)[0].uncertainty_m
        for characteristic in scenario.characteristics
    }
    return PlanEvaluation(
        station_poses=list(station_poses),
        per_characteristic_uncertainty_m=per_characteristic_uncertainty_m,
        network_solve=network_solve,
    )


# An ObjectiveFn takes one evaluation (plus the scenario it was computed
# under, for anything an objective needs beyond the evaluation itself --
# e.g. C needs each characteristic's ClusterAssignment/DecisionRule to
# compute risk) and returns a single value the search minimises.
ObjectiveFn = Callable[[PlanEvaluation, PlanningScenario], float]


def make_weighted_uncertainty_objective(weights: np.ndarray) -> ObjectiveFn:
    """Objectives A1 (`weights=uniform_weights(...)`) and B
    (`weights=risk_derived_weights(...)`): `sum(weight_i * uncertainty_i)`
    over projected characteristic uncertainties.
    """

    def objective(evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
        return float(
            sum(
                w * evaluation.per_characteristic_uncertainty_m[c.name]
                for w, c in zip(weights, scenario.characteristics)
            )
        )

    return objective


def objective_a0_trace_covariance(evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
    """Objective A0: the field's actual criterion -- the trace of the
    full 3D coordinate covariance (a sum of *variances*, not the
    projected, tolerance-direction uncertainties A1 and B use), summed
    over every target point a characteristic in this scenario involves.

    This is genuinely different from A1: A1 already incorporates step 4's
    tolerance-direction projection (a goal-oriented criterion on a linear
    quantity of interest), whereas A0 is blind to tolerance direction
    entirely -- the literal trace-of-covariance criterion Schmitt et al.
    (2016) use, and the quantity Cai (2013) and Wang, Forbes & Maropoulos
    (2014)'s summed-uncertainty criteria approximate. Including A0
    alongside A1 in the ablation makes explicit that "Plan A" was never
    literally the field's own criterion (see
    docs/step6_headline_experiment.md's ablation section).
    """
    target_indices = sorted({index for c in scenario.characteristics for index in c.target_indices})
    return float(
        sum(np.trace(evaluation.network_solve.target_point_covariance(i)) for i in target_indices)
    )


def objective_c_direct_risk(evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
    """Objective C: direct minimisation of total expected decision risk,
    with no linearised weight proxy. Each candidate's score is the sum,
    over every characteristic, of that characteristic's *actual* global
    risk (PFA + PFR) computed from its own achieved uncertainty for this
    exact candidate -- not a fixed-reference-point slope (contrast with
    B, `make_weighted_uncertainty_objective(risk_derived_weights(...))`).

    Cost-symmetric (implicitly K_FA = K_FR = 1, i.e. this minimises total
    error *probability*, not total error *cost*): asymmetric consequence
    costs (aerospace's K_FA >> K_FR, say) would multiply each term before
    summing, but eliciting real cost figures is a data-gathering exercise
    (CLAUDE.md never invents a reference number it doesn't have) rather
    than a modelling one, so it is not built into this function --
    a caller with real K_FA/K_FR figures can weight the two probabilities
    before combining, in a variant of this function or a per-cluster
    extension of `risk.jcgm106.RiskResult`.
    """
    total = 0.0
    for characteristic in scenario.characteristics:
        assignment = scenario.assignments[characteristic.name]
        decision_rule = scenario.decision_rules[characteristic.name]
        uncertainty_m = evaluation.per_characteristic_uncertainty_m[characteristic.name]
        risk = evaluate_conformity_risk(assignment, uncertainty_m, decision_rule)
        total += risk.probability_false_acceptance + risk.probability_false_rejection
    return total


def constrained_objective(
    objective_fn: ObjectiveFn, max_uncertainty_m: Optional[float]
) -> ObjectiveFn:
    """Wrap any `ObjectiveFn` with a hard cap: a candidate where *any*
    characteristic's projected uncertainty exceeds `max_uncertainty_m` is
    disqualified (scored `+inf`) regardless of how well it does on the
    objective itself.

    Exists because a linearised risk weight (objective B) has no natural
    floor: once a characteristic's weight is driven low enough (a
    high-capability process on the flat part of its risk curve -- step
    5), nothing in the objective stops that characteristic's own
    uncertainty from growing without bound in exchange for an
    arbitrarily small gain elsewhere, right up to where the small-
    perturbation linearisation the weight was computed under stops being
    a good approximation at all. A cap -- e.g. a minimum test uncertainty
    ratio (TUR) or an absolute uncertainty ceiling a quality system
    requires regardless of computed risk -- keeps every candidate the
    search can choose within a regime a metrologist would actually
    accept. `max_uncertainty_m=None` (the default via `grid_search_station_placement`)
    applies no cap, reproducing the unconstrained objective exactly.
    """
    if max_uncertainty_m is None:
        return objective_fn

    def wrapped(evaluation: PlanEvaluation, scenario: PlanningScenario) -> float:
        if any(u > max_uncertainty_m for u in evaluation.per_characteristic_uncertainty_m.values()):
            return float("inf")
        return objective_fn(evaluation, scenario)

    return wrapped
