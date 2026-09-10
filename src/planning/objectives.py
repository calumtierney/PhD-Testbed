"""The two objectives compared in step 6, the headline experiment
(CLAUDE.md §5, step 6):

  A. **Uniform weights** -- minimise the plain sum of characteristic
     uncertainties. Applied at the characteristic level (post step 4's
     projection) rather than the raw coordinate level, this is the
     A-optimal criterion (CLAUDE.md §1: "minimising the trace of the
     coordinate variance matrix") the field currently uses.
  B. **Risk-derived weights** -- minimise a WEIGHTED sum of the same
     uncertainties, where each characteristic's weight is the local
     sensitivity of its own expected decision risk (global PFA + PFR,
     step 5) to its own uncertainty.

Both objectives are evaluated over *exactly* the same pipeline --
`evaluate_plan` below runs the same network solve, the same per-
characteristic projection, the same risk model -- for either. Only the
weight vector multiplying the per-characteristic uncertainties before
summing differs (CLAUDE.md §1: "replacing the objective function, not the
optimiser"). See docs/step6_headline_experiment.md for the full picture.

**Why a sensitivity weight, not the risk itself.** The natural first idea
-- weight each characteristic by its own risk -- doesn't quite make
sense as an *optimisation* weight: risk is what the search is trying to
reduce, and a station arrangement changes *uncertainty*, not risk
directly (risk is a function of uncertainty, via the step 5 integral).
The quantity that actually says "how much would reducing this
characteristic's uncertainty by a little help" is the local slope,
`d(risk)/d(uncertainty)`, evaluated at a realistic operating uncertainty
-- exactly what CLAUDE.md §5 step 6 asks for: weight each characteristic
"by the sensitivity of expected decision cost to its uncertainty". A
characteristic on the flat part of its risk curve (a high-capability
process -- step 5's gate) has a near-zero slope: driving its uncertainty
down barely moves its risk, so it should barely influence where stations
go. A characteristic on the steep part (a marginal process) has a large
slope: small uncertainty improvements there move risk substantially, so
it should pull disproportionately on the plan.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from characteristics.characteristic import Characteristic, evaluate_characteristic
from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from network.solve import NetworkSolveResult, exact_observations, solve_network
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
    """

    target_points_m: np.ndarray
    tracker: LaserTracker
    characteristics: List[Characteristic]
    assignments: Dict[str, ClusterAssignment]
    decision_rules: Dict[str, DecisionRule]
    anchor_pose: InstrumentPose


def uniform_weights(scenario: PlanningScenario) -> np.ndarray:
    """Objective A's weights: 1 for every characteristic."""
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
    would depend on that candidate's own result), which would make A and
    B no longer comparable as "the same optimiser, different objective"
    (CLAUDE.md §3) -- the weights have to be fixed before the search
    starts, exactly as CLAUDE.md §5 step 6 describes them ("compute those
    weights from the risk layer", as a preprocessing step, not inside the
    search loop).

    Normalising to mean 1 puts B on the same total scale as A (which
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
    """The outcome of scoring one candidate set of station poses.

    Attributes
    ----------
    station_poses : the additional (non-anchor) stations that were scored.
    weighted_objective : sum(weight_i * uncertainty_m_i) -- what the
        search minimises.
    per_characteristic_uncertainty_m : characteristic name -> uncertainty,
        unweighted -- for reporting (CLAUDE.md §5 step 6 asks for this
        explicitly), independent of which objective produced this plan.
    network_solve : the full NetworkSolveResult this was computed from
        (design covariance -- see `evaluate_plan`'s docstring).
    """

    station_poses: List[InstrumentPose]
    weighted_objective: float
    per_characteristic_uncertainty_m: Dict[str, float]
    network_solve: NetworkSolveResult


def evaluate_plan(
    station_poses: List[InstrumentPose], scenario: PlanningScenario, weights: np.ndarray
) -> PlanEvaluation:
    """Score one candidate set of (non-anchor) station poses under `weights`.

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
    """
    all_poses = [scenario.anchor_pose] + list(station_poses)
    observations = exact_observations(scenario.target_points_m, all_poses)
    network_solve = solve_network(
        observations, scenario.target_points_m.copy(), all_poses, scenario.tracker, anchor_index=0
    )

    per_characteristic_uncertainty_m = {}
    weighted_objective = 0.0
    for characteristic, weight in zip(scenario.characteristics, weights):
        uncertainty_m = evaluate_characteristic(characteristic, network_solve)[0].uncertainty_m
        per_characteristic_uncertainty_m[characteristic.name] = uncertainty_m
        weighted_objective += weight * uncertainty_m

    return PlanEvaluation(
        station_poses=list(station_poses),
        weighted_objective=weighted_objective,
        per_characteristic_uncertainty_m=per_characteristic_uncertainty_m,
        network_solve=network_solve,
    )
