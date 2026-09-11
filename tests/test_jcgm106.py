"""Tests for risk.jcgm106.

test_high_capability_process_shows_near_zero_risk_regardless_of_uncertainty
and test_marginal_process_risk_is_strongly_sensitive_to_uncertainty
together are the gate for build step 5 (CLAUDE.md §5): a high-capability
process shows near-zero risk even when uncertainty is moderately large; a
process sitting near a tolerance limit shows risk that is strongly
sensitive to uncertainty.
"""
import numpy as np
import pytest

from characteristics.characteristic import Characteristic, evaluate_characteristic
from characteristics.tolerances import ProfileTolerance
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker, scene_point_covariances
from risk.jcgm106 import (
    ClusterAssignment,
    DecisionRule,
    FeatureCluster,
    ProcessPrior,
    evaluate_characteristic_risk,
    evaluate_conformity_risk,
)

# A ±50 um tolerance band about nominal -- CLAUDE.md §4d's own reference
# scale ("uncertainty at 5 m vs +-50 um tolerance").
_TOLERANCE = DecisionRule.simple_acceptance(-50e-6, 50e-6)

# Uncertainty magnitudes actually produced by this codebase's earlier
# steps (step 1/4's Table 4 numbers run 1.2-10.3 um for a single
# station), extended a little further to show the trend clearly.
_UNCERTAINTIES_M = [1e-6, 2e-6, 5e-6, 8e-6, 11e-6, 14e-6]


def test_process_prior_rejects_nonpositive_std():
    with pytest.raises(ValueError):
        ProcessPrior(mean_m=0.0, std_m=0.0)


def test_cluster_assignment_rejects_confidence_outside_unit_interval():
    cluster = FeatureCluster("A", ProcessPrior(0.0, 5e-6))
    with pytest.raises(ValueError):
        ClusterAssignment("c1", cluster, confidence=1.5)


def test_cpk_matches_hand_calculation():
    prior = ProcessPrior(mean_m=40e-6, std_m=6e-6)
    cpk = prior.cpk(tolerance_lower_m=-50e-6, tolerance_upper_m=50e-6)
    # min((50-40)/(3*6), (40-(-50))/(3*6)) = min(10/18, 90/18)
    assert cpk == pytest.approx(10.0 / 18.0)


def test_decision_rule_defaults_to_simple_acceptance():
    rule = DecisionRule(tolerance_lower_m=-50e-6, tolerance_upper_m=50e-6)
    assert rule.acceptance_lower_m == -50e-6
    assert rule.acceptance_upper_m == 50e-6


def test_decision_rule_from_zone_width():
    rule = DecisionRule.from_zone_width(1e-4)
    assert rule.tolerance_lower_m == -5e-5
    assert rule.tolerance_upper_m == 5e-5


def test_decision_rule_rejects_inverted_limits():
    with pytest.raises(ValueError):
        DecisionRule(tolerance_lower_m=50e-6, tolerance_upper_m=-50e-6)


def test_high_capability_process_shows_near_zero_risk_regardless_of_uncertainty():
    """Process centred on nominal, std 5 um against a +-50 um tolerance:
    Cpk ~ 3.33, comfortably 'high-capability'. Across the whole realistic
    uncertainty range, total risk should stay tiny."""
    cluster = FeatureCluster("high-capability cluster", ProcessPrior(mean_m=0.0, std_m=5e-6))
    assignment = ClusterAssignment("hole A profile", cluster, confidence=0.9)

    for uncertainty_m in _UNCERTAINTIES_M:
        result = evaluate_conformity_risk(assignment, uncertainty_m, _TOLERANCE)
        total_risk = result.probability_false_acceptance + result.probability_false_rejection
        assert total_risk < 1e-3, (
            f"expected near-zero risk at uncertainty={uncertainty_m*1e6:.1f}um, got {total_risk:.2e}"
        )
        assert result.process_cpk > 3.0
        assert result.cluster_assignment_confidence == 0.9


def test_marginal_process_risk_is_strongly_sensitive_to_uncertainty():
    """Process sitting near the upper tolerance limit (mean 40 um against
    a +-50 um tolerance, Cpk ~ 0.56): total risk should grow substantially
    -- not just detectably -- as uncertainty increases across the same
    realistic range."""
    cluster = FeatureCluster("marginal cluster", ProcessPrior(mean_m=40e-6, std_m=6e-6))
    assignment = ClusterAssignment("hole B profile", cluster, confidence=0.7)

    total_risks = []
    for uncertainty_m in _UNCERTAINTIES_M:
        result = evaluate_conformity_risk(assignment, uncertainty_m, _TOLERANCE)
        total_risks.append(result.probability_false_acceptance + result.probability_false_rejection)

    assert result.process_cpk < 1.0
    # Risk should rise monotonically (or very close to it) with uncertainty...
    assert all(b >= a - 1e-9 for a, b in zip(total_risks, total_risks[1:]))
    # ...and by a lot: at least an order of magnitude from smallest to largest uncertainty.
    assert total_risks[-1] > 10 * total_risks[0]
    assert total_risks[-1] > 0.1  # not just "bigger", genuinely substantial


def test_high_capability_vs_marginal_contrast_at_matched_uncertainty():
    """The actual gate, stated as a direct contrast: at the *same*
    uncertainty, the high-capability process's risk is negligible next to
    the marginal process's."""
    high_capability = ClusterAssignment(
        "c1", FeatureCluster("A", ProcessPrior(mean_m=0.0, std_m=5e-6))
    )
    marginal = ClusterAssignment("c2", FeatureCluster("B", ProcessPrior(mean_m=40e-6, std_m=6e-6)))

    for uncertainty_m in _UNCERTAINTIES_M:
        high_cap_result = evaluate_conformity_risk(high_capability, uncertainty_m, _TOLERANCE)
        marginal_result = evaluate_conformity_risk(marginal, uncertainty_m, _TOLERANCE)
        high_cap_total = high_cap_result.probability_false_acceptance + high_cap_result.probability_false_rejection
        marginal_total = marginal_result.probability_false_acceptance + marginal_result.probability_false_rejection
        assert high_cap_total < marginal_total


def test_false_acceptance_dominates_when_process_is_centred_outside_tolerance():
    """Symmetric check on the other failure mode: a process centred
    *outside* tolerance (a systematically bad process) should show
    meaningful false acceptance, not false rejection -- confirms both
    integrals are wired to the right side of the tolerance limits."""
    cluster = FeatureCluster("bad process", ProcessPrior(mean_m=60e-6, std_m=5e-6))
    assignment = ClusterAssignment("c3", cluster)
    result = evaluate_conformity_risk(assignment, uncertainty_m=8e-6, decision_rule=_TOLERANCE)
    assert result.probability_false_acceptance > 0.01
    assert result.probability_false_rejection < result.probability_false_acceptance


def test_zero_uncertainty_gives_zero_risk_for_a_conforming_process():
    """A perfect measurement (uncertainty = 0) of a process safely inside
    tolerance should carry no risk at all -- the true value is what's
    measured, and it's always in tolerance by construction of the prior's
    support being effectively within limits at this Cpk."""
    cluster = FeatureCluster("A", ProcessPrior(mean_m=0.0, std_m=1e-6))  # Cpk ~ 16.7
    assignment = ClusterAssignment("c4", cluster)
    result = evaluate_conformity_risk(assignment, uncertainty_m=0.0, decision_rule=_TOLERANCE)
    assert result.probability_false_acceptance == pytest.approx(0.0, abs=1e-12)
    assert result.probability_false_rejection == pytest.approx(0.0, abs=1e-12)


def test_evaluate_characteristic_risk_chains_onto_step4():
    """The risk layer should chain directly onto a step 4
    CharacteristicPointUncertainty without the caller re-extracting a
    bare uncertainty_m float by hand."""
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=np.array([[2.5, 0.0, 0.0]]), instrument_pose=pose)
    covariances = scene_point_covariances(scene, tracker)

    characteristic = Characteristic(
        name="profile",
        target_indices=[0],
        tolerance=ProfileTolerance(zone_width_m=1e-4, surface_normal=np.array([0.0, 1.0, 0.0])),
    )
    point_uncertainty = evaluate_characteristic(characteristic, covariances)[0]

    cluster = FeatureCluster("A", ProcessPrior(mean_m=0.0, std_m=10e-6))
    assignment = ClusterAssignment(characteristic.name, cluster)
    decision_rule = DecisionRule.from_zone_width(characteristic.tolerance.zone_width_m)

    result = evaluate_characteristic_risk(point_uncertainty, assignment, decision_rule)
    assert result.probability_false_acceptance >= 0.0
    assert result.probability_false_rejection >= 0.0
