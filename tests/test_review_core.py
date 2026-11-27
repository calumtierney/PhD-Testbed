"""The plan review (review.core): gates 2-6 of the plan-review brief, and what each number means.

Gate 2  round trip        a plan from `plan_qif_cell`, written as a plan file and reviewed, reproduces the planner's own numbers
Gate 3  frame equivalence the same plan in `file` and `cell` coordinates gives an identical review
Gate 4  same scorer       every review in a comparison uses one scorer and one risk-model instance
Gate 5  infeasibility     reported with the right reason and never repaired
Gate 6  unmeasured        priced exactly as `risk.decision.unmeasured_risk` / `planning.process_evidence` price it
"""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from instruments.laser_tracker import LaserTracker
from interchange.plan_file import PlanStation, plan_file_from_stations
from interchange.qif import read_qif
from planning.cell import CellRiskModel, PartFrameScorer, layout_poses
from planning.inspection import _escape_probability
from planning.process_evidence import ProcessEvidenceEvaluator
from planning.qif_cell import build_qif_cell, plan_qif_cell
from review.core import (
    build_context, compare_plans, format_comparison, review_plan, review_plans, review_with_context,
)
from review.counterparts import algorithm_counterparts
from review.demo import demo_plan
from review.scene import ReviewScene, SceneError
from review.settings import ReviewSettings
from risk.decision import best_rule_risk, unmeasured_risk

SAMPLE = Path(__file__).resolve().parent.parent / "scenes" / "qif" / "SheetMetal_QIF_Results_6_samples.QIF"
CREATED = "2025-11-27T00:00:00Z"
NOISE = ReviewSettings("noise_only")


def make_plan(name, positions, frame="cell", role="synthetic", **kwargs):
    return plan_file_from_stations(name, role, "test", CREATED, frame, positions, **kwargs)


@pytest.fixture(scope="module")
def factory():
    return ReviewScene.from_factory_cell()


@pytest.fixture(scope="module")
def qif():
    cell = build_qif_cell(read_qif(SAMPLE), tracker=NOISE.build_tracker())
    plan = plan_qif_cell(cell, station_counts=(2,), objectives=("A1", "risk"), rule="simple", seed=0,
                         de=dict(maxiter=6, popsize=6, tol=1e-8, workers=1))
    return cell, ReviewScene.from_qif_cell(cell, "sheet-metal sample"), plan


def assert_same(a, b, path="", skip=()):
    """Recursive equality; floats to a relative 1e-9."""
    if path in skip:
        return
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}", skip)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]", skip)
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


# -- gate 2: the round trip -----------------------------------------------------------------
def test_gate2_a_planners_plan_reviewed_reproduces_the_planners_own_numbers(qif):
    cell, scene, planned = qif
    context = build_context(scene, NOISE)
    checked = 0
    for key, entry in planned["plans"].items():
        if not entry["feasible"]:
            continue
        plan = make_plan(key, [s["position_cell_m"] for s in entry["stations"]],
                         track_coordinates=[(s["s_m"], s["height_m"]) for s in entry["stations"]], role="algorithm")
        review = review_with_context(context, plan)
        assert review.feasible and review.findings == []
        assert review.n_shots == entry["n_shots"] and review.n_readings == entry["n_shots"]  # one face: readings = shots
        by_name = {c["name"]: c for c in review.characteristics}
        for name, u_um in entry["u_um"].items():
            assert by_name[name]["u_um"] == pytest.approx(u_um, rel=1e-12)
            assert by_name[name]["tur"] == pytest.approx(entry["tur"][name], rel=1e-12)
            assert by_name[name]["risk"] == pytest.approx(entry["risk"][name], rel=1e-12)
        assert review.totals["expected_wrong_decisions"] == pytest.approx(entry["total_risk"], rel=1e-12)
        assert review.totals["worst_tur"] == pytest.approx(entry["worst_tur"], rel=1e-12)
        assert review.time_min == pytest.approx(entry["time_min"], rel=1e-12)
        checked += 1
    assert checked >= 2


# -- gate 3: frame equivalence ----------------------------------------------------------------
def test_gate3_the_same_plan_in_file_and_cell_coordinates_gives_an_identical_review(qif):
    cell, scene, planned = qif
    entry = planned["plans"]["risk|2"]
    in_cell = make_plan("p", [s["position_cell_m"] for s in entry["stations"]], frame="cell")
    in_file = make_plan("p", [s["position_file_m"] for s in entry["stations"]], frame="file")
    assert in_cell.content_hash() != in_file.content_hash()  # different files...
    context = build_context(scene, NOISE)
    a, b = review_with_context(context, in_cell).to_dict(), review_with_context(context, in_file).to_dict()
    assert a["plan"]["plan_hash"] == b["plan"]["plan_hash"]  # ...the same physical plan
    assert a["plan"]["frame"] == "cell" and b["plan"]["frame"] == "file"
    assert_same(a, b, skip={".plan.frame", ".plan.file_hash"})


def test_a_file_frame_plan_cannot_be_reviewed_against_a_scene_with_no_file_frame(factory):
    plan = make_plan("p", [[0.0, 0.0, 1.0]], frame="file")
    with pytest.raises(SceneError, match="no file frame"):
        review_plan(factory, plan, NOISE)


# -- gate 4: one scorer, one risk model ----------------------------------------------------------
def test_gate4_every_review_in_a_comparison_uses_one_scorer_and_one_risk_model(factory, monkeypatch):
    scorer_ids, model_ids = set(), set()
    real_score, real_curve = PartFrameScorer.score_visibility, CellRiskModel.per_characteristic

    def spy_score(self, *args, **kwargs):
        scorer_ids.add(id(self))
        return real_score(self, *args, **kwargs)

    def spy_curve(self, *args, **kwargs):
        model_ids.add(id(self))
        return real_curve(self, *args, **kwargs)

    monkeypatch.setattr(PartFrameScorer, "score_visibility", spy_score)
    monkeypatch.setattr(CellRiskModel, "per_characteristic", spy_curve)
    plans = [demo_plan(factory, n) for n in (3, 4, 5)]
    context, reviews = review_plans(factory, plans, NOISE)
    assert scorer_ids == {id(context.scorer)} and model_ids == {id(context.risk_model)}
    assert len({r.context_id for r in reviews}) == 1
    assert compare_plans(reviews)["context_id"] == context.context_id


def test_reviews_scored_with_different_settings_cannot_be_compared(factory):
    plan = demo_plan(factory, 4)
    a = review_plan(factory, plan, NOISE)
    b = review_plan(factory, plan, ReviewSettings("npl_residual"))
    with pytest.raises(ValueError, match="different context_id"):
        compare_plans([a, b])


def test_the_tracker_model_is_a_required_setting_with_no_default():
    with pytest.raises(TypeError):
        ReviewSettings()
    with pytest.raises(ValueError, match="tracker_model"):
        ReviewSettings("whatever")
    with pytest.raises(ValueError, match="no default"):
        ReviewSettings.from_dict({"decision_rule": "simple"})
    assert ReviewSettings.from_dict(NOISE.to_dict()) == NOISE


def test_the_instrument_model_is_printed_and_recorded(factory):
    plans = [demo_plan(factory, 4)]
    for settings, words in ((NOISE, "sensor noise only"), (ReviewSettings("npl_residual"), "NPL geometric parameters")):
        context, reviews = review_plans(factory, plans, settings)
        text = format_comparison(compare_plans(reviews))
        assert words in text and words in reviews[0].provenance["tracker_label"]
        assert reviews[0].provenance["settings"]["tracker_model"] == settings.tracker_model
    npl = review_plan(factory, plans[0], ReviewSettings("npl_residual")).provenance["tracker"]
    assert npl["parameter_set_hash"] is not None and "consider mode" in npl["geometric_parameters"]
    assert review_plan(factory, plans[0], NOISE).provenance["tracker"]["parameter_set_hash"] is None


def test_the_npl_model_changes_the_numbers_and_the_review_says_which_one_it_used(factory):
    plan = demo_plan(factory, 4)
    noise = review_plan(factory, plan, NOISE)
    npl = review_plan(factory, plan, ReviewSettings("npl_residual"))
    assert noise.context_id != npl.context_id
    # parameter errors only add uncertainty
    for a, b in zip(noise.characteristics, npl.characteristics):
        assert b["u_um"] > a["u_um"]


# -- gate 5: infeasibility is reported, never repaired ----------------------------------------------
def loops4(scene):
    return [p.position_m for p in layout_poses([(15.9, 1.28), (27.5, 1.45), (45.0, 1.17), (54.4, 1.18)], scene.scenario, scene.track)]


def test_gate5_a_station_that_sees_too_few_targets_is_named_and_the_plan_is_not_repaired(factory):
    positions = [list(p) for p in loops4(factory)]
    positions.append([400.0, 400.0, 1.0])  # far outside the 30 m range: sees nothing
    plan = make_plan("with a lost station", positions)
    review = review_plan(factory, plan, NOISE)
    assert not review.feasible and review.characteristics is None and review.totals is None
    thin = [f for f in review.findings if f["kind"] == "station_thin"]
    assert [f["station"] for f in thin] == [5] and "station 5" in thin[0]["detail"] and thin[0]["severity"] == "blocking"
    assert review.n_stations == 5 and [s["index"] for s in review.stations] == [1, 2, 3, 4, 5]  # no station dropped
    assert review.stations[4]["position_cell_m"] == [400.0, 400.0, 1.0]  # nor moved
    comparison = compare_plans([review])
    assert comparison["rows"][0]["feasible"] is False and comparison["rows"][0]["expected_wrong_decisions"] is None
    assert "INFEASIBLE" in format_comparison(comparison) and "station 5" in format_comparison(comparison)


def test_gate5_a_station_sharing_fewer_than_three_targets_with_the_network_is_untied(factory):
    context = build_context(factory, NOISE)
    positions = loops4(factory)[:2]
    poses = [dataclasses.replace(p) for p in layout_poses([(15.9, 1.28), (27.5, 1.45)], factory.scenario, factory.track)]
    visible = context.scorer.visible(poses)
    first = [int(i) for i in np.nonzero(visible[0])[0]][:8]
    second = [int(i) for i in np.nonzero(visible[1] & ~visible[0])[0]][:4]
    assert len(second) >= 3, "the scene should give station 2 targets that station 1 cannot see"
    names = factory.target_names
    plan = make_plan("two islands", positions, targets=[[names[i] for i in first], [names[i] for i in second]])
    review = review_plan(factory, plan, NOISE)
    assert not review.feasible
    assert [f["station"] for f in review.findings if f["kind"] == "station_untied"] == [2]
    assert review.stations[1]["n_targets_measured"] == len(second)  # the listed shots were kept as listed


def test_gate5_a_characteristic_no_station_sees_is_a_blocking_finding_only_if_required(factory):
    one_station = loops4(factory)[:1]  # the first loop station faces one side of the part only
    plan = make_plan("one station", [one_station[0].tolist()])
    relaxed = review_plan(factory, plan, NOISE)
    strict = review_plan(factory, plan, ReviewSettings("noise_only", require_all_characteristics=True))
    missing = [f for f in relaxed.findings if f["kind"] == "characteristic_not_seen"]
    assert missing, "this station should not see every characteristic"
    assert all(f["severity"] == "warning" and f["characteristic"] for f in missing)
    assert strict.feasible is False and strict.characteristics is None
    assert {f["characteristic"] for f in strict.findings if f["kind"] == "characteristic_not_seen" and f["severity"] == "blocking"} == \
        {f["characteristic"] for f in missing}
    assert relaxed.feasible  # priced, not rejected (gate 6 checks the price)


def test_gate5_unknown_target_names_and_unseeable_listed_targets_are_reported(factory):
    context = build_context(factory, NOISE)
    positions = [p.tolist() for p in loops4(factory)]
    unknown = make_plan("typo", positions, targets=[["no_such_target"], None, None, None])
    review = review_with_context(context, unknown)
    assert not review.feasible and review.findings[0]["kind"] == "unknown_target" and review.findings[0]["target"] == "no_such_target"
    # listing a target the station cannot see: reported with its reason, shot not counted
    poses = layout_poses([(15.9, 1.28)], factory.scenario, factory.track)
    hidden = int(np.nonzero(~context.scorer.visible(poses)[0])[0][0])
    plan = make_plan("lists a hidden target", [positions[0]], targets=[[factory.target_names[hidden]]])
    review = review_with_context(context, plan)
    finding = next(f for f in review.findings if f["kind"] == "listed_target_not_visible")
    assert finding["station"] == 1 and finding["target"] == factory.target_names[hidden] and "cannot see" in finding["detail"]
    assert review.stations[0]["n_targets_measured"] == 0


def test_a_station_that_lists_its_targets_measures_exactly_those_it_can_see(factory):
    context = build_context(factory, NOISE)
    positions = [p.tolist() for p in loops4(factory)]
    everything = review_with_context(context, make_plan("all visible", positions))
    poses = layout_poses([(15.9, 1.28), (27.5, 1.45), (45.0, 1.17), (54.4, 1.18)], factory.scenario, factory.track)
    visible = context.scorer.visible(poses)
    keep = [int(i) for i in np.nonzero(visible[1])[0]][:10]
    subset = review_with_context(context, make_plan("lists some", positions, targets=[None, [factory.target_names[i] for i in keep], None, None]))
    assert subset.stations[1]["n_targets_measured"] == 10 and subset.stations[1]["n_targets_visible"] == int(visible[1].sum())
    assert everything.stations[1]["n_targets_measured"] == int(visible[1].sum())
    assert subset.n_shots == everything.n_shots - (int(visible[1].sum()) - 10)
    assert not any(f["kind"] == "listed_target_not_visible" for f in subset.findings)


# -- gate 6: unmeasured characteristics ------------------------------------------------------------------
def test_gate6_an_unmeasured_characteristic_is_priced_as_process_evidence_and_unmeasured_risk_price_it(factory):
    context = build_context(factory, NOISE)
    plan = make_plan("one station", [loops4(factory)[0].tolist()])
    review = review_with_context(context, plan)
    unmeasured = [c for c in review.characteristics if not c["measured"]]
    assert unmeasured and review.totals["n_unmeasured"] == len(unmeasured)
    evaluator = ProcessEvidenceEvaluator(factory.scenario)
    limits = factory.tolerance_limits_m()
    for c in unmeasured:
        k = factory.characteristic_names.index(c["name"])
        price = unmeasured_risk(context.knowledge[k], limits[k, 0], limits[k, 1])
        assert c["risk"] == pytest.approx(price, rel=1e-12) and c["risk_exact"] == pytest.approx(price, rel=1e-12)
        assert c["u_um"] is None and c["tur"] is None and c["acceptance_um"] is None
        nonconformance = context.knowledge[k].nonconformance(limits[k, 0], limits[k, 1])
        if nonconformance <= 0.5:  # accepting unseen is the better decision: process_evidence's price is the same number
            assert c["decision"] == "accepted unseen" and c["pfr"] == 0.0
            assert c["risk"] == pytest.approx(evaluator.escape_price(c["name"]), rel=1e-9)
        else:
            assert c["decision"] == "rejected unseen" and c["pfa"] == 0.0
    assert review.totals["risk_from_unmeasured"] == pytest.approx(sum(c["risk"] for c in unmeasured), rel=1e-12)


# -- what the numbers are --------------------------------------------------------------------------------
def test_totals_are_the_sum_of_the_characteristics_and_escape_follows_planning_inspection(factory):
    review = review_plan(factory, demo_plan(factory, 4), NOISE)
    rows, totals = review.characteristics, review.totals
    assert totals["expected_wrong_decisions"] == pytest.approx(sum(c["risk"] for c in rows), rel=1e-12)
    assert totals["expected_wrong_decisions_exact"] == pytest.approx(sum(c["pfa"] + c["pfr"] for c in rows), rel=1e-12)
    assert totals["expected_false_accepts"] == pytest.approx(sum(c["pfa"] for c in rows), rel=1e-12)
    assert totals["escape_probability"] == pytest.approx(_escape_probability(np.array([c["pfa"] for c in rows])), rel=1e-12)
    assert totals["risk_model_vs_exact_relative_gap"] < 0.05  # the tabulated curve against the direct integral
    for c in rows:
        assert c["acceptance_um"] == pytest.approx([-c["tolerance_half_um"], c["tolerance_half_um"]])  # simple acceptance
        assert c["tur"] == pytest.approx(c["tolerance_half_um"] / (2.0 * c["u_um"]))


def test_the_conformance_probability_rule_reports_its_acceptance_limits(factory):
    """Exercises the 'best' rule's PFA/PFR/acceptance path without tabulating its risk curves (about 70 s): the
    context borrows the simple rule's risk model, so only the exact split and the limits are asserted."""
    simple = build_context(factory, NOISE)
    best = dataclasses.replace(simple, settings=ReviewSettings("noise_only", decision_rule="best"))
    review = review_with_context(best, demo_plan(factory, 4))
    limits = factory.tolerance_limits_m()
    for k, c in enumerate(review.characteristics):
        pfa, pfr, acceptance = best_rule_risk(simple.knowledge[k], limits[k, 0], limits[k, 1], c["u_um"] * 1e-6, 0.5)
        assert (c["pfa"], c["pfr"]) == pytest.approx((pfa, pfr), rel=1e-9)
        assert c["acceptance_um"][0] == (None if not np.isfinite(acceptance[0]) else pytest.approx(acceptance[0] * 1e6, rel=1e-9))


def test_two_faces_double_the_readings_and_shrink_uncertainty_under_noise_alone(factory):
    positions = [p.tolist() for p in loops4(factory)]
    front = review_plan(factory, make_plan("front", positions), NOISE)
    both = review_plan(factory, make_plan("both", positions, faces=[("front", "back")] * 4), NOISE)
    assert both.n_readings == 2 * front.n_readings and both.n_shots == front.n_shots
    assert both.time_min["per_observation_model_min"] == pytest.approx(2 * front.time_min["per_observation_model_min"])
    for a, b in zip(front.characteristics, both.characteristics):
        assert b["u_um"] < a["u_um"]


def test_stations_that_are_different_trackers_do_not_share_parameter_errors(factory):
    positions = [p.tolist() for p in loops4(factory)]
    npl = ReviewSettings("npl_residual")
    one_tracker = review_plan(factory, make_plan("one", positions), npl)
    four_trackers = review_plan(factory, make_plan("four", positions, instrument_ids=[0, 1, 2, 3]), npl)
    assert one_tracker.totals["expected_wrong_decisions"] != pytest.approx(four_trackers.totals["expected_wrong_decisions"], rel=1e-6)
    noise_one = review_plan(factory, make_plan("one", positions), NOISE)
    noise_four = review_plan(factory, make_plan("four", positions, instrument_ids=[0, 1, 2, 3]), NOISE)
    assert noise_one.totals["expected_wrong_decisions"] == pytest.approx(noise_four.totals["expected_wrong_decisions"], rel=1e-9)  # nothing to share


def test_the_elevation_limits_setting_reaches_the_scorer(factory):
    context = build_context(factory, ReviewSettings("noise_only", elevation_limits="api_t3"))
    limits = context.scenario.visibility_limits
    assert np.rad2deg(limits.min_elevation_rad) == pytest.approx(-60.0) and np.rad2deg(limits.max_elevation_rad) == pytest.approx(77.0)
    assert build_context(factory, NOISE).scenario.visibility_limits.min_elevation_rad is None
    assert "elevation range" in context.provenance["elevation_limits"]


def test_the_placement_and_placeholder_warnings_are_printed(qif, factory):
    cell, scene, planned = qif
    entry = planned["plans"]["risk|2"]
    plan = make_plan("p", [s["position_cell_m"] for s in entry["stations"]])
    text = format_comparison(compare_plans([review_plan(scene, plan, NOISE)]))
    assert scene.placement_assumed and "PLACEMENT ASSUMED" in text
    assert "assumed: laid out by rule" in text
    text = format_comparison(compare_plans(review_plans(factory, [demo_plan(factory, 4)], NOISE)[1]))
    assert "PLACEHOLDER process statistics" in text and "synthetic" in text
    assert "station 1, the first listed" in text


# -- the demo and the algorithm counterparts --------------------------------------------------------------------
def test_the_demo_plan_is_labelled_synthetic_everywhere(factory):
    plan = demo_plan(factory, 4)
    assert plan.author_role == "synthetic" and "SYNTHETIC" in plan.source and "Not an engineer's plan" in plan.source
    _, reviews = review_plans(factory, [plan], NOISE)
    text = format_comparison(compare_plans(reviews))
    assert "synthetic" in text and "engineer" not in text.lower()


def test_counterparts_are_searched_with_one_optimiser_reviewed_by_one_context_and_the_given_plan_is_untouched(factory):
    settings = ReviewSettings("noise_only", de_maxiter=8, de_popsize=6)
    plan = demo_plan(factory, 3)
    context = build_context(factory, settings)
    calls = []
    result = algorithm_counterparts(factory, plan, settings, context=context, objectives=("A1", "risk"),
                                    progress=lambda done, total, label: calls.append((done, total, label)))
    assert result.given.to_dict() == review_with_context(context, plan).to_dict()  # reviewed as it is, never re-optimised
    assert calls and calls[-1][0] == calls[-1][1] and [c[0] for c in calls] == sorted(c[0] for c in calls)
    kinds = {c.kind for c in result.counterparts}
    assert "same_count" in kinds and all(c.plan.author_role == "algorithm" for c in result.counterparts)
    assert all(r.context_id == context.context_id for r in result.reviews())
    compare_plans(result.reviews())  # comparable by construction
    for c in result.counterparts:
        again = review_with_context(context, c.plan)
        assert_same(again.to_dict(), c.review.to_dict(), skip={".plan.created"})
        if c.kind == "same_count":
            assert c.n_stations == plan.n_stations
        if c.kind == "one_fewer":
            assert c.n_stations == plan.n_stations - 1
        if c.kind == "iso_risk":
            assert c.review.totals["expected_wrong_decisions"] <= result.given.totals["expected_wrong_decisions"] * (1 + 1e-9)
            assert c.n_stations <= plan.n_stations
    with pytest.raises(ValueError, match="unknown objectives"):
        algorithm_counterparts(factory, plan, settings, objectives=("nonsense",))


def test_an_infeasible_given_plan_has_no_iso_risk_target_and_is_not_repaired(factory):
    settings = ReviewSettings("noise_only", de_maxiter=4, de_popsize=5)
    plan = make_plan("lost", [loops4(factory)[0].tolist(), [400.0, 400.0, 1.0]])
    result = algorithm_counterparts(factory, plan, settings, objectives=("risk",))
    assert not result.given.feasible and result.given.n_stations == 2
    assert any("iso-risk: the given plan is infeasible" in m for m in result.not_found)
    assert not any(c.kind == "iso_risk" for c in result.counterparts)


def test_there_is_no_nothing_measured_reference_line(factory):
    """Under placeholder priors a 'risk if nothing is measured' line beats every plan (measurement creates false rejections faster than it
    prevents false accepts; planning.process_evidence), so it misleads more than it informs and the review does not offer it."""
    _, reviews = review_plans(factory, [demo_plan(factory, 4)], NOISE)
    comparison = compare_plans(reviews)
    assert "risk_if_nothing_is_measured" not in comparison and "statements" not in comparison
    assert "risk_if_nothing_is_measured" not in reviews[0].totals
    text = format_comparison(comparison).lower()
    assert "nothing is measured" not in text and "measuring nothing" not in text and "note:" not in text
    assert "risk_from_unmeasured" in reviews[0].totals  # an unmeasured characteristic's own price is still reported
