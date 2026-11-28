"""review.explain: the chain behind every figure, pinned to the review it explains."""
import json

import numpy as np
import pytest

from instruments.laser_tracker import LaserTracker
from interchange.plan_file import plan_file_from_stations
from review.core import build_context, measurement_pattern, review_with_context
from review.demo import demo_plan
from review.explain import explain_plan
from review.scene import ReviewScene
from review.settings import ReviewSettings


@pytest.fixture(scope="module")
def factory():
    return ReviewScene.from_factory_cell()


@pytest.fixture(scope="module", params=["noise_only", "npl_residual"])
def explained(request, factory):
    context = build_context(factory, ReviewSettings(request.param))
    plan = demo_plan(factory, 4)
    review = review_with_context(context, plan)
    return context, plan, review, explain_plan(context, plan, review, n_parts=100_000)


def test_the_explanations_uncertainty_is_the_reviews_for_every_characteristic(explained):
    context, plan, review, e = explained
    assert e.feasible and len(e.chain) == 6
    for chain, row in zip(e.chain, review.characteristics):
        assert chain["name"] == row["name"]
        assert chain["part_frame"]["reconciles_with_review"]
        assert chain["part_frame"]["u_um"] == pytest.approx(row["u_um"], rel=1e-9)
        d = chain["decision"]
        assert (d["pfa"], d["pfr"], d["risk"], d["tur"]) == (row["pfa"], row["pfr"], row["risk"], row["tur"])


def test_the_visibility_link_counts_what_the_review_measured(explained):
    context, plan, review, e = explained
    pattern = measurement_pattern(context, plan)
    for chain in e.chain:
        t = chain["target_index"]
        measuring = [s for s in chain["visibility"]["stations"] if s["measured"]]
        assert len(measuring) == chain["visibility"]["n_measuring"] == int(pattern.visible[:, t].sum()) == len(chain["one_shot"])
        for s in chain["visibility"]["stations"]:
            assert (s["reason"] == "measured") == s["measured"]
    focus = next(c for c in e.chain if c["name"] == "cluster2_profile_0")  # the demo plan's stations 2-4 face its surface edge-on or from behind
    assert [s["reason"] for s in focus["visibility"]["stations"]] == ["measured", "incidence_too_steep", "incidence_too_steep", "incidence_too_steep"]
    assert [s["incidence_deg"] > 80 for s in focus["visibility"]["stations"][1:]] == [True, True, True]


def test_one_reading_decomposes_into_range_and_the_two_angles():
    """A reading's error ellipsoid has semi-axes sigma_d along the beam, range x cos(elevation) x sigma_az and range x sigma_el across it
    (the numbers link 2 of the explanation prints)."""
    tracker = LaserTracker()
    point = np.array([3.0, 2.0, 1.0])
    d, elevation = np.linalg.norm(point), np.arcsin(point[2] / np.linalg.norm(point))
    expected = sorted([tracker.sigma_d_m, d * np.cos(elevation) * tracker.sigma_theta_rad, d * tracker.sigma_phi_rad])
    np.testing.assert_allclose(np.sqrt(np.linalg.eigvalsh(tracker.covariance_local(point))), expected, rtol=1e-9)


def test_a_target_only_the_anchor_measures_has_its_one_reading_as_its_anchor_frame_error(factory):
    context = build_context(factory, ReviewSettings("noise_only"))
    e = explain_plan(context, demo_plan(factory, 4), n_parts=20_000)
    for chain in e.chain:
        if chain["visibility"]["n_measuring"] == 1 and chain["one_shot"][0]["station"] == 1:
            assert chain["network"]["anchor_frame_sigma_along_tolerance_um"] == pytest.approx(chain["one_shot"][0]["along_tolerance_um"], rel=1e-6)
    text = e.to_text("cluster2_profile_0")
    assert "Only station 1, the anchor, measures this target" in text


def test_the_parameter_share_is_zero_for_sensor_noise_and_positive_for_the_npl_model(explained):
    context, plan, review, e = explained
    for chain in e.chain:
        share = chain["part_frame"]["geometric_parameter_variance_share"]
        if context.settings.tracker_model == "noise_only":
            assert share == pytest.approx(0.0, abs=1e-9)
        else:
            assert 0.01 < share < 0.6
        assert chain["part_frame"]["sensor_noise_variance_share"] + share == pytest.approx(1.0)


def test_the_what_drives_tables_reproduce_the_risk_and_show_the_offset_matters(explained):
    context, plan, review, e = explained
    for chain in e.chain:
        d = e.drivers[chain["name"]]
        row = next(r for r in review.characteristics if r["name"] == chain["name"])
        assert d["available"]
        assert d["risk_now"] == pytest.approx(row["risk_exact"], rel=1e-9)
        assert next(r for r in d["vs_u"] if r["u_factor"] == 1.0)["risk"] == pytest.approx(row["risk_exact"], rel=1e-9)
        assert d["vs_offset"][-1]["risk"] == pytest.approx(row["risk_exact"], rel=1e-9) and d["vs_offset"][0]["mean_um"] == 0.0
        risks = [r["risk"] for r in d["vs_u"]]
        assert risks == sorted(risks)  # a worse measurement never helps
    marginal = e.drivers["cluster2_profile_0"]
    assert marginal["vs_offset"][0]["risk"] < 1e-4 * marginal["risk_now"]           # centred, the same measurement is harmless
    assert marginal["same_cpk_centred"]["ratio_actual_to_centred"] > 20              # the same Cpk, with the margin in spread, is far safer
    assert "same_cpk_centred" not in e.drivers["cluster1_profile_0"]                 # already centred: nothing to compare


def test_outputs_roll_up_to_the_totals_and_shares_sum_to_one(explained):
    context, plan, review, e = explained
    rows = e.outputs["rows"]
    assert sum(r["risk"] for r in rows) == pytest.approx(review.totals["expected_wrong_decisions"], rel=1e-12)
    assert sum(r["share_of_total"] for r in rows) == pytest.approx(1.0)
    assert e.focus == max(rows, key=lambda r: r["risk"])["name"] == "cluster2_profile_2"  # the default focus: where the risk is


def test_the_independent_checks_run_and_pass(explained):
    context, plan, review, e = explained
    assert len(e.checks) == 24 and all(c["passed"] for c in e.checks)


def test_the_text_names_the_chain_the_placeholders_and_the_checks(explained):
    context, plan, review, e = explained
    text = e.to_text("cluster2_profile_0")
    for section in ("WHAT THE HEADLINE FIGURE IS", "1. INPUTS", "2. THE CHAIN", "3. ONE CHARACTERISTIC IN FULL", "4. OUTPUTS", "5. WHAT DRIVES",
                    "6. INDEPENDENT CHECKS", "Link 1 - visibility", "Link 5 - the decision"):
        assert section in text, section
    assert "<-- PLACEHOLDER" in text and "SUM over the characteristics" in text and "24 of 24 checks agree" in text
    assert "[matches the review]" in text and "DOES NOT match" not in text
    assert "the offset process is" in text
    assert json.loads(json.dumps(e.to_dict(), default=float))["plan"]["name"] == plan.name


def test_an_infeasible_plan_is_explained_by_its_reasons_and_an_unknown_characteristic_is_refused(factory):
    context = build_context(factory, ReviewSettings("noise_only"))
    lost = plan_file_from_stations("lost", "engineer", "t", "2025-11-27T00:00:00Z", "cell", [[0.0, -4.0, 1.5], [900.0, 900.0, 1.0]])
    e = explain_plan(context, lost)
    assert not e.feasible and e.chain == [] and e.checks == []
    assert "INFEASIBLE" in e.to_text() and "station 2" in e.to_text()
    with pytest.raises(ValueError, match="not a measured characteristic"):
        explain_plan(context, demo_plan(factory, 4), characteristic="nope", n_parts=1000)


def test_an_unmeasured_characteristic_is_listed_in_the_outputs_as_priced_not_chained(factory):
    context = build_context(factory, ReviewSettings("noise_only"))
    one = plan_file_from_stations("one", "engineer", "t", "2025-11-27T00:00:00Z", "cell", [list(demo_plan(factory, 4).stations[0].position_m)])
    e = explain_plan(context, one, n_parts=20_000)
    unmeasured = [r for r in e.outputs["rows"] if not r["measured"]]
    assert unmeasured and len(e.chain) == 6 - len(unmeasured)
    assert "priced at its process prior" in e.to_text()
