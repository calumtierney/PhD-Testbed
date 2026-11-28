"""Gate 8: Streamlit's AppTest runs the Plan review tab on the demo scene without error, and the tab does what it says: no run before the
instrument model and role are chosen, the review is logged, the sign-off is logged."""
import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("LVM_TESTBED_LOG_DIR", str(tmp_path / "logs"))
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception
    return at


def log_records(tmp_path):
    path = tmp_path / "logs" / "review_log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_gate8_the_plan_review_tab_runs_on_the_demo_scene_without_error(app, tmp_path):
    # the other two tabs are untouched and the third is there
    assert [t.label for t in app.tabs] == ["Tracker error ellipsoid (step 1)", "Conformity risk (step 5)", "Plan review"]
    run = app.button(key="pr_run")
    assert run.disabled  # nothing chosen yet: no instrument model, no role, no plan
    assert log_records(tmp_path) == []

    assert app.radio(key="pr_marginal_process").value == "typical"  # the built-in cell's default marginal process: +10 um / 10 um, Cpk 1.33
    app.checkbox(key="pr_add_demo").check().run()
    assert app.button(key="pr_run").disabled  # a plan alone is not enough: the instrument model has no default
    app.selectbox(key="pr_tracker_model").select("noise_only").run()
    assert app.button(key="pr_run").disabled  # and a role is required
    app.selectbox(key="pr_role").select("metrologist").run()
    assert not app.button(key="pr_run").disabled and not app.exception

    app.button(key="pr_run").click().run()
    assert not app.exception, [e.value for e in app.exception]
    text = " ".join(m.value for m in app.markdown)
    assert "Sensor noise only" in text or "sensor noise only" in text  # the instrument model is printed above the result
    assert any("PLACEHOLDER" in w.value for w in app.warning)
    assert any(df.value.shape[0] == 1 for df in app.dataframe)  # the plan table (one demo plan)
    records = log_records(tmp_path)
    assert [r["event"] for r in records] == ["scene_loaded", "plan_reviewed", "comparison"]
    assert all(r["reviewer_role"] == "metrologist" for r in records)

    # the explanation: the chain behind the figures, for the highest-risk characteristic of the one plan
    assert not app.button(key="pr_explain_run").disabled
    app.button(key="pr_explain_run").click().run()
    assert not app.exception, [e.value for e in app.exception]
    explanation = " ".join(c.value for c in app.code)
    for section in ("HOW THE NUMBERS FOR", "1. INPUTS", "3. ONE CHARACTERISTIC IN FULL", "5. WHAT DRIVES", "24 of 24 checks agree"):
        assert section in explanation, section
    assert "<-- PLACEHOLDER" in explanation

    # the sign-off writes to the log. Form widgets commit together with the submit button, so set them, then submit once; option 0 is "none of
    # these", option 1 the one plan
    app.selectbox(key="pr_signoff_choice").select_index(1)
    app.text_area(key="pr_signoff_reason").input("the demonstration plan is only for the test")
    app.button(key="pr_signoff_submit").click().run()
    assert not app.exception
    records = log_records(tmp_path)
    assert records[-1]["event"] == "sign_off" and records[-1]["payload"]["chosen_plan_hash"] == records[2]["plan_hashes"][0]
    assert records[-1]["payload"]["reason"] == "the demonstration plan is only for the test"


def test_a_sign_off_without_a_reason_is_refused_and_writes_nothing(app, tmp_path):
    app.checkbox(key="pr_add_demo").check().run()
    app.selectbox(key="pr_tracker_model").select("noise_only").run()
    app.selectbox(key="pr_role").select("planner").run()
    app.button(key="pr_run").click().run()
    n = len(log_records(tmp_path))
    app.selectbox(key="pr_signoff_choice").select_index(0)  # "none of these", but no reason given
    app.button(key="pr_signoff_submit").click().run()
    assert any("give a reason" in e.value for e in app.error)
    assert len(log_records(tmp_path)) == n
    app.button(key="pr_signoff_submit").click().run()  # nothing chosen at all
    assert any("Choose a plan" in e.value for e in app.error)
    assert len(log_records(tmp_path)) == n


def test_the_tab_refuses_to_run_when_the_log_directory_is_inside_the_repository(monkeypatch):
    from review.provenance import REPO_ROOT
    monkeypatch.setenv("LVM_TESTBED_LOG_DIR", str(REPO_ROOT / "results" / "logs"))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("inside the repository" in e.value for e in at.error)
    assert not (REPO_ROOT / "results" / "logs").exists()
