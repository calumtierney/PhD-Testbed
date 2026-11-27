"""Gate 7 of the plan-review brief: the review log round-trips, is append-only, refuses an unknown schema version and
refuses a directory inside the repository. Plus what is and is not written to it."""
import json
import os
import stat
from pathlib import Path

import pytest

from interchange.qif import read_qif
from planning.qif_cell import build_qif_cell
from review.core import compare_plans, review_plans
from review.demo import demo_plan
from review.log import (
    ENV_VAR, LOG_FILE_NAME, LOG_SCHEMA_VERSION, ExecutionResult, LogError, ReviewLog, comparison_id, default_log_dir,
    execution_results, is_inside_repository, load_log, parse_execution_result, resolve_log_dir,
)
from review.provenance import REPO_ROOT
from review.scene import ReviewScene
from review.settings import ReviewSettings

SAMPLE = REPO_ROOT / "scenes" / "qif" / "SheetMetal_QIF_Results_6_samples.QIF"
NOISE = ReviewSettings("noise_only")


@pytest.fixture(scope="module")
def factory():
    return ReviewScene.from_factory_cell()


@pytest.fixture(scope="module")
def compared(factory):
    context, reviews = review_plans(factory, [demo_plan(factory, 4), demo_plan(factory, 3)], NOISE)
    return reviews, compare_plans(reviews)


def lines(path: Path):
    return path.read_bytes().splitlines(keepends=True)


# -- round trip --------------------------------------------------------------------------------------
def test_gate7_the_log_round_trips(tmp_path, factory, compared):
    reviews, comparison = compared
    log = ReviewLog(tmp_path, reviewer_role="metrologist")
    written = log.log_session(factory, NOISE, reviews, comparison)
    sign = log.log_sign_off(comparison, reviews[1].plan["plan_hash"], "fewer stations, risk close enough")
    records = load_log(tmp_path)
    assert [r["event"] for r in records] == ["scene_loaded", "plan_reviewed", "plan_reviewed", "comparison", "sign_off"]
    assert records == json.loads(json.dumps(written + [sign]))  # what was written is what is read, as JSON
    for record in records:
        assert record["schema_version"] == LOG_SCHEMA_VERSION and record["reviewer_role"] == "metrologist"
        assert record["timestamp_utc"].endswith("Z") and len(record["record_id"]) == 32
        assert "reviewer_name" not in record  # a role, never a name unless the reviewer adds one
    assert len({r["record_id"] for r in records}) == len(records)
    scene_hash = factory.scene_hash()
    assert all(r["scene_hash"] == scene_hash for r in records)
    reviewed = records[1]["payload"]["review"]
    assert reviewed["feasible"] and reviewed["totals"]["expected_wrong_decisions"] == reviews[0].totals["expected_wrong_decisions"]
    assert records[1]["plan_hashes"] == [reviews[0].plan["plan_hash"]]
    assert records[1]["provenance"]["tracker_model"] == "noise_only" and records[1]["settings"]["tracker_model"] == "noise_only"
    comparison_record = records[3]
    assert comparison_record["payload"]["comparison_id"] == comparison_id(comparison) == records[4]["payload"]["comparison_id"]
    assert comparison_record["plan_hashes"] == [r.plan["plan_hash"] for r in reviews]
    assert records[4]["payload"] == {"comparison_id": comparison_id(comparison), "chosen_plan_hash": reviews[1].plan["plan_hash"],
                                     "reason": "fewer stations, risk close enough"}


def test_a_reviewer_name_is_recorded_only_if_the_reviewer_adds_one(tmp_path, factory, compared):
    reviews, _ = compared
    ReviewLog(tmp_path, "engineer", reviewer_name="A. Person").log_plan_reviewed(reviews[0], factory)
    assert load_log(tmp_path)[0]["reviewer_name"] == "A. Person"
    with pytest.raises(LogError, match="role"):
        ReviewLog(tmp_path, reviewer_role="  ")


def test_a_sign_off_names_one_of_the_compared_plans_or_none_and_gives_a_reason(tmp_path, compared):
    _, comparison = compared
    log = ReviewLog(tmp_path, "planner")
    log.log_sign_off(comparison, None, "none of them: both leave the marginal cluster at 2:1")
    assert load_log(tmp_path)[0]["payload"]["chosen_plan_hash"] is None
    with pytest.raises(LogError, match="not one of the compared"):
        log.log_sign_off(comparison, "0" * 64, "x")
    with pytest.raises(LogError, match="reason"):
        log.log_sign_off(comparison, comparison["rows"][0]["plan_hash"], "   ")
    assert len(load_log(tmp_path)) == 1  # refused sign-offs wrote nothing


# -- append-only -----------------------------------------------------------------------------------------
def test_gate7_the_log_is_append_only(tmp_path, factory, compared):
    reviews, comparison = compared
    log = ReviewLog(tmp_path, "metrologist")
    log.log_plan_reviewed(reviews[0], factory)
    before = log.path.read_bytes()
    log.log_comparison(comparison, factory)
    after = log.path.read_bytes()
    assert after.startswith(before) and len(after) > len(before)  # the existing line is untouched
    other = ReviewLog(tmp_path, "planner")  # a second writer (another session) appends, it does not replace
    other.log_sign_off(comparison, None, "none")
    final = log.path.read_bytes()
    assert final.startswith(after)
    assert all(line.endswith(b"\n") and json.loads(line) for line in final.splitlines(keepends=True))
    assert len(final.splitlines()) == 3 and stat.S_IMODE(os.stat(log.path).st_mode) == 0o600


def test_every_event_type_appends_exactly_one_line(tmp_path, factory, compared):
    reviews, comparison = compared
    log = ReviewLog(tmp_path, "metrologist")
    steps = [lambda: log.log_scene_loaded(factory, NOISE), lambda: log.log_plan_reviewed(reviews[0], factory),
             lambda: log.log_comparison(comparison, factory), lambda: log.log_sign_off(comparison, None, "r")]
    for n, step in enumerate(steps, start=1):
        step()
        assert len(lines(log.path)) == n


def test_appending_with_an_unknown_event_is_refused(tmp_path):
    with pytest.raises(LogError, match="unknown event"):
        ReviewLog(tmp_path, "x").append("something_else", {})
    assert not (tmp_path / LOG_FILE_NAME).exists()


# -- unknown schema versions and malformed lines ------------------------------------------------------------
def test_gate7_an_unknown_schema_version_is_refused_and_the_line_is_named(tmp_path, factory, compared):
    reviews, _ = compared
    log = ReviewLog(tmp_path, "metrologist")
    log.log_plan_reviewed(reviews[0], factory)
    record = json.loads(lines(log.path)[0])
    record["schema_version"] = LOG_SCHEMA_VERSION + 1
    with open(log.path, "ab") as handle:
        handle.write((json.dumps(record) + "\n").encode())
    with pytest.raises(LogError, match=r"line 2: schema_version 2 is not supported"):
        load_log(tmp_path)


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.pop("payload"), "record keys"),
    (lambda r: r.update(extra=1), "record keys"),
    (lambda r: r.update(event="mystery"), "unknown event"),
    (lambda r: r.pop("schema_version"), "schema_version None"),
])
def test_malformed_records_are_refused(tmp_path, factory, compared, mutate, match):
    reviews, _ = compared
    log = ReviewLog(tmp_path, "metrologist")
    log.log_plan_reviewed(reviews[0], factory)
    record = json.loads(lines(log.path)[0])
    mutate(record)
    log.path.write_text(json.dumps(record) + "\n")
    with pytest.raises(LogError, match=match):
        load_log(tmp_path)


def test_invalid_json_is_refused_and_an_absent_log_is_empty(tmp_path):
    assert load_log(tmp_path / "nothing_here") == []
    (tmp_path / LOG_FILE_NAME).write_text("{not json\n")
    with pytest.raises(LogError, match="line 1: not valid JSON"):
        load_log(tmp_path)


# -- where it may live -------------------------------------------------------------------------------------------
def test_gate7_a_log_directory_inside_the_repository_is_refused(tmp_path, monkeypatch):
    inside = REPO_ROOT / "review_logs_must_not_be_here"
    with pytest.raises(LogError, match="inside the repository"):
        ReviewLog(inside, "x")
    with pytest.raises(LogError, match="inside the repository"):
        ReviewLog(REPO_ROOT, "x")
    with pytest.raises(LogError, match="inside the repository"):
        load_log(REPO_ROOT / "results")
    monkeypatch.setenv(ENV_VAR, str(REPO_ROOT / "results" / "logs"))
    with pytest.raises(LogError, match=f"\\${ENV_VAR}"):
        resolve_log_dir()
    assert not inside.exists()
    link = tmp_path / "sneaky"
    link.symlink_to(REPO_ROOT / "results")  # a symlink into the repository is still inside it
    with pytest.raises(LogError, match="inside the repository"):
        ReviewLog(link / "logs", "x")


def test_the_environment_variable_sets_the_directory_and_the_default_is_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "logs"))
    assert resolve_log_dir() == (tmp_path / "logs").resolve()
    assert ReviewLog(reviewer_role="x").directory == (tmp_path / "logs").resolve()
    monkeypatch.delenv(ENV_VAR)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert not is_inside_repository(default_log_dir())
    assert resolve_log_dir() == (tmp_path / "data" / "lvm_testbed" / "review_logs").resolve()
    monkeypatch.setenv("XDG_DATA_HOME", str(REPO_ROOT / "data"))  # a default that would land in the repository is refused too
    with pytest.raises(LogError, match="inside the repository"):
        resolve_log_dir()


# -- what is written ----------------------------------------------------------------------------------------------
def test_scene_names_are_left_out_of_the_log_by_default_and_kept_on_request(tmp_path):
    cell = build_qif_cell(read_qif(SAMPLE))
    scene = ReviewScene.from_qif_cell(cell, "customer_part_FILE.qif")
    from interchange.plan_file import plan_file_from_stations
    track = scene.track  # a QIF scene needs a plan on its own track: three stations evenly spaced round it
    positions = []
    for k in range(3):
        x, y = track.position_xy_m((k + 0.5) * track.perimeter_m / 3)
        positions.append([x, y, 1.5])
    plan = plan_file_from_stations("Customer wing demo plan", "synthetic", "exported from CustomerSoft v9", "2025-11-27T00:00:00Z", "cell", positions)
    context, reviews = review_plans(scene, [plan], NOISE)
    comparison = compare_plans(reviews)
    secrets = [p.name for p in cell.planned] + [p.feature_name for p in cell.planned] + [c.name for c in cell.clusters] + \
        ["customer_part_FILE.qif", "Customer wing demo plan", "CustomerSoft"]

    private = ReviewLog(tmp_path / "private", "metrologist")
    private.log_session(scene, NOISE, reviews, comparison)
    text = private.path.read_text()
    assert [s for s in secrets if s in text] == []
    assert "char_0" in text and "cluster_0" in text
    records = load_log(tmp_path / "private")
    review = records[1]["payload"]["review"]
    assert review["feasible"] and [c["name"] for c in review["characteristics"]][:2] == ["char_0", "char_1"]
    assert any(f["kind"] == "characteristic_not_seen" and f["characteristic"].startswith("char_") for f in review["findings"])
    assert review["plan"]["name"] == "redacted" and review["plan"]["plan_hash"] == reviews[0].plan["plan_hash"]
    assert all("members" not in c for c in records[0]["payload"]["scene"]["clusters"])

    named = ReviewLog(tmp_path / "named", "metrologist", log_names=True)
    named.log_session(scene, NOISE, reviews, comparison)
    text = named.path.read_text()
    assert "customer_part_FILE.qif" in text and "Customer wing demo plan" in text and cell.planned[0].name in text


def test_the_numbers_are_the_same_whether_or_not_names_are_logged(tmp_path, factory, compared):
    reviews, comparison = compared
    a, b = ReviewLog(tmp_path / "a", "x"), ReviewLog(tmp_path / "b", "x", log_names=True)
    a.log_plan_reviewed(reviews[0], factory)
    b.log_plan_reviewed(reviews[0], factory)
    ra, rb = load_log(tmp_path / "a")[0]["payload"]["review"], load_log(tmp_path / "b")[0]["payload"]["review"]
    assert ra["totals"] == rb["totals"] and ra["stations"] == rb["stations"] and ra["feasible"] == rb["feasible"]
    for ca, cb in zip(ra["characteristics"], rb["characteristics"]):
        assert {k: v for k, v in ca.items() if k not in ("name", "cluster")} == {k: v for k, v in cb.items() if k not in ("name", "cluster")}


def test_an_infeasible_review_is_logged_with_its_reasons(tmp_path, factory):
    from interchange.plan_file import plan_file_from_stations
    plan = plan_file_from_stations("lost", "engineer", "s", "2025-11-27T00:00:00Z", "cell", [[0.0, -4.0, 1.5], [900.0, 900.0, 1.0]])
    _, reviews = review_plans(factory, [plan], NOISE)
    ReviewLog(tmp_path, "metrologist").log_plan_reviewed(reviews[0], factory)
    review = load_log(tmp_path)[0]["payload"]["review"]
    assert review["feasible"] is False and review["characteristics"] is None
    assert any(f["kind"] == "station_thin" for f in review["findings"])


# -- execution results (Loop 2) -------------------------------------------------------------------------------------
GOOD_RESULT = {
    "plan_hash": "a" * 64, "scene_hash": "b" * 64, "executed_utc": "2026-01-15T09:30:00Z",
    "network_residuals": {"rms_um": 3.1, "max_um": 9.8, "n_observations": 126},
    "characteristics": [{"index": 0, "measured_deviation_um": 4.2, "decision": "accept", "outcome": "conforming"},
                        {"index": 1, "measured_deviation_um": None, "decision": "not_decided", "outcome": "unknown"}],
    "notes": "pilot",
}


def test_an_execution_result_has_a_schema_a_writer_and_a_reader(tmp_path):
    result = parse_execution_result(GOOD_RESULT)
    assert isinstance(result, ExecutionResult) and result.characteristics[1]["measured_deviation_um"] is None
    log = ReviewLog(tmp_path, "metrologist")
    log.log_execution_result(result, NOISE.to_dict())
    records = load_log(tmp_path)
    assert records[0]["event"] == "execution_result" and records[0]["plan_hashes"] == ["a" * 64] and records[0]["scene_hash"] == "b" * 64
    assert execution_results(records) == [result]


@pytest.mark.parametrize("change, match", [
    ({"plan_hash": "nothash"}, "SHA-256"),
    ({"executed_utc": "yesterday"}, "ISO 8601"),
    ({"network_residuals": {"rms_um": 1.0}}, "exactly"),
    ({"network_residuals": {"rms_um": float("nan"), "max_um": 1.0, "n_observations": 3}}, "finite"),
    ({"characteristics": []}, "non-empty"),
    ({"characteristics": [{"index": 0, "measured_deviation_um": 1.0, "decision": "maybe", "outcome": "unknown"}]}, "decision"),
    ({"characteristics": [{"index": 0, "measured_deviation_um": 1.0, "decision": "accept", "outcome": "fine"}]}, "outcome"),
    ({"characteristics": [{"index": 0, "measured_deviation_um": 1.0, "decision": "accept"}]}, "exactly"),
    ({"surprise": 1}, "unknown key"),
])
def test_execution_results_are_validated_strictly(change, match):
    with pytest.raises(LogError, match=match):
        parse_execution_result({**GOOD_RESULT, **change})


def test_duplicate_characteristic_indices_in_an_execution_result_are_refused():
    rows = [GOOD_RESULT["characteristics"][0], dict(GOOD_RESULT["characteristics"][0])]
    with pytest.raises(LogError, match="distinct"):
        parse_execution_result({**GOOD_RESULT, "characteristics": rows})


def test_a_log_with_a_bad_execution_result_is_refused_on_reading(tmp_path):
    log = ReviewLog(tmp_path, "metrologist")
    log.append("execution_result", {**GOOD_RESULT, "plan_hash": "short"}, scene_hash="b" * 64)
    with pytest.raises(LogError, match="SHA-256"):
        load_log(tmp_path)
