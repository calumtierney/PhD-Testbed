"""The command-line twin of the Plan review tab (viz.review_plans)."""
import json

import pytest

from review.log import load_log
from review.provenance import REPO_ROOT
from viz.review_plans import main


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    directory = tmp_path / "logs"
    monkeypatch.setenv("LVM_TESTBED_LOG_DIR", str(directory))
    return directory


def test_the_tracker_model_is_required_and_has_no_default(capsys):
    with pytest.raises(SystemExit):
        main(["factory", "--demo", "--no-log"])
    assert "--tracker-model" in capsys.readouterr().err


def test_a_review_is_logged_and_printed_and_exported(tmp_path, log_dir, capsys):
    out = tmp_path / "export"
    status = main(["factory", "--demo", "--tracker-model", "noise_only", "--reviewer-role", "metrologist", "--out", str(out)])
    printed = capsys.readouterr().out
    assert status == 0
    assert "sensor noise only" in printed and "synthetic" in printed and "PLACEHOLDER" in printed and "logged to" in printed
    assert [r["event"] for r in load_log(log_dir)] == ["scene_loaded", "plan_reviewed", "comparison"]
    bundle = json.loads((out / "comparison.json").read_text())
    assert bundle["comparison"]["rows"][0]["author_role"] == "synthetic" and bundle["reviews"][0]["feasible"]
    assert {p.name for p in out.iterdir()} == {"comparison.json", "plan_view.png", "plan_view.svg", "risk_by_characteristic.png", "risk_by_characteristic.svg"}
    assert (out / "plan_view.svg").read_text().lstrip().startswith(("<?xml", "<svg"))


def test_every_review_needs_a_role_unless_it_is_a_dry_run(log_dir, capsys):
    assert main(["factory", "--demo", "--tracker-model", "noise_only"]) == 2
    assert "reviewer-role" in capsys.readouterr().err
    assert main(["factory", "--demo", "--tracker-model", "noise_only", "--no-log"]) == 0
    assert not log_dir.exists()  # a dry run writes nothing


def test_a_log_or_export_directory_inside_the_repository_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LVM_TESTBED_LOG_DIR", str(REPO_ROOT / "results" / "logs"))
    assert main(["factory", "--demo", "--tracker-model", "noise_only", "--reviewer-role", "x"]) == 2
    assert "inside the repository" in capsys.readouterr().err
    assert main(["factory", "--demo", "--tracker-model", "noise_only", "--no-log", "--out", str(REPO_ROOT / "results" / "review")]) == 2
    assert "inside the repository" in capsys.readouterr().err
    assert not (REPO_ROOT / "results" / "review").exists() and not (REPO_ROOT / "results" / "logs").exists()


def test_a_bad_plan_file_is_reported_not_guessed(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "units": "mm"}))
    assert main(["factory", str(bad), "--tracker-model", "noise_only", "--no-log"]) == 2
    assert "missing required key" in capsys.readouterr().err  # the reader's own message; nothing was guessed or defaulted
    wrong_units = tmp_path / "mm.json"
    wrong_units.write_text(json.dumps({"schema_version": 1, "name": "p", "author_role": "engineer", "source": "s", "created": "2025-11-27T00:00:00Z",
                                       "frame": "cell", "units": "mm", "stations": [{"position_m": [0, 0, 1]}]}))
    assert main(["factory", str(wrong_units), "--tracker-model", "noise_only", "--no-log"]) == 2
    assert "only metres" in capsys.readouterr().err
