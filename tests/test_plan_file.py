"""Tests for interchange.plan_file: strict reading of a plan, defaults only where the format allows them."""
import copy
import json

import pytest

from interchange.plan_file import (
    PlanFileError, PlanStation, parse_plan, plan_file_from_stations, read_plan_file, read_plan_from_spatialanalyzer, write_plan_file,
)

BASE = {
    "schema_version": 1, "name": "demo", "author_role": "synthetic", "source": "test", "created": "2025-11-27T10:00:00Z",
    "frame": "cell", "units": "m",
    "stations": [
        {"position_m": [1.0, -4.0, 1.2]},
        {"position_m": [8.0, 4.5, 1.8], "instrument_id": 1, "faces": ["front", "back"], "track": {"s_m": 22.0, "height_m": 1.8},
         "targets": ["a", "b", "c"]},
    ],
}


def mutated(**changes):
    data = copy.deepcopy(BASE)
    for key, value in changes.items():
        data[key] = value
    return data


def test_a_valid_plan_reads_with_only_the_documented_defaults():
    plan = parse_plan(BASE)
    first, second = plan.stations
    assert first == PlanStation((1.0, -4.0, 1.2))  # instrument 0, front face, no track, no target list
    assert first.faces == ("front",) and first.instrument_id == 0 and first.targets is None
    assert second.instrument_id == 1 and second.faces == ("front", "back") and (second.s_m, second.height_m) == (22.0, 1.8)
    assert second.targets == ("a", "b", "c")
    assert plan.n_stations == 2 and plan.author_role == "synthetic"


def test_an_off_track_position_is_accepted_and_the_track_entry_is_optional():
    plan = parse_plan(mutated(stations=[{"position_m": [-40.0, 55.0, 0.3]}, {"position_m": [3.0, 3.0, 3.0]}]))
    assert plan.stations[0].s_m is None  # nothing makes a station stand on the track


def test_write_then_read_round_trips_exactly(tmp_path):
    plan = parse_plan(BASE)
    path = tmp_path / "plan.json"
    write_plan_file(plan, path)
    assert read_plan_file(path) == plan
    assert read_plan_file(path.read_bytes()) == plan  # an upload, read from memory
    assert json.loads(path.read_text())["units"] == "m"


@pytest.mark.parametrize("change, match", [
    ({"units": "mm"}, "only metres"),
    ({"units": "in"}, "only metres"),
    ({"frame": "world"}, "frame"),
    ({"author_role": "boss"}, "author_role"),
    ({"schema_version": 2}, "schema_version"),
    ({"stations": []}, "non-empty"),
    ({"created": "yesterday"}, "ISO 8601"),
    ({"extra": 1}, "unknown key"),
    ({"name": ""}, "must not be empty"),
])
def test_top_level_refusals(change, match):
    with pytest.raises(PlanFileError, match=match):
        parse_plan(mutated(**change))


@pytest.mark.parametrize("key", ["schema_version", "name", "author_role", "source", "created", "frame", "units", "stations"])
def test_every_top_level_key_is_required(key):
    data = copy.deepcopy(BASE)
    del data[key]
    with pytest.raises(PlanFileError):
        parse_plan(data)


@pytest.mark.parametrize("station, match", [
    ({"position_m": [1.0, 2.0]}, "three numbers"),
    ({"position_m": [1.0, 2.0, "3"]}, "expected a number"),
    ({"position_m": [1.0, 2.0, float("nan")]}, "finite"),
    ({"position_m": [1.0, 2.0, float("inf")]}, "finite"),
    ({"position_m": [1.0, 2.0, True]}, "expected a number"),
    ({}, "position_m is required"),
    ({"position_m": [0, 0, 0], "instrument_id": -1}, "non-negative"),
    ({"position_m": [0, 0, 0], "instrument_id": 1.5}, "non-negative"),
    ({"position_m": [0, 0, 0], "faces": []}, "non-empty"),
    ({"position_m": [0, 0, 0], "faces": ["top"]}, "not one of"),
    ({"position_m": [0, 0, 0], "faces": ["front", "front"]}, "twice"),
    ({"position_m": [0, 0, 0], "track": {"s_m": 1.0}}, "both"),
    ({"position_m": [0, 0, 0], "track": {"s_m": 1.0, "height_m": 1.0, "x": 2}}, "unknown key"),
    ({"position_m": [0, 0, 0], "targets": []}, "non-empty"),
    ({"position_m": [0, 0, 0], "targets": ["a", "a"]}, "twice"),
    ({"position_m": [0, 0, 0], "colour": "red"}, "unknown key"),
])
def test_station_refusals(station, match):
    with pytest.raises(PlanFileError, match=match):
        parse_plan(mutated(stations=[station]))


def test_json_that_is_not_a_plan_is_refused(tmp_path):
    for payload, match in ((b"[1, 2]", "JSON object"), (b"not json", "valid UTF-8 JSON"), (b"\xff\xfe", "valid UTF-8 JSON")):
        with pytest.raises(PlanFileError, match=match):
            read_plan_file(payload)
    big = tmp_path / "big.json"
    big.write_bytes(b" " * 200)
    with pytest.raises(PlanFileError, match="larger than"):
        read_plan_file(big, max_bytes=100)


def test_the_hash_ignores_labels_but_not_what_is_scored():
    plan = parse_plan(BASE)
    relabelled = parse_plan(mutated(name="other", source="elsewhere", created="2030-01-01T00:00:00Z", author_role="engineer"))
    assert plan.content_hash() == relabelled.content_hash()
    no_track = copy.deepcopy(BASE)
    del no_track["stations"][1]["track"]
    assert parse_plan(no_track).content_hash() == plan.content_hash()  # the track coordinates are a label
    moved = copy.deepcopy(BASE)
    moved["stations"][0]["position_m"][0] += 1e-9
    assert parse_plan(moved).content_hash() != plan.content_hash()
    other_frame = mutated(frame="file")
    assert parse_plan(other_frame).content_hash() != plan.content_hash()
    reordered = copy.deepcopy(BASE)
    reordered["stations"].reverse()
    assert parse_plan(reordered).content_hash() != plan.content_hash()  # the first station is the anchor


def test_the_spatialanalyzer_adapter_is_a_documented_stub_not_a_guess():
    with pytest.raises(NotImplementedError, match="TODO\\(source\\)"):
        read_plan_from_spatialanalyzer(b"anything")
    assert "target" in read_plan_from_spatialanalyzer.__doc__ and "frame" in read_plan_from_spatialanalyzer.__doc__


def test_plan_file_from_stations_validates_like_the_reader():
    plan = plan_file_from_stations("p", "algorithm", "s", "2025-01-01T00:00:00Z", "cell", [[0, 0, 1], [1, 1, 1]],
                                   track_coordinates=[(0.0, 1.0), None], instrument_ids=[0, 1], faces=[("front",), ("front", "back")])
    assert plan.stations[0].s_m == 0.0 and plan.stations[1].s_m is None and plan.stations[1].faces == ("front", "back")
    with pytest.raises(PlanFileError):
        plan_file_from_stations("p", "algorithm", "s", "2025-01-01T00:00:00Z", "cell", [[0, 0, float("nan")]])
