"""A measurement plan as a file: how an engineer's plan, or an algorithm's, gets into the testbed.

    {"schema_version": 1, "name": "...", "author_role": "engineer", "source": "free text",
     "created": "2025-11-27T10:00:00Z", "frame": "cell", "units": "m",
     "stations": [{"position_m": [x, y, z], "instrument_id": 0, "faces": ["front"],
                   "track": {"s_m": 12.0, "height_m": 1.25}, "targets": ["name", "..."]}]}

What a plan is, for the review (`review.core`): where each tracker stands, which
physical tracker it is, which faces it reads, and optionally which targets it
actually measures from there. Nothing else: no times, no sequence, no
instrument settings. A station is a *position*; the instrument's orientation is
not in the file (see "Orientation" below).

**Strictness.** A plan file is an input to a risk calculation, so a misread value
changes a decision. The reader refuses anything it does not recognise: unknown
keys at any level, units other than metres, a frame other than `file` or `cell`,
non-finite numbers, empty station lists, duplicate faces. The only defaults are
the two the format allows: `instrument_id` (0: one tracker for every station)
and `faces` (`["front"]`). Everything else must be stated.

**Frames.** `cell` coordinates are the testbed cell's (metres; z up). `file`
coordinates are those of the QIF document the scene was built from; they are
converted with the scene's part-to-cell placement, which a scene built without
a supplied placement *assumes* (`planning.qif_cell.default_placement`), and the
review says so. A built-in scene has no file frame: a plan in `file` coordinates
cannot be reviewed against it.

**Engineers do not stand on our track.** A station is any point in the cell;
the optional `track` entry (arc length and height on `geometry.station_track`)
is a convenience of plans the testbed itself made, kept so a plan can be drawn
in track coordinates and nothing more: when both are given, the position is what
is scored.

**Orientation.** The file has no orientation. The review builds each station as
a *levelled* tracker (its standing axis vertical, as on a tripod) turned to face
the centroid of the scene's targets. With sensor noise alone the turn about the
vertical cannot change any covariance: the azimuth encoder has the same noise at
every angle. With the NPL geometric parameters it can, because the encoder
harmonics and axis tilts are fixed in the instrument's frame: a few tenths of a
percent on the factory-cell scene (`docs/step1b_npl_geometric_model.md`). So the
convention is fixed and printed with every review, not left to chance, and a real
tracker's heading, if it matters, would have to come from the plan source.

**Adapters.** What a particular measurement-planning package can export is not
known here and is not guessed. `read_plan_from_spatialanalyzer` is the place
such an adapter goes; see its docstring for what it needs.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

SCHEMA_VERSION = 1
AUTHOR_ROLES = ("engineer", "algorithm", "synthetic")
FRAMES = ("file", "cell")
FACE_NAMES = ("front", "back")
MAX_FILE_BYTES = 5_000_000

_TOP_KEYS = {"schema_version", "name", "author_role", "source", "created", "frame", "units", "stations"}
_STATION_KEYS = {"position_m", "instrument_id", "track", "faces", "targets"}
_TRACK_KEYS = {"s_m", "height_m"}


class PlanFileError(ValueError):
    """A plan file the reader refuses, with the reason."""


@dataclass(frozen=True)
class PlanStation:
    """One tracker position. `targets` is None when the plan does not say what is
    measured from here (every target the station can see is then assumed to be
    measured), or a tuple of target names (exactly those are measured, if the
    station can see them)."""

    position_m: Tuple[float, float, float]
    instrument_id: int = 0
    s_m: Optional[float] = None
    height_m: Optional[float] = None
    faces: Tuple[str, ...] = ("front",)
    targets: Optional[Tuple[str, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"position_m": list(self.position_m), "instrument_id": self.instrument_id,
                               "faces": list(self.faces)}
        if self.s_m is not None:
            out["track"] = {"s_m": self.s_m, "height_m": self.height_m}
        if self.targets is not None:
            out["targets"] = list(self.targets)
        return out


@dataclass(frozen=True)
class PlanFile:
    """A plan: metadata plus stations, in the order listed. **The first station is
    the one the others are tied to** (the anchor of the review's network), so list
    the station the frame is set from first. Datum-referenced results do not depend
    on the choice; whether the others can be tied to it does."""

    name: str
    author_role: str
    source: str
    created: str
    frame: str
    stations: Tuple[PlanStation, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "author_role": self.author_role,
                "source": self.source, "created": self.created, "frame": self.frame, "units": "m",
                "stations": [s.to_dict() for s in self.stations]}

    def content_hash(self) -> str:
        """SHA-256 of what a review scores: frame, and each station's position,
        instrument, faces and targets, in order. Not the name, author, source or
        creation time: the same stations are the same plan whoever wrote them."""
        content = {"frame": self.frame, "stations": [s.to_dict() for s in self.stations]}
        # the track coordinates are a label, not part of what is scored
        for station in content["stations"]:
            station.pop("track", None)
        return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def n_stations(self) -> int:
        return len(self.stations)


# -- reading ---------------------------------------------------------------------------
def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanFileError(f"{where}: expected a number, got {value!r}")
    if not math.isfinite(value):
        raise PlanFileError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _string(value: Any, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PlanFileError(f"{where}: expected text, got {value!r}")
    if not allow_empty and not value.strip():
        raise PlanFileError(f"{where}: must not be empty")
    return value


def _no_unknown(mapping: Dict[str, Any], allowed: set, where: str):
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PlanFileError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _station(raw: Any, index: int) -> PlanStation:
    where = f"stations[{index}]"
    if not isinstance(raw, dict):
        raise PlanFileError(f"{where}: expected an object")
    _no_unknown(raw, _STATION_KEYS, where)
    if "position_m" not in raw:
        raise PlanFileError(f"{where}: position_m is required")
    position = raw["position_m"]
    if not isinstance(position, list) or len(position) != 3:
        raise PlanFileError(f"{where}.position_m: expected three numbers [x, y, z] in metres")
    position_m = tuple(_finite(v, f"{where}.position_m") for v in position)

    instrument_id = raw.get("instrument_id", 0)
    if isinstance(instrument_id, bool) or not isinstance(instrument_id, int) or instrument_id < 0:
        raise PlanFileError(f"{where}.instrument_id: expected a non-negative integer, got {instrument_id!r}")

    s_m = height_m = None
    if "track" in raw:
        track = raw["track"]
        if not isinstance(track, dict):
            raise PlanFileError(f"{where}.track: expected an object with s_m and height_m")
        _no_unknown(track, _TRACK_KEYS, f"{where}.track")
        if set(track) != _TRACK_KEYS:
            raise PlanFileError(f"{where}.track: needs both s_m and height_m")
        s_m, height_m = _finite(track["s_m"], f"{where}.track.s_m"), _finite(track["height_m"], f"{where}.track.height_m")

    faces = raw.get("faces", ["front"])
    if not isinstance(faces, list) or not faces:
        raise PlanFileError(f"{where}.faces: expected a non-empty list drawn from {list(FACE_NAMES)}")
    for face in faces:
        if face not in FACE_NAMES:
            raise PlanFileError(f"{where}.faces: {face!r} is not one of {list(FACE_NAMES)}")
    if len(set(faces)) != len(faces):
        raise PlanFileError(f"{where}.faces: a face is listed twice")

    targets = None
    if "targets" in raw:
        listed = raw["targets"]
        if not isinstance(listed, list) or not listed:
            raise PlanFileError(f"{where}.targets: expected a non-empty list of target names (omit the key to mean 'whatever is visible')")
        names = tuple(_string(t, f"{where}.targets") for t in listed)
        if len(set(names)) != len(names):
            raise PlanFileError(f"{where}.targets: a target is listed twice")
        targets = names
    return PlanStation(position_m, instrument_id, s_m, height_m, tuple(faces), targets)


def parse_plan(data: Any) -> PlanFile:
    """Validate a decoded JSON document and return the plan. Raises `PlanFileError`."""
    if not isinstance(data, dict):
        raise PlanFileError("a plan file is a JSON object")
    if "schema_version" not in data:
        raise PlanFileError("schema_version is required")
    if data["schema_version"] != SCHEMA_VERSION:
        raise PlanFileError(f"schema_version {data['schema_version']!r} is not supported (this reader reads {SCHEMA_VERSION})")
    _no_unknown(data, _TOP_KEYS, "plan")
    missing = sorted(_TOP_KEYS - set(data))
    if missing:
        raise PlanFileError(f"plan: missing required key(s) {missing}")
    if data["units"] != "m":
        raise PlanFileError(f"units: only metres ('m') are accepted, got {data['units']!r}; convert before writing the file")
    if data["author_role"] not in AUTHOR_ROLES:
        raise PlanFileError(f"author_role: {data['author_role']!r} is not one of {list(AUTHOR_ROLES)}")
    if data["frame"] not in FRAMES:
        raise PlanFileError(f"frame: {data['frame']!r} is not one of {list(FRAMES)}")
    created = _string(data["created"], "created")
    try:
        datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        raise PlanFileError(f"created: {created!r} is not an ISO 8601 date-time") from None
    stations = data["stations"]
    if not isinstance(stations, list) or not stations:
        raise PlanFileError("stations: expected a non-empty list")
    return PlanFile(
        name=_string(data["name"], "name"), author_role=data["author_role"], source=_string(data["source"], "source", True),
        created=created, frame=data["frame"], stations=tuple(_station(s, i) for i, s in enumerate(stations)),
    )


def read_plan_file(source: Union[str, Path, bytes], max_bytes: int = MAX_FILE_BYTES) -> PlanFile:
    """Read a plan file from a path or from bytes (an upload, held in memory).
    Refuses files over `max_bytes`: a plan is a few kilobytes."""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise PlanFileError(f"{path.name}: larger than {max_bytes} bytes; a plan file is a few kilobytes")
        raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise PlanFileError(f"plan file larger than {max_bytes} bytes")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanFileError(f"not valid UTF-8 JSON: {error}") from None
    return parse_plan(data)


def write_plan_file(plan: PlanFile, path: Union[str, Path]) -> None:
    """Write `plan` as a plan file. Refuses nothing it could not have read back:
    the plan is validated by round-tripping through `parse_plan` first."""
    document = plan.to_dict()
    parse_plan(json.loads(json.dumps(document)))
    Path(path).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


# -- adapters ----------------------------------------------------------------------------
def read_plan_from_spatialanalyzer(source: Union[str, Path, bytes]) -> PlanFile:
    """Adapter slot for a SpatialAnalyzer export. **Not implemented**: what
    SpatialAnalyzer can export for a plan is not known here, and a format is not
    guessed (CLAUDE.md §6: never fabricate a reference).

    TODO(source): to write this adapter the following are needed, from a real
    export of a real plan:

      * the file format (and version) and where it records instrument stations
        (an "instrument" or "station" object with a transform?), and which of
        its objects are planned stations and which are only history;
      * the coordinate frame the positions are in (the job's, the part's, the
        aircraft's), the length unit, and the transform to the CAD / QIF frame
        the scene was built from;
      * how the instrument's identity is recorded (one tracker or several, so
        `instrument_id` can be filled) and which face each reading was taken on;
      * what a station's measured targets are called and how those names map to
        the QIF features / characteristics (target naming), so `targets` can be
        filled and the review can check each is visible;
      * whether stations are listed in measurement order, so the first-station
        (anchor) convention of `PlanFile` can be honoured.

    An adapter returns a `PlanFile` with `author_role="engineer"` and a `source`
    string naming the export; `review` then needs nothing else.
    """
    raise NotImplementedError("read_plan_from_spatialanalyzer: see the docstring for what an adapter needs (TODO(source))")


def plan_file_from_stations(name: str, author_role: str, source: str, created: str, frame: str, positions_m, track_coordinates=None,
                            instrument_ids=None, faces=None, targets=None) -> PlanFile:
    """Build a `PlanFile` from parallel lists. `track_coordinates` is a list of `(s_m, height_m)` or None entries;
    `faces` a list of tuples; `targets` a list of name tuples or None entries. Validated by `parse_plan`."""
    stations = []
    for i, position in enumerate(positions_m):
        station = {"position_m": [float(v) for v in position]}
        if instrument_ids is not None:
            station["instrument_id"] = int(instrument_ids[i])
        if track_coordinates is not None and track_coordinates[i] is not None:
            station["track"] = {"s_m": float(track_coordinates[i][0]), "height_m": float(track_coordinates[i][1])}
        if faces is not None and faces[i] is not None:
            station["faces"] = list(faces[i])
        if targets is not None and targets[i] is not None:
            station["targets"] = list(targets[i])
        stations.append(station)
    return parse_plan({"schema_version": SCHEMA_VERSION, "name": name, "author_role": author_role, "source": source,
                       "created": created, "frame": frame, "units": "m", "stations": stations})
