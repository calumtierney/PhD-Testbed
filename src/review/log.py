"""The review log: an append-only record of what was compared, what each plan delivered and which one a
reviewer would sign off. **The schema matters more than the interface.** This record is what later work
(including any learning) will be built on, so it is written from day one, whatever front end wrote it.

**Format.** JSON Lines, one record per event, appended with a single write and never rewritten. Every record:

    {"schema_version": 1, "event": "scene_loaded|plan_reviewed|comparison|sign_off|execution_result",
     "timestamp_utc": "...Z", "record_id": "<uuid4 hex>", "reviewer_role": "...", ["reviewer_name": "..."],
     "scene_hash": "...", "plan_hashes": ["..."], "settings": {...}, "provenance": {...}, "payload": {...}}

  scene_loaded      payload: the scene's summary (counts, hashes, clusters; notes only if names are logged).
  plan_reviewed     payload: {"review": <the full PlanReview>}.
  comparison        payload: {"comparison_id", "context_id", "plan_hashes", "rows", "ranking...", "statements",
                    "counterparts": [{plan_hash, kind, objective, n_stations}], "search": {...}}. The per-plan
                    reviews are their own `plan_reviewed` records, joined by plan hash.
  sign_off          payload: {"comparison_id", "chosen_plan_hash" (or null: none acceptable), "reason" (free text)}.
  execution_result  payload: an `ExecutionResult` (below): the Loop 2 input of the thesis (Ch2 §2.2.1), defined and
                    readable now, with no user interface yet.

**What is logged, and what is not.** Plan hashes are frame-independent (`review.core`); settings and provenance
(instrument model, decision rule, priors and whether they are placeholders, commit, scene hash) are in every record
that carries a review. Reviewer *role* is recorded; a *name* only if the reviewer adds one. Names from the scene
(characteristics, targets, the file or plan name, free-text sources and notes) are **left out by default**: a QIF
file's names may belong to a customer. Characteristics are then `char_<k>` (their index, with kind and cluster
aliases), and the scene's own free text is dropped. `ReviewLog(log_names=True)` keeps them. The redaction is by
alias substitution and is best effort: a name written inside a reviewer's own free text is the reviewer's to
leave out, and the safe place for the log is wherever access is controlled.

**Where it lives.** In the directory `LVM_TESTBED_LOG_DIR`, else the user's data directory (outside the repository).
**A directory inside this repository's working tree is refused**: a log may hold a customer's plans.
No network calls, no telemetry; this module opens one local file.

**Reading it back.** `load_log` returns the records and refuses a line with an unknown schema version, an unknown
event, a missing key or invalid JSON, naming the line.

**What `planning.surrogate` would need from these records, and what is missing.** The surrogate learns a plan
evaluator's outputs (coverage, log escape probability, log expected false rejects, rigidity) from station positions
(`layout_features`: the K stations sorted by x, flattened and padded), with a grouped train/validation/test split
so near-duplicate layouts never straddle the split (CLAUDE.md §6). A `plan_reviewed` record has the positions
(`stations[].position_cell_m`), the outcomes (`characteristics[].measured`, `totals.escape_probability`,
`totals.expected_false_rejects`, `feasible`) and a `scene_hash` to group by. Missing, in order of importance:
  1. *Real labels.* Everything in `plan_reviewed` is the model's own prediction; a surrogate trained on it can at
     best reproduce the model. Learning something the model does not contain needs `execution_result` records:
     measured deviations, decisions and outcomes for plans that were executed.
  2. *A scene descriptor.* The surrogate's features are positions only and assume one scene; plans from different
     scenes cannot be pooled without one (part extent, number and type of characteristics, cluster statistics: the
     `scene_loaded` summary has the ingredients, not a fixed-length vector).
  3. *Groups.* Engineers' plans are not generated as jittered families, so the group for the split has to be the
     scene (hold out whole scenes) or the plan's author; `scene_hash` is recorded for that.
  4. *Targets for this evaluator.* The surrogate was built for the wing-box inspection evaluator; the review's
     quantities (a datum-referenced uncertainty per characteristic) are different targets and need their own head.
  5. *Volume.* A handful of reviewed plans is not a training set.
"""
import json
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from review.provenance import REPO_ROOT, digest

LOG_SCHEMA_VERSION = 1
EVENTS = ("scene_loaded", "plan_reviewed", "comparison", "sign_off", "execution_result")
ENV_VAR = "LVM_TESTBED_LOG_DIR"
LOG_FILE_NAME = "review_log.jsonl"
_RECORD_KEYS = {"schema_version", "event", "timestamp_utc", "record_id", "reviewer_role", "scene_hash", "plan_hashes", "settings",
                "provenance", "payload"}
_OPTIONAL_RECORD_KEYS = {"reviewer_name"}


class LogError(ValueError):
    """A log problem: a refused directory, or a line `load_log` will not read."""


# -- where the log lives -----------------------------------------------------------------------
def default_log_dir() -> Path:
    """The user's data directory for this tool, outside the repository: `$XDG_DATA_HOME` or `~/.local/share` on
    Linux, `~/Library/Application Support` on macOS, `%LOCALAPPDATA%` on Windows, then `lvm_testbed/review_logs`."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "lvm_testbed" / "review_logs"


def is_inside_repository(directory: Path) -> bool:
    """Whether `directory` (which need not exist yet) lies inside this repository's working tree, symlinks resolved."""
    resolved = Path(directory).expanduser().resolve()
    root = REPO_ROOT.resolve()
    return resolved == root or root in resolved.parents


def resolve_log_dir(directory: Optional[os.PathLike] = None) -> Path:
    """The log directory: the argument, else `LVM_TESTBED_LOG_DIR`, else the user's data directory. Refuses one inside the
    repository's working tree, and says so."""
    if directory is not None:
        chosen, origin = Path(directory), "the directory given"
    elif os.environ.get(ENV_VAR):
        chosen, origin = Path(os.environ[ENV_VAR]), f"${ENV_VAR}"
    else:
        chosen, origin = default_log_dir(), "the default user data directory"
    chosen = chosen.expanduser()
    if is_inside_repository(chosen):
        raise LogError(f"{origin} ({chosen}) is inside the repository's working tree ({REPO_ROOT}); a review log may hold a customer's "
                       f"plans and must not be where it could be committed. Set {ENV_VAR} to a directory outside the repository.")
    return chosen.resolve()


# -- redaction -----------------------------------------------------------------------------------
def _alias_map(scene) -> Dict[str, str]:
    """name -> alias for every scene name a log should not carry by default."""
    aliases: Dict[str, str] = {}
    for k, name in enumerate(scene.characteristic_names):
        aliases[name] = f"char_{k}"
    for i, name in enumerate(scene.target_names):
        aliases.setdefault(name, f"target_{i}")
    for i, name in scene.feature_names:
        aliases.setdefault(name, f"target_{i}")
    for j, cluster in enumerate(dict.fromkeys(scene.clusters)):
        aliases[cluster] = f"cluster_{j}"
    return aliases


def _substitute(text: str, aliases: Dict[str, str]) -> str:
    for name in sorted(aliases, key=len, reverse=True):
        if name:
            text = text.replace(name, aliases[name])
    return text


def _scrub(obj: Any, aliases: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        return _substitute(obj, aliases)
    if isinstance(obj, dict):
        return {(_substitute(k, aliases) if isinstance(k, str) else k): _scrub(v, aliases) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, aliases) for v in obj]
    return obj


def redact_review(review: Dict[str, Any], scene) -> Dict[str, Any]:
    """A `PlanReview.to_dict()` with the scene's names replaced by aliases and its free text dropped (see the module docstring)."""
    aliases = _alias_map(scene)
    out = _scrub(review, aliases)
    out["plan"]["name"] = "redacted"
    out["plan"]["source"] = "redacted"
    out["provenance"] = _redact_provenance(out["provenance"])
    return out


def _redact_provenance(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Provenance with the scene's label and free-text notes dropped and the clusters' member lists (characteristic names) removed.
    Expects names already scrubbed from the strings that remain."""
    prov = dict(provenance)
    prov["scene"] = {**prov["scene"], "label": "redacted"}
    prov["scene_notes"] = []
    prov["process_priors"] = [{k: v for k, v in c.items() if k != "members"} for c in prov["process_priors"]]
    return prov


def redact_scene_summary(summary: Dict[str, Any], scene) -> Dict[str, Any]:
    out = _scrub(summary, _alias_map(scene))
    out["label"] = "redacted"
    out["notes"] = []
    out["clusters"] = [{k: v for k, v in c.items() if k != "members"} for c in out["clusters"]]
    return out


# -- execution results (Loop 2) -------------------------------------------------------------------
DECISIONS = ("accept", "reject", "not_decided")
OUTCOMES = ("conforming", "nonconforming", "unknown")
_RESIDUAL_KEYS = ("rms_um", "max_um", "n_observations")
_CHARACTERISTIC_KEYS = {"index", "measured_deviation_um", "decision", "outcome"}


@dataclass(frozen=True)
class ExecutionResult:
    """What came back from executing a plan: the Loop 2 input (Ch2 §2.2.1). Characteristics are by *index* (the order
    of the scene's characteristics, as in `PlanReview.characteristics`), so no name needs to be logged.

    network_residuals : the network adjustment's achieved residuals: `rms_um`, `max_um`, `n_observations`.
    characteristics   : one entry per characteristic: `index`, `measured_deviation_um` (or None if not measured),
                        `decision` ("accept", "reject" or "not_decided") and `outcome` ("conforming", "nonconforming"
                        or "unknown": what later inspection or service showed, where known).
    """

    plan_hash: str
    scene_hash: str
    executed_utc: str
    network_residuals: Dict[str, float]
    characteristics: Tuple[Dict[str, Any], ...]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"plan_hash": self.plan_hash, "scene_hash": self.scene_hash, "executed_utc": self.executed_utc,
                "network_residuals": dict(self.network_residuals), "characteristics": [dict(c) for c in self.characteristics],
                "notes": self.notes}


def _number(value: Any, where: str, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LogError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def parse_execution_result(data: Any) -> ExecutionResult:
    """Validate and read an `execution_result` payload. Strict: unknown keys, missing keys and unknown enumerations are refused."""
    if not isinstance(data, dict):
        raise LogError("an execution result is an object")
    keys = {"plan_hash", "scene_hash", "executed_utc", "network_residuals", "characteristics", "notes"}
    if set(data) - keys:
        raise LogError(f"execution_result: unknown key(s) {sorted(set(data) - keys)}")
    if keys - {"notes"} - set(data):
        raise LogError(f"execution_result: missing key(s) {sorted(keys - {'notes'} - set(data))}")
    for key in ("plan_hash", "scene_hash"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise LogError(f"execution_result.{key}: expected a SHA-256 hex digest")
    try:
        datetime.fromisoformat(str(data["executed_utc"]).replace("Z", "+00:00"))
    except ValueError:
        raise LogError(f"execution_result.executed_utc: {data['executed_utc']!r} is not an ISO 8601 date-time") from None
    residuals = data["network_residuals"]
    if not isinstance(residuals, dict) or set(residuals) != set(_RESIDUAL_KEYS):
        raise LogError(f"execution_result.network_residuals: needs exactly {list(_RESIDUAL_KEYS)}")
    parsed_residuals = {k: _number(residuals[k], f"network_residuals.{k}") for k in _RESIDUAL_KEYS}
    if not isinstance(data["characteristics"], list) or not data["characteristics"]:
        raise LogError("execution_result.characteristics: expected a non-empty list")
    rows, seen = [], set()
    for i, row in enumerate(data["characteristics"]):
        where = f"execution_result.characteristics[{i}]"
        if not isinstance(row, dict) or set(row) != _CHARACTERISTIC_KEYS:
            raise LogError(f"{where}: needs exactly {sorted(_CHARACTERISTIC_KEYS)}")
        if isinstance(row["index"], bool) or not isinstance(row["index"], int) or row["index"] < 0 or row["index"] in seen:
            raise LogError(f"{where}.index: expected a distinct non-negative integer")
        seen.add(row["index"])
        if row["decision"] not in DECISIONS or row["outcome"] not in OUTCOMES:
            raise LogError(f"{where}: decision must be one of {list(DECISIONS)} and outcome one of {list(OUTCOMES)}")
        rows.append({"index": row["index"], "measured_deviation_um": _number(row["measured_deviation_um"], where, True),
                     "decision": row["decision"], "outcome": row["outcome"]})
    notes = data.get("notes", "")
    if not isinstance(notes, str):
        raise LogError("execution_result.notes: expected text")
    return ExecutionResult(data["plan_hash"], data["scene_hash"], str(data["executed_utc"]), parsed_residuals, tuple(rows), notes)


# -- the log --------------------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ReviewLog:
    """Append records to the review log. One instance per session; every method appends one record (or several) and returns
    them. Nothing here reads or rewrites an existing line."""

    def __init__(self, directory: Optional[os.PathLike] = None, reviewer_role: str = "unspecified",
                 reviewer_name: Optional[str] = None, log_names: bool = False):
        if not isinstance(reviewer_role, str) or not reviewer_role.strip():
            raise LogError("reviewer_role must be a non-empty string (a role, such as 'metrologist'; not a name)")
        self.directory = resolve_log_dir(directory)
        self.reviewer_role = reviewer_role
        self.reviewer_name = reviewer_name
        self.log_names = log_names

    @property
    def path(self) -> Path:
        return self.directory / LOG_FILE_NAME

    def append(self, event: str, payload: Dict[str, Any], scene_hash: Optional[str] = None, plan_hashes: Sequence[str] = (),
               settings: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Append one record with a single write to a file opened for appending. Returns the record."""
        if event not in EVENTS:
            raise LogError(f"unknown event {event!r}; the events are {list(EVENTS)}")
        record: Dict[str, Any] = {
            "schema_version": LOG_SCHEMA_VERSION, "event": event, "timestamp_utc": _now(), "record_id": uuid.uuid4().hex,
            "reviewer_role": self.reviewer_role, "scene_hash": scene_hash, "plan_hashes": list(plan_hashes),
            "settings": settings or {}, "provenance": provenance or {}, "payload": payload}
        if self.reviewer_name:
            record["reviewer_name"] = self.reviewer_name
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False, default=float) + "\n"
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return record

    # -- the five events -------------------------------------------------------------------------
    def log_scene_loaded(self, scene, settings=None) -> Dict[str, Any]:
        summary = scene.summary()
        payload = {"scene": summary if self.log_names else redact_scene_summary(summary, scene)}
        return self.append("scene_loaded", payload, summary["scene_hash"], (), settings.to_dict() if settings else None)

    def log_plan_reviewed(self, review, scene) -> Dict[str, Any]:
        data = review.to_dict()
        shown = data if self.log_names else redact_review(data, scene)
        return self.append("plan_reviewed", {"review": shown}, review.provenance["scene"]["scene_hash"], [review.plan["plan_hash"]],
                           review.provenance["settings"], shown["provenance"])

    def log_comparison(self, comparison: Dict[str, Any], scene, counterparts=None, search: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """`comparison` is `review.core.compare_plans`' result; `counterparts` an iterable of `review.counterparts.Counterpart`."""
        aliases = {} if self.log_names else _alias_map(scene)
        plan_hashes = [row["plan_hash"] for row in comparison["rows"]]
        rows = _scrub(comparison["rows"], aliases)
        if not self.log_names:
            for row in rows:
                row["name"] = "redacted"
        payload = {
            "comparison_id": comparison_id(comparison), "context_id": comparison["context_id"], "plan_hashes": plan_hashes,
            "rows": rows, "ranking_by_expected_wrong_decisions": comparison["ranking_by_expected_wrong_decisions"],
            "risk_if_nothing_is_measured": comparison["risk_if_nothing_is_measured"], "statements": comparison["statements"],
            "counterparts": [{"plan_hash": c.review.plan["plan_hash"], "kind": c.kind, "objective": c.objective, "n_stations": c.n_stations}
                             for c in (counterparts or ())],
            "search": search or {}}
        prov = comparison["provenance"]
        prov = prov if self.log_names else _redact_provenance(_scrub(prov, aliases))
        return self.append("comparison", payload, comparison["provenance"]["scene"]["scene_hash"], plan_hashes,
                           comparison["settings"], prov)

    def log_sign_off(self, comparison: Dict[str, Any], chosen_plan_hash: Optional[str], reason: str) -> Dict[str, Any]:
        """The reviewer's choice among the compared plans (None: none is acceptable) and why. `reason` is the reviewer's own text and is
        logged as written."""
        hashes = [row["plan_hash"] for row in comparison["rows"]]
        if chosen_plan_hash is not None and chosen_plan_hash not in hashes:
            raise LogError("chosen_plan_hash is not one of the compared plans")
        if not isinstance(reason, str) or not reason.strip():
            raise LogError("a sign-off needs a reason (free text)")
        payload = {"comparison_id": comparison_id(comparison), "chosen_plan_hash": chosen_plan_hash, "reason": reason}
        return self.append("sign_off", payload, comparison["provenance"]["scene"]["scene_hash"], hashes, comparison["settings"])

    def log_execution_result(self, result: ExecutionResult, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        parse_execution_result(result.to_dict())  # refuse what the reader would refuse
        return self.append("execution_result", result.to_dict(), result.scene_hash, [result.plan_hash], settings)

    def log_session(self, scene, settings, reviews: Sequence, comparison: Dict[str, Any], counterparts=None,
                    search: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """What one comparison writes: the scene, each plan's review, and the comparison. Returns the records."""
        records = [self.log_scene_loaded(scene, settings)]
        records += [self.log_plan_reviewed(r, scene) for r in reviews]
        records.append(self.log_comparison(comparison, scene, counterparts, search))
        return records


def comparison_id(comparison: Dict[str, Any]) -> str:
    """Identifies a comparison: the context (scene, settings, scorer) and the plans compared, in order."""
    return digest(comparison["context_id"], [row["plan_hash"] for row in comparison["rows"]])


# -- reading ----------------------------------------------------------------------------------------
def load_log(directory: Optional[os.PathLike] = None) -> List[Dict[str, Any]]:
    """Every record in the log, oldest first. Refuses (naming the line) invalid JSON, a missing or extra key, an unknown event
    and any `schema_version` other than the one this reader knows (`LOG_SCHEMA_VERSION`). An absent log is an empty one."""
    path = resolve_log_dir(directory) / LOG_FILE_NAME
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise LogError(f"{path.name} line {number}: not valid JSON ({error})") from None
            if not isinstance(record, dict) or record.get("schema_version") != LOG_SCHEMA_VERSION:
                version = record.get("schema_version") if isinstance(record, dict) else None
                raise LogError(f"{path.name} line {number}: schema_version {version!r} is not supported (this reader reads {LOG_SCHEMA_VERSION})")
            keys = set(record)
            if not _RECORD_KEYS <= keys or keys - _RECORD_KEYS - _OPTIONAL_RECORD_KEYS:
                raise LogError(f"{path.name} line {number}: record keys are {sorted(keys)}, expected {sorted(_RECORD_KEYS)}")
            if record["event"] not in EVENTS:
                raise LogError(f"{path.name} line {number}: unknown event {record['event']!r}")
            if record["event"] == "execution_result":
                parse_execution_result(record["payload"])
            records.append(record)
    return records


def execution_results(records: Iterable[Dict[str, Any]]) -> List[ExecutionResult]:
    """The `execution_result` records, parsed (the Loop 2 inputs)."""
    return [parse_execution_result(r["payload"]) for r in records if r["event"] == "execution_result"]
