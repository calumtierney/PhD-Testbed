"""Exporting a comparison: one JSON document (the comparison, every plan's full review, the counterparts' provenance) and, from
`viz`, its figures.

The export is the reviewer's own copy and carries names as given (unlike the log, which leaves scene names out by default). It is
written only where the reviewer says, and **never inside this repository's working tree** (the same refusal as the log): an export
may hold a customer's plans, and `results/` is for the testbed's own experiments.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from review.core import PlanReview
from review.log import LogError, is_inside_repository


def comparison_bundle(comparison: Dict[str, Any], reviews: Sequence[PlanReview], counterparts=None,
                      search: Optional[Dict[str, Any]] = None, not_found: Sequence[str] = ()) -> Dict[str, Any]:
    """Everything a comparison produced, as one JSON-serialisable dict."""
    return {
        "comparison": comparison,
        "reviews": [r.to_dict() for r in reviews],
        "counterparts": [{"plan_hash": c.review.plan["plan_hash"], "kind": c.kind, "objective": c.objective, "n_stations": c.n_stations,
                          "plan_file": c.plan.to_dict()} for c in (counterparts or ())],
        "counterparts_not_found": list(not_found), "search": search or {},
    }


def bundle_json(bundle: Dict[str, Any]) -> str:
    return json.dumps(bundle, indent=1, allow_nan=False, default=float)


def write_bundle(directory: os.PathLike, bundle: Dict[str, Any], files: Optional[Dict[str, bytes]] = None) -> List[str]:
    """Write `comparison.json` (the bundle) and any `files` (name -> bytes, e.g. the figures) into `directory`. Refuses a directory
    inside the repository's working tree."""
    directory = Path(directory).expanduser()
    if is_inside_repository(directory):
        raise LogError(f"{directory} is inside the repository's working tree; an export may hold a customer's plans. Choose a directory outside it.")
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    (directory / "comparison.json").write_text(bundle_json(bundle), encoding="utf-8")
    written.append(str(directory / "comparison.json"))
    for name, data in (files or {}).items():
        (directory / name).write_bytes(data)
        written.append(str(directory / name))
    return written
