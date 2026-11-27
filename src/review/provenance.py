"""Where a review's numbers came from: hashes, the commit, and what the instrument model was."""
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

from instruments.laser_tracker import LaserTracker

REPO_ROOT = Path(__file__).resolve().parents[2]


def digest(*parts: Any) -> str:
    """SHA-256 of numbers, arrays, text and nested lists/dicts, in a fixed encoding
    (floats are hashed by their IEEE-754 bytes, so a different value is a different
    hash, and a re-read file with the same values is the same)."""
    h = hashlib.sha256()

    def feed(obj):
        if obj is None:
            h.update(b"N")
        elif isinstance(obj, (bool, np.bool_)):
            h.update(b"B1" if obj else b"B0")
        elif isinstance(obj, (int, np.integer)):
            h.update(b"I" + str(int(obj)).encode())
        elif isinstance(obj, (float, np.floating)):
            h.update(b"F" + np.float64(obj).tobytes())
        elif isinstance(obj, str):
            h.update(b"S" + obj.encode() + b"\0")
        elif isinstance(obj, np.ndarray):
            h.update(b"A" + str(obj.shape).encode() + np.ascontiguousarray(obj, dtype="<f8").tobytes())
        elif isinstance(obj, dict):
            h.update(b"{")
            for key in sorted(obj):
                feed(str(key))
                feed(obj[key])
            h.update(b"}")
        elif isinstance(obj, (list, tuple)):
            h.update(b"[")
            for item in obj:
                feed(item)
            h.update(b"]")
        else:
            raise TypeError(f"cannot hash {type(obj).__name__}")

    for part in parts:
        feed(part)
    return h.hexdigest()


def settings_digest(settings_dict: Dict[str, Any]) -> str:
    return digest(settings_dict)


def git_state() -> Dict[str, Any]:
    """The commit this code is at and whether the working tree has uncommitted changes. Reads the
    local repository only (no network); `unknown` where this is not a git checkout."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def tracker_description(tracker: LaserTracker) -> Dict[str, Any]:
    """What the instrument model is, for a record: the sensor noise, the floor, and which geometric
    parameter set (if any), identified by a hash of its values and covariance."""
    params = tracker.geometric_params
    carries = params is not None and bool(params.covariance.any())
    return {
        "sigma_d_um": tracker.sigma_d_m * 1e6,
        "sigma_azimuth_arcsec": float(np.rad2deg(tracker.sigma_theta_rad) * 3600.0),
        "sigma_elevation_arcsec": float(np.rad2deg(tracker.sigma_phi_rad) * 3600.0),
        "systematic_floor_um": tracker.systematic_std_m * 1e6,
        "geometric_parameters": ("none (sensor noise only)" if not carries else
                                 "NPL 16-parameter model, consider mode (shared by every station of a tracker)"),
        "parameter_set_hash": None if not carries else digest(params.values, params.covariance),
        "floor_excludes_geometric_parameters": tracker.floor_excludes_geometric_parameters,
    }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
