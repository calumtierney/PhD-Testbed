"""A demonstration plan, clearly labelled as what it is: synthetic.

`demo_plan` builds a plan by a stated rule that looks at nothing about the part: stations evenly spaced in
arc length along the scene's track, at one height. Its `author_role` is "synthetic" and its source says so.
It exists so the comparison can run end to end before a real plan is available. **It is not an engineer's
plan**, and no output calls it one.
"""
from datetime import datetime, timezone

from interchange.plan_file import PlanFile, plan_file_from_stations
from review.scene import ReviewScene


def demo_plan(scene: ReviewScene, n_stations: int = 4, height_m: float = 1.25, start_fraction: float = 0.5) -> PlanFile:
    """`n_stations` stations evenly spaced in arc length round the scene's track, the k-th at
    s = (k + `start_fraction`) * perimeter / n_stations, all at `height_m`, in cell coordinates."""
    if n_stations < 1:
        raise ValueError("n_stations must be at least 1")
    track = scene.track
    positions, coordinates = [], []
    for k in range(n_stations):
        s = (k + start_fraction) * track.perimeter_m / n_stations
        x, y = track.position_xy_m(s)
        positions.append([x, y, height_m])
        coordinates.append((s, height_m))
    rule = (f"SYNTHETIC demonstration: {n_stations} stations evenly spaced in arc length round the track at {height_m:g} m, "
            f"s = (k + {start_fraction:g}) x perimeter / {n_stations}; a rule that does not look at the part. Not an engineer's plan.")
    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return plan_file_from_stations(f"synthetic: {n_stations} evenly spaced at {height_m:g} m", "synthetic", rule, created, "cell",
                                   positions, track_coordinates=coordinates)
