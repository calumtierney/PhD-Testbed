"""Candidate station poses: reducing station placement to a search space a
conventional optimiser can actually search (CLAUDE.md §5, step 6).

CLAUDE.md §3: the optimiser itself is deliberately conventional and held
constant across the step 6 comparison -- grid search, or a standard GA
once grid search's `C(n_candidates, n_stations)` cost stops being
tractable (`planning.search.genetic_algorithm_station_placement`; see
that module's docstring for when that crossover happens). Both need
station placement cut down from its full 6 degrees of freedom (3
position + 3 orientation) to something a conventional optimiser can
actually search. This module offers two ways to do that:

  - `hemisphere_candidate_poses` -- 2 DOF per station: a fixed standoff
    `radius_m` from a scene's target centroid, azimuth and elevation on
    the hemisphere around it, boresight aimed at the centroid. Good for
    a compact scene where "which side, how high, at a sensible standoff"
    is genuinely the only open question -- CLAUDE.md §5 step 6's original
    two-cluster scene, a few metres across.
  - `floor_grid_candidate_poses` -- 3 DOF per station: real (x, y)
    floor position plus a station height, each boresighted at an aim
    point, again reducing the open question to "where on the floor" and
    "how tall a tripod" rather than a full 6-DOF search. Needed once a
    scene is large enough, or occluded enough, that *distance* to the
    part -- not just angle around a fixed ring -- is itself part of what
    a placement decision should be free to vary (a 30 m wing is not well
    served by one fixed standoff radius: a station near enough to see one
    end clearly may be much too far from the other).

Either way, this isn't just a convenience simplification -- it mirrors
how real LVM planning is usually done too: from a candidate set of
plausible, reachable, roughly-facing-the-part viewpoints, not a search
over all of free space (an unconstrained search would happily suggest
standing inside the part, or facing away from it).
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose
from geometry.station_track import StationTrack


def look_at_orientation(position_m: np.ndarray, aim_point_m: np.ndarray) -> np.ndarray:
    """Orientation whose local +x axis (this codebase's boresight
    direction -- geometry.spherical's theta = phi = 0) points from
    `position_m` straight at `aim_point_m`.

    Public (not `_`-prefixed): both candidate generators below use it,
    and so does `planning.search.genetic_algorithm_station_placement`,
    which needs to turn a raw (x, y, z) position the optimiser is
    searching over into a full pose the same way a candidate list does.
    """
    x_axis = aim_point_m - position_m
    x_axis = x_axis / np.linalg.norm(x_axis)
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference_up, x_axis)) > 0.97:
        reference_up = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference_up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def hemisphere_candidate_poses(
    centre_m: np.ndarray,
    radius_m: float,
    azimuths_deg: Sequence[float],
    elevations_deg: Sequence[float],
) -> List[InstrumentPose]:
    """Every (azimuth, elevation) combination, as a station pose standing
    `radius_m` back from `centre_m` and boresighted at it.

    Azimuth 0 deg is along +x from the centre; elevation 0 deg is level
    with the centre, 90 deg is directly overhead. `len(azimuths_deg) *
    len(elevations_deg)` poses are returned, in azimuth-major order.
    """
    centre_m = np.asarray(centre_m, dtype=float)
    poses = []
    for az_deg in azimuths_deg:
        for el_deg in elevations_deg:
            az = np.deg2rad(az_deg)
            el = np.deg2rad(el_deg)
            # Direction *from the centre out to the station*: elevation 0
            # is level with the centre, elevation 90 points straight up,
            # so the station itself ends up above the centre, looking
            # down -- not below it looking up.
            direction_centre_to_station = np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
            )
            position_m = centre_m + radius_m * direction_centre_to_station
            poses.append(
                InstrumentPose(
                    position_m=position_m,
                    orientation=look_at_orientation(position_m, centre_m),
                )
            )
    return poses


def floor_grid_candidate_poses(
    aim_point_m: np.ndarray,
    x_positions_m: Sequence[float],
    y_positions_m: Sequence[float],
    station_heights_m: Sequence[float],
    exclusion_radius_m: float = 0.0,
) -> List[InstrumentPose]:
    """Every (x, y, height) combination on an explicit floor-plan grid, as
    a candidate station boresighted at `aim_point_m` -- see the module
    docstring for why this exists alongside `hemisphere_candidate_poses`.

    `len(x_positions_m) * len(y_positions_m) * len(station_heights_m)`
    candidates are returned (minus any dropped by `exclusion_radius_m`),
    in x-major, then y-, then height-order. Unlike
    `hemisphere_candidate_poses`, nothing here assumes the candidates sit
    at one shared distance from the part: `x_positions_m`/`y_positions_m`
    are free to span the whole reachable floor area, letting a search
    trade off distance against angle, not just search angle at a fixed
    distance.

    `exclusion_radius_m` drops any candidate within this distance of
    `aim_point_m` -- guards against a grid point landing on top of (or
    inside) the part itself, which both `look_at_orientation` (a
    zero-length boresight vector) and the spherical observation model (a
    near-zero range) handle badly. 0.0 (the default) applies no
    exclusion; pass something like half the part's smallest dimension
    for a grid that might otherwise span the part's own footprint.
    """
    aim_point_m = np.asarray(aim_point_m, dtype=float)
    poses = []
    for x in x_positions_m:
        for y in y_positions_m:
            for z in station_heights_m:
                position_m = np.array([x, y, z])
                if np.linalg.norm(position_m - aim_point_m) < exclusion_radius_m:
                    continue
                poses.append(
                    InstrumentPose(
                        position_m=position_m,
                        orientation=look_at_orientation(position_m, aim_point_m),
                    )
                )
    return poses


def levelled_orientation(position_m: np.ndarray, aim_point_m: np.ndarray) -> np.ndarray:
    """Orientation of a *levelled* tracker: local z is world up (the
    instrument's vertical axis is gravity), and local x is the horizontal
    direction from `position_m` towards `aim_point_m`.

    A real laser tracker is levelled on its tripod, so only its yaw is free;
    it does not tilt to point at the part, and the aim point's height is
    irrelevant. This matters because the error model's azimuth and elevation
    encoders have different noise (0.485 vs 0.694 arcsec): a tilted frame
    swaps the roles of the two and so rotates the larger lateral error.

    **Yaw itself has no effect on any covariance this codebase computes.** The
    azimuth encoder covers 360 degrees with the same noise at every angle, so
    turning the whole instrument about the vertical leaves the global-frame
    covariance of every target unchanged (tests/test_station_track.py checks
    this to rounding error). The choice of aim point here therefore only fixes
    which way the local x axis points on a figure; it cannot change a result.

    When `aim_point_m` is at the station's own height this is identical to
    `look_at_orientation`. If the aim is directly above or below the station
    the horizontal direction is undefined and local x defaults to +x.
    """
    horizontal = np.array([aim_point_m[0] - position_m[0], aim_point_m[1] - position_m[1], 0.0])
    norm = np.linalg.norm(horizontal)
    x_axis = horizontal / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


@dataclass
class TrackPose(InstrumentPose):
    """An `InstrumentPose` that also remembers where on its track it stands:
    arc length `s_m`, `height_m` and which `segment` (see
    `geometry.station_track`). Usable anywhere an `InstrumentPose` is, so
    results can be reported in track coordinates, not only (x, y, z)."""

    s_m: float = 0.0
    height_m: float = 0.0
    segment: str = ""


def track_s_values(track: StationTrack, spacing_m: float) -> np.ndarray:
    """Arc-length positions `spacing_m` apart around the whole track.

    The perimeter is rarely a whole number of spacings, so the number of
    points is round(perimeter / spacing) and they are spread *uniformly*
    (actual spacing = perimeter / n, within a few percent of `spacing_m`):
    the gap across the s = 0 join is the same as everywhere else. s = 0 is
    included. Blocked intervals are not applied here."""
    n = max(1, int(round(track.perimeter_m / spacing_m)))
    return np.arange(n) * (track.perimeter_m / n)


def track_candidate_poses(
    track: StationTrack,
    spacing_m: float,
    station_heights_m: Sequence[float],
    aim_point_m: Optional[np.ndarray] = None,
    aim_height_m: float = 1.25,
    orientation_mode: str = "levelled",
) -> List[TrackPose]:
    """Candidate stations every `spacing_m` along `track` (uniformly, see
    `track_s_values`) at each of `station_heights_m`, skipping blocked
    intervals. Ordered by s, then by height; each carries `(s_m, height_m,
    segment)`.

    `orientation_mode`:
      "levelled" (the default here) -- a real, levelled tracker: local z up,
          local x horizontal towards the aim point (`levelled_orientation`).
      "look_at" -- tilted so the boresight points exactly at the aim point
          (`look_at_orientation`, what the ring candidates do).

    `aim_point_m` None means the nearest point on the part's centreline
    (x clamped to the part's extent, y = the part's centre, z =
    `aim_height_m`; a placeholder, TODO(source): mid-height of the real part).
    It sets only the local x direction on a figure when levelled, and has no
    effect on any result (yaw invariance); it does matter for "look_at".
    """
    if orientation_mode not in ("levelled", "look_at"):
        raise ValueError(f"orientation_mode must be 'levelled' or 'look_at', got {orientation_mode!r}")
    make_orientation = levelled_orientation if orientation_mode == "levelled" else look_at_orientation
    poses: List[TrackPose] = []
    for s in track_s_values(track, spacing_m):
        if track.is_blocked(s):
            continue
        x, y = track.position_xy_m(s)
        for height in station_heights_m:
            position = np.array([x, y, height])
            if aim_point_m is None:
                aim = np.array([np.clip(x, track.part_x_min_m, track.part_x_max_m), track.part_centre_y_m, aim_height_m])
            else:
                aim = np.asarray(aim_point_m, dtype=float)
            poses.append(
                TrackPose(
                    position_m=position, orientation=make_orientation(position, aim),
                    s_m=float(s), height_m=float(height), segment=track.segment(s),
                )
            )
    return poses
