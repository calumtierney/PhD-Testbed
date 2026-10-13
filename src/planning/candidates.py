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
from typing import List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose


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
