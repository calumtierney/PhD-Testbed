"""Candidate station poses: reducing station placement to a search space a
conventional optimiser can actually search (CLAUDE.md §5, step 6).

CLAUDE.md §3: the optimiser itself is deliberately conventional and held
constant across the step 6 comparison -- grid search (a GA is the
documented fallback "if grid is too slow", not needed here; see
docs/step6_headline_experiment.md). Grid search is only tractable in a
low-dimensional space, and an unconstrained station -- position (3 DOF)
plus orientation (3 DOF) -- is 6-dimensional per station. This module
cuts that to 2 DOF per station: a station stands at a fixed standoff
`radius_m` from a scene's target centroid, at some (azimuth, elevation)
on the hemisphere around it, boresight aimed directly at the centroid.

This isn't just a convenience simplification -- it mirrors how real LVM
planning is usually done too: from a candidate set of plausible,
reachable, roughly-facing-the-part viewpoints, not a search over all of
free space (an unconstrained search would happily suggest standing inside
the part, or facing away from it). Fixing the standoff distance and
aiming direction encodes exactly that domain knowledge, leaving azimuth
and elevation -- "which side, how high" -- as the genuinely open
question a planner has to answer.
"""
from typing import List, Sequence

import numpy as np

from geometry.pose import InstrumentPose


def _look_at_orientation(position_m: np.ndarray, aim_point_m: np.ndarray) -> np.ndarray:
    """Orientation whose local +x axis (this codebase's boresight
    direction -- geometry.spherical's theta = phi = 0) points from
    `position_m` straight at `aim_point_m`."""
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
                    orientation=_look_at_orientation(position_m, centre_m),
                )
            )
    return poses
