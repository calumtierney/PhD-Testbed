"""Visibility: can a target actually be measured from a given instrument pose?

Build step 2 (CLAUDE.md §5), the second link in the chain (§2):

    station pose -> visibility -> instrument error model -> ...

This module answers "can it be measured at all", strictly before the
question step 1 answers ("how precisely"). A target can fail visibility
for three independent geometric reasons, checked in this order:

  1. **Range** -- the target is further from the instrument than its
     maximum working range.
  2. **Incidence** -- the beam meets the target's own surface too close
     to edge-on. A laser tracker's return signal comes from light
     reflecting/scattering off (or through, for a retroreflector) the
     target; hit a surface at a shallow enough angle and the return is
     unreliable or absent, regardless of range. This needs to know which
     way the target's surface faces -- its normal -- which is why
     `Scene.target_normals_m` exists.
  3. **Occlusion** -- something in the scene physically blocks the
     straight-line path from the instrument to the target (a wall, a
     fixture, another part of the structure being measured).

Range and incidence are cheap, O(1)-per-target checks; occlusion is the
expensive one; O(n_triangles) per target, since it means testing the
sightline against every triangle in the obstacle mesh (see
`geometry.mesh`). Checking range and incidence first, and only ray
casting for targets that still might be visible, means an out-of-range or
grazing target never pays for a mesh test it can't possibly need. This is
purely a performance ordering, not a physical one -- CLAUDE.md §2 lists
occlusion first because it's conceptually the most obviously "geometric"
of the three, but the physics doesn't require checking them in any
particular order, since all three are independent yes/no conditions.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from geometry.mesh import segment_intersects_mesh
from geometry.scene import Scene


class VisibilityReason(Enum):
    """Why a target is or isn't measurable. VISIBLE is the only "pass"."""

    VISIBLE = "visible"
    OUT_OF_RANGE = "out_of_range"
    INCIDENCE_TOO_STEEP = "incidence_too_steep"
    OCCLUDED = "occluded"


@dataclass(frozen=True)
class VisibilityLimits:
    """The instrument's operating envelope for visibility purposes.

    Attributes
    ----------
    max_range_m : float
        Targets further than this from the instrument are unmeasurable
        regardless of what else is true about them.
    max_incidence_angle_rad : float
        The largest angle, between the beam and the target surface
        normal, at which the instrument can still get a usable
        measurement. 0 rad is straight-on (beam parallel to the normal);
        pi/2 rad is a beam grazing along the surface.
    """

    max_range_m: float
    max_incidence_angle_rad: float


@dataclass(frozen=True)
class VisibilityResult:
    """The outcome of a visibility check for one target."""

    reason: VisibilityReason

    @property
    def is_visible(self) -> bool:
        return self.reason is VisibilityReason.VISIBLE


def check_target_visibility(
    instrument_position_m: np.ndarray,
    target_point_m: np.ndarray,
    limits: VisibilityLimits,
    target_normal_m: np.ndarray = None,
    obstacles=None,
) -> VisibilityResult:
    """Visibility of a single target from a single instrument position.

    `target_normal_m` (outward unit surface normal) is required to run
    the incidence check; if it's None, that check is skipped entirely
    (treated as "unconstrained"), not treated as a failure. `obstacles`
    (a `geometry.mesh.TriangleMesh`) is required to run the occlusion
    check; if it's None, there is nothing to be occluded by.
    """
    instrument_position_m = np.asarray(instrument_position_m, dtype=float)
    target_point_m = np.asarray(target_point_m, dtype=float)

    beam_m = target_point_m - instrument_position_m
    range_m = np.linalg.norm(beam_m)
    if range_m > limits.max_range_m:
        return VisibilityResult(VisibilityReason.OUT_OF_RANGE)

    if target_normal_m is not None:
        # Angle between the surface normal (pointing away from the
        # material, out into free space) and the direction back from the
        # target to the instrument. A beam arriving straight-on has the
        # instrument sitting right along the normal (angle ~ 0); a beam
        # grazing the surface has the instrument almost in the surface's
        # own plane (angle ~ pi/2). Using -beam_m (target -> instrument)
        # rather than beam_m (instrument -> target) is what makes
        # "straight-on" come out as 0 rather than pi.
        target_normal_m = np.asarray(target_normal_m, dtype=float)
        cos_incidence = np.dot(-beam_m, target_normal_m) / range_m
        incidence_angle_rad = np.arccos(np.clip(cos_incidence, -1.0, 1.0))
        if incidence_angle_rad > limits.max_incidence_angle_rad:
            return VisibilityResult(VisibilityReason.INCIDENCE_TOO_STEEP)

    if obstacles is not None and segment_intersects_mesh(
        instrument_position_m, target_point_m, obstacles
    ):
        return VisibilityResult(VisibilityReason.OCCLUDED)

    return VisibilityResult(VisibilityReason.VISIBLE)


def scene_visibility(scene: Scene, limits: VisibilityLimits) -> list:
    """Visibility of every target in a scene, from its one instrument station.

    Explicit loop over targets, matching `instruments.laser_tracker.
    scene_point_covariances` -- see that function's docstring for why
    (CLAUDE.md §6: clarity over vectorisation while target counts are
    small).

    Returns
    -------
    list[VisibilityResult], one per target, in `scene.target_points_m` order.
    """
    results = []
    for i in range(scene.n_points):
        normal_m = None if scene.target_normals_m is None else scene.target_normals_m[i]
        results.append(
            check_target_visibility(
                scene.instrument_pose.position_m,
                scene.target_points_m[i],
                limits,
                target_normal_m=normal_m,
                obstacles=scene.obstacles,
            )
        )
    return results
