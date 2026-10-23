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


def visibility_matrix(
    station_positions_m: np.ndarray,
    target_points_m: np.ndarray,
    limits: VisibilityLimits,
    target_normals_m: np.ndarray = None,
    obstacles=None,
    occlusion_target_offset_m: float = 0.0,
) -> np.ndarray:
    """Which stations can measure which targets, as an (S, N) boolean matrix.

    The same three checks as `check_target_visibility` -- range, incidence,
    occlusion -- applied to every station-target pair at once with array
    operations (the occlusion test is the same Moller-Trumbore segment/
    triangle test as `geometry.mesh.segment_intersects_mesh`, batched over
    all targets and all triangles per station). Exists because planning a
    ~500-point inspection scores tens of thousands of candidate layouts,
    and the scalar version loops in Python over every target and every
    triangle; `tests/test_visibility.py` pins this to the scalar version.

    `occlusion_target_offset_m` lifts the *target end* of each sightline
    this far along the target's own normal before the occlusion test (and
    only for it). A point that lies on the part's own surface is the
    endpoint of its sightline; a mesh that approximates that surface can
    sit a fraction of a millimetre on the wrong side of it and block the
    ray against the part's own skin. Default 0.0 reproduces the scalar
    function exactly; callers whose obstacle mesh is a faceted stand-in
    for the surface the points lie on should pass a small positive offset.
    """
    stations = np.atleast_2d(np.asarray(station_positions_m, dtype=float))
    targets = np.atleast_2d(np.asarray(target_points_m, dtype=float))
    n_stations, n_targets = len(stations), len(targets)

    beams = targets[None, :, :] - stations[:, None, :]  # (S, N, 3), station -> target
    range_m = np.linalg.norm(beams, axis=2)
    visible = range_m <= limits.max_range_m

    if target_normals_m is not None:
        normals = np.atleast_2d(np.asarray(target_normals_m, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_incidence = np.einsum("snk,nk->sn", -beams, normals) / range_m
        incidence = np.arccos(np.clip(cos_incidence, -1.0, 1.0))
        visible &= incidence <= limits.max_incidence_angle_rad

    if obstacles is None or obstacles.n_triangles == 0:
        return visible

    aim = targets
    if occlusion_target_offset_m != 0.0 and target_normals_m is not None:
        aim = targets + occlusion_target_offset_m * np.atleast_2d(np.asarray(target_normals_m, dtype=float))

    v0 = obstacles.vertices_m[obstacles.faces[:, 0]]
    edge1 = obstacles.vertices_m[obstacles.faces[:, 1]] - v0  # (K, 3)
    edge2 = obstacles.vertices_m[obstacles.faces[:, 2]] - v0
    epsilon, t_margin = 1e-9, 1e-6

    for s in range(n_stations):
        candidates = np.nonzero(visible[s])[0]  # ray-cast only what range/incidence left
        if len(candidates) == 0:
            continue
        direction = aim[candidates] - stations[s]  # (M, 3)
        h = np.cross(direction[:, None, :], edge2[None, :, :])  # (M, K, 3)
        a = np.einsum("kj,mkj->mk", edge1, h)
        parallel = np.abs(a) < epsilon
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.0 / a
            s_vec = stations[s] - v0  # (K, 3)
            u = f * np.einsum("kj,mkj->mk", s_vec, h)
            q = np.cross(s_vec, edge1)  # (K, 3)
            v = f * np.einsum("mj,kj->mk", direction, q)
            t = f * np.einsum("kj,kj->k", edge2, q)[None, :]
        with np.errstate(invalid="ignore"):  # NaN where the ray is parallel; masked out by ~parallel
            hit = (
                ~parallel & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
                & (t > t_margin) & (t < 1.0 - t_margin)
            )
        visible[s, candidates[hit.any(axis=1)]] = False
    return visible
