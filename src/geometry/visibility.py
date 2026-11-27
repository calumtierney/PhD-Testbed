"""Visibility: can a target actually be measured from a given instrument pose?

Build step 2 (CLAUDE.md §5), the second link in the chain (§2):

    station pose -> visibility -> instrument error model -> ...

This module answers "can it be measured at all", strictly before the
question step 1 answers ("how precisely"). A target can fail visibility
for four independent geometric reasons, checked in this order:

  1. **Range** -- the target is further from the instrument than its
     maximum working range.
  2. **Elevation** (optional) -- the beam points outside the tracker's
     mechanical elevation range. A laser tracker's head tilts about a
     horizontal axis through a limited arc (Hughes et al. 2011 §8: -60
     to +77 degrees for the API T3), so it cannot look straight up at a
     ceiling target or steeply down at the floor beneath it. Azimuth is
     a full 360 degrees and never limits anything. Elevation is measured
     in the *instrument's own frame* (above its local x-y plane, the same
     convention as `geometry.spherical`), so it needs the station's
     orientation; a levelled tracker's local z is world up. Off by
     default (`VisibilityLimits.min_elevation_rad`/`max_elevation_rad`
     None) so every result computed before this check existed is
     unchanged.
  3. **Incidence** -- the beam meets the target's own surface too close
     to edge-on. A laser tracker's return signal comes from light
     reflecting/scattering off (or through, for a retroreflector) the
     target; hit a surface at a shallow enough angle and the return is
     unreliable or absent, regardless of range. This needs to know which
     way the target's surface faces -- its normal -- which is why
     `Scene.target_normals_m` exists.
  4. **Occlusion** -- something in the scene physically blocks the
     straight-line path from the instrument to the target (a wall, a
     fixture, another part of the structure being measured).

Range, elevation and incidence are cheap, O(1)-per-target checks;
occlusion is the expensive one; O(n_triangles) per target, since it means
testing the sightline against every triangle in the obstacle mesh (see
`geometry.mesh`). Checking the cheap three first, and only ray casting
for targets that still might be visible, means an out-of-range or
grazing target never pays for a mesh test it can't possibly need. This is
purely a performance ordering, not a physical one -- CLAUDE.md §2 lists
occlusion first because it's conceptually the most obviously "geometric"
of them, but the physics doesn't require checking them in any particular
order, since all are independent yes/no conditions.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from geometry.mesh import segment_intersects_mesh
from geometry.scene import Scene


class VisibilityReason(Enum):
    """Why a target is or isn't measurable. VISIBLE is the only "pass"."""

    VISIBLE = "visible"
    OUT_OF_RANGE = "out_of_range"
    ELEVATION_OUT_OF_RANGE = "elevation_out_of_range"
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
    min_elevation_rad, max_elevation_rad : float or None
        The instrument's mechanical elevation range, measured above the
        instrument's local x-y plane (so a levelled tracker's horizon is 0
        and straight up is +pi/2). A target whose beam elevation lies outside
        [min, max] is unmeasurable. None (the default) means that side is
        unconstrained, and None on both reproduces the behaviour from before
        this check existed. See `API_T3_MIN_ELEVATION_RAD` for the values.
    """

    max_range_m: float
    max_incidence_angle_rad: float
    min_elevation_rad: Optional[float] = None
    max_elevation_rad: Optional[float] = None

    def __post_init__(self):
        for name in ("min_elevation_rad", "max_elevation_rad"):
            value = getattr(self, name)
            if value is not None and not -np.pi / 2 <= value <= np.pi / 2:
                raise ValueError(f"{name} must lie in [-pi/2, pi/2] rad (an elevation), got {value!r}")
        if (
            self.min_elevation_rad is not None
            and self.max_elevation_rad is not None
            and self.min_elevation_rad >= self.max_elevation_rad
        ):
            raise ValueError(
                f"min_elevation_rad ({self.min_elevation_rad!r}) must be below max_elevation_rad "
                f"({self.max_elevation_rad!r})"
            )

    @property
    def has_elevation_limits(self) -> bool:
        return self.min_elevation_rad is not None or self.max_elevation_rad is not None


# Mechanical elevation range of the instrument in the Hughes et al. 2011
# network test (API T3): -60 to +77 degrees (CLAUDE.md §4f, "Network ...
# elevations no higher than 57 degrees (mechanical range -60 to +77)").
# TODO(source): GKN's own tracker (Leica AT-series or similar) has its own
# range; replace these when the instrument is known. Passed explicitly to
# `VisibilityLimits`, never applied silently.
API_T3_MIN_ELEVATION_RAD = float(np.deg2rad(-60.0))
API_T3_MAX_ELEVATION_RAD = float(np.deg2rad(77.0))


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
    instrument_orientation: np.ndarray = None,
) -> VisibilityResult:
    """Visibility of a single target from a single instrument position.

    `instrument_orientation` ((3, 3), `v_global = orientation @ v_local`, as
    `geometry.pose.InstrumentPose.orientation`) is used only by the elevation
    check, and only when `limits` carries elevation limits. None means the
    identity: the instrument's local z is world up, i.e. a levelled tracker.

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

    if limits.has_elevation_limits and range_m > 0.0:
        orientation = np.eye(3) if instrument_orientation is None else np.asarray(instrument_orientation, dtype=float)
        local_z_m = float(orientation[:, 2] @ beam_m)  # beam component along the instrument's vertical axis
        elevation_rad = np.arcsin(np.clip(local_z_m / range_m, -1.0, 1.0))
        if limits.min_elevation_rad is not None and elevation_rad < limits.min_elevation_rad:
            return VisibilityResult(VisibilityReason.ELEVATION_OUT_OF_RANGE)
        if limits.max_elevation_rad is not None and elevation_rad > limits.max_elevation_rad:
            return VisibilityResult(VisibilityReason.ELEVATION_OUT_OF_RANGE)

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
                instrument_orientation=scene.instrument_pose.orientation,
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
    station_orientations: np.ndarray = None,
) -> np.ndarray:
    """Which stations can measure which targets, as an (S, N) boolean matrix.

    The same checks as `check_target_visibility` -- range, elevation,
    incidence, occlusion -- applied to every station-target pair at once with array
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

    `station_orientations` ((S, 3, 3), one `InstrumentPose.orientation` per
    station) is read only when `limits` has elevation limits; None means every
    station is levelled with local z = world up.
    """
    stations = np.atleast_2d(np.asarray(station_positions_m, dtype=float))
    targets = np.atleast_2d(np.asarray(target_points_m, dtype=float))
    n_stations, n_targets = len(stations), len(targets)

    beams = targets[None, :, :] - stations[:, None, :]  # (S, N, 3), station -> target
    range_m = np.linalg.norm(beams, axis=2)
    visible = range_m <= limits.max_range_m

    if limits.has_elevation_limits:
        if station_orientations is None:
            up_axes = np.tile(np.array([0.0, 0.0, 1.0]), (n_stations, 1))
        else:
            orientations = np.asarray(station_orientations, dtype=float).reshape(n_stations, 3, 3)
            up_axes = orientations[:, :, 2]  # each station's local z axis, in the global frame
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_elevation = np.einsum("snk,sk->sn", beams, up_axes) / range_m
        elevation = np.arcsin(np.clip(sin_elevation, -1.0, 1.0))
        # A zero-range pair has no defined elevation: NaN, and the scalar check
        # skips the elevation test there, so match it rather than fail the pair.
        defined = range_m > 0.0
        if limits.min_elevation_rad is not None:
            visible &= ~defined | (elevation >= limits.min_elevation_rad)
        if limits.max_elevation_rad is not None:
            visible &= ~defined | (elevation <= limits.max_elevation_rad)

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

    # Moller-Trumbore, with every per-ray dot product rewritten as a single
    # matrix product against a per-triangle vector precomputed once, using the
    # scalar triple product a . (b x c) = b . (c x a):
    #   det  a = edge1 . (d x edge2)  = d . (edge2 x edge1)
    #   u  ~ s . (d x edge2)          = d . (edge2 x s)
    #   v  ~ d . (s x edge1)
    #   t  ~ edge2 . (s x edge1)
    # so a batch of M rays against K triangles is three (M,3)@(3,K) products
    # instead of an (M,K,3) cross-product array. Same test, same numbers to
    # round-off (tests/test_visibility.py pins it to the scalar version).
    det_vec = np.cross(edge2, edge1)  # (K, 3)
    for s in range(n_stations):
        candidates = np.nonzero(visible[s])[0]  # ray-cast only what range/incidence left
        if len(candidates) == 0:
            continue
        direction = aim[candidates] - stations[s]  # (M, 3)
        s_vec = stations[s] - v0  # (K, 3)
        q = np.cross(s_vec, edge1)  # (K, 3)
        a = direction @ det_vec.T  # (M, K)
        parallel = np.abs(a) < epsilon
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.0 / a
            u = f * (direction @ np.cross(edge2, s_vec).T)
            v = f * (direction @ q.T)
            t = f * np.einsum("kj,kj->k", edge2, q)[None, :]
        with np.errstate(invalid="ignore"):  # NaN where the ray is parallel; masked out by ~parallel
            hit = (
                ~parallel & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
                & (t > t_margin) & (t < 1.0 - t_margin)
            )
        visible[s, candidates[hit.any(axis=1)]] = False
    return visible
