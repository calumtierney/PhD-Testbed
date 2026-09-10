"""Tolerance types: what direction(s) each GD&T tolerance actually constrains.

CLAUDE.md §5, step 4. See `characteristic.py`'s module docstring for the
geometric explanation of *why* projection is the right operation, and
`docs/step4_characteristic_layer_physics.md` for the full derivation.
Three tolerance types to start, per the build order: position, flatness,
parallelism.

Every tolerance type here exposes one method, `constraint_directions`,
returning a `(k, 3)` array of orthonormal unit row vectors -- the
direction(s), in the global frame, that the tolerance's zone actually
restricts movement along. `k` is 1 for a tolerance defined by a single
plane-pair (flatness, parallelism) and up to 3 for a position tolerance,
depending on whether it's a spherical (3D, isotropic) or cylindrical (2D,
an axis excluded) zone. `characteristic.py` uses this to project a
point's full 3x3 covariance down onto only the directions that matter.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from characteristics.datum import Datum


def _orthonormal_basis_perpendicular_to(axis: np.ndarray) -> np.ndarray:
    """Two orthonormal unit vectors spanning the plane perpendicular to
    `axis`. Used for a cylindrical position-tolerance zone: the zone
    constrains everything perpendicular to its own axis, nothing along it.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return np.stack([u, v])


@dataclass(frozen=True)
class PositionTolerance:
    """A position tolerance: the feature must lie within a zone of size
    `zone_diameter_m` centred on its true position.

    `axis_direction`, if given, makes the zone a *cylinder* along that
    axis -- the classic case of a hole toleranced only in the plane
    perpendicular to its own axis, with depth along the axis left
    unconstrained by this callout. Left as `None` (the default), the zone
    is a *sphere*: every direction is constrained equally (isotropic).
    """

    zone_diameter_m: float
    axis_direction: Optional[np.ndarray] = None

    def constraint_directions(self, datum: Optional[Datum] = None) -> np.ndarray:
        # datum accepted for interface consistency with the other
        # tolerance types (Characteristic calls every tolerance the same
        # way); a position tolerance's own callout direction doesn't
        # depend on a datum's orientation, so it's unused here.
        if self.axis_direction is None:
            return np.eye(3)  # spherical zone: constrained equally in x, y, z
        return _orthonormal_basis_perpendicular_to(self.axis_direction)  # cylindrical zone


@dataclass(frozen=True)
class FlatnessTolerance:
    """A flatness tolerance: the feature's points must lie between two
    parallel planes `zone_width_m` apart. Points are free to move
    *within* those planes -- flatness only constrains deviation along the
    surface's own normal direction.

    `surface_normal` is the measured surface's own (nominal) normal --
    contrast with `ParallelismTolerance`, which uses an externally
    supplied datum's normal instead of the surface's own.
    """

    zone_width_m: float
    surface_normal: np.ndarray

    def constraint_directions(self, datum: Optional[Datum] = None) -> np.ndarray:
        normal = np.asarray(self.surface_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        return normal[None, :]


@dataclass(frozen=True)
class ParallelismTolerance:
    """A parallelism tolerance: like flatness, the feature's points must
    lie between two parallel planes `zone_width_m` apart -- but those
    planes are oriented parallel to an external datum, not to the
    surface's own best-fit orientation. Constrains deviation along the
    *datum's* normal, which need not coincide with the surface's own.
    """

    zone_width_m: float

    def constraint_directions(self, datum: Optional[Datum]) -> np.ndarray:
        if datum is None:
            raise ValueError(
                "ParallelismTolerance needs a datum to define its constraint "
                "direction -- pass one via Characteristic(datum=...)"
            )
        return datum.normal[None, :]
