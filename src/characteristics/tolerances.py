"""Tolerance types: what direction(s) each GD&T tolerance actually constrains.

CLAUDE.md §5, step 4. See `characteristic.py`'s module docstring for the
geometric explanation of *why* projection is the right operation, and
`docs/step4_characteristic_layer_physics.md` for the full derivation.
Three tolerance types to start, per the build order: position, profile
(see the naming note below), and parallelism.

Every tolerance type here exposes two methods:

  - `constraint_directions` -- a `(k, 3)` array of orthonormal unit row
    vectors, the direction(s), in the global frame, that the tolerance's
    zone actually restricts movement along. `k` is 1 for a tolerance
    defined by a single plane-pair (profile, parallelism) and up to 3 for
    a position tolerance, depending on whether it's a spherical (3D,
    isotropic) or cylindrical (2D, an axis excluded) zone.
    `characteristic.py` uses this to project a point's full 3x3
    covariance down onto only the directions that matter.
  - `additional_variance_m2` -- extra variance (in the same projected
    subspace) that isn't captured by projecting the *point's* covariance
    alone. Zero for `PositionTolerance` and `ProfileTolerance`; nonzero
    for `ParallelismTolerance` when its datum carries an establishment
    uncertainty (`Datum.establishment_uncertainty_m`) -- see that class's
    docstring. Defaults to 0.0 everywhere, so a caller who hasn't supplied
    a datum uncertainty sees no change in behaviour.

**Naming note -- what "profile" means here, and why it isn't "flatness".**
A real GD&T flatness callout is unilateral (a single zone of width `t`,
`0` to `t`, never `+-t/2`), evaluated from *many* measured points on a
surface by a minimum-zone or least-squares fit, and its uncertainty is
therefore a property of the whole point set's covariance (points measured
from a shared station share correlated error, which partly cancels in a
best-fit calculation) -- not Gaussian, not zero-mean, and not reducible to
one point's variance along a normal. What this module actually computes
-- one point's *signed* deviation along a nominal direction, projected
from that one point's own covariance -- is a different, narrower thing:
the uncertainty of that point's normal (profile) deviation from its
designed position. That's a legitimate characteristic in its own right
(a profile-of-a-surface-point tolerance, GPS-valid as a bilateral zone
about a true profile), but it is not flatness, and calling it that
overstates what's modelled. `ProfileTolerance` is named, and documented,
accordingly; a caller wanting true multi-point flatness needs the
order-statistics-of-correlated-points treatment `characteristic.py`'s
module docstring already flags as unbuilt.
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

    Reminder (see `characteristic.py` and `risk.jcgm106`'s module
    docstrings): for a zone with more than one constrained direction, the
    RSS/trace summary this module's caller computes is a Gaussian
    approximation to what is really a chi-distributed radial deviation
    (2 or 3 degrees of freedom). It is not the exact multivariate
    conformity calculation; treat it as directionally indicative only.
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

    def additional_variance_m2(self, datum: Optional[Datum] = None) -> float:
        return 0.0


@dataclass(frozen=True)
class ProfileTolerance:
    """A profile-of-a-point tolerance: this *one point's* signed deviation
    along `surface_normal` must lie within a zone `zone_width_m` wide.
    Points are free to move within the plane perpendicular to that normal
    -- only the out-of-plane deviation is constrained.

    See the module docstring's naming note: this is deliberately not
    called "flatness". It is the uncertainty of one point's normal
    deviation from its nominal position -- the right building block for a
    true multi-point flatness or profile-of-a-surface callout, but not
    that callout itself (which needs several points' *joint* covariance
    and a minimum-zone or least-squares fit, not a single point's
    variance).

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

    def additional_variance_m2(self, datum: Optional[Datum] = None) -> float:
        return 0.0


@dataclass(frozen=True)
class ParallelismTolerance:
    """A parallelism tolerance: like a profile callout, the feature's
    points must lie between two parallel planes `zone_width_m` apart --
    but those planes are oriented parallel to an external datum, not to
    the surface's own best-fit orientation. Constrains deviation along
    the *datum's* normal, which need not coincide with the surface's own.

    Requires a datum (raises if none is given -- a parallelism callout is
    meaningless without one). If that datum carries an
    `establishment_uncertainty_m` (Datum's own docstring explains what
    this is and isn't), it is added to the projected point variance in
    quadrature via `additional_variance_m2`, so a parallelism
    characteristic's reported uncertainty reflects both "how well was the
    point measured" and "how well was the datum established" -- not just
    the former.
    """

    zone_width_m: float

    def constraint_directions(self, datum: Optional[Datum]) -> np.ndarray:
        if datum is None:
            raise ValueError(
                "ParallelismTolerance needs a datum to define its constraint "
                "direction -- pass one via Characteristic(datum=...)"
            )
        return datum.normal[None, :]

    def additional_variance_m2(self, datum: Optional[Datum]) -> float:
        if datum is None:
            raise ValueError(
                "ParallelismTolerance needs a datum -- pass one via Characteristic(datum=...)"
            )
        return datum.establishment_uncertainty_m**2
