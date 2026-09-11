"""The characteristic layer (CLAUDE.md §5, step 4): turning a network
solve's covariance into the uncertainty of a specific toleranced feature.

This is where the project starts to diverge from existing tools (CLAUDE.md
§1). Everything before this step treats "uncertainty" as a property of a
*point* -- a full 3x3 covariance, never reduced to a scalar (§3). This
step asks a different question: given that covariance, how uncertain is
one particular *characteristic* -- one toleranced position, profile, or
parallelism callout on that point? The answer depends on which tolerance
it is, even for the exact same point, measured the exact same way.

**The geometric idea, before any code.** A GD&T tolerance defines a zone
-- a region of space the toleranced feature must fall inside. Different
tolerance types are shaped, and *oriented*, differently:

  - A **position** tolerance (with no axis given) is a sphere: it cares
    about deviation in every direction equally.
  - A position tolerance *with* an axis (e.g. a hole's position,
    specified only in the plane perpendicular to the hole's own axis) is
    a cylinder: it cares about deviation in two directions, and not at
    all about deviation along the third (the hole's depth).
  - A **profile** tolerance (one point's normal deviation -- see
    `tolerances.py`'s naming note for why this isn't called "flatness")
    is a pair of parallel planes, oriented to the *measured surface's
    own* nominal orientation at that point: it cares about deviation
    along one direction only (the surface's normal) and not at all about
    deviation within the planes (sideways).
  - A **parallelism** tolerance is also a pair of parallel planes, one
    direction only -- but that direction comes from an *external datum*,
    not from the surface being measured. A surface that's nominally flat
    but slightly tilted relative to its datum has a profile normal and a
    parallelism (datum) normal that point in genuinely different
    directions.

A point's measurement uncertainty is a full 3D object -- the covariance
ellipsoid from step 1 or step 3, which (for a laser tracker) is a
flattened disc, thin along the beam and wide across it. A tolerance zone,
though, only "sees" the component of that ellipsoid lying along the
direction(s) its own zone actually constrains -- everything else is
uncertainty in a direction the tolerance simply doesn't care about, and
doesn't contribute to whether the feature passes or fails that specific
callout.

**The operation that captures this is projection.** For a tolerance
constraining a single direction `d` (a unit vector), the variance *in
that direction* is the classic linear projection of a Gaussian:
`Var(d . x) = d^T Cov(x) d`. For a tolerance constraining several
directions at once (e.g. a spherical position zone, constraining all
three), stack the unit directions as the rows of a matrix `D` (shape
`(k, 3)`); the projected `k x k` covariance is `D Cov(x) D^T`. This is
the object `evaluate_characteristic` below actually returns -- per
CLAUDE.md §3, the projected covariance is what's authoritative, not a
single number pulled out of it. A scalar summary (`sqrt(trace(...))`, the
RSS spread across whichever directions the tolerance kept) is attached
alongside purely as a reporting convenience, the same pattern step 3's
`mean_positional_uncertainty_m` uses, for exactly the same reason: useful
for a headline number or a plot, not a replacement for the full object
carried through the chain.

**Why this makes position and parallelism differ, physically, on the same
point.** A laser tracker's per-point ellipsoid has one small axis (along
its beam) and (up to) two large axes (across it) -- step 1's anisotropy.
A spherical position tolerance's three constraint directions span the
*whole* ellipsoid, so its projected variance picks up the large
transverse axes in full: `trace` of the whole covariance. A parallelism
tolerance's *one* constraint direction is fixed by the datum, with no
relationship at all to where the tracker happened to be standing. If that
datum direction happens to line up with the beam, parallelism inherits
the ellipsoid's *small* axis and comes out far more precise than
position, from the exact same measurement. If it lines up with a
transverse direction instead, parallelism inherits a *large* axis and
looks about as uncertain as position (only using one axis, whereas
position combines all three). Either way, the two tolerances are reading
different projections of one physical ellipsoid -- CLAUDE.md §3's
"projected differently" claim, made concrete. See
`docs/step4_characteristic_layer_physics.md` for a worked numeric example
and `tests/test_characteristics.py` for the gate this reasoning predicts.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from characteristics.datum import Datum
from characteristics.tolerances import ParallelismTolerance, PositionTolerance, ProfileTolerance

Tolerance = Union[PositionTolerance, ProfileTolerance, ParallelismTolerance]


@dataclass(frozen=True)
class Characteristic:
    """One toleranced thing to be verified.

    Attributes
    ----------
    name : a human-readable identifier (e.g. "hole A position").
    target_indices : which target point(s) this characteristic involves,
        as indices into whatever covariance source `evaluate_characteristic`
        is given. A single index for most position/parallelism/profile
        characteristics; several is supported for a callout spanning
        multiple measured points, but only as a per-point breakdown (see
        the module docstring's note on what step 4 does and doesn't do
        with several points).
    tolerance : the tolerance type and its zone size -- a
        `PositionTolerance`, `ProfileTolerance`, or `ParallelismTolerance`.
    datum : the external reference `ParallelismTolerance` measures
        against; ignored by `PositionTolerance` and `ProfileTolerance`
        (both define their own constraint direction without one), kept
        optional here rather than only on `ParallelismTolerance` so every
        characteristic carries the same four fields the build order
        (CLAUDE.md §5, step 4) specifies: point(s), tolerance type,
        tolerance limits, datum reference.
    """

    name: str
    target_indices: List[int]
    tolerance: Tolerance
    datum: Optional[Datum] = None


@dataclass(frozen=True)
class CharacteristicPointUncertainty:
    """The projected uncertainty of one point within a characteristic.

    Attributes
    ----------
    target_index : which point this is.
    constraint_directions : (k, 3) ndarray -- the unit direction(s) the
        tolerance actually constrains, in the global frame.
    projected_covariance_m2 : (k, k) ndarray -- the point's covariance,
        projected onto `constraint_directions`. The authoritative result;
        CLAUDE.md §3 -- keep this, not just `uncertainty_m`, where a
        downstream step (risk, §5 step 5) needs the real shape.
    uncertainty_m : float -- sqrt(trace(projected_covariance_m2)), an RSS
        summary for reporting/plotting. See the module docstring's note on
        why this is a convenience, not the authoritative value.
    """

    target_index: int
    constraint_directions: np.ndarray
    projected_covariance_m2: np.ndarray
    uncertainty_m: float


def _point_covariance(covariance_source, target_index: int) -> np.ndarray:
    """Accepts either an (N, 3, 3) array of per-point covariances (as
    `instruments.laser_tracker.scene_point_covariances` returns) or any
    object exposing `.target_point_covariance(i)` (as `network.solve.
    NetworkSolveResult` does) -- so a characteristic can be evaluated
    straight off a single-station measurement or a full network solve
    without the caller converting between them."""
    if hasattr(covariance_source, "target_point_covariance"):
        return covariance_source.target_point_covariance(target_index)
    return np.asarray(covariance_source)[target_index]


def evaluate_characteristic(
    characteristic: Characteristic, covariance_source
) -> List[CharacteristicPointUncertainty]:
    """Project each of a characteristic's points' covariance onto the
    directions its tolerance actually constrains.

    `covariance_source` is anything `_point_covariance` accepts (see
    above). Returns one `CharacteristicPointUncertainty` per entry in
    `characteristic.target_indices`, in order.

    This deliberately stops at a per-point breakdown rather than
    combining several points into one characteristic-wide number (e.g. a
    true multi-point flatness callout's classic max-minus-min across
    several correlated points is a harder statistical question -- order
    statistics of correlated Gaussians -- than a direct projection, and
    isn't needed to show that different tolerance types on the *same*
    point project differently, which is this step's gate). A multi-point
    aggregate is a natural next elaboration, not built here (CLAUDE.md §9).

    Any `additional_variance_m2` the tolerance itself contributes (e.g. a
    `ParallelismTolerance`'s datum establishment uncertainty -- see that
    class's docstring) is added in quadrature to the point's own projected
    variance, spread evenly across the diagonal of the `(k, k)` projected
    covariance. For the single-direction tolerances that currently define
    a nonzero `additional_variance_m2` this is exact (k=1: it just adds to
    the one entry); it is not a claim about how such a term would combine
    with a multi-direction (position) tolerance, which none currently use.
    """
    directions = characteristic.tolerance.constraint_directions(characteristic.datum)
    additional_variance_m2 = characteristic.tolerance.additional_variance_m2(characteristic.datum)
    n_directions = directions.shape[0]
    results = []
    for target_index in characteristic.target_indices:
        covariance_m2 = _point_covariance(covariance_source, target_index)
        projected_covariance_m2 = directions @ covariance_m2 @ directions.T
        if additional_variance_m2 > 0.0:
            projected_covariance_m2 = projected_covariance_m2 + (
                additional_variance_m2 / n_directions
            ) * np.eye(n_directions)
        uncertainty_m = float(np.sqrt(np.trace(projected_covariance_m2)))
        results.append(
            CharacteristicPointUncertainty(
                target_index=target_index,
                constraint_directions=directions,
                projected_covariance_m2=projected_covariance_m2,
                uncertainty_m=uncertainty_m,
            )
        )
    return results
