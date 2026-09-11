"""Datum: an external geometric reference a tolerance can be measured against.

Kept deliberately minimal for this build step (CLAUDE.md §5, step 4):
a name, an orientation (a plane's normal), and one scalar for how well
established that normal is. A full GD&T datum reference frame -- an
origin, an axis system, degrees of freedom removed in turn by
primary/secondary/tertiary datum features -- is a substantially bigger
feature than step 4 needs in order to show that different tolerance types
project the same covariance onto different directions (its actual job --
see `characteristic.py`'s module docstring). If a later step needs the
full datum reference frame (e.g. to compute a characteristic's *nominal*
value, not just its uncertainty), extend this then rather than guessing
its shape now.

**Why establishment uncertainty is not optional to ignore.** A datum is
itself established from measurements of one or more datum features (a
face, a set of holes, an ERS constellation) -- it is not a mathematical
given with zero uncertainty. A parallelism callout referencing this datum
inherits that uncertainty on top of the measured point's own: reporting
only the point's projected covariance and pretending the datum plane
itself is known exactly understates the true characteristic uncertainty.
`establishment_uncertainty_m` is a deliberately simple placeholder for
that contribution -- a single equivalent linear standard uncertainty
along the datum's normal, added in quadrature by
`ParallelismTolerance.additional_variance_m2` -- not a full propagation
of the datum-fitting covariance through to an arbitrary measured point
(which would need the datum feature's own point covariance and its
fitted-plane sensitivity, i.e. the GUM Supplement 1/2 Monte Carlo or
linearised-fit treatment this codebase does not yet build). Defaults to
0.0 (no datum uncertainty modelled) so existing callers are unaffected
until they choose to supply a value.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Datum:
    """An external reference plane a tolerance can be measured against.

    Attributes
    ----------
    name : str -- a human-readable identifier (e.g. "A", "datum plane A").
    normal : (3,) ndarray -- the datum plane's orientation in the global
        frame, stored as a unit vector regardless of the input's own
        magnitude.
    establishment_uncertainty_m : float -- standard uncertainty (not
        expanded) of the datum plane's own position along its normal,
        as established from whatever datum feature(s) define it. See the
        module docstring for what this is and is not a substitute for.
        Defaults to 0.0 -- "this datum is treated as exactly known",
        which should be stated explicitly by a caller, not assumed
        silently by omission.
    """

    name: str
    normal: np.ndarray
    establishment_uncertainty_m: float = 0.0

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("datum normal must be nonzero")
        object.__setattr__(self, "normal", normal / norm)
        if self.establishment_uncertainty_m < 0.0:
            raise ValueError(
                f"establishment_uncertainty_m must be >= 0, got {self.establishment_uncertainty_m}"
            )
