"""Datum: an external geometric reference a tolerance can be measured against.

Kept deliberately minimal for this build step (CLAUDE.md §5, step 4):
just a name and an orientation (a plane's normal). A full GD&T datum
reference frame -- an origin, an axis system, degrees of freedom removed
in turn by primary/secondary/tertiary datum features -- is a
substantially bigger feature than step 4 needs in order to show that
different tolerance types project the same covariance onto different
directions (its actual job -- see `characteristic.py`'s module
docstring). If a later step needs the full datum reference frame (e.g. to
compute a characteristic's *nominal* value, not just its uncertainty),
extend this then rather than guessing its shape now.
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
    """

    name: str
    normal: np.ndarray

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("datum normal must be nonzero")
        object.__setattr__(self, "normal", normal / norm)
