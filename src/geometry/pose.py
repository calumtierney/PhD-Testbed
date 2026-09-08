"""Instrument pose: where a station sits and how its local frame is oriented."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class InstrumentPose:
    """One instrument station: position and orientation in the global frame.

    Attributes
    ----------
    position_m : (3,) ndarray
        Instrument origin, expressed in the global (scene) frame.
    orientation : (3, 3) ndarray
        Rotation matrix whose columns are the instrument's local x, y, z
        axes, each expressed in the global frame. A global-frame vector
        ``v_global`` and its local-frame counterpart ``v_local`` are related
        by ``v_global = orientation @ v_local``. Defaults to the identity,
        i.e. the instrument's local frame is aligned with the global frame.
    """

    position_m: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.position_m = np.asarray(self.position_m, dtype=float)
        self.orientation = np.asarray(self.orientation, dtype=float)
        if self.position_m.shape != (3,):
            raise ValueError(f"position_m must have shape (3,), got {self.position_m.shape}")
        if self.orientation.shape != (3, 3):
            raise ValueError(f"orientation must have shape (3, 3), got {self.orientation.shape}")

    def to_local(self, point_global_m: np.ndarray) -> np.ndarray:
        """Express a global-frame point in the instrument's local frame."""
        return self.orientation.T @ (np.asarray(point_global_m, dtype=float) - self.position_m)

    def covariance_to_global(self, covariance_local: np.ndarray) -> np.ndarray:
        """Rotate a 3x3 covariance matrix from the local frame to the global frame."""
        return self.orientation @ covariance_local @ self.orientation.T
