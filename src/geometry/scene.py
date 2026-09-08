"""Scene: the target points to be measured, plus the instrument observing them.

Step 1 (CLAUDE.md §5) supports exactly one instrument station and no
obstacles. Occlusion meshes arrive in step 2, multiple stations in step 3 --
do not add either here.
"""
from dataclasses import dataclass

import numpy as np

from geometry.pose import InstrumentPose


@dataclass
class Scene:
    """A set of target points and the single instrument station observing them.

    Attributes
    ----------
    target_points_m : (N, 3) ndarray
        Target coordinates in the global frame.
    instrument_pose : InstrumentPose
        The one station observing every target in this scene.
    """

    target_points_m: np.ndarray
    instrument_pose: InstrumentPose

    def __post_init__(self):
        self.target_points_m = np.atleast_2d(np.asarray(self.target_points_m, dtype=float))
        if self.target_points_m.ndim != 2 or self.target_points_m.shape[1] != 3:
            raise ValueError(
                f"target_points_m must have shape (N, 3), got {self.target_points_m.shape}"
            )

    @property
    def n_points(self) -> int:
        return self.target_points_m.shape[0]
