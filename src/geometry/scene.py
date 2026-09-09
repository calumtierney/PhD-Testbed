"""Scene: the target points to be measured, plus the instrument observing them.

Step 1 (CLAUDE.md §5) supported exactly one instrument station, no
obstacles and no surface normals -- none of those are needed to propagate
sensor noise to a Cartesian covariance. Step 2 (visibility) needs two more
things to decide whether a target can be seen at all, so both are added
here as optional fields that default to "not modelled", leaving every
step 1 use of Scene unchanged:

  - `target_normals_m` -- the target surface's own orientation, needed for
    the incidence-angle check (how face-on the beam hits the surface).
  - `obstacles` -- blocking geometry, needed for the occlusion check.

Multiple instrument stations are still out of scope; that's step 3.
"""
from dataclasses import dataclass

import numpy as np

from geometry.mesh import TriangleMesh
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
    target_normals_m : (N, 3) ndarray or None
        Outward unit surface normal at each target -- the direction the
        target's surface faces, away from the material it's measured on.
        Used only by the visibility check's incidence-angle test (step 2);
        the step 1 error model ignores it entirely. None means "no
        incidence check requested" rather than "flat, facing the tracker".
    obstacles : TriangleMesh or None
        Blocking geometry for the visibility check's occlusion test (step
        2). None means an empty scene -- nothing can occlude anything.
    """

    target_points_m: np.ndarray
    instrument_pose: InstrumentPose
    target_normals_m: np.ndarray = None
    obstacles: TriangleMesh = None

    def __post_init__(self):
        self.target_points_m = np.atleast_2d(np.asarray(self.target_points_m, dtype=float))
        if self.target_points_m.ndim != 2 or self.target_points_m.shape[1] != 3:
            raise ValueError(
                f"target_points_m must have shape (N, 3), got {self.target_points_m.shape}"
            )

        if self.target_normals_m is not None:
            self.target_normals_m = np.atleast_2d(np.asarray(self.target_normals_m, dtype=float))
            if self.target_normals_m.shape != self.target_points_m.shape:
                raise ValueError(
                    f"target_normals_m must have shape {self.target_points_m.shape} "
                    f"(one normal per target point), got {self.target_normals_m.shape}"
                )

    @property
    def n_points(self) -> int:
        return self.target_points_m.shape[0]
