"""Laser tracker instrument error model (build step 1, CLAUDE.md §5).

A laser tracker measures a target in its own local spherical coordinates:
range d (distance along the beam), azimuth theta and elevation phi (the two
angles that steer the beam). These three raw measurements are corrupted by
three essentially independent noise sources:

  - Range, d, is measured interferometrically / by absolute distance meter
    -- a very precise process. sigma_d is small and, critically, does NOT
    grow with distance: the interferometer is just as precise at 20 m as
    at 2 m.
  - Azimuth and elevation, theta and phi, are measured by angular encoders.
    An encoder has a fixed *angular* precision, sigma_theta / sigma_phi,
    typically a few microradians. But an angular error only becomes a
    *position* error once multiplied by the distance to the target: the
    same angular wobble swings the far end of a long beam much further off
    target than the near end of a short one. The angular channel's
    contribution to position error is therefore d * sigma_theta and
    d * sigma_phi, and grows linearly with range.

Consequently the two directions transverse to the beam carry a position
uncertainty that scales with range, while the one direction along the beam
carries a position uncertainty that is roughly constant. Even fairly close
in, the angular channel already dominates, because encoders are simply far
noisier in relative terms than the interferometer. The measurement error
ellipsoid is therefore a flattened disc (an oblate spheroid): thin along
the beam, wide across it -- ~10:1 at a few metres' range, per Hughes et al.
(2011) and consistent with Schmitt et al.'s point that a laser tracker's
uncertainty cannot be summarised by a single `A + B*L` number, because that
number would have to describe two physically different error mechanisms at
once. This is exactly why CLAUDE.md §3 insists uncertainty is carried as a
full covariance matrix and never reduced to a scalar before the
characteristic layer.

See docs/step1_laser_tracker_physics.md for the full derivation, the
worked geometry of *why* the transverse directions come out mutually
orthogonal and orthogonal to the beam (so the noise ellipsoid's principal
axes are exactly range/azimuth/elevation), and an explicit note on the
default noise values used here.
"""
from dataclasses import dataclass

import numpy as np

from geometry.pose import InstrumentPose
from geometry.scene import Scene
from geometry.spherical import cartesian_to_spherical, spherical_jacobian

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

# Calibrated so that, at the CLAUDE.md §4/§5 reference range of 2.5 m, the
# transverse standard deviation d * sigma_theta comes out at 40 um and the
# radial standard deviation sigma_d at 4 um -- reproducing the ~10:1
# anisotropy Hughes et al. (2011) report. This is NOT a direct quote of
# their headline a-priori figures (sigma_d ~ 1.2 um, angular sigma ~ 0.3 to
# 0.7 arcsec); those figures, applied naively at 2.5 m, give a smaller
# absolute spread (~5:1, a few um lateral). The discrepancy and the
# reasoning behind this calibration are logged in
# docs/step1_laser_tracker_physics.md -- TODO(source): confirm against the
# primary paper's a posteriori table directly (not accessible from this
# session; NPL's eprints host is blocked by the network egress policy here).
DEFAULT_SIGMA_D_M = 4.0e-6
DEFAULT_SIGMA_THETA_ARCSEC = 3.3
DEFAULT_SIGMA_PHI_ARCSEC = 3.3


@dataclass(frozen=True)
class LaserTracker:
    """Spherical-noise error model for a laser tracker.

    Attributes
    ----------
    sigma_d_m : range_m standard deviation (interferometer / ADM noise).
    sigma_theta_rad : azimuth_rad standard deviation (encoder noise).
    sigma_phi_rad : elevation_rad standard deviation (encoder noise).
    """

    sigma_d_m: float = DEFAULT_SIGMA_D_M
    sigma_theta_rad: float = DEFAULT_SIGMA_THETA_ARCSEC * ARCSEC_TO_RAD
    sigma_phi_rad: float = DEFAULT_SIGMA_PHI_ARCSEC * ARCSEC_TO_RAD

    def covariance_local(self, point_local_m: np.ndarray) -> np.ndarray:
        """3x3 Cartesian covariance of a measured point, in the instrument's
        own local frame.

        Propagates the independent spherical-coordinate variances
        (sigma_d^2, sigma_theta^2, sigma_phi^2) through the spherical ->
        Cartesian Jacobian: Sigma_xyz = J @ Sigma_spherical @ J.T. This is
        first-order (linearised) error propagation, valid as long as the
        angular noise is small compared to one radian -- true here by
        several orders of magnitude.
        """
        d, theta, phi = cartesian_to_spherical(point_local_m)
        jacobian = spherical_jacobian(d, theta, phi)
        sigma_spherical = np.diag(
            [self.sigma_d_m**2, self.sigma_theta_rad**2, self.sigma_phi_rad**2]
        )
        return jacobian @ sigma_spherical @ jacobian.T

    def covariance_global(self, point_global_m: np.ndarray, pose: InstrumentPose) -> np.ndarray:
        """3x3 Cartesian covariance of a measured point, in the global (scene) frame."""
        point_local_m = pose.to_local(point_global_m)
        covariance_local = self.covariance_local(point_local_m)
        return pose.covariance_to_global(covariance_local)


def scene_point_covariances(scene: Scene, tracker: LaserTracker) -> np.ndarray:
    """Per-point Cartesian covariance for every target in a scene.

    Explicit loop over points rather than a vectorised batch computation:
    per CLAUDE.md §6, clarity of the physics wins over vectorisation while
    the codebase is this young, and scene sizes here (tens to low
    thousands of targets) make this loop's cost negligible.

    Returns
    -------
    (N, 3, 3) ndarray -- covariances[i] is the global-frame covariance of
    scene.target_points_m[i].
    """
    covariances = np.empty((scene.n_points, 3, 3))
    for i in range(scene.n_points):
        covariances[i] = tracker.covariance_global(
            scene.target_points_m[i], scene.instrument_pose
        )
    return covariances
