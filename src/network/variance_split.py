"""How much of a tracker's coordinate variance is the geometric parameters'.

Hughes et al. (2011, section 7) simulate 1000 points 1-5 m from the tracker at
random azimuth and elevation, front and back face, propagate the parameter
uncertainties and the sensor noise, and find that about 80% of the variance
comes from the random effects of the angle sensors and the remaining ~20% from
the 14 geometric parameters and lambda, mu (0.1-3.8% each). That is the most
direct check that the parameter covariance is implemented correctly: it
exercises the model's sensitivities, the instrument frame and the variance
bookkeeping with nothing else (one station, no network).

This does it through `network.design` -- one station, half the points on each
face -- with the covariance `hughes_2011_residual()` gives. The paper does not
say how the random angles are distributed; uniform in angle (the literal
reading) is the default, uniform in direction (area on the sphere) the
alternative, and both are reported.
"""
from typing import Dict, Optional

import numpy as np

from geometry.spherical import spherical_to_cartesian
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import BACK, FRONT, N_PARAMETERS, PARAMETER_NAMES, NPLErrorParameters
from network.design import design_point_covariances


def _trace_sum(points: np.ndarray, tracker: LaserTracker, face: str) -> float:
    result = design_point_covariances(points, np.zeros((1, 3)), np.eye(3)[None], tracker, np.ones((1, len(points)), bool),
                                      0, None, [(face,)])
    return float(np.einsum("tii->", result.point_covariance_m2))


def random_points(n_points: int, rng: np.random.Generator, range_m=(1.0, 5.0), elevation_deg=(-60.0, 77.0),
                  distribution: str = "uniform_angle") -> np.ndarray:
    """Targets in the tracker's frame: range uniform in `range_m`, azimuth uniform,
    elevation uniform in angle ("uniform_angle") or in direction on the sphere
    ("uniform_direction": sin(elevation) uniform) within `elevation_deg`."""
    low, high = np.deg2rad(elevation_deg)
    if distribution == "uniform_angle":
        elevation = rng.uniform(low, high, n_points)
    elif distribution == "uniform_direction":
        elevation = np.arcsin(rng.uniform(np.sin(low), np.sin(high), n_points))
    else:
        raise ValueError(f"distribution must be 'uniform_angle' or 'uniform_direction', got {distribution!r}")
    return np.array([spherical_to_cartesian(d, az, el) for d, az, el in zip(
        rng.uniform(*range_m, n_points), rng.uniform(-np.pi, np.pi, n_points), elevation)])


def parameter_variance_split(params: NPLErrorParameters, tracker: Optional[LaserTracker] = None, n_points: int = 1000,
                             seed: int = 0, distribution: str = "uniform_angle") -> Dict:
    """Shares of the total coordinate variance (summed over the points and the
    three coordinates) from the sensor noise and from each parameter alone.

    Returns `{"parameters": share, "noise": share, "range_noise": share,
    "angle_noise": share, "per_parameter": {name: share}, "additivity_error": ...}`
    -- shares of the total (noise + parameters). Per-parameter shares come from
    running the design once per parameter with only that parameter's variance, and
    add up to the all-parameters share when the covariance is diagonal
    (`additivity_error` is the difference, zero then).
    """
    tracker = LaserTracker() if tracker is None else tracker
    points = random_points(n_points, np.random.default_rng(seed), distribution=distribution)
    halves = (points[: n_points // 2], points[n_points // 2:])
    faces = (FRONT, BACK)
    noise = sum(_trace_sum(h, tracker, f) for h, f in zip(halves, faces))

    def consider(covariance: np.ndarray) -> float:
        carrier = LaserTracker(tracker.sigma_d_m, tracker.sigma_theta_rad, tracker.sigma_phi_rad, tracker.systematic_std_m,
                               NPLErrorParameters(params.values, covariance))
        return sum(_trace_sum(h, carrier, f) for h, f in zip(halves, faces)) - noise

    total_parameters = consider(params.covariance)
    per_parameter = {}
    for j, name in enumerate(PARAMETER_NAMES):
        covariance = np.zeros((N_PARAMETERS, N_PARAMETERS))
        covariance[j, j] = params.covariance[j, j]
        per_parameter[name] = consider(covariance) if covariance.any() else 0.0
    total = noise + total_parameters
    d = np.linalg.norm(points, axis=1)
    elevation = np.arcsin(points[:, 2] / d)
    range_noise = len(points) * tracker.sigma_d_m**2
    angle_noise = float(np.sum((d * np.cos(elevation) * tracker.sigma_theta_rad) ** 2 + (d * tracker.sigma_phi_rad) ** 2))
    return {
        "parameters": total_parameters / total, "noise": noise / total, "range_noise": range_noise / total,
        "angle_noise": angle_noise / total, "per_parameter": {k: v / total for k, v in per_parameter.items()},
        "additivity_error": (sum(per_parameter.values()) - total_parameters) / total,
    }
