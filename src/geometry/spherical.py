"""Spherical <-> Cartesian conversions for a single instrument's local frame.

Convention (this is how a laser tracker actually reports a shot, not the
physics-textbook polar-angle-from-the-z-axis convention):

    d      range   -- distance from the instrument origin to the target [m]
    theta  azimuth  -- angle in the local x-y (horizontal) plane, measured
                        from the local x-axis towards the local y-axis [rad]
    phi    elevation -- angle above the local x-y plane [rad]
                        (phi = 0 on the horizontal plane, +pi/2 straight up)

x = d * cos(phi) * cos(theta)
y = d * cos(phi) * sin(theta)
z = d * sin(phi)

All angles in radians, all lengths in metres (SI internally; see CLAUDE.md
§6 on units).
"""
import numpy as np


def spherical_to_cartesian(d, theta, phi):
    """Convert one (range, azimuth, elevation) reading to a local-frame point.

    Parameters
    ----------
    d : range_m, distance from the instrument origin.
    theta : azimuth_rad.
    phi : elevation_rad.

    Returns
    -------
    (3,) ndarray -- point in the instrument's local Cartesian frame.
    """
    cos_phi = np.cos(phi)
    x = d * cos_phi * np.cos(theta)
    y = d * cos_phi * np.sin(theta)
    z = d * np.sin(phi)
    return np.array([x, y, z])


def cartesian_to_spherical(point_local_m):
    """Convert a local-frame Cartesian point to (range, azimuth, elevation).

    Parameters
    ----------
    point_local_m : (3,) array_like, point in the instrument's local frame.

    Returns
    -------
    d, theta, phi : range_m, azimuth_rad, elevation_rad.

    Raises
    ------
    ValueError if the point coincides with the instrument origin (range
    zero has no defined azimuth/elevation).
    """
    x, y, z = point_local_m
    d = np.sqrt(x**2 + y**2 + z**2)
    if d == 0.0:
        raise ValueError(
            "point coincides with the instrument origin: range is zero, "
            "azimuth and elevation are undefined"
        )
    theta = np.arctan2(y, x)
    phi = np.arcsin(np.clip(z / d, -1.0, 1.0))
    return d, theta, phi


def spherical_jacobian(d, theta, phi):
    """3x3 Jacobian d(x, y, z) / d(d, theta, phi), evaluated at the point.

    Column i is the direction (in the local Cartesian frame) that the
    target moves per unit change of spherical coordinate i, with its
    physical scale folded in:

      - column 0 (d/range)      : unit vector along the beam (radial),
                                   scale 1 -- range moves the point 1:1.
      - column 1 (theta/azimuth): unit vector "around" the beam,
                                   scale d*cos(phi) -- a fixed azimuthal
                                   swing sweeps out more distance the
                                   farther out (and closer to the target)
                                   the point is.
      - column 2 (phi/elevation): unit vector "up" the beam,
                                   scale d -- a fixed elevation swing
                                   always sweeps out d radians of arc.

    These three columns are mutually orthogonal at every point (see
    tests/test_spherical.py::test_local_frame_is_orthogonal) -- they are
    exactly the local radial/azimuthal/elevation unit vectors of a
    spherical coordinate system. That orthogonality is what makes the
    laser tracker's noise ellipsoid (see instruments/laser_tracker.py)
    diagonal in this frame: range noise, azimuth noise and elevation noise
    each move the measured point along one of these three independent
    directions, so they don't mix.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    return np.array(
        [
            [cos_p * cos_t, -d * cos_p * sin_t, -d * sin_p * cos_t],
            [cos_p * sin_t, d * cos_p * cos_t, -d * sin_p * sin_t],
            [sin_p, 0.0, d * cos_p],
        ]
    )


def cartesian_to_spherical_back_face(point_local_m):
    """The *back-face* reading of a point: (range, azimuth, elevation).

    A laser tracker can point its beam at the same target in two ways. In
    the **front face** the standing axis is at azimuth theta and the transit
    axis at elevation phi, with phi in [-pi/2, +pi/2] -- what
    `cartesian_to_spherical` returns (its `arcsin` can only return that
    range). In the **back face** the head is turned round by half a
    revolution and the beam is steered over the top: azimuth theta + pi,
    elevation pi - phi, which for an ordinary target lies in
    [pi/2, 3*pi/2] -- "more than vertical". A perfect instrument gives the
    same point from both faces (`spherical_to_cartesian` of either reading
    returns it); a real one does not, and the disagreement between the two
    faces is how the NPL geometric errors are separated
    (`instruments.npl_model`, `docs/step1b_npl_geometric_model.md`).

    Azimuth is wrapped to (-pi, pi], like the front-face value.
    """
    d, theta, phi = cartesian_to_spherical(point_local_m)
    theta_back = theta + np.pi
    if theta_back > np.pi:
        theta_back -= 2.0 * np.pi
    return d, theta_back, np.pi - phi
