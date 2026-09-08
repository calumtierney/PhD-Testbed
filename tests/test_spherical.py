"""Tests for geometry.spherical: coordinate conversions and their Jacobian."""
import numpy as np
import pytest

from geometry.spherical import cartesian_to_spherical, spherical_jacobian, spherical_to_cartesian


def test_round_trip_conversion():
    d, theta, phi = 5.3, 0.7, 0.3
    point = spherical_to_cartesian(d, theta, phi)
    d2, theta2, phi2 = cartesian_to_spherical(point)
    assert d2 == pytest.approx(d)
    assert theta2 == pytest.approx(theta)
    assert phi2 == pytest.approx(phi)


def test_jacobian_matches_finite_difference():
    """The analytic Jacobian must agree with a central-difference numerical
    one -- a direct check on the algebra in spherical_jacobian, independent
    of any physics claim."""
    d, theta, phi = 3.1, 0.4, 0.2
    jacobian = spherical_jacobian(d, theta, phi)

    eps = 1e-6
    numeric = np.zeros((3, 3))
    for col, delta in enumerate([(eps, 0, 0), (0, eps, 0), (0, 0, eps)]):
        p_plus = spherical_to_cartesian(d + delta[0], theta + delta[1], phi + delta[2])
        p_minus = spherical_to_cartesian(d - delta[0], theta - delta[1], phi - delta[2])
        numeric[:, col] = (p_plus - p_minus) / (2 * eps)

    assert np.allclose(jacobian, numeric, atol=1e-6)


def test_local_frame_is_orthogonal():
    """The three Jacobian columns (radial, azimuthal, elevation directions)
    are mutually orthogonal at every point, not just at boresight. This is
    the geometric fact that makes the laser tracker's noise ellipsoid (see
    instruments/laser_tracker.py) diagonal in this frame everywhere in the
    scene, not only directly in front of the instrument.
    """
    for d, theta, phi in [(4.0, 1.1, -0.4), (2.5, 0.0, 0.0), (10.0, -2.0, 0.6)]:
        jacobian = spherical_jacobian(d, theta, phi)
        gram = jacobian.T @ jacobian
        off_diagonal = gram - np.diag(np.diag(gram))
        assert np.allclose(off_diagonal, 0.0, atol=1e-10)


def test_range_zero_is_undefined():
    with pytest.raises(ValueError):
        cartesian_to_spherical(np.zeros(3))
