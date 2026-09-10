"""Tests for characteristics.tolerances and characteristics.datum."""
import numpy as np
import pytest

from characteristics.datum import Datum
from characteristics.tolerances import (
    FlatnessTolerance,
    ParallelismTolerance,
    PositionTolerance,
    _orthonormal_basis_perpendicular_to,
)


def test_datum_normalizes_its_normal():
    datum = Datum(name="A", normal=np.array([0.0, 0.0, 5.0]))
    assert np.allclose(datum.normal, [0.0, 0.0, 1.0])


def test_datum_rejects_zero_normal():
    with pytest.raises(ValueError):
        Datum(name="A", normal=np.zeros(3))


def test_orthonormal_basis_perpendicular_to_axis():
    for axis in [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1.0, 1.0])]:
        basis = _orthonormal_basis_perpendicular_to(axis)
        assert basis.shape == (2, 3)
        unit_axis = axis / np.linalg.norm(axis)
        # both basis vectors perpendicular to the axis
        assert abs(np.dot(basis[0], unit_axis)) < 1e-10
        assert abs(np.dot(basis[1], unit_axis)) < 1e-10
        # and to each other, each unit length
        assert abs(np.dot(basis[0], basis[1])) < 1e-10
        assert np.linalg.norm(basis[0]) == pytest.approx(1.0)
        assert np.linalg.norm(basis[1]) == pytest.approx(1.0)


def test_position_tolerance_isotropic_by_default():
    tolerance = PositionTolerance(zone_diameter_m=1e-4)
    directions = tolerance.constraint_directions()
    assert np.allclose(directions, np.eye(3))


def test_position_tolerance_cylindrical_with_axis():
    tolerance = PositionTolerance(zone_diameter_m=1e-4, axis_direction=np.array([1.0, 0, 0]))
    directions = tolerance.constraint_directions()
    assert directions.shape == (2, 3)
    # neither direction has any x-component -- the axis is unconstrained
    assert np.allclose(directions[:, 0], 0.0)


def test_flatness_tolerance_uses_surface_normal():
    tolerance = FlatnessTolerance(zone_width_m=5e-5, surface_normal=np.array([0, 0, 3.0]))
    directions = tolerance.constraint_directions()
    assert directions.shape == (1, 3)
    assert np.allclose(directions[0], [0, 0, 1.0])


def test_parallelism_tolerance_uses_datum_normal():
    datum = Datum(name="A", normal=np.array([1.0, 0, 0]))
    tolerance = ParallelismTolerance(zone_width_m=5e-5)
    directions = tolerance.constraint_directions(datum)
    assert np.allclose(directions, [[1.0, 0, 0]])


def test_parallelism_tolerance_requires_a_datum():
    tolerance = ParallelismTolerance(zone_width_m=5e-5)
    with pytest.raises(ValueError):
        tolerance.constraint_directions(None)
