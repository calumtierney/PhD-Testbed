"""Tests for geometry.mesh: STL loading and ray/triangle intersection."""
import numpy as np
import pytest

from geometry.mesh import TriangleMesh, load_stl, segment_intersects_mesh


def test_load_binary_stl_wall_fixture():
    """scenes/wall.stl: the two-triangle synthetic wall used by the step 2
    gate test -- loading it should give 6 vertices (STL doesn't share
    vertices between triangles) and 2 faces."""
    mesh = load_stl("scenes/wall.stl")
    assert mesh.vertices_m.shape == (6, 3)
    assert mesh.faces.shape == (2, 3)
    # every vertex should sit on the wall's plane, x = 1.5
    assert np.allclose(mesh.vertices_m[:, 0], 1.5)


def test_load_ascii_stl():
    mesh = load_stl("scenes/single_triangle_ascii.stl")
    assert mesh.vertices_m.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
    assert np.allclose(mesh.vertices_m, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_triangle_mesh_rejects_bad_shapes():
    with pytest.raises(ValueError):
        TriangleMesh(vertices_m=np.zeros((3, 2)), faces=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError):
        TriangleMesh(vertices_m=np.zeros((3, 3)), faces=np.array([[0, 1, 5]]))  # index 5 doesn't exist


def test_triangle_accessor():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    mesh = TriangleMesh(vertices_m=vertices, faces=faces)
    assert mesh.n_triangles == 2
    assert np.allclose(mesh.triangle(1), [[0, 0, 0], [1, 0, 0], [0, 0, 1]])


def _single_triangle_mesh():
    # One triangle spanning x in [0,1], y in [0,1] at z = 1, useful as a
    # simple "floor at height 1" obstacle for direct intersection tests.
    vertices = np.array([[0, 0, 1], [2, 0, 1], [0, 2, 1]], dtype=float)
    faces = np.array([[0, 1, 2]])
    return TriangleMesh(vertices_m=vertices, faces=faces)


def test_segment_crossing_the_triangle_intersects():
    mesh = _single_triangle_mesh()
    origin = np.array([0.3, 0.3, 0.0])
    target = np.array([0.3, 0.3, 2.0])  # straight through the triangle's interior
    assert segment_intersects_mesh(origin, target, mesh) is True


def test_segment_missing_the_triangle_does_not_intersect():
    mesh = _single_triangle_mesh()
    origin = np.array([5.0, 5.0, 0.0])
    target = np.array([5.0, 5.0, 2.0])  # well outside the triangle's footprint
    assert segment_intersects_mesh(origin, target, mesh) is False


def test_segment_stopping_short_of_the_triangle_does_not_intersect():
    mesh = _single_triangle_mesh()
    origin = np.array([0.3, 0.3, 0.0])
    target = np.array([0.3, 0.3, 0.5])  # target is well below the triangle, z=1
    assert segment_intersects_mesh(origin, target, mesh) is False


def test_target_touching_the_mesh_is_not_self_occluded():
    """A target sitting exactly on (or just past) the obstacle plane is
    the endpoint of the segment (t ~ 1), not an obstruction between the
    instrument and itself -- the t_margin in segment_intersects_mesh
    exists precisely to exclude this."""
    mesh = _single_triangle_mesh()
    origin = np.array([0.3, 0.3, 0.0])
    target = np.array([0.3, 0.3, 1.0])  # lands exactly on the triangle's plane
    assert segment_intersects_mesh(origin, target, mesh) is False
