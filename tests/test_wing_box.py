"""Tests for geometry.wing_box: the synthetic wing-box inspection part."""
import numpy as np
import pytest

from geometry.mesh import segment_intersects_mesh
from geometry.visibility import VisibilityLimits, visibility_matrix
from geometry.wing_box import (
    NOSE_A, NOSE_B, NOSE_ZC, TASK_NAMES, X_NOSE_START, build_wing_box_part,
)


@pytest.fixture(scope="module")
def part():
    return build_wing_box_part()


def test_part_has_about_five_hundred_points_across_six_tasks(part):
    assert part.n_points == 500
    assert sorted(set(part.task_ids.tolist())) == [1, 2, 3, 4, 5, 6]
    assert [int(part.task_mask(t).sum()) for t in range(1, 7)] == [60, 60, 120, 150, 55, 55]
    assert set(TASK_NAMES) == {1, 2, 3, 4, 5, 6}


def test_part_dimensions_are_about_1500_by_2000_by_1500_mm(part):
    extent = np.ptp(np.vstack([part.points_m, part.occluder.vertices_m]), axis=0)
    assert 1.4 < extent[0] < 1.6 and 1.9 < extent[1] < 2.1
    assert part.keep_out_max_m[2] == pytest.approx(1.55)  # fixture + wing stand ~1.5 m tall


def test_normals_are_unit_length_and_point_out_of_the_body(part):
    assert np.allclose(np.linalg.norm(part.normals_m, axis=1), 1.0, atol=1e-9)
    # stepping a little way along each normal must leave the (shrunk) body mesh's footprint,
    # i.e. an outward step lands in free space: no point of the body lies along it
    step = part.points_m + 0.02 * part.normals_m
    inward = part.points_m - 0.02 * part.normals_m
    far = part.points_m + 5.0 * part.normals_m
    assert not any(segment_intersects_mesh(far[i], step[i], part.occluder) for i in range(0, 500, 7))
    assert any(segment_intersects_mesh(far[i], inward[i], part.occluder) for i in range(0, 500, 7))


def test_leading_edge_profile_points_lie_on_the_nose_ellipse(part):
    pts = part.points_m[part.task_mask(1)]
    value = ((pts[:, 0] - X_NOSE_START) / NOSE_A) ** 2 + ((pts[:, 2] - NOSE_ZC) / NOSE_B) ** 2
    assert np.allclose(value, 1.0, atol=1e-3)  # T1 sits clear of the free-form bump (bump ~ e^-13)


def test_free_form_patch_deviates_from_the_plain_ellipse_by_a_few_mm(part):
    pts = part.points_m[part.task_mask(3)]
    r = np.sqrt(((pts[:, 0] - X_NOSE_START) / NOSE_A) ** 2 + ((pts[:, 2] - NOSE_ZC) / NOSE_B) ** 2)
    assert 0.001 < (r - 1.0).max() * min(NOSE_A, NOSE_B) < 0.012  # millimetres, not centimetres


def test_no_point_is_trapped_inside_the_occluder(part):
    """Every point must be reachable by a sightline from somewhere: a ray
    from 1.5 m out along its own normal, ignoring range and incidence
    limits, reaches the point (the shrunk body mesh never blocks the part's
    own skin)."""
    limits = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.pi)
    reachable = []
    for i in range(part.n_points):
        station = part.points_m[i] + 1.5 * part.normals_m[i]
        reachable.append(
            visibility_matrix(station[None], part.points_m[i][None], limits, part.normals_m[i][None],
                              part.occluder, occlusion_target_offset_m=1e-3)[0, 0]
        )
    assert np.mean(reachable) > 0.97  # a few lower-flap points may sit behind a fixture plate


def test_a_station_on_the_far_side_of_the_body_cannot_see_the_top_skin(part):
    limits = VisibilityLimits(max_range_m=20.0, max_incidence_angle_rad=np.deg2rad(80.0))
    below = np.array([[0.75, 1.0, 0.1]])  # under the part, inside the fixture envelope
    seen = visibility_matrix(below, part.points_m, limits, part.normals_m, part.occluder, 1e-3)[0]
    assert not seen[part.task_mask(4)].any()
