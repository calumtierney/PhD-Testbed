"""A synthetic, parametric stand-in for the assembly wing box of the digital
twin example (Figure 9.13): a part about 1.5 m x 2.0 m x 1.5 m on a
vertical-plate fixture, with six inspection tasks and ~500 inspection
points.

**This is not the real part.** No CAD file or inspection-point list was
available, so the outline, the point counts per task and the fixture are
representative inventions that follow the figure and the task list. They
are chosen only to give the right *kind* of geometry (a rounded leading
edge, a thin tapered flap, a flat top skin, a free-form patch, a fixture
that blocks low sightlines); no result should be read as a statement about
the real part's measurability. `build_wing_box_part()` returns plain arrays
and a `TriangleMesh`, so real data replaces it by loading the inspection
points, their normals and an STL (`geometry.mesh.load_stl`) in its place.

Frame: x along the chord (flap at x = 0, leading edge nose at x = 1.5),
y along the span (0 to 2 m), z up, floor at z = 0.

The six tasks (task ids 1-6, as in the source description):

  T1  profile of the leading edge at the right end   (y = 0.10 m)
  T2  profile of the leading edge at the left end    (y = 1.90 m)
  T3  one free-form surface on the leading edge      (mid-span patch)
  T4  the flat surface on the top
  T5  profile of the flap at the right end           (y = 0.10 m)
  T6  profile of the flap at the left end            (y = 1.90 m)

The point counts per task (60, 60, 120, 150, 55, 55 = 500) are a placeholder
split of "around 500"; the source gives only the total.

**The occluder mesh** is the wing body, shrunk inward by `BODY_SHRINK_M`,
plus three fixture plates. The shrink matters: the inspection points lie on
the true curved skin, the mesh is a faceted approximation of it, and a facet
that sits even a fraction of a millimetre outside a point would block that
point's own sightline against the part's own skin. Shrinking by several
millimetres (more than the facet sag) puts every point outside the mesh, so
only genuine occluders -- the far side of the body, the fixture -- block.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from geometry.mesh import TriangleMesh, box_mesh

Z_TOP, Z_BOTTOM = 1.40, 1.00
X_FLAP_ROOT, X_NOSE_START = 0.30, 1.15
NOSE_A, NOSE_B, NOSE_ZC = 0.35, 0.20, 1.20  # nose ellipse: semi-axes in x, z; centre height
FLAP_TE_UPPER, FLAP_TE_LOWER = 1.22, 1.18  # trailing-edge skin heights at x = 0
SPAN_M = 2.0
Y_RIGHT_END, Y_LEFT_END = 0.10, 1.90
BODY_SHRINK_M = 0.004
FIXTURE_PLATE_Y_M = (0.30, 0.985, 1.67)
FIXTURE_PLATE_THICKNESS_M = 0.03

# Free-form feature on the leading edge (T3): a smooth outward bump, part of
# the *nominal* surface (the CAD shape the points are compared against).
BUMP_AMPLITUDE_M, BUMP_Y_CENTRE_M, BUMP_Y_WIDTH_M, BUMP_PSI_WIDTH_RAD = 0.006, 1.0, 0.25, 0.6

TASK_NAMES: Dict[int, str] = {
    1: "T1 leading-edge profile, right end",
    2: "T2 leading-edge profile, left end",
    3: "T3 free-form leading-edge surface",
    4: "T4 flat top surface",
    5: "T5 flap profile, right end",
    6: "T6 flap profile, left end",
}


@dataclass(frozen=True)
class WingBoxPart:
    """The synthetic part.

    points_m (500, 3), normals_m (500, 3, unit, outward), task_ids (500,)
    in 1..6; `occluder` blocks sightlines; `keep_out_min_m/max_m` is the
    box a tracker cannot stand inside; `station_bounds_m` the (x, y, z)
    box a tracker may be placed in (tripod to raised platform).
    """

    points_m: np.ndarray
    normals_m: np.ndarray
    task_ids: np.ndarray
    occluder: TriangleMesh
    keep_out_min_m: np.ndarray
    keep_out_max_m: np.ndarray
    station_bounds_m: Tuple[Tuple[float, float], ...]

    @property
    def n_points(self) -> int:
        return self.points_m.shape[0]

    def task_mask(self, task_id: int) -> np.ndarray:
        return self.task_ids == task_id


def _nose_surface(psi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nominal leading-edge surface S(psi, y), psi in [-pi/2, pi/2]
    (-pi/2 = lower tangent point, 0 = nose tip, +pi/2 = upper tangent
    point): an elliptical nose plus the smooth free-form bump."""
    psi, y = np.broadcast_arrays(np.asarray(psi, float), np.asarray(y, float))
    x0 = X_NOSE_START + NOSE_A * np.cos(psi)
    z0 = NOSE_ZC + NOSE_B * np.sin(psi)
    # outward unit normal of the ellipse in the x-z plane
    nx, nz = np.cos(psi) / NOSE_A, np.sin(psi) / NOSE_B
    norm = np.hypot(nx, nz)
    nx, nz = nx / norm, nz / norm
    bump = BUMP_AMPLITUDE_M * np.exp(
        -(((y - BUMP_Y_CENTRE_M) / BUMP_Y_WIDTH_M) ** 2) - ((psi / BUMP_PSI_WIDTH_RAD) ** 2)
    )
    return np.stack([x0 + bump * nx, y, z0 + bump * nz], axis=-1)


def _nose_normals(psi: np.ndarray, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Outward unit normals of `_nose_surface`, from the cross product of
    its two tangent vectors (central differences)."""
    d_psi = (_nose_surface(psi + h, y) - _nose_surface(psi - h, y)) / (2 * h)
    d_y = (_nose_surface(psi, y + h) - _nose_surface(psi, y - h)) / (2 * h)
    n = np.cross(d_y, d_psi)  # (dS/dy x dS/dpsi) points outward for this parametrisation
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    centre = np.stack([np.full_like(np.asarray(psi, float), X_NOSE_START), y, np.full_like(np.asarray(psi, float), NOSE_ZC)], axis=-1)
    outward = np.einsum("...i,...i->...", n, _nose_surface(psi, y) - centre) > 0
    return np.where(outward[..., None], n, -n)


def _flap_surface(x: np.ndarray, y: float, upper: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Points and outward normals on the tapered flap's upper or lower skin."""
    slope = (Z_TOP - FLAP_TE_UPPER) / X_FLAP_ROOT if upper else (Z_BOTTOM - FLAP_TE_LOWER) / X_FLAP_ROOT
    z0 = FLAP_TE_UPPER if upper else FLAP_TE_LOWER
    z = z0 + slope * x
    pts = np.stack([x, np.full_like(x, y), z], axis=1)
    n = np.array([-slope, 0.0, 1.0 if upper else -1.0])
    n = n / np.linalg.norm(n)
    return pts, np.tile(n, (len(x), 1))


def _section_outline(shrink_m: float) -> np.ndarray:
    """Closed (x, z) outline of the wing section, clockwise, offset inward
    by `shrink_m` along averaged vertex normals."""
    psi = np.linspace(np.pi / 2, -np.pi / 2, 26)[1:-1]
    nose = np.stack([X_NOSE_START + NOSE_A * np.cos(psi), NOSE_ZC + NOSE_B * np.sin(psi)], axis=1)
    outline = np.vstack([
        [[0.0, FLAP_TE_UPPER], [X_FLAP_ROOT, Z_TOP], [X_NOSE_START, Z_TOP]],
        nose,
        [[X_NOSE_START, Z_BOTTOM], [X_FLAP_ROOT, Z_BOTTOM], [0.0, FLAP_TE_LOWER]],
    ])
    edges = np.roll(outline, -1, axis=0) - outline
    edge_normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)  # outward for a clockwise outline
    edge_normals /= np.linalg.norm(edge_normals, axis=1, keepdims=True)
    vertex_normals = edge_normals + np.roll(edge_normals, 1, axis=0)
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return outline - shrink_m * vertex_normals


def _body_mesh(shrink_m: float) -> TriangleMesh:
    """The wing body as an extruded convex prism, shrunk inward by `shrink_m`."""
    outline = _section_outline(shrink_m)
    n = len(outline)
    y_lo, y_hi = shrink_m, SPAN_M - shrink_m
    verts = np.vstack([
        np.column_stack([outline[:, 0], np.full(n, y_lo), outline[:, 1]]),
        np.column_stack([outline[:, 0], np.full(n, y_hi), outline[:, 1]]),
    ])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]  # side quad
    for i in range(1, n - 1):
        faces += [[0, i, i + 1], [n, n + i, n + i + 1]]  # fan-triangulated end caps (convex outline)
    return TriangleMesh(vertices_m=verts, faces=np.array(faces))


def _merge_meshes(meshes) -> TriangleMesh:
    verts, faces, offset = [], [], 0
    for m in meshes:
        verts.append(m.vertices_m)
        faces.append(m.faces + offset)
        offset += len(m.vertices_m)
    return TriangleMesh(vertices_m=np.vstack(verts), faces=np.vstack(faces))


def build_wing_box_part() -> WingBoxPart:
    """Build the synthetic wing box (see the module docstring)."""
    points, normals, tasks = [], [], []

    def add(pts, nrm, task):
        points.append(pts)
        normals.append(nrm)
        tasks.append(np.full(len(pts), task))

    psi_profile = np.linspace(-np.pi / 2, np.pi / 2, 60)
    for task, y in ((1, Y_RIGHT_END), (2, Y_LEFT_END)):
        yy = np.full_like(psi_profile, y)
        add(_nose_surface(psi_profile, yy), _nose_normals(psi_profile, yy), task)

    psi_grid, y_grid = np.meshgrid(np.deg2rad(np.linspace(-60, 60, 8)), np.linspace(0.5, 1.5, 15), indexing="ij")
    psi_flat, y_flat = psi_grid.ravel(), y_grid.ravel()
    add(_nose_surface(psi_flat, y_flat), _nose_normals(psi_flat, y_flat), 3)

    xs, ys = np.meshgrid(np.linspace(0.40, 1.05, 10), np.linspace(0.20, 1.80, 15), indexing="ij")
    top = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, Z_TOP)], axis=1)
    add(top, np.tile([0.0, 0.0, 1.0], (len(top), 1)), 4)

    x_flap = np.linspace(0.28, 0.02, 27)
    for task, y in ((5, Y_RIGHT_END), (6, Y_LEFT_END)):
        up_pts, up_n = _flap_surface(x_flap, y, upper=True)
        lo_pts, lo_n = _flap_surface(x_flap, y, upper=False)
        te_pt = np.array([[0.0, y, 0.5 * (FLAP_TE_UPPER + FLAP_TE_LOWER)]])
        add(np.vstack([up_pts, lo_pts, te_pt]), np.vstack([up_n, lo_n, [[-1.0, 0.0, 0.0]]]), task)

    plates = [
        box_mesh(np.array([0.25, y0, 0.0]), np.array([1.25, y0 + FIXTURE_PLATE_THICKNESS_M, Z_BOTTOM - 0.01]))
        for y0 in FIXTURE_PLATE_Y_M
    ]
    return WingBoxPart(
        points_m=np.vstack(points),
        normals_m=np.vstack(normals),
        task_ids=np.concatenate(tasks),
        occluder=_merge_meshes([_body_mesh(BODY_SHRINK_M)] + plates),
        keep_out_min_m=np.array([-0.15, -0.15, 0.0]),
        keep_out_max_m=np.array([1.65, SPAN_M + 0.15, 1.55]),
        station_bounds_m=((-3.5, 5.0), (-3.5, 5.5), (0.8, 3.0)),
    )
