"""Triangle mesh obstacles, and segment/triangle intersection for occlusion.

Build step 2 (CLAUDE.md §5): the only use of a mesh in this codebase so
far is as *blocking* geometry for the visibility check in
`geometry.visibility` -- something a laser beam can be stopped by (a wall,
a fixture, a wing spar). It is not the part being measured; the points
being measured are `Scene.target_points_m`, independent of any mesh.

**Loading -- a design decision, not the only reasonable one.** STL meshes
are parsed directly here (both the binary and ASCII variants of the
format), rather than by pulling in an external mesh library such as
`trimesh` or `numpy-stl`. The STL format is simple enough that a ~40 line
parser is easy to read and verify by eye, and CLAUDE.md §6 asks that new
dependencies be added only when needed and justified -- a whole mesh
library, with its own dependency tree, is a lot of surface area for
"read some triangles out of a file". The tradeoff: this parser is
minimal and doesn't handle the exotic corners of the format (colour
extensions, multiple `solid` blocks in one ASCII file, self-intersecting
or non-manifold meshes) -- if a mesh from a real CAD export doesn't load
cleanly, that's the first thing to suspect, and the fix is more parser,
not a rewrite.

**Acceleration -- deliberately not built yet.** `segment_intersects_mesh`
tests every triangle in the mesh for every query: O(n_triangles) per
query, so O(n_targets * n_triangles) for a whole scene. That's fine for
the target counts (tens to low thousands) and small synthetic meshes step
2's gate needs. If scenes grow to dense CAD meshes (many thousands of
triangles) checked against many targets, this is the first place a
spatial index belongs -- bucket the triangles once when the mesh is
built (a bounding volume hierarchy, or a uniform grid), then only test
the ray against the buckets it actually passes through, rather than every
triangle in the mesh. Not built here: step 2's gate is about correctness
(is an occluded point correctly flagged?), not scale, and CLAUDE.md §6
prefers clarity first, optimising once tests pass.
"""
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TriangleMesh:
    """A triangle-soup mesh: a flat list of vertices and, per triangle,
    which three vertices form it.

    Attributes
    ----------
    vertices_m : (V, 3) ndarray
        Vertex coordinates, in whatever frame the mesh was authored in
        (the scene's global frame, here -- CLAUDE.md keeps SI units
        throughout, so metres).
    faces : (F, 3) int ndarray
        Each row is three indices into `vertices_m`, the corners of one
        triangle. No assumption is made about winding order or vertex
        sharing between triangles -- STL itself doesn't share vertices
        (see `load_stl`), so this format doesn't either.
    """

    vertices_m: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        self.vertices_m = np.asarray(self.vertices_m, dtype=float)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices_m.ndim != 2 or self.vertices_m.shape[1] != 3:
            raise ValueError(f"vertices_m must have shape (V, 3), got {self.vertices_m.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3), got {self.faces.shape}")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices_m)):
            raise ValueError("faces reference a vertex index outside vertices_m")

    @property
    def n_triangles(self) -> int:
        return self.faces.shape[0]

    def triangle(self, i: int) -> np.ndarray:
        """The three (3,) vertex positions of triangle i, as a (3, 3) array."""
        return self.vertices_m[self.faces[i]]


def load_stl(path) -> TriangleMesh:
    """Load a triangle mesh from an STL file, binary or ASCII.

    STL has no header byte that unambiguously says which variant a file
    is (a binary file's 80-byte header is free-form text and can
    legally, if confusingly, start with the word "solid" too). The
    reliable test is arithmetic: a binary file's size must equal
    ``84 + 50 * triangle_count`` exactly, where ``triangle_count`` is the
    uint32 stored at byte 80. If that check passes, it's binary; if not,
    it's parsed as ASCII text.
    """
    path = Path(path)
    data = path.read_bytes()

    if len(data) >= 84:
        declared_count = struct.unpack_from("<I", data, 80)[0]
        expected_binary_size = 84 + 50 * declared_count
        if len(data) == expected_binary_size:
            return _parse_stl_binary(data, declared_count)

    return _parse_stl_ascii(data.decode("ascii", errors="replace"))


def _parse_stl_binary(data: bytes, n_triangles: int) -> TriangleMesh:
    # Binary STL, after the 80-byte header and 4-byte triangle count: each
    # triangle is 50 bytes -- a facet normal (3 float32, ignored: we don't
    # trust an author-supplied normal over the one implied by vertex
    # winding, and nothing here uses it yet), three vertices (3 float32
    # each), and a 2-byte "attribute byte count" almost nothing uses.
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attribute_byte_count", "<u2"),
        ]
    )
    records = np.frombuffer(data, dtype=dtype, count=n_triangles, offset=84)

    # STL triangles don't share vertices -- each triangle brings its own
    # three, even where two triangles touch in space. We keep that
    # structure rather than deduplicating: deduplication is an
    # optimisation (fewer vertices to store), not something correctness
    # here depends on.
    vertices_m = np.empty((3 * n_triangles, 3))
    vertices_m[0::3] = records["v0"]
    vertices_m[1::3] = records["v1"]
    vertices_m[2::3] = records["v2"]
    faces = np.arange(3 * n_triangles).reshape(n_triangles, 3)
    return TriangleMesh(vertices_m=vertices_m, faces=faces)


def _parse_stl_ascii(text: str) -> TriangleMesh:
    # ASCII STL is line-oriented: "vertex x y z" lines carry all the
    # geometry we need; "facet normal ...", "outer loop", "endloop",
    # "endfacet", "solid ..."/"endsolid" are structural and can be
    # skipped, since three consecutive "vertex" lines always make up one
    # triangle regardless of the surrounding keywords.
    vertices = []
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "vertex":
            if len(tokens) != 4:
                raise ValueError(f"malformed STL vertex line: {line!r}")
            vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])

    if len(vertices) % 3 != 0:
        raise ValueError(
            f"ASCII STL has {len(vertices)} vertex lines, not a multiple of 3 "
            "(every triangle needs exactly three)"
        )

    vertices_m = np.array(vertices, dtype=float) if vertices else np.empty((0, 3))
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return TriangleMesh(vertices_m=vertices_m, faces=faces)


def _ray_triangle_intersection_t(
    origin_m: np.ndarray,
    direction_m: np.ndarray,
    v0_m: np.ndarray,
    v1_m: np.ndarray,
    v2_m: np.ndarray,
    epsilon: float = 1e-9,
):
    """Möller-Trumbore ray/triangle intersection.

    Solves ``origin + t * direction = v0 + u * (v1 - v0) + v * (v2 - v0)``
    for the parametric distance ``t`` and the barycentric coordinates
    ``u, v``, without first computing the triangle's plane explicitly.

    `direction_m` need not be a unit vector -- when it's ``target - origin``
    (as `segment_intersects_mesh` below uses it), ``t`` comes out as the
    fraction of that segment: ``t = 0`` at the origin, ``t = 1`` at the
    target, which is exactly the range a caller wants to test.

    Returns
    -------
    float or None -- the parametric distance ``t`` if the ray hits inside
    the triangle, otherwise None (parallel to the triangle's plane, or a
    hit outside the triangle's bounds).
    """
    edge1 = v1_m - v0_m
    edge2 = v2_m - v0_m
    h = np.cross(direction_m, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
        return None  # ray (near-)parallel to the triangle's plane

    f = 1.0 / a
    s = origin_m - v0_m
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(direction_m, q)
    if v < 0.0 or u + v > 1.0:
        return None

    return f * np.dot(edge2, q)


def segment_intersects_mesh(
    origin_m: np.ndarray,
    target_m: np.ndarray,
    mesh: TriangleMesh,
    t_margin: float = 1e-6,
) -> bool:
    """Does the straight line segment from `origin_m` to `target_m` cross
    any triangle in `mesh`?

    Explicit loop over triangles, not a vectorised batch test -- see the
    module docstring's note on acceleration: this is the O(n_triangles)
    per query cost that a spatial index would later cut down, kept as a
    plain loop for now because it's the version that's easiest to read
    and trust.

    `t_margin` excludes intersections at the very ends of the segment
    (`t` within `t_margin` of 0 or 1): a hit at `t = 0` would just be the
    instrument's own mount, and a hit at `t = 1` would be the target
    point itself brushing the mesh, neither of which is "something in
    the way".
    """
    direction_m = np.asarray(target_m, dtype=float) - np.asarray(origin_m, dtype=float)
    for i in range(mesh.n_triangles):
        v0_m, v1_m, v2_m = mesh.triangle(i)
        t = _ray_triangle_intersection_t(origin_m, direction_m, v0_m, v1_m, v2_m)
        if t is not None and t_margin < t < 1.0 - t_margin:
            return True
    return False
