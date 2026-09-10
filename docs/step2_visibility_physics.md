# Step 2: visibility — geometry, not statistics

Companion to `src/geometry/mesh.py` and `src/geometry/visibility.py`.

Step 1 asked "given that a point is measured, how precise is the
measurement?" Step 2 asks the question that has to be answered first: "can
this point be measured from here at all?" Nothing in this step involves a
probability distribution — it's pure geometry, three independent yes/no
tests, per CLAUDE.md §2's chain:

```
station pose -> visibility -> instrument error model -> ...
```

## The three conditions

**Range.** Every instrument has a working envelope; beyond it, the
returned signal is too weak, or doesn't exist. Simple: distance from
instrument to target compared to a limit.

**Incidence.** A laser tracker's beam has to actually hit — and, for the
non-retroreflector case a general instrument's beam has to usefully
reflect or scatter off — the target surface. A beam meeting a surface
close to edge-on either doesn't return enough signal to lock onto, or
returns something contaminated by nearby geometry. The angle that matters
is the one between the beam and the surface's own local normal:

- **0** (beam along the normal) — straight-on, the best case.
- **near π/2** (beam almost in the surface's own plane) — grazing, the
  worst case.

This is the first place surface *orientation*, not just position, enters
the codebase. `Scene.target_normals_m` carries it: one outward unit
normal per target, "outward" meaning pointing away from the material the
target sits on, out into the space an instrument would occupy to see it
face-on. Concretely, `check_target_visibility` computes

```
cos(incidence) = dot(instrument_position - target_point, target_normal) / range
```

using the direction *from the target back to the instrument* (not
instrument-to-target) dotted with the outward normal — that's what makes
"straight-on" come out as `cos(incidence) = 1`, i.e. incidence angle 0,
rather than π.

If a target's normal isn't supplied (`target_normal_m=None`), the
incidence check is skipped, not failed — "no normal given" means "not
constraining this", the same convention `Scene.target_normals_m = None`
already uses for "no incidence check requested" scene-wide.

**Occlusion.** Something between the instrument and the target blocks
line of sight entirely — different in kind from the first two: it's not a
property of the target or the instrument alone, but of everything else in
the scene. This needs a representation of "everything else": a triangle
mesh (`geometry.mesh.TriangleMesh`), and a test of whether the straight
segment from instrument to target crosses any triangle in it
(`geometry.mesh.segment_intersects_mesh`, using the Möller–Trumbore
ray/triangle intersection — a standard, closed-form solve for the point
where a ray crosses a triangle's plane, then a cheap check that the point
actually lands inside the triangle rather than just its infinite plane).

## Why range and incidence are checked before occlusion

All three conditions are logically independent — a target can fail any
one of them regardless of the other two, so there's no physical reason
they have to be checked in a particular order. But their *cost* differs
enormously: range and incidence are each one dot product and a comparison
— O(1) per target. Occlusion means testing the sightline against every
triangle in the obstacle mesh — O(n_triangles) per target (see the
"acceleration" note in `geometry/mesh.py` for what would speed this up
later, deliberately not built yet).

So `check_target_visibility` checks range first, then incidence, and only
reaches for the ray cast if the target has survived both — an
out-of-range or grazing target never pays for a mesh test it can't
possibly need to fail. `test_range_checked_before_expensive_occlusion` in
`tests/test_visibility.py` pins this ordering down (a target that's both
out of range and would be occluded is reported `OUT_OF_RANGE`).

## What "occluded" doesn't mean

The segment test excludes intersections right at its own two ends (a
`t_margin` around `t=0` and `t=1` in `geometry.mesh.
segment_intersects_mesh`). Without that margin, a target sitting exactly
on — or numerically just past — the obstacle's own surface would register
as blocking its own sightline, which isn't a real occlusion, just floating
point noise at the boundary. `test_target_touching_the_mesh_is_not_self_occluded`
in `tests/test_mesh.py` checks this directly. This only matters for a
target that coincides with the *obstacle* mesh; the obstacle mesh in this
codebase represents blocking geometry (a wall, a fixture), not the part
being measured, so in practice it shouldn't come up often — but it's
cheap to get right and expensive to debug later if it's wrong.

## The gate

CLAUDE.md §5, step 2: *"a target placed behind a wall is reported
unmeasurable; the same target with the wall removed is measurable."*
`scenes/wall.stl` is a small synthetic wall (two triangles, a flat
rectangle in the plane x = 1.5 m) built for exactly this test. With the
wall present, a target at (3, 0, 0) — straight through the wall from an
instrument at the origin — comes back `OCCLUDED`; with `obstacles=None`
(the wall "removed"), the same target, same position, comes back
`VISIBLE`. See `tests/test_visibility.py::
test_target_behind_wall_is_occluded_wall_removed_is_visible`.
