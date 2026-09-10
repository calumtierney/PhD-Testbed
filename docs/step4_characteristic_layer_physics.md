# Step 4 physics: why the same point gives different tolerances different uncertainty

Companion to `src/characteristics/characteristic.py` and
`src/characteristics/tolerances.py`. Read this before the code, or to
check the physics independently of it. This is the step CLAUDE.md §1
calls out as where the project actually diverges from existing tools, so
it's worth being slow and concrete about it.

## The idea, without any algebra yet

A tolerance zone is a region of space. Different GD&T tolerance types
draw that region differently:

- **Position** (no axis given): a sphere around the true location. Every
  direction counts equally.
- **Position with an axis** (a hole toleranced only in cross-section): a
  cylinder. Two directions count; the third (along the hole) doesn't.
- **Flatness**: two parallel planes, oriented however the *measured
  surface itself* best fits. One direction counts (perpendicular to those
  planes); the other two (within the planes) don't.
- **Parallelism**: also two parallel planes, one direction — but that
  direction comes from an *external datum*, not from the surface being
  measured. A surface can be flat and still not parallel to something
  else, so this direction can genuinely differ from flatness's.

None of this is about how precisely the point was measured. It's about
which part of "how precisely" a given tolerance is even asking about.

## The measurement, unchanged from step 1

The point itself carries one physical object: a 3x3 covariance
describing its uncertainty in *every* direction — from step 1, a laser
tracker's covariance is a flattened disc, thin along the beam
(`sigma_d`, tiny) and wide across it (`d * sigma_theta`, `d * sigma_phi`,
much bigger). A tolerance doesn't change that covariance. It changes
which slice of it gets to matter.

## The operation: projection

For a tolerance that cares about one direction `d` (flatness,
parallelism), the quantity that actually gets compared to the tolerance
limit is the point's displacement *along* `d`: a single scalar,
`d . (measured - true)`. That scalar is a linear combination of a
Gaussian vector, so it's Gaussian too, with variance

```
Var(d . x) = d^T Cov(x) d
```

— exactly the classic "variance of a linear combination" result. For a
tolerance that cares about several directions at once (a spherical
position zone cares about all three), stack the unit directions as rows
of a matrix `D` (shape `(k, 3)`); the *projected* covariance is

```
Cov_projected = D Cov(x) D^T          (k x k)
```

`characteristics.characteristic.evaluate_characteristic` computes exactly
this, then reports `sqrt(trace(Cov_projected))` alongside it as an RSS
summary — the total spread across however many directions the tolerance
kept, for a quick number or a plot, while `Cov_projected` itself (not the
scalar) is what CLAUDE.md §3 says must be carried forward if anything
downstream needs the real shape.

## Worked example, with real numbers

Take the exact scene `tests/test_characteristics.py`'s gate test uses: one
target at `(2.5, 0, 0)`, one `LaserTracker` at the origin, default (Table
4) noise. Boresight geometry (`theta = phi = 0`) means the covariance is
already diagonal in the global x/y/z axes — no rotation needed to see the
principal directions by eye:

```
Cov = diag(1.216^2, 5.878^2, 8.413^2)  um^2
      (x: along the beam -- radial, tiny
       y: azimuth-lateral
       z: elevation-lateral)
```

Three tolerances, three different constraint directions, on this exact
same covariance:

| Tolerance | Constraint direction(s) | Projected variance | uncertainty_m |
|---|---|---|---|
| Parallelism, datum normal = x (along the beam) | `[1,0,0]` | `1.216^2` | **1.22 µm** |
| Flatness, surface normal = y | `[0,1,0]` | `5.878^2` | **5.88 µm** |
| Position, spherical (no axis) | `[1,0,0],[0,1,0],[0,0,1]` | `1.216^2+5.878^2+8.413^2` | **10.33 µm** |

The parallelism callout, by pure geometric luck of its datum pointing
straight down the tracker's own beam, inherits the *tiny* radial axis and
comes out nearly ten times tighter than the position callout on the exact
same point, from the exact same measurement. Flip the datum's normal to
a transverse direction instead and parallelism would inherit a *large*
axis and look nearly as loose as position. Nothing about the instrument
changed between these three rows — only which slice of one physical
ellipsoid each tolerance is asking about.

This is CLAUDE.md §3's claim ("a hole-position tolerance and a
parallelism tolerance on the same feature... carry different risk
because they project the error ellipsoid differently") made numeric.
It's also the reason existing tools' A-optimal planning (minimise the
trace, i.e. effectively always answering the *position*-shaped question)
can misallocate measurement effort: a plan optimised to minimise
`trace(Cov)` is implicitly optimising for the spherical-position case,
even for characteristics that are actually flatness or parallelism
callouts reading a much smaller (or, in principle, larger) slice of that
same covariance. Step 6, the headline experiment, is where this gets
tested against an actual planning comparison.

## Why the rotation-invariance test matters

`tests/test_characteristics.py::test_position_tolerance_trace_is_rotation_invariant`
checks something that looks almost too obvious to test: a spherical
position tolerance's projected variance equals `trace(Cov)` exactly, no
matter which orthonormal basis is used to represent "all three
directions" (`PositionTolerance` happens to use the plain x/y/z axes,
`np.eye(3)`). This is true because `trace(D Cov D^T) = trace(Cov D^T D) =
trace(Cov)` whenever `D` is a complete orthonormal basis (`D^T D = I`) —
projecting onto *some* full basis can't lose or gain total variance, it
can only redistribute it across the `k x k` matrix. It's a cheap,
exact correctness check on the projection machinery itself, independent
of any physical claim about the scene.

## What step 4 deliberately doesn't do yet

A `Characteristic` can hold several target points (for a flatness callout
spanning multiple measured points on one surface), and
`evaluate_characteristic` will happily project each of them — but it
returns a *per-point* breakdown, not a single combined number. Real GD&T
flatness is usually evaluated as the range (max minus min) of several
points' deviation from a best-fit plane, and the uncertainty of a range
statistic over *correlated* Gaussian variables (the points share network
solve history, so their projected deviations are not independent) is a
genuinely harder problem — order statistics, not a direct projection.
That's a plausible next elaboration, not something this build step's gate
needs (CLAUDE.md §9): the gate is that different tolerance types project
one point's covariance differently, which the worked example above shows
directly.
