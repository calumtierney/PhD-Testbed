# Step 1 physics: why the laser tracker's error ellipsoid is flattened

Companion to `src/instruments/laser_tracker.py`. Read this before reading
the Jacobian code, or to check the physics independently of it.

## The measurement, and where the noise enters

A laser tracker does not measure (x, y, z) directly. It measures three
things, each with its own, essentially independent, noise:

| Quantity | How it's measured | Typical noise character |
|---|---|---|
| range `d` | laser interferometer / absolute distance meter | small, and **independent of distance** |
| azimuth `theta` | angular encoder | small **angle**, independent of distance |
| elevation `phi` | angular encoder | small **angle**, independent of distance |

The interferometer counts wavelengths of light along the beam — it is one
of the most precise length measurements that exists, and its precision
doesn't degrade as the target moves further away. The encoders, by
contrast, resolve an *angle*: a fixed angular precision, regardless of
range.

## Why a fixed angular error becomes range-dependent

The three coordinates are related to Cartesian position through the usual
spherical geometry: pointing a beam of length `d` along the direction
`(theta, phi)` lands at

```
x = d cos(phi) cos(theta)
y = d cos(phi) sin(theta)
z = d sin(phi)
```

Nudge `theta` by a small angle `dtheta`, holding `d` fixed. The point at
the end of the beam sweeps sideways by a *distance*, not just an angle:
roughly `d * cos(phi) * dtheta` (the arc length subtended by that angle at
radius `d cos(phi)` — the horizontal distance out to the target). The same
angular wobble sweeps a bigger physical distance the longer the beam is.

Nudge `phi` similarly and the point sweeps vertically by roughly
`d * dphi`.

Nudge `d` itself, holding the direction fixed, and the point simply moves
`dd` along the beam — no distance-dependent amplification, because range
*is* the distance being measured.

So the three noise sources contribute position error in three different,
mutually perpendicular directions, with three different scalings:

| Direction | Contribution | Scales with range? |
|---|---|---|
| along the beam (radial) | `sigma_d` | no |
| across the beam, azimuthal | `d * sigma_theta` | yes, linearly |
| across the beam, elevation | `d * sigma_phi` | yes, linearly |

Even at a modest range, `d * sigma_theta` and `d * sigma_phi` overtake
`sigma_d`, because the encoders are so much noisier in relative terms than
the interferometer. The result is an error ellipsoid that is thin along
the beam and wide across it — a flattened disc, not a sphere. That's the
anisotropy Hughes et al. (2011) report, and it's why CLAUDE.md §3 insists
on carrying the full covariance matrix rather than a scalar: the same
instrument, at the same point, is several times *more* precise along the
beam than across it.

## Why the three directions don't mix (the Jacobian is exactly diagonalisable)

It isn't just that these three error contributions happen to point in
different directions — they are **exactly mutually orthogonal**, at any
target position, not only directly in front of the tracker. The three
columns of the spherical-to-Cartesian Jacobian
(`geometry.spherical.spherical_jacobian`) are the standard local
radial/azimuthal/elevation unit vectors of a spherical coordinate system,
and those are orthogonal by construction (`tests/test_spherical.py::
test_local_frame_is_orthogonal` checks this directly).

That orthogonality means: if `J` is the Jacobian and the spherical
measurement noise is diagonal (`d`, `theta`, `phi` are independent), then

```
Sigma_local = J @ diag(sigma_d^2, sigma_theta^2, sigma_phi^2) @ J.T
```

has its eigenvectors exactly along the radial/azimuthal/elevation
directions, with eigenvalues `sigma_d^2`, `(d cos(phi) sigma_theta)^2`,
and `(d sigma_phi)^2`. No approximation is needed to see that the ellipsoid
axes line up with "along the beam" and "across the beam" — it falls
straight out of the geometry of spherical coordinates, and
`tests/test_laser_tracker.py::test_off_axis_point_radial_axis_still_exact`
checks it numerically away from boresight, where the algebra is otherwise
messy.

## The noise values: Hughes et al. (2011) Table 4, used directly

CLAUDE.md §4a gives the model's inputs — a posteriori sensor noise
standard deviations, fitted from a real network measurement on an API T3:

| Sensor | Value | As used here |
|---|---|---|
| Distance | σ = 1.216 µm | `sigma_d_m` — a fixed length noise |
| Azimuth | σ = 0.485 arcsec (= 2.351 µm at 1 m) | `sigma_theta_rad` |
| Elevation | σ = 0.694 arcsec (= 3.365 µm at 1 m) | `sigma_phi_rad` |

These are used **exactly as reported**, with no fitting or retuning.
Propagated through the Jacobian above at 2.5 m (CLAUDE.md §4b), boresight
(`theta = phi = 0`, so `cos(phi) = 1`):

```
radial            sigma_d                      = 1.216 um
azimuth lateral   d * sigma_theta = 2.5 * 2.351 = 5.878 um  ("~5.9 um")
elevation lateral d * sigma_phi   = 2.5 * 3.365 = 8.413 um  ("~8.4 um")
```

giving an anisotropy ratio of about 4.8:1 (azimuth) to 6.9:1 (elevation)
against the radial direction — the "roughly 5:1 to 7:1" gate in §4b. This
is a genuine model *output*: nothing here was chosen to hit that number,
it falls out of propagating the measured sensor noise through the
Jacobian. `tests/test_laser_tracker.py::
test_boresight_anisotropy_matches_hughes_table4_prediction` checks it.

Azimuth and elevation noise are *not* equal (0.485 vs 0.694 arcsec) — the
elevation axis on this instrument is noisier, so the ellipsoid isn't
axisymmetric about the beam the way a simplified "one angular sigma"
model would suggest; it's a proper triaxial ellipsoid with three distinct
semi-axes.

The one input CLAUDE.md §4a itself flags as unresolved: the paper's
distance-channel angle figure, 0.312 arcsec, would subtend 1.513 µm at
1 m if treated the way the azimuth and elevation rows are — but the
distance row's own length figure is 1.216 µm, not 1.513 µm. That
inconsistency is in the source table, not introduced here. This code uses
`sigma_d = 1.216 µm` as a fixed length noise (distance doesn't need an
angle-to-length conversion in the first place) and does not use the 0.312
arcsec figure anywhere. `TODO(source)`: work out what that figure means.

## A separate, larger number: what Hughes et al. actually *observed*

CLAUDE.md §4c quotes a different set of figures from the same paper: 40 µm
lateral / 4 µm radial, unshielded, at 2.5 m — a 10:1 ratio. It would be a
mistake to treat that as a second, conflicting sensor-noise gate. It is a
**live distribution width**, not a fitted standard deviation, and — by the
paper's own account — it includes:

- the sensor noise modelled above,
- **atmospheric beam bending** (the air path refracts unevenly as it
  heats and cools, deflecting the beam sideways — a real physical effect,
  not sensor noise), and
- SMR nest repeatability (~4 µm — how precisely the retroreflector seats
  in its nest each time it's picked up and placed back), which turns out
  to dominate the *radial* direction, not the interferometer.

None of that extra physics is in the step 1 model, which is deliberately
just the spherical sensor noise (CLAUDE.md §5, step 1: "no occlusion,
spherical error model"). So step 1 should **not** be expected to
reproduce 40/4 µm, and it doesn't.

The evidence that the sensor-noise model above is nonetheless correct:
shielding the air path (removing most of the atmospheric term) brings the
observed ratio down from 10:1 to about 5.8:1 — which lands right inside
the 5–7:1 the sensor-noise-only model predicts. The unshielded 40 µm isn't
a different truth to reconcile; it's what you get once an atmospheric
term, which is a later step's physics (CLAUDE.md §4c's "gate for the
environmental step"), is added on top of what's built here.

## Summary for step 1

- **Gate (§4b):** lateral 5–9 µm, radial ~1.2 µm, ratio ~5:1–7:1, at
  2.5 m, from Table 4 sensor noise alone. **Met** — see
  `test_boresight_anisotropy_matches_hughes_table4_prediction`.
- **Not a step 1 gate (§4c):** the 40 µm / 4 µm unshielded scatter. That
  requires an atmospheric term and nest repeatability, neither of which
  belongs in the sensor error model. Revisit when a later step adds
  environmental effects.
