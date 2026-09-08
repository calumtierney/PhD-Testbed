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
contrast, resolve an *angle*: a fixed angular precision, say a few
microradians, regardless of range.

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
"~10:1 anisotropy" Hughes et al. (2011) report, and it's why CLAUDE.md §3
insists on carrying the full covariance matrix rather than a scalar: the
same instrument, at the same point, is roughly ten times *more* precise
along the beam than across it.

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

## The default noise values, and an honest gap

CLAUDE.md §4 quotes Hughes et al. (2011): **~40 µm lateral vs ~4 µm
radial at 2.5 m** — a 10:1 ratio. The Session 1 kickoff prompt also quotes
the paper's headline a-priori noise figures: `sigma_d` roughly 1.2 µm,
angular sigmas roughly 0.3 to 0.7 arcsec.

Those two things are in tension. Applying the headline angular figure
literally at 2.5 m:

```
0.5 arcsec = 2.42e-6 rad
lateral = 2.5 m * 2.42e-6 rad ≈ 6 um     (not ~40 um)
```

`sigma_d = 1.2 um` doesn't match the ~4 um radial figure either. Both
numbers are in the right qualitative direction (lateral clearly bigger
than radial) but off by roughly 5-6x in magnitude from the reference
table's 40/4 µm.

One plausible resolution: "a posteriori" noise (fitted from residuals of
a real network measurement, which is what Hughes et al. actually report)
tends to run larger than "a priori" (nominal encoder spec), because it
picks up unmodelled effects — air turbulence, mount stability, etc. — that
a pure encoder datasheet number doesn't include. That would make the
0.3-0.7 arcsec figure a spec-sheet number and the 40/4 µm figure a
real-network-residual number, which are allowed to disagree.

This repository could not confirm which is right: the primary source
(Hughes, Forbes, Lewis, Sun, Veal, Nasr, *Meas. Sci. Technol.* 22(4),
2011) is not accessible from this session — NPL's eprints host
(`eprintspublications.npl.co.uk`) is blocked by the environment's network
egress policy, and the paper is otherwise paywalled.

**What this codebase does about it** (per CLAUDE.md §6, "never fabricate a
reference number; mark clearly as placeholder"):

- `LaserTracker`'s defaults (`sigma_d_m = 4.0e-6`, `sigma_theta_rad =
  sigma_phi_rad = 3.3 arcsec`) are **calibrated to reproduce the CLAUDE.md
  §4 gate numbers exactly** (40/4 µm, 10:1, at 2.5 m) — because that gate,
  not the headline spec sheet figures, is what step 1 is required to pass.
- This is called out explicitly as a calibration, not a literal quote, in
  `laser_tracker.py`'s module docstring, with a `TODO(source)`.
- `test_anisotropy_direction_is_robust_to_literal_headline_defaults`
  confirms the *qualitative* result (lateral > radial, ellipsoid flattened
  along the beam) holds even under the literal 1.2 µm / 0.5 arcsec
  figures — only the *absolute magnitude* is sensitive to which numbers
  are used.

**Action for you (Calum):** if you have direct access to Hughes et al.
(2011) Table [wherever the 40/4 µm and a-posteriori sigma values are
reported), it would be worth confirming which noise values they actually
used to get 40/4 µm at their test range, and updating the `TODO(source)`
in `laser_tracker.py` accordingly — the calibration above matches the
target numbers but has not been checked against the paper's own stated
sigma_d/sigma_theta/sigma_phi.
