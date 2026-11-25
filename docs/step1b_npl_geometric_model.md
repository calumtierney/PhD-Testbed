# Step 1b: the NPL geometric model — what a tracker's mechanism does to a reading

Status: **the forward model is built and gated** (parameter set, readings from
true position and back, sensitivities, two-face physics). The network layers
that carry the parameter uncertainty into target covariance (consider and
estimate modes) are not yet built; this document is extended as they land.

Code: `src/instruments/npl_model.py`. Source: Hughes, Forbes, Lewis, Sun, Veal
and Nasr, *Laser tracker error determination using a network measurement*,
Meas. Sci. Technol. 22 (2011) 045103. Values: CLAUDE.md §4f.

## Why a second error model

Step 1 used the paper's *noise*: independent random errors on range and the
two angles (Table 4). That is the error you cannot remove by looking again.
The paper's subject is the other kind: the instrument's *mechanism* is not
perfect. The two rotation axes are not exactly at right angles and through
one point, the beam does not leave the gimbal exactly where it should, and
the angle encoders are not perfectly uniform. These produce errors that
repeat: the same shot gives the same wrong answer. A manufacturer measures
them and compensates inside the tracker; what is left is *known only to a few
micrometres and arcseconds*, and that leftover is one unknown per physical
instrument. It is shared by every shot, at every station the instrument is
moved to. It correlates the errors of different targets, so it cannot be
represented by an independent per-point term, which is what
`LaserTracker.systematic_std_m` is, and says so.

## What each parameter is, physically

Frame: standing axis z, azimuth θ from +x towards +y, elevation φ above the
horizontal (as `geometry.spherical`). With every parameter zero the model is
exactly the spherical conversion (a test).

| Parameter | What is wrong with the instrument | What a metrologist sees on a two-face check |
|---|---|---|
| e_x | the transit (elevation) axis does not intersect the standing axis: offset along x | elevation disagreement between faces, ∝ sin φ / range |
| b_y,0, b_z,0 | the beam leaves the gimbal offset sideways / vertically | a lateral / vertical offset: angle errors that fall as 1/range, opposite on the two faces |
| α | the transit axis is tilted out of the horizontal (the "trunnion-axis" error of a theodolite) | azimuth disagreement −2α tan φ: zero at the horizon, large near the zenith |
| γ | the beam is not at right angles to the transit axis (the "collimation error") | azimuth disagreement +2γ / cos φ: never zero, grows towards the zenith |
| a_A,q, b_A,q | azimuth encoder eccentricity (q = 1) and scale (q = 2) | odd harmonics show as an azimuth disagreement −2(a cos θ + b sin θ); even harmonics cancel between faces |
| a_E,0 | elevation zero offset | elevation sum 2a_E,0 at every elevation |
| a_E,q, b_E,q | elevation encoder harmonics | b_E,1 → 2b sin φ, a_E,2 → 2a cos 2φ; a_E,1 and b_E,2 cancel between faces |
| λ, μ | range offset and scale: d = (1 + μ)d* + λ | none: invisible to a two-face test |

`results/two_face_signatures.png` shows these fingerprints drawn from the
model with the closed-form laws laid over them. The laws are derived and
tested in `tests/test_npl_model.py::test_two_face_laws`, independently of
the paper's numbers, and pin the sign of every geometric parameter. The
figure is also why a network test is needed: λ, μ, the second azimuth
harmonic, a_E,1 and b_E,2 leave no trace in a two-face measurement at all.

## The model, and the places it differs from the paper

Equations 3–5 and 8–10 of the paper, written in this repository's
conventions (`npl_model.py` docstring has the full statement): the target is
at x = b(θ*, φ*) + d* n_B(θ*, φ*), where b is the point the beam leaves from
and n_B its direction after the head turns by the true azimuth θ* and the
beam is tipped by the true elevation φ* about the tilted, offset transit
axis. Readings are d = (1+μ)d* + λ, θ = θ* + Σ Fourier(θ*), φ = φ* + a_E,0 +
Σ Fourier(φ*).

- **Fourier terms at the true angle.** The paper evaluates them at the true
  angle in the forward model (4–5) but at the *observed* angle in its
  corrections (13, 16–17). The difference is second order in the encoder
  errors: under 1 nm out to 6.4 m (3 nm at 30 m), about a thousandth of the
  uncertainty of the points. This code uses the true angle throughout, so the
  correction `observed_to_true` is the exact inverse of the forward model;
  `convention="paper"` reproduces the paper's version (a test quantifies it).
- **Two solutions.** Equation 10 has two: the front face (θ*, φ*) and the back
  face near (θ*+π, π−φ*). `geometry.spherical.cartesian_to_spherical_back_face`
  gives the ideal back-face reading (the existing `cartesian_to_spherical`,
  using `arcsin`, cannot return an elevation above 90°).
- **A blind spot.** Because the beam is offset from the standing axis and
  tilted, no beam passes through the vertical axis through the instrument
  origin: a target there is unreachable, by about the component of the source
  offset perpendicular to the tilt, of order 100 µm at the paper's values.
  Pointing straight up reaches a point ~100 µm to one side. `solve_true_coordinates`
  raises a clear error for a target in the blind spot. This is not a convergence
  failure; it is geometry. It does not affect any planning scene.
- **The existing ideal Jacobian is singular at the zenith**
  (`geometry.spherical.spherical_jacobian`: the azimuth column is ∝ cos φ),
  and `network.design.shot_information_global` clamps cos φ at 1e-12 there.
  Not changed here (it would alter existing results); a test records the
  fact. The NPL forward map itself stays smooth, as the paper says: its
  azimuth column is small (a few hundred µm per radian at 3 m) but not zero.
  Note also that the visibility layer has no elevation limit, though the
  tracker's mechanical range is −60° to +77°.

## Round trip and solver

`true_to_observed` (target → readings) needs a solve; `observed_to_true`
(readings → target) is explicit. The solve is Newton's method from the ideal
spherical solution of the same face, finite-difference Jacobian, minimum-norm
least-squares step, step halving. Converged when the beam point matches the
target to 0.1 pm in every coordinate; at the paper's magnitudes it takes at
most 4 iterations (mean about 2), up to 8 for a beam pointing straight up.
Round trip with the uncompensated Hughes values is exact to about 5×10⁻¹⁴ m
(rounding), including 89.9° and straight-up beams, and at ten times the
paper's errors.

Sensitivities `observation_point_jacobian` and `observation_parameter_jacobian`
are central finite differences of the solve. The parameter steps (1 µm for
lengths, 1e-6 for μ, 1 arcsec for angles) are justified in `PARAMETER_STEPS`:
the readings are nearly linear in each parameter, so a central difference over
the parameter's own uncertainty scale is far above rounding and far below any
nonlinearity; halving or doubling the step changes nothing above 1e-6 of the
largest entry (a test).

## How to put in a different tracker

The parameter set is a description of *one physical instrument*, not constants
buried in a function. Fill an `NPLErrorParameters` from a network-test fit with
`NPLErrorParameters.from_fit(values, covariance, label)` or read one with
`from_json(path)` (a file written by `to_json`; the parameter names must match
exactly, in order, in metres and radians). `hughes_2011_uncompensated()` is
what the paper's API T3 did with its compensation off; `hughes_2011_residual()`
is the same instrument compensated with those values (zero mean, same
covariance) and is the one a plan should use, but it is optimistic for a tracker
between characterisations (drift is not in it). Replace both with GKN's own
tracker when its network test is available.

## Placeholders and open questions

- μ: the paper prints 1.000, which can only mean 1 + μ; used as μ = 0.
  `TODO(source)`.
- The correlations the paper mentions (γ–b_y, a_E,0–b_z, a_E,2–a_E,0) have no
  published magnitudes; covariances built from Tables 2–3 are diagonal.
- Table 4's "A" and "R" columns (σ_D,A = 1.216 µm, σ_D,R = 0.312″, …). For the
  two angle sensors the columns are one quantity in two units (0.485″ subtends
  2.351 µm at 1 m); for distance they are not (0.312″ would subtend 1.513 µm).
  Open question for NPL; Wang, Forbes and Maropoulos (2014) list the same
  absolute/relative pairs as an 8-parameter tracker model, which may help.
