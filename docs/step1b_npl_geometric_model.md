# Step 1b: the NPL geometric model — what a tracker's mechanism does to a reading

Status: **built and gated.** The forward model, its inverse and the two-face
physics (gates 1-4); the consider term in the fast design covariance and in the
dense solve; estimate mode with priors and a rank test; the Hughes et al.
network test reproduced on a reconstructed layout (gate 5) and the paper's
variance split (gate 6); the noise-only/full-model side-by-side on the existing
experiments. **Not adopted as a default anywhere**: with `geometric_params`
unset every result is what it was. Which default is yours to decide; the numbers
for that decision are in "Side by side" below.

Code: `src/instruments/npl_model.py` (the model), `src/network/design.py` and
`src/network/solve.py` (consider and estimate modes), `src/network/hughes_network.py`
(gate 5), `src/network/variance_split.py` (gate 6). Source: Hughes, Forbes, Lewis, Sun, Veal
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


## Two modes, as built

A tracker's parameter errors are one 16-vector per *physical tracker*, with
covariance V_c, shared by every shot at every station it occupies. To the
network a parameter error is a bias on every shot's pseudo-measurement: the
point an ideal conversion of the corrected readings reports is shifted by
R_s P_k delta_c, with P_k (3 x 16) from `npl_model.reading_sensitivities`
(finite differences of explicit maps) or, for a compensated tracker whose
parameter values are zero, `shot_shift_zero_mean` (closed form, 30 times
faster, equal to 1e-9). P = J_sph G_c - F_c: encoder errors move the reported
point through the spherical Jacobian; axis offsets and tilts move the beam and
so move it the other way.

**(A) Consider mode, the planning default.** The tracker is compensated; the
parameters are *not* re-estimated; their errors are uncertain but fixed. The
target covariance is Sigma = N^-1 + S V S^T with S = N^-1 Q, Q having 16 columns
per tracker, accumulated per instrument id. `network.design` does this as a
low-rank update of the block-arrow elimination (linear in shots; algebra in the
module docstring); `network.solve` does it densely from the Jacobian. They agree
to 1e-5 of the largest entry (the dense solve's Jacobian is a finite difference);
the sensitivity S matches nonlinear network solves of data from a perturbed
tracker for every one of the 16 parameters (`tests/test_design_geometric.py`).
This is how a tracker is used on the shop floor with the manufacturer's
compensation applied inside the software.

**(B) Estimate mode.** The 16 parameters of each tracker are unknowns of the
network solve under a prior (`ParameterPrior`: flat, fixed, or a mean and
covariance per parameter; the paper's equation 18). The posterior covariance
comes from the full normal matrix after a Jacobi-scaled rank test: a network that
cannot determine a parameter (no length reference and no prior on mu) or a pose
raises `NetworkRankError` naming the weakest directions, where `pinv` would give
the undetermined direction zero variance. With the same prior, the estimate
posterior is never worse than consider mode in the matrix sense (a test). This
is how the network test itself works and what fitting GKN's own tracker needs.

**Which for planning.** (A), unless the real workflow re-estimates the
parameters: a plan with two-face and birdbath shots whose data are used to
self-calibrate (some USMN runs can), or a tracker not characterised for a long
time. (B) then gives a smaller covariance; (A) overstates it.

Instrument identity is per station (`instrument_ids`; default one tracker);
faces are per station (`("front",)` or `("front", "back")`). A second face
doubles the information (halves the noise-only covariance exactly) but a
different bias per shot: collimation error (gamma) cancels between faces to
second order, a range offset does not (tests).

## Gate 5: the Hughes network test

`results/npl_network_test.png`, `viz/run_network_test_reproduction.py`. The
layout is a reconstruction (the paper gives no coordinates): 8 wall nests, 1
floor nest, 6 bench nests; positions 0-4 as described; two birdbath lines
(horizontal 6-8, 45-degree diagonal 9-10, each through trackers 0 and 3 or 4);
126 readings (63 of 75 pairs on both faces), ranges 0.27-6.4 m, maximum elevation
56.5 degrees, mean cos 2phi 0.68 (paper: 0.69). Not tuned to anything below.

| | This reconstruction | Paper |
|---|---|---|
| Recovery over 100 seeds, within 1 / 2 / 3 fitted sigma | 68.5% / 96.3% / 99.9% | 68.3% / 95.4% / 99.7% expected |
| Fitted parameter uncertainties / Tables 2-3 | 0.56 to 1.10 | 1 |
| Target uncertainty (RSS) | 1.6 - 5.9 um | 2.1 - 4.9 um |
| Station position uncertainty (RSS) | 1.0 - 4.7 um | 1.1 - 2.2 um |
| Pointing uncertainty | 0.15 - 0.43 arcsec | 0.33 - 0.45 arcsec |
| Correlations the paper names | gamma-b_y -0.79, a_E0-b_z +0.59, a_E2-a_E0 -0.81 | "significant", no values |
| a_E2-a_E0 as the maximum elevation rises 57 -> 75 degrees | -0.81 -> -0.74 | "because no elevation exceeded 57 degrees" |

**The lambda-e_x correlation did not reproduce.** The paper adds the birdbath
lines because lambda and e_x can be strongly correlated. In this layout the
correlation coefficient is moderate with the lines (-0.31) and without (-0.23),
two faces or one. The reason: e_x acts on the range as e_x(1 - cos phi) on the
front face and e_x(1 + cos phi) on the back, while lambda acts equally on both,
so any elevation spread and the second face separate them. What the lines do
do is tighten lambda: removing them raises sigma(lambda) from 0.38 to 0.52 um
(front face only: 0.58 to 0.81, and sigma(e_x) from 2.6 to 5.8 um), while
removing four other readings from the same positions changes it by 3% (10%
front-only). So the lines matter, but not through the correlation coefficient the
paper's description suggests. It may be that the paper's network, which we only
know in outline, had less elevation spread than this one.

## Gate 6: the variance split

`results/npl_variance_split.json`. 1000 points, 1-5 m, random azimuth and
elevation within -60 to +77 degrees, half on each face, `hughes_2011_residual()`:
the geometric parameters carry **19.2%** of the coordinate variance (paper:
about 20%); the angle sensors **80.0%** (paper: about 80%); range noise 0.8%.
Per parameter the shares run from 0.02% (e_x) to 6.3% (a_E,0) against the
paper's 0.1-3.8%: the overall figure matches, the per-parameter band does not.
The most likely reason is the correlations the paper mentions and does not
quantify (this covariance is diagonal); as a what-if, not an input, putting
a_E,0-a_E,2 and the other two named correlations in at 0.5-0.9 in either sign
moves the overall share by only +-2 points, so the 19.2% stands either way, and
how the paper defines a per-parameter share is not stated.

## What the parameter errors do to the existing experiments

Consider mode, `hughes_2011_residual()` (zero mean, Tables 2-3 uncertainties,
no correlations), one tracker, the per-point characteristic layer as it
stands. `viz/run_npl_side_by_side.py`, `results/npl_side_by_side.json`.

**How big, per point** (parameter share of each characteristic's variance, min /
median / max, and the parameter contribution in quadrature):

| Scene, plan | Share of variance | Contribution |
|---|---|---|
| Headline two-cluster, A1 plan, 2 stations | 39 / 41 / 42% | 5.7 - 6.5 um |
| Long part on the track (anchor frame), A1, 3 stations | 4.2 / 4.4 / 4.6% | 5.9 - 13.1 um |
| Factory cell, risk-planned 4 stations, per point in the anchor frame | 27 / 41 / 45% | 3.5 - 7.0 um |
| Same plan, datum-referenced (joint covariance) | 6 / 15 / 41% | 1.2 - 1.5 um |

The last two rows are the same layout, same tracker: reading each point
alone attributes three to five times the parameter uncertainty that the
datum-referenced deviation actually suffers, because the common errors cancel
(next section).

**Headline experiment** (2 added stations on the 24-pose ring): all four
objectives choose the *same* plan under both models. Under the full model the
same plans are worse: A0/A1 sum u 44.3 -> 57.5 um, PFR 0.199 -> 0.348; B/C sum u
46.6 -> 59.3 um, PFR 0.193 -> 0.343 (PFA stays ~0). The ordering of objectives is
unchanged (B and C ahead of A0 and A1: by 3.2% of risk noise-only, by 1.4% under the full model).

**Station-track grid stage** (1.0 m spacing, 3 added stations, exhaustive; the
GA stages and the 0.5 m grid were not re-run): A1, B and C choose the same plan;
A0's changes by one station height (+x loop station 2.4 -> 1.0 m). Risk
1.777 -> 1.818 (+2.3%), sum u 265 -> 271 um. The anchor-frame uncertainties here
are 27-70 um, so a 6-13 um parameter contribution barely moves them.

**Speed** (200 evaluations each; the closed-form shift is in): per plan through
`evaluate_plan` (headline) 9.8 -> 14.1 ms (1.4x; objective C 110 -> 117 ms,
1.06x); the track path is unchanged within timing noise (35.5 -> 34.0 ms; C
about 140 ms either way); the fast design scorers the cell experiments use
1.30 -> 1.85 ms (DatumCellScorer, 1.4x) and 1.01 -> 1.62 ms (per point, 1.6x).
Before the closed-form shift the fast scorers were 5-8 times slower; the
finite-difference route is kept for parameters with nonzero values.

## Section 5: what the per-point characteristic layer does with this

`tests/test_design_geometric.py::test_common_parameter_errors_cancel_in_the_distance_between_nearby_targets`.
Two targets 0.3 m apart, measured from the same trackers, have parameter-induced
errors correlated at 0.999; the parameter-induced variance of their separation
is 0.14% of the sum of their individual variances (1.6% at 1 m, 22% at 3 m, where
the correlation is 0.83). The noise part cancels far less (about 20%), because
it is independent per shot except through the station poses.

So anything defined *relative to other measured points* -- a hole position
against its datum features, a distance, a step -- benefits from the common
errors cancelling, and `evaluate_characteristic`, which reads only a point's
own 3 x 3 covariance (`target_point_covariance(i)`), overstates its parameter
uncertainty by up to three orders of magnitude for nearby points. It could
therefore change which plans win for the wrong reason.

Which consumers are affected:

  * the **factory-cell experiments** (`planning.cell.DatumCellScorer`,
    `PartFrameScorer`) already use the joint covariance and the S-transformation
    to the datum (`network.design.datum_referenced_functional`); the consider
    term enters through `design_joint_covariance` and the cancellation happens
    automatically. No change needed;
  * the **headline and track experiments** (`planning.objectives.evaluate_plan`
    and `evaluate_characteristic`) read points one at a time in the anchor
    frame, so they carry the full parameter variance. That is exactly what the
    side-by-side above measures;
  * the wing-box inspection experiment (`planning/inspection*.py`) was not touched.

`RelativeCharacteristic` has since been built (below).

**Built: `RelativeCharacteristic`** (`characteristics.characteristic`,
`tests/test_relative_characteristic.py`). It sits beside `Characteristic`, holding a
target index, the indices of its datum targets and a tolerance.
`evaluate_relative_characteristic(characteristic, covariance_source, points_m, subset=None)`
takes a joint covariance source -- `design_joint_covariance`'s output over `subset`, or a
`NetworkSolveResult`, which now has `joint_point_covariance(indices)` -- applies
`network.design.datum_referenced_functional` (the S-transformation to the datum frame fitted
to the datum targets) and projects onto the tolerance's constraint directions. It returns the
3 x 3 datum-referenced covariance (the authoritative object), the projection, the
RSS summary `evaluate_characteristic` reports and the worst-direction standard deviation
`PartFrameScorer` reports. `evaluate_characteristic` and its tests are untouched; a per-point
characteristic stays correct for a quantity that really is defined in the tracker's frame.

What the tests show:

  * for a profile tolerance it reproduces `DatumCellScorer` and `PartFrameScorer` to 1e-9 (with
    and without a placement floor, which the scorers add on top and the layer does not);
  * five points inside a 0.3 m cluster, three trackers, Hughes residual covariance: the
    parameter-induced variance relative to the datum is under a tenth of the same quantity in
    the anchor frame (the sensor noise does not cancel nearly as much);
  * the same evaluation from a block-diagonal covariance (the per-point blocks
    `evaluate_characteristic` works from) overstates the relative uncertainty by more than
    1.5x: the cancellation lives in the off-diagonal blocks, so it needs the joint covariance;
  * a network solve and the fast design covariance agree; two tolerances on one point still
    give different uncertainties; a parallelism datum's establishment variance is added as in
    `evaluate_characteristic`;
  * too few or collinear datum targets, or an unobserved point or datum target, raise
    `ValueError` rather than returning a number: a deviation from a frame that was never
    realised is undefined, not zero.

**Not wired in.** The headline and track experiments still call `evaluate_characteristic`
(anchor frame); they have not been switched to `RelativeCharacteristic`, so their parameter
share is still the per-point one measured above. Doing that changes what those experiments
mean (which datum targets define the part frame in the headline scene is a modelling choice,
and the headline scene has none), so it is a separate decision.

## Placeholders

Those GKN's own network test will replace (fill an `NPLErrorParameters` with
`from_fit` / `from_json`; run `hughes_2011_network_scenario`'s machinery on
their layout):

  * the parameter values and covariance (`hughes_2011_*` are one API T3 in IFM
    mode, from 2011);
  * the layout: every coordinate, the 12 obscured pairs and the station yaws of
    `network.hughes_network` (TODO(source));
  * the sensor noise (Table 4 is that T3's a posteriori noise);
  * the prior on mu (5e-7, the paper's value, traceable to the laser wavelength).

Those that need another source:

  * mu itself (printed "1.000": read as 1 + mu, mu ~ 0);
  * the correlations the paper names but does not quantify (the covariances here
    are diagonal; the reconstruction suggests gamma-b_y about -0.8, a_E0-b_z about
    +0.6, a_E2-a_E0 about -0.8, but that is this layout, not a source);
  * the 0.312 arcsec distance row of Table 4 (CLAUDE.md §4a; open question for NPL);
  * what `hughes_2011_residual` should be for a factory-compensated tracker
    between characterisations (drift is not in it);
  * the elevation limits of the instrument that will actually be used
    (`geometry.visibility.API_T3_MIN_ELEVATION_RAD`/`MAX`, -60 to +77 degrees, are the API
    T3's; GKN's tracker has its own);
  * everything in `systematic_std_m` (refraction, thermal drift, nest
    repeatability, gravity, fixturing), which stays a placeholder floor.

## Elevation limits in the visibility layer

A laser tracker's head tilts through a limited arc (Hughes et al. 2011 §8, API T3: -60 to +77
degrees above the instrument's horizontal plane) and azimuth is a full circle, so a tracker
cannot look straight up at a ceiling target or steeply down at the floor under it.
`geometry.visibility.VisibilityLimits` takes optional `min_elevation_rad` and
`max_elevation_rad` (default `None`: unconstrained, which reproduces every earlier result),
and `check_target_visibility` / `visibility_matrix` take the station's orientation
(`instrument_orientation`, `station_orientations`). A target outside the range reports
`VisibilityReason.ELEVATION_OUT_OF_RANGE`.

Elevation is measured in the **instrument's own frame** (above its local x-y plane, the
convention of `geometry.spherical`), so the answer depends on the pose's orientation: for a
levelled tracker the local z is world up. With no orientation given the check assumes a
levelled tracker. The planning call sites (`planning.objectives`, the cell scorers) pass each
pose's own orientation. `levelled_orientation` and the track candidates are levelled;
`hemisphere_candidate_poses` (the headline ring and the wing experiment's ring) are
`look_at` poses, tilted to point at the part, in which the limit would be measured from the
boresight and so never bind. The limit only means what it says on levelled poses.
`API_T3_MIN_ELEVATION_RAD` / `API_T3_MAX_ELEVATION_RAD` are named constants to pass in
explicitly: nothing enables them silently, and GKN's tracker's range is `TODO(source)`.

Does it bind? Measured, levelled stations:

| Scene | Candidates | Visible pairs without -> with the limit |
|---|---|---|
| Factory cell (28 reference nests + part), track, 0.8-3.0 m | 228 | 6220 -> 6220 |
| Long part alone, same candidates | 228 | 1846 -> 1846 |
| Wing, random stations in the search bounds (1-10 m high) | 2000 | 10083 -> 9499 (-5.8%) |

On the track scenes the limit changes nothing: the stations are low and the targets are
metres away. On the wing it does: a station high above the part (above 6 m: 6686 -> 6192
pairs) is looking down more steeply than -60 degrees at the targets under it. So switching it
on would change the wing experiment's feasible set and nothing else measured here.
