# CLAUDE.md — LVM Measurement Planning Testbed

Persistent context for this repository. Read this before doing anything.

---

## 1. What this project is

PhD research, Calum Tierney, School of Mechanical and Aerospace Engineering, Queen's University Belfast. Supervisor: Prof. Paul Maropoulos. Industrial partners: GKN Aerospace, AMIC.

**Domain:** large-volume metrology (LVM) — measuring aerospace structures from 2 m to 30 m with laser trackers, photogrammetry and similar instruments.

**The problem:** deciding which instruments to use, where to put them, and in what order, is a slow expert task. Existing tools automate it by minimising measurement uncertainty.

**The contribution, in one sentence:**

> Large-scale metrology plans measurements A-optimally (minimising the trace of the coordinate variance matrix). This project plans them *goal-oriented*, where the quantity of interest is a conformity decision.

In practice this means replacing the objective function, not the optimiser. Prior work already supports weighted objectives; nobody has a principled way to set the weights. We derive them from JCGM 106 decision risk.

**This repository** builds the simulation testbed that makes that comparison possible.

---

## 2. The core chain

Everything in this codebase implements one chain:

```
station pose
  → visibility (occlusion, range, incidence angle)
  → instrument error model
  → coordinate covariance  [existing tools stop here and sum the diagonal]
  → characteristic uncertainty (covariance projected onto the tolerance direction)
  → conformity decision risk (JCGM 106)
```

Keep this chain visible in the module structure. Each link is a separate, separately testable module.

---

## 3. Non-negotiable technical decisions

These come from primary literature and are not up for redesign without discussion.

**Uncertainty is a full covariance matrix, never a scalar.**
Laser tracker uncertainty is strongly anisotropic — the sensor noise model alone (§4a/4b) gives roughly 5:1 to 7:1 lateral-to-radial at 2.5 m, and the full observed scatter including atmospheric and nest effects (§4c) reaches roughly 10:1 unshielded. Schmitt et al.'s CIRP keynote states explicitly that distance uncertainty cannot be summarised by an `A + B·L` formula and that independence assumptions give a poor guide to measurement strategy capability. A scalar uncertainty makes the risk calculation wrong in a direction-dependent way. Never reduce to a magnitude before the characteristic layer.

**Risk depends on direction.**
A hole-position tolerance and a parallelism tolerance on the same feature, measured from the same station, carry different risk because they project the error ellipsoid differently. The characteristic layer must know what direction each tolerance constrains.

**The optimiser is deliberately conventional and held constant.**
The experiment varies the objective, not the search method. Use grid search or a standard GA. Do not invent an optimiser — a novel optimiser would make results unattributable.

**Process capability priors are per feature-cluster, not per characteristic.**
LVM is small-batch; classical SPC needs ~50 samples and aerospace does not supply them. Priors are elicited for clusters of similar features sharing a production process (Schmitt et al.). Cluster assignment carries its own uncertainty and must be reported.

**Exploit the block-diagonal structure before reaching for surrogates.**
Each sensor reading involves one target, so the normal equations are block-diagonal: `O(m·n₀²)`, scaling roughly linearly in target count rather than cubically. A slow baseline would inflate any later surrogate speedup claim.

---

## 4. Reference numbers for validation

**Read this section carefully. An earlier version of this file conflated two different kinds of quantity and produced an unachievable gate. The distinction below is the whole point.**

### 4a. Model inputs — Hughes et al. 2011, Table 4

A posteriori sensor noise standard deviations from the network fit on an API T3:

| Sensor | Value (length) | Value (angle) |
|---|---|---|
| Distance | σ = 1.216 µm | (0.312 arcsec — see note) |
| Azimuth | 2.351 µm at 1 m | σ = 0.485 arcsec |
| Elevation | 3.365 µm at 1 m | σ = 0.694 arcsec |

For both angular sensors the two columns are the *same quantity* expressed two ways: 0.485 arcsec subtends 2.351 µm at 1 m, and 0.694 arcsec subtends 3.365 µm at 1 m. Both check out exactly. Use the angular values and scale with range.

Note: the distance row does not follow that pattern (0.312 arcsec would subtend 1.513 µm at 1 m, not 1.216 µm). This is unresolved. Use σ_d = 1.216 µm as a fixed length noise and do not use the 0.312 arcsec figure until someone works out what it means. `TODO(source)`.

### 4b. What the model should therefore predict — this is the step 1 gate

At 2.5 m range, propagating 4a through the spherical-to-Cartesian Jacobian:

| Quantity | Expected 1σ |
|---|---|
| Lateral, azimuth-driven | ≈ 5.9 µm |
| Lateral, elevation-driven | ≈ 8.4 µm |
| Radial (along beam) | ≈ 1.2 µm |
| **Anisotropy ratio** | **roughly 5:1 to 7:1** |

**Gate:** lateral σ between about 5 and 9 µm, radial σ about 1.2 µm, ratio in the 5–7:1 band, with the long axes of the ellipsoid perpendicular to the beam. Do not tune the inputs to hit a target — the inputs are fixed by 4a and the ratio is an output.

### 4c. Observed quantities requiring physics not yet in the model

These are **live distribution widths** from Hughes et al. Section 8, not standard deviations, and they include effects the step 1 model does not contain. They are gates for *later* steps, not step 1.

| Observation | Value | What it includes |
|---|---|---|
| Lateral width at 2.5 m, unshielded | 40 µm | Sensor noise **plus atmospheric beam bending** plus nest repeatability |
| Same, with the air path shielded | 23 µm | Mostly sensor noise and nest repeatability |
| Width along beam | 4 µm | Dominated by nest repeatability, not the interferometer |
| SMR nest repeatability | ≈ 4 µm | — |

The paper states the scatter is a mixture of atmospheric beam bending, angular sensor accuracy and target nest repeatability. Note that shielding drops the ratio from 10:1 to about 5.8:1, which lands squarely on the 5–7:1 the sensor model predicts. That agreement is the evidence the model is right; the unshielded 40 µm is what you get once turbulence is added.

**Gate for the environmental step (later):** adding an atmospheric term should take the predicted lateral spread from the shielded regime to roughly the unshielded observation, not the other way round.

### 4d. Other reference values

| Quantity | Value | Source |
|---|---|---|
| Angular share of coordinate variance (1000 pts, 1–5 m) | ~80%, model parameters ~20% | Hughes et al. 2011 |
| Leica AT901 MPE | 15 µm + 6 µm/m at 2σ | Francis et al. 2016 |
| Uncertainty at 5 m vs ±50 µm tolerance | 45 µm — consumes 90% of tolerance | Francis et al. 2016 |
| Mean uncertainty vs station count (1→4) | 26.5 → 16.6 → 13.7 → 10.7 µm | Wang, Forbes & Maropoulos 2014 |
| Thermal stabilisation error, first hours | 20–80 µm | Muñoz et al. 2016 |
| Gravity contribution to a 22 µm gear tolerance | 8 µm | Schmitt et al. 2016 |
| Network test scale | 15 targets, 5 stations, 126 obs, 55 min | Hughes et al. 2011 |
| Target location uncertainty from that fit | 2.1–4.9 µm | Hughes et al. 2011 |

**Instrument error model:** the NPL 14-parameter geometric model (Hughes et al. 2011, Table 1) — range offset λ, scale μ, transit axis offset eₓ, beam offsets b_y,0 and b_z,0, transit and beam axis angles α and γ, plus first and second order Fourier terms on each encoder. It returns a variance matrix V_h, not just values. The geometric parameters are a *step 3* concern; step 1 needs only the 4a noise values.

### 4e. Rule

If a computed result disagrees with 4b, the code is wrong. If it disagrees with 4c, check first whether the missing physics explains the gap — usually it does. **Never adjust an input in 4a to make an output match.** Inputs are measured; outputs are predictions; the whole value of the testbed is that the second follows from the first.

---

## 5. Build order

Build the whole chain thin before making any link thick. Each step must pass its gate before the next begins.

| Step | Deliverable | Gate |
|---|---|---|
| 1 | Single tracker, N points, no occlusion, spherical error model → per-point covariance | Reproduces ~10:1 anisotropy at 2.5 m |
| 2 | Ray-cast occlusion on triangle mesh; range and incidence-angle limits | Points behind an obstacle are correctly unmeasurable |
| 3 | Multi-station weighted least-squares network solve → full covariance | Uncertainty falls with station count and flattens after 4–5 |
| 4 | Characteristic layer: tolerances, datum schemes, covariance projected onto tolerance direction | Two different tolerances on one feature yield different uncertainties |
| 5 | JCGM 106 risk layer: PFA and PFR from uncertainty, tolerance limits and process prior | A high-capability characteristic shows near-zero risk regardless of uncertainty |
| 6 | Optimiser wrapper; run uniform-weight vs risk-derived-weight comparison | **The headline experiment** |

---

## 6. Conventions and constraints

**Language:** Python 3. Libraries: numpy, scipy, matplotlib, pytest. Add others only when needed and say why.

**The author's background:** aerospace engineering, strong on metrology and the physics, developing in Python. Explain *why* a piece of code is structured as it is, not only what it does. Prefer clear, boring code over clever code. Prefer explicit loops over dense vectorisation where the physics is easier to follow — optimise later, once tests pass.

**Physical framing:** explain results the way a metrologist would. "The ellipsoid is flattened perpendicular to the beam because the angular encoders are worse than the interferometer" is better than "the covariance has a large eigenvalue in the tangential direction."

**Units:** SI internally, metres and radians. Convert only for display. Put units in every docstring and variable name where ambiguous (`range_m`, `angle_rad`, `u_expanded_um`).

**Testing:** pytest, tests written alongside code, not after. Every physical claim gets a test against a reference number from §4 where one exists.

**Validation discipline — important.** A previous project reported an inflated F1 score because evaluation data leaked into training. Any predictive model in this repo must use a held-out test set, and the split must be visible in the code. If you are ever unsure whether an evaluation is contaminated, stop and say so.

**Never fabricate a reference number.** If a value is needed and not in §4, ask or mark it clearly as a placeholder with a `TODO(source)` comment.

---

## 7. Glossary

- **LVM** — large-volume metrology
- **Laser tracker** — instrument measuring range plus two angles to a reflector; range is far more accurate than the angles
- **SMR** — spherically mounted retroreflector; the target the tracker follows
- **Station** — one position the instrument is placed at; a plan usually needs several
- **MPE** — maximum permissible error; the manufacturer's specification, a scalar, insufficient for our purposes
- **Network solve / bundle adjustment** — least-squares fit solving all station poses and target coordinates together, returning a covariance
- **USMN** — Unified Spatial Metrology Network; the commercial equivalent in SpatialAnalyzer, used as a validation reference
- **Characteristic** — one toleranced thing to be verified (a hole position, a flatness, a gap)
- **GD&T / PMI** — geometric dimensioning and tolerancing / product manufacturing information; the machine-readable tolerance data in a CAD model
- **JCGM 106** — the standard defining conformity decision risk under measurement uncertainty
- **PFA / PFR** — probability of false acceptance (a bad part passes) / false rejection (a good part fails)
- **Guard band** — acceptance limit pulled inside the tolerance limit to control risk
- **Cpk** — process capability index; how much room the process has inside the tolerance
- **A-optimality** — minimising the trace of the variance matrix; what the field currently does
- **Goal-oriented design** — minimising uncertainty in a downstream quantity of interest; what this project does

---

## 8. Repository layout

```
src/
  geometry/       scene, meshes, poses, ray casting
  instruments/    error models (tracker first)
  network/        least-squares solve, covariance
  characteristics/ tolerances, datums, projection onto tolerance direction
  risk/           JCGM 106 conformity risk
  planning/       optimiser wrapper, objectives (uniform vs risk-weighted)
tests/
scenes/           test geometry, starting synthetic
docs/             literature notes, derivations
results/          experiment outputs, never edited by hand
```

---

## 9. Working style

- Small commits, each one leaving the tests passing.
- When a design decision has more than one reasonable answer, state the options and the tradeoff rather than silently picking one.
- Flag anything that contradicts §3 or §4 immediately — those sections encode findings from primary sources and a contradiction usually means a bug.
- If a requested feature would take the project outside the six-step ladder in §5, say so before building it.
