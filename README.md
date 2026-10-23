# LVM Measurement Planning Testbed

## Running this

```bash
pip install -r requirements.txt && pip install -e .
pytest
python -m viz.make_figures
```

(Full step-by-step VS Code / Windows setup, the interactive `streamlit run
app.py` explorer, and a troubleshooting table: [`SETUP.md`](SETUP.md).)

Simulation testbed for goal-oriented measurement planning in large-volume
metrology (LVM). Full project context, technical decisions, reference
numbers and the build order live in [`CLAUDE.md`](CLAUDE.md) — read that
first, especially if you're picking this repository up in a new session.

## The idea in one sentence

Existing large-volume metrology planning tools choose instrument stations
to minimise measurement uncertainty (A-optimal design). This project plans
stations instead to minimise the risk of a wrong accept/reject decision on
the parts being measured — the same instruments and station budget, aimed
at the thing an inspection plan actually exists to get right.

## The six-link chain

Every part of this codebase implements one pipeline, station pose to
decision risk. Each link below is a separate, separately testable module
under `src/`:

1. **Station pose** — where an instrument (laser tracker, photogrammetry
   camera, ...) is placed to observe the scene.
2. **Visibility** (`src/geometry`) — is a target point actually measurable
   from that pose? Occlusion, maximum range, maximum incidence angle.
3. **Instrument error model** (`src/instruments`) — the instrument's own
   measurement noise, propagated into a 3x3 Cartesian covariance matrix
   *per point*. This is never a scalar: real instrument noise is strongly
   anisotropic (see `docs/step1_laser_tracker_physics.md`).
4. **Coordinate covariance** (`src/network`) — combine observations from
   every station of a shared target set into a single network solve
   (weighted nonlinear least squares), giving a full covariance over all
   target coordinates. *Existing tools stop here and sum the diagonal.*
5. **Characteristic uncertainty** (`src/characteristics`) — project that
   covariance onto the direction a specific GD&T tolerance actually
   constrains. A position tolerance and a parallelism tolerance on the
   same feature see different uncertainties from the same measurement.
   (One tolerance type here is named `ProfileTolerance`, not "flatness" —
   it's one point's signed normal deviation, not true multi-point,
   minimum-zone GD&T flatness; see `characteristics/tolerances.py`'s
   naming note.)
6. **Conformity decision risk** (`src/risk`) — JCGM 106: turn a
   characteristic's uncertainty, its tolerance limits and a process
   capability prior into a probability of false acceptance / false
   rejection.

`src/planning` wraps a conventional optimiser around station placement and
compares four objectives over that chain (CLAUDE.md §5, step 6 — the
headline experiment): the field's actual A-optimal criterion (trace of
the raw coordinate covariance), uniform weights on projected
characteristic uncertainty, linearised risk-derived weights, and direct
nonlinear risk minimisation. That comparison is the experiment the
project exists to run.

## Status

Build proceeds one rung of the ladder in `CLAUDE.md` §5 at a time, each
gated by a reproduced physics result before the next begins. Currently
complete:

- **step 1** — single-tracker spherical error model, propagated to a
  per-point Cartesian covariance (`docs/step1_laser_tracker_physics.md`).
- **step 2** — visibility: range, incidence-angle and mesh-occlusion
  checks for whether a target can be measured at all, before step 1's
  error model ever runs (`docs/step2_visibility_physics.md`).
- **step 3** — the multi-station network solve: target coordinates and
  station poses estimated jointly by weighted nonlinear least squares,
  returning the full covariance over every target coordinate
  (`docs/step3_network_solve_physics.md`).
- **step 4** — the characteristic layer: position, profile and
  parallelism tolerances each project a point's covariance onto the
  direction they actually constrain, so the same measurement gives
  different tolerance types different uncertainties
  (`docs/step4_characteristic_layer_physics.md`).
- **step 5** — the JCGM 106 conformity risk layer: global probability of
  false acceptance/rejection from a characteristic's uncertainty, its
  tolerance limits, and a process-capability prior held per
  feature-cluster (not per characteristic — CLAUDE.md §3)
  (`docs/step5_risk_layer_physics.md`).
- **step 6 — the headline experiment** — a grid-search optimiser wrapper
  compares four station-placement objectives (A0: trace of the raw
  coordinate covariance, the field's actual criterion; A1: uniform
  weights on projected characteristic uncertainty; B: linearised
  risk-derived weights; C: direct nonlinear risk minimisation) with
  everything else held identical. Result: the risk-based objectives
  differ from A0/A1, in the direction the hypothesis predicts (effort
  pulled from a high-capability cluster towards a marginal one; global
  risk down, at the cost of higher traditional "summed uncertainty"),
  modestly in this scene, for reasons the write-up explains, plus an
  iso-risk comparison (risk-weighting reaches a given risk target with
  fewer stations at some targets) and a guard against unbounded
  degradation (`docs/step6_headline_experiment.md`). A second scenario
  (`planning.experiment.mixed_tolerance_two_cluster_scenario`) checks
  whether A0/A1 and B/C's coincidence in the main scene was a general
  equivalence or just that scene's symmetry, by mixing position,
  profile and parallelism tolerances together — the answer depends on
  candidate-grid resolution as well as the tolerance mix (same doc).
  The full six-step chain is now built end to end.
- **towards a more realistic headline experiment** — three extensions so
  far: (1) step 2's occlusion/visibility layer is now wired into
  planning itself (`PlanningScenario.obstacles`/`visibility_limits`,
  `objectives.evaluate_plan`), so a candidate plan that leaves a target
  unseen, or a station unable to see enough of the scene to pin its own
  pose, is correctly disqualified rather than silently solved; (2)
  `planning.candidates.floor_grid_candidate_poses` generalises station
  placement beyond a fixed-radius ring to a real (x, y, height) grid, and
  `planning.search.genetic_algorithm_station_placement` adds the
  standard-GA search CLAUDE.md §3 names as the fallback once that grid's
  combinatorics stop being grid-search-tractable (`scipy.optimize.
  differential_evolution`, no new dependency); (3)
  `planning.experiment.wing_four_point_scenario` puts all of this
  together on a single ~30 m wing (CLAUDE.md §1's own upper range bound)
  with four named fittings and a support-tower occluder under mid-span.
  The full A0/A1/B/C comparison on this scene (`planning.experiment.
  run_wing_experiment`) is written up in `docs/step6_headline_experiment.md`'s
  "The wing scenario" section. **That section carries a correction:** the
  first version of this result (a 19% / 6.6% risk reduction for objective C)
  came from layouts that were not valid networks -- a station tied to the
  rest by a single shared target has a gauge freedom that `solve_network`'s
  `pinv` silently reports as zero variance, and the GA had learned to use
  it. With rigid connectivity enforced (`network.design`), all four
  objectives land within 1.6% of each other (total risk ~0.80-0.87; three
  stations over ~30 m is a weak-measurement regime), which is the size of
  the GA's own imprecision. The wing-box experiment below is where the
  project's hypothesis is now tested.

- **the wing-box inspection experiment** (`docs/wingbox_experiment.md`) —
  a separate experiment on a ~500-point, six-task wing box (a *synthetic
  stand-in*: no CAD or point list was available), comparing exactly two
  plans on station count: **U**, the conventional uncertainty-driven
  (A-optimal) plan, and **R**, the risk-driven plan held to a decision-risk
  limit. On the stand-in, at ±50 µm, plan U needs 9 stations and 1151
  shots; plan R held to *no more risk than U actually delivers* needs 3
  stations and 679 shots, with 29× fewer expected false rejections, by
  measuring the marginal leading-edge tasks tightly and leaving 56
  lower-flap points (priced at their process's escape risk) unmeasured. The
  doc is explicit about what that depends on: the saving survives changes of
  tolerance (±25 to ±250 µm) and prior shape, shrinks to 5 stations if the
  cluster assignment is only 50% reliable, and **vanishes if the hard-to-see
  flaps are the marginal cluster or if policy requires ≥75% of every task
  measured**. No published aerospace limit for acceptable false-accept risk
  was found, so the limit is *not invented*: R is held to U's own realised
  risk, or to what a TUR 4:1 rule guarantees, with sweeps. The
  conventional hard TUR 4:1 rule on every point could not be met at all. A
  gradient-boosted **surrogate of the digital twin** (`planning.surrogate`,
  scikit-learn — added to `requirements.txt` with its reason) predicts
  coverage and escape probability to ~3% on layouts it never saw (split by
  base layout, plus a held-out region of the station box), but not false
  rejections; used naively as a GA objective it is exploited, and with model
  management it reaches the same station counts with 29–61× fewer exact
  evaluations — but no wall-clock saving at this evaluator's ~5–10 ms, which
  the doc says plainly. Outside the six-step ladder of CLAUDE.md §5; run
  with `python -m viz.make_wingbox_figures --stage experiment|seeds|surrogate|figures`
  (about 2.5 hours in all; figures `results/panel10*.png`).

**Visualization** (`src/visualization`, not one of the six steps — see
its module docstring): matplotlib plotting for scenes, per-point error
ellipsoids, visibility results and network-solve outcomes, so a
metrology engineer can look at what the chain produced rather than read
arrays. Static figures only (no new dependency, no GUI/web layer); see
"Visualizing results" below.

## Development

```bash
pip install -e ".[dev]"
pytest
```

`pytest` (well under a minute) is the fast, deterministic check on every
physics gate in `CLAUDE.md` §5 steps 1-5, RNG-seeded so it reproduces
identically run to run.

To independently reproduce every figure under `results/` — including the
step 6 headline-experiment numbers `docs/step6_headline_experiment.md`
reports — rather than just trusting them:

```bash
python -m viz.make_figures
```

Takes 2-3 minutes (objective C's per-candidate risk integration
dominates the two `run_headline_experiment` calls — see
`planning.experiment.run_headline_experiment`'s own docstring). It
prints what it wrote and rewrites every `results/panel*.png` file in
place; steps 1-5's panels use fixed RNG seeds, and the step 6 panels
solve from noiseless `exact_observations` at the true geometry rather
than simulating and fitting random noise, so there is no RNG anywhere in
this command — the output should match the committed files exactly, not
just approximately (`git diff results/` after running it should be
empty).

The wing scenario's figures (`panel9*_wing_*.png`,
`docs/step6_headline_experiment.md`'s "The wing scenario" section) are
*not* part of the command above — they're regenerated separately:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python -m viz.make_wing_figures
```

This one is genuinely slow (15-25 minutes, a GA search run four times
over two scenarios — see `viz/make_wing_figures.py`'s own module
docstring for why it needs those environment variables to actually be
*fast*, not slower than leaving them unset) and, unlike the command
above, not bit-for-bit reproducible: `scipy.optimize.differential_evolution`
is a genetic algorithm, not an exhaustive search, and re-running it can
land on slightly different station positions and objective values than
the ones committed — `docs/step6_headline_experiment.md`'s "Getting a
trustworthy GA comparison took three attempts" section is the honest
account of what that did and didn't fix.

```
src/
  geometry/         scene, poses, spherical<->Cartesian, meshes, ray casting, visibility
  instruments/       instrument error models (laser tracker first)
  network/           multi-station least-squares solve
  characteristics/    tolerances, datums, covariance projection onto the tolerance direction
  risk/               JCGM 106 conformity risk: global PFA/PFR per characteristic
  planning/           grid-search station placement; uniform vs risk-derived-weight objectives
  visualization/       matplotlib plotting of scenes and results (not a ladder step)
tests/               pytest, alongside the code it tests
scenes/              test geometry (synthetic to start)
docs/                literature notes, derivations
results/             experiment outputs — never edited by hand
```

## Visualizing results

```python
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker
from visualization.plotting import plot_scene

scene = Scene(target_points_m=my_points, instrument_pose=InstrumentPose(position_m=my_station))
fig, ax = plot_scene(scene, tracker=LaserTracker(), title="my scene")
fig.savefig("results/my_scene.png")  # or fig.show() in a notebook
```

`plot_scene` also takes `visibility_results` (from `geometry.visibility.
scene_visibility`) to colour targets by why they can/can't be measured,
`network.solve` results plot with `visualization.plotting.
plot_network_result` / `plot_uncertainty_vs_station_count`,
characteristic comparisons (step 4 — several tolerance types on one
point) plot with `plot_characteristic_comparison`, conformity risk
(step 5 — `risk.jcgm106`) plots with `plot_risk_vs_uncertainty` (risk
against uncertainty, one line per process/cluster, log-scaled since a
high-capability and a marginal process's risk typically differ by many
orders of magnitude at the same uncertainty), and the step 6 headline
comparison (uniform vs risk-weighted plans) plots with
`plot_plan_comparison`. See `src/visualization/plotting.py`'s module
docstring for the exaggeration convention used for covariance ellipsoids
(they're micrometre-scale next to a metre-scale scene, so are drawn
scaled up and always labelled with the factor).
