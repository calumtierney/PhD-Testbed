# LVM Measurement Planning Testbed

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
6. **Conformity decision risk** (`src/risk`) — JCGM 106: turn a
   characteristic's uncertainty, its tolerance limits and a process
   capability prior into a probability of false acceptance / false
   rejection.

`src/planning` wraps a conventional optimiser around station placement and
compares two objectives over that chain: minimise uncertainty (uniform
weights) versus minimise decision risk (risk-derived weights). That
comparison is the experiment the project exists to run — see `CLAUDE.md`
§5, step 6.

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
- **step 4** — the characteristic layer: position, flatness and
  parallelism tolerances each project a point's covariance onto the
  direction they actually constrain, so the same measurement gives
  different tolerance types different uncertainties
  (`docs/step4_characteristic_layer_physics.md`).

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

```
src/
  geometry/         scene, poses, spherical<->Cartesian, meshes, ray casting, visibility
  instruments/       instrument error models (laser tracker first)
  network/           multi-station least-squares solve
  characteristics/    tolerances, datums, covariance projection onto the tolerance direction
  risk/               JCGM 106 conformity risk (step 5+)
  planning/           optimiser wrapper, objectives (step 6)
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
plot_network_result` / `plot_uncertainty_vs_station_count`, and
characteristic comparisons (step 4 — several tolerance types on one
point) plot with `plot_characteristic_comparison`. See
`src/visualization/plotting.py`'s module docstring for the exaggeration
convention used for covariance ellipsoids (they're micrometre-scale next
to a metre-scale scene, so are drawn scaled up and always labelled with
the factor).
