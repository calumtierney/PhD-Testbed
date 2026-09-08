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
complete: **step 1** — single-tracker spherical error model, propagated to
a per-point Cartesian covariance.

## Development

```bash
pip install -e ".[dev]"
pytest
```

```
src/
  geometry/         scene, poses, spherical<->Cartesian, ray casting (step 2+)
  instruments/       instrument error models (laser tracker first)
  network/           multi-station least-squares solve (step 3+)
  characteristics/    tolerances, datums, covariance projection (step 4+)
  risk/               JCGM 106 conformity risk (step 5+)
  planning/           optimiser wrapper, objectives (step 6)
tests/               pytest, alongside the code it tests
scenes/              test geometry (synthetic to start)
docs/                literature notes, derivations
results/             experiment outputs — never edited by hand
```
