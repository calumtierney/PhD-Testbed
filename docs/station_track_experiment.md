# The station-track experiment: a long part in a factory cell

**Status: first complete version, on a placeholder cell.** Every dimension of
the part, the jig, the track, the tie points and the tripod heights is a
stand-in (§8 lists them all, each marked `TODO(source)` in the code). The
numbers below are statements about this testbed on that stand-in, not about a
real GKN cell.

The earlier experiments put the trackers on a ring (`docs/step6_headline_experiment.md`)
or on a free floor grid (the 30 m wing). A real cell for a long part is neither:
trackers stand on stands, in two rows along the long sides of the part, and can
be carried round either end. This experiment replaces the ring by that
**station track** and asks the headline question again -- do the four
objectives (A0 trace of the covariance, A1 uniform weights, B risk-derived
weights, C direct expected risk) choose different stations? -- on a scene
where where a tracker may stand is constrained by a physical layout.

## 1. What was built

| Piece | Where | What |
|---|---|---|
| `StationTrack` | `src/geometry/station_track.py` | Pure geometry: a closed stadium path round the part (two straight rows, a semicircular loop at each end), parameterised by arc length `s_m`; blocked intervals; segment names. No physics, no import from `planning`. |
| `track_candidate_poses`, `levelled_orientation`, `TrackPose` | `src/planning/candidates.py` | Candidates every `spacing_m` along the track at each height, each carrying `(s_m, height_m, segment)`. Orientation `"levelled"` (default) or `"look_at"`. |
| `candidate_poses=` on `run_headline_experiment`, `iso_risk_station_counts` | `src/planning/experiment.py` | Optional; the default builds the identical 24-pose ring. |
| `track_genetic_algorithm_station_placement`, `TrackStationCost` | `src/planning/search.py` | The GA over `(s_m, height_m)` per station: same driver (`run_station_search`, now `dims_per_station`-aware), same `INFEASIBLE_PENALTY`, same seeding. |
| `prefilter_visibility` on `grid_search_station_placement` | `src/planning/search.py` | Opt-in speed-up that changes no result (tested): reuses each candidate's visibility row. |
| `long_part_two_cluster_scenario`, `long_part_track` | `src/planning/experiment.py` | The scene (§3). |
| `grid_all_objectives`, `run_track_ga`, `summarise_plan`, `cross_score`, `time_evaluations` | `src/planning/track_experiment.py` | The experiment's searches and reporting. |
| runner and figures | `viz/run_station_track_experiment.py`, `viz/make_station_track_experiment_figures.py`, `viz/make_station_track_figure.py` | Results JSON and panels `results/panel11a-c*`, `results/station_track_cell.png`. |
| tests | `tests/test_station_track.py`, `tests/test_station_track_scenario.py`, one in `tests/test_planning.py` | Geometry, candidates, the physics gates of §4, the GA. |

### The track

`s = 0` is the −x end of the −y row; `s` increases anticlockwise seen from above:
`-y row`, `+x loop`, `+y row`, `-x loop`. With equal standoffs each loop is a
semicircle about the midpoint of that end of the footprint, radius
`width/2 + standoff`; with unequal standoffs it is centred between the two
rows (the simplest continuous closed path; clearance from the footprint never
falls below the smaller standoff). For the 15 × 2.5 m part and a 3 m standoff
the perimeter is 56.7 m.

### Levelled orientation, and why it hardly matters

A real tracker is levelled on its stand; only its yaw is free. **Yaw alone
changes no covariance** (the azimuth encoder has the same noise at every angle;
tested to rounding error), so for levelled poses the aim point cannot change a
result. **Tilting the frame does** change it (it swaps the roles of the azimuth
and elevation encoders, 0.485 vs 0.694 arcsec). The ring's candidates are
tilted (`look_at`: boresight at the centroid, up to 20° at the high tier), which
is slightly unphysical for a levelled instrument.

*Side check on the existing ring experiment (report only, nothing adopted):*
the identical 24-pose ring and scene with levelled orientations. **Every
objective picks the same stations.** Total risk moves from 0.19898 to 0.19922
(A0, A1), and 0.19268 to 0.19269 (B, C); summed uncertainty by at most 0.07%.
The existing headline is insensitive to this.

## 2. Regression

The ring experiment is unchanged: the full existing suite passes unchanged, and
`python -m viz.make_figures` rewrites panels 1-8 byte-identically
(`git diff results/` empty for them) after the `candidate_poses` refactor.

## 3. The scene

`long_part_two_cluster_scenario`, parameters and constants `LONG_PART_*`:

* **Part**: box 15 × 2.5 m, underside 0.5 m, top 2.0 m, the visibility layer's
  only obstacle (`box_mesh`). A tracker cannot see through it.
* **Clusters**: six profile tolerances (±50 µm), three per cluster, the existing
  GKN-calibrated priors (high-capability μ 0, σ 8.33 µm, Cpk 2.0; marginal μ
  38 µm, σ 3 µm, Cpk 1.33) and confidence 0.9. High-capability on the **+y**
  face near x = 4 m (normal +y), marginal on the **−y** face near x = 11 m
  (normal −y): opposite long faces, offset along the part.
* **Tie points** (ERS-style nests, no tolerance, as in `wing_four_point_scenario`):
  **5 on each end face and 5 along each long face**. The first plan-view figure
  showed that with 3 per end face the loop stations were only a thin bridge
  between the two sides; the user chose more end-face ties. That makes 20 ties
  and 26 targets in all. Top-face ties are impossible: from a floor track they
  are beyond the 80° incidence limit.
* **Visibility**: on; range 30 m (CLAUDE.md §1), incidence 80° (the repo's
  convention, not a literature value), occlusion by the part.
* **Anchor**: fixed on the track at the **−x tip** (the apex of the −x loop,
  `track.tip_minus_x_s_m`, s = 50.0 m), height 1.5 m, the same for every
  objective. It sits there because that is where a reference station sees the
  −x end-face ties; the sensitivity case puts it at the +x tip.
* **Candidates**: heights (1.0, 2.4) m on the grid, [0.8, 3.0] m for the GA.

### What a plan has to be

A plan is feasible only if every target is seen by some station and every
added station shares ≥3 targets with a chain leading to the anchor (the rigid-
network rule the repo already enforces). On this scene that is very restrictive:

* A row station sees only its own long face. The far face is behind the part and
  fails the 80° incidence limit first (with the incidence rule switched off it
  is **occluded**, which `tests/test_station_track_scenario.py` checks both ways).
  No row station sees an end face (grazing incidence).
* An end-loop station sees one end face **and one long face**, never both long
  faces. No station anywhere sees both long faces.
* **Consequently no plan with one or two added stations is feasible** (exhaustive:
  0 of 6,441 pairs at 1 m spacing, for each of four anchor positions), and every
  feasible three-station plan contains an end-loop station. Three is the
  smallest search there is. (The brief expected a 2-station grid cross-check;
  none exists on this scene, so the cross-check is at three added stations.)

## 4. Physics gates on the new scene (tests, all passing)

* single-station uncertainty rises with range, in proportion across the beam (angle
  noise × range) and not at all along it (the interferometer), so always
  lateral > 3 × radial;
* a station on the +y row cannot see the −y face (incidence, and occlusion once
  incidence is relaxed);
* end-loop stations see an end face and one long face; no station sees both
  long faces; row stations see no end face;
* mean target uncertainty falls with every added station (greedy, 3 → 7 added)
  and the last station buys less than half of what the first did;
* one and two added stations can never cover and tie the part; three can.

## 5. Budget, and the spacing chosen

Measured on 200 random feasible 3-station plans: `evaluate_plan` (visibility,
network solve, projection) **42 ms**; objective call **A0 0.015 ms, A1 0.002 ms,
B 0.002 ms, C 134 ms**. So a full evaluation costs 42 ms for A0/A1/B and 176 ms
for C.

The brief's rule (finer spacing only if a full four-objective grid stays under
~30 min) cannot be applied at two stations, which are infeasible. Applied to the
smallest real grid, three added stations, 1.0 m spacing (114 candidates, 240,464
combinations, 208 feasible) took 41 s; the extrapolation to 0.5 m (8× the
combinations) was 329 s. **0.5 m was chosen** (228 candidates, 1,898,400
combinations, 1,440 feasible, measured 284 s for all four objectives). Four-
station grids (106 M combinations at 0.5 m) are not attempted; the GA does 4
stations.

## 6. Results (anchor at the −x tip)

**Summed risk is large.** PFA is essentially 0 (≈5e-5) in every plan; all the
risk is **false rejection**. With stations on the end loops the characteristics
are 10-15 m from the nearest tracker, lateral uncertainty is 30-60 µm against a
±50 µm tolerance, and the decision rule has no guard band, so many conforming
parts are rejected. Inputs are fixed (CLAUDE.md §4a/§4e): this is the model's
prediction for a long part on this layout, not something to tune away. The
"Σ risk" below is the expected number of wrong decisions over six characteristics.

### Three added stations (+ anchor), GA, `(s_m, height_m, segment)`

| Objective | Stations | Σu (µm) | PFA | PFR | Σ risk |
|---|---|---|---|---|---|
| A0 | (15.9, 0.81, +x loop) (44.2, 0.80, −x loop) (54.0, 2.02, −x loop) | 278.4 | 0.00004 | 1.9041 | **1.9041** |
| A1 | (16.7, 1.23, +x loop) (44.2, 1.13, −x loop) (54.4, 1.19, −x loop) | 240.1 | 0.00005 | 1.7106 | 1.7106 |
| B | (16.7, 1.22, +x loop) (44.2, 1.11, −x loop) (54.4, 1.19, −x loop) | 240.1 | 0.00005 | 1.7108 | 1.7109 |
| C | (16.7, 1.23, +x loop) (44.2, 1.16, −x loop) (54.4, 1.18, −x loop) | 240.1 | 0.00005 | 1.7105 | **1.7105** |

### Four added stations, GA

| Objective | Stations | Σu (µm) | PFA | PFR | Σ risk |
|---|---|---|---|---|---|
| A0 | (15.9, 0.86, +x) (27.3, 2.97, +x) (46.1, 0.82, −x) (53.7, 2.30, −x) | 178.7 | 0.00005 | 1.2475 | **1.2475** |
| A1 | (15.9, 1.25, +x) (27.5, 1.36, +x) (45.0, 1.18, −x) (54.4, 1.19, −x) | 134.8 | 0.00005 | 1.0510 | 1.0511 |
| B | (16.4, 1.22, +x) (27.5, 1.06, +x) (45.0, 1.18, −x) (54.4, 1.19, −x) | 134.9 | 0.00005 | 1.0512 | 1.0513 |
| C | (15.9, 1.28, +x) (27.5, 1.45, +x) (45.0, 1.17, −x) (54.4, 1.18, −x) | 134.8 | 0.00005 | 1.0510 | **1.0511** |

Per-characteristic uncertainty and risk are in `results/panel11c_track_uncertainty_risk.png`:
the high-capability cluster sits at 13-17 µm (4 stations) and ~0 risk; the
marginal cluster at 28-32 µm and ~0.34-0.36 risk each. **The marginal cluster
carries essentially all the risk at 4 stations.**

**What the results say.**

1. **A1, B and C pick the same plan** here (differences ≤0.03% in risk are the
   GA's own imprecision). **A0, the field's criterion, picks a different one and
   delivers more risk: +11.3% at 3 stations, +18.7% at 4** (1.9041 vs 1.7105;
   1.2475 vs 1.0511). The difference shows up in the marginal cluster (the
   bars): at 3 stations A0 leaves it at 57-62 µm against 46-50 µm for the
   others, while the high-capability cluster is about the same in both; and
   the marginal cluster is where the risk is. Cross-scoring confirms every search found
   the best of its own objective: each diagonal is the column minimum (`cross_score`
   in the JSON).
2. **Every chosen station is on an end loop. No plan uses a row station**, at 3
   or at 4 stations, under any objective. This is the §3 feasibility structure
   at work (the loops are the only places that see an end face and a long face),
   and the search confirms it rather than having been told. A row station is close to its face but sees only that face, so it cannot supply the
   ties the network needs.
3. **Where the added station goes** (`panel11b_track_heatmap.png`: a fourth station
   added to C's best 3-station plan, scanned over the whole track and heights
   0.8-3.0 m, scored under A0 and under C). Only the two end loops score well,
   and the best spot is on the +x loop for both: **A0 at (27.5 m, 2.6 m), C at
   (27.0 m, 1.6 m)**. They are 0.5 m apart in s but 1.0 m apart in height; the
   minima are not the same cell, but fall in the same stretch of the track. So
   on this scene the two heat maps' minima nearly coincide in position, and
   differ mainly in height. The contribution is visible mainly in the *level* of
   risk each plan delivers (finding 1), not in a different region of the track.
4. **GA versus grid.** At three added stations the GA's value is within the stated
   tolerance (GA ≤ 1.05 × grid at 0.5 m) for every objective; in fact it is lower
   everywhere (ratio A0 0.878, A1 0.915, B 0.830, C 0.964) because it searches
   between grid points and heights down to 0.8 m. A0's GA sits on the 0.8 m height
   bound for two stations (3 added: 0.81, 0.80; 4 added: 0.82, 0.86): the real
   optimum for that objective may be below the placeholder bound.
   The unseeded GA for C at 3 stations was beaten under its own metric by A1's
   winner, so it was re-run seeded with all winners (the GA is a good plan, not a
   provably best one); that is recorded in the JSON (`reseeded_objectives`).
   Search effort: `maxiter` 120, `popsize` 25 (× the number of searched numbers),
   4 cores: about 30 min for all four objectives at 3 added stations, 31 min at 4.

### Iso-risk station counts

One and two added stations are infeasible whatever the target, so the question is
3 versus 4. Counts are taken from the GA winners (a 4-station grid is not run).
Smallest number of added stations whose best plan reaches a target total risk:

| Target total risk | A0 | A1 | B | C |
|---|---|---|---|---|
| ≤ 1.71053 | 4 | 4 | 4 | **3** |
| ≤ 1.31382 | 4 | 4 | 4 | 4 |
| ≤ 1.15616 | **none ≤ 4** | 4 | 4 | 4 |
| ≤ 1.10361 | none | 4 | 4 | 4 |
| ≤ 1.05106 | none | none | none | 4 |

A0 never reaches a risk below ~1.25 within four stations, while A1/B/C do. The
differences between A1, B and C in the first and last rows (≤0.01%) are the GA's
noise, not a finding: read the table as "A0 needs more stations (or cannot get
there) and A1/B/C agree".

### Anchor sensitivity (anchor at the +x tip)

Three added stations, same search, anchor moved to the opposite tip (the part
itself unchanged); the grid at 0.5 m confirms the GA's ordering (ratios 0.93,
0.97, 0.93, 0.96).

| Objective | Stations | Σ risk |
|---|---|---|
| A0 | (15.9, 2.97, +x) (24.9, 0.80, +x) (44.4, 2.98, −x) | 2.6425 |
| A1 | (15.9, 1.15, +x) (26.1, 1.21, +x) (45.0, 1.26, −x) | 1.9783 |
| B | (18.4, 1.19, +x) (27.5, 1.16, +x) (55.8, 1.27, −x) | **2.4952** |
| C | (15.9, 1.15, +x) (26.1, 1.21, +x) (45.0, 1.27, −x) | **1.9783** |

* **The datum matters**: C's best risk is 1.7105 with the −x anchor and 1.9783 with
  the +x anchor, 16% more, because the anchor determines which loop must supply the
  tie to the rest of the network and with which targets.
* **B diverges from C in this case** (2.4952 vs 1.9783, +26%; the grid agrees:
  B at (18.1, 27.1, 55.7) scores 2.471 under C). B's weights concentrate on the marginal
  cluster, so B drives the marginal cluster's uncertainty down to 17-20 µm and
  lets the high-capability cluster go to 85-91 µm. Per characteristic, B's plan
  has risk 0.56-0.59 on the high-capability ones (C's plan: 0.28-0.31) and 0.24-0.27
  on the marginal ones (C's plan: 0.35-0.38): B saves about 0.3 in total on the
  marginal cluster and loses about 0.8 on the other. The weights were linearised at 8 µm;
  uncertainties here are far from that, which is consistent with (not proof of)
  why the linearisation misleads. With the −x anchor B and C coincide.
* **A1 = C again** in both anchor cases.

### Does any chosen plan use an end-loop station?

Yes, all of them, exclusively (§6, finding 2): in 12 of 12 plans, at 3 and 4
stations, every added station is on an end loop; none is on a row.

## 7. How to reproduce

```
python -m pytest
python -m viz.make_figures                       # panels 1-8, must leave git diff results/ empty
python -m viz.make_station_track_figure          # results/station_track_cell.png (seconds)
python -m viz.run_station_track_experiment       # results/station_track_results.json (about 1.5-2 h, 4 cores)
python -m viz.make_station_track_experiment_figures   # results/panel11a-c*.png/.svg (a few minutes)
```

The runner pins BLAS threads before `workers=-1` (see its docstring) and writes
the JSON after every stage. Seeds are fixed (`seed=0`).

## 8. Placeholders needing GKN data

All are named constants or parameters (`planning.experiment`, `LONG_PART_*`),
marked `TODO(source)`:

1. part length (15 m), width (2.5 m), underside height (0.5 m), top height (2.0 m);
2. track standoff (3 m) and blocked intervals (none: aisles, jig legs, safety zones);
3. tripod/stand heights: grid (1.0, 2.4) m, GA range [0.8, 3.0] m (the GA's best
   plans sit on 0.8 m for A0, so the lower bound matters);
4. anchor: where on the track (−x tip) and its height (1.5 m);
5. where the two clusters sit along the part (x ≈ 4 m and 11 m) and on which
   faces, and the heights of their targets (1.0-1.5 m);
6. number and positions of the tie points: 5 per end face, 5 per long face
   at x = 1.5, 5.5, 8.5, 9.5, 13.5 m, heights 1.0-1.5 m (the real nest layout);
7. the aim height used for `track_candidate_poses` defaults (1.25 m; affects
   figures only when levelled);
8. the incidence limit (80°, a repo convention, not a literature value) and the
   maximum range (30 m, CLAUDE.md §1).

## 9. Caveats worth saying aloud

* Risk this large (≈ 1 expected wrong decision in 6 at four stations) comes from
  one tracker stationed 10-15 m from its targets with no guard band; it says
  the plan the cell *forces* is a poor plan for a ±50 µm tolerance, which is a
  result about the layout (and a prompt to look at stations closer to the part,
  guard banding, and the tie-point layout), not a defect to be tuned.
* The "every target must be observed, including every tie" rule is the repo's
  existing rule; it is what makes one and two stations infeasible and sends every
  station to the loops. A cell where ties are optional would behave differently.
* GA results are good plans, not provably the best; the seeded re-runs and the
  grid cross-check above are the evidence.
