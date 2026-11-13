# The factory-cell experiment: reference network, datum-referenced uncertainty, decision rules

**Status: first complete version on a synthetic cell.** Every dimension and every
process prior is a placeholder (`TODO(source)`); §7 lists them. This is the
experiment behind the draft paper (`docs/paper/README.md`).

It continues the station-track work (`docs/station_track_experiment.md`,
`docs/process_evidence_experiment.md`, `docs/conditions_sweep.md`), which found
that on the track as first modelled, risk-based planning saved nothing against a
planner working to uncertainty along each tolerance (A1), and that the conditions
under which the objective matters are that clusters *compete for stations* and
*differ in how their risk responds to uncertainty*. Three things were unrealistic
in that model, and each is addressed here.

## 1. What was added

| What | Where | Why |
|---|---|---|
| **Reference network**: 28 nests on pillars and end walls of the cell, solved as unknowns in the bundle adjustment | `planning.cell.factory_cell_scenario`, `ers_nests` | Real cells are tied together through nests on the cell structure (ERS), not through the part. Without it only end-loop stations can tie the two sides. |
| **Datum-referenced uncertainty**: each characteristic's uncertainty relative to the rigid best fit of the part's datum targets (S-transformation), every station planned | `network.design.design_joint_covariance`, `datum_referenced_functional`; `planning.cell.DatumCellScorer` | A deviation is toleranced against the part's datums, not a tracker. The earlier fixed "anchor" station made far characteristics inherit its long-range angular error. The result is independent of which station the network is solved in (tested). |
| **Process knowledge from n builds**: Student-t predictive prior; mixtures for assignment doubt | `risk.decision.ProcessKnowledge` | A process is never known exactly in small-batch work. n = None reproduces `risk.jcgm106`. |
| **Decision rules by conformance probability** (JCGM 106): acceptance limits from the predicted uncertainty | `risk.decision.acceptance_interval`, `best_rule_risk` | Simple acceptance is far from risk-optimal when uncertainty is large relative to the process spread. |
| Risk curves, objectives, search, time model | `planning.cell` | A layout scores in ~1.5 ms, so every objective is searched at every station count. |

Tests: `tests/test_decision.py` (21), `tests/test_cell.py` (14). The scorer is pinned to
the full network solve; the joint covariance to the solve's cross-covariances (1e-7);
the datum functional to rigid-motion invariance; the risk integrals to `risk.jcgm106`,
a conjugate posterior and a brute-force integral.

## 2. Runs

```
python -m viz.run_cell_experiment --frame datum               # main: results/cell_results_datum.json (~40 min, 4 cores)
python -m viz.run_cell_experiment --frame datum --no-network  # same cell without the nests
python -m viz.run_cell_experiment                             # anchor-frame convention of the earlier experiments
python -m viz.run_cell_sensitivity                            # results/cell_sensitivity.json (~35 min)
python -m viz.make_paper_figures                              # results/paper_fig1..7 (.png, .svg)
```

Objectives: A0 (trace), A1 (sum of u), B (weights at 8 µm), T (worst u, the TUR plan),
C (risk, simple acceptance, process known), R<n> (risk, conformance-probability rule,
n builds), R<n>free (same, characteristics may go unobserved). Same differential
evolution for all (popsize 25, maxiter 150, seed 0), cross-scored, with a seeded re-run
wherever another objective's layout beats an objective's own winner.

## 3. Results (datum frame, ±50 µm; risk = expected wrong decisions over six characteristics)

**The reference network.** Without it, four stations (all on end loops) give u = 7.0-8.3 µm
and risk 0.21; with it, 1.6-5.3 µm and 0.0014-0.0043: 50-140 times lower. Two stations with
the network (0.034) beat six without (0.071).

**Objectives at equal stations** (risk as objective C defines it):

| Stations | A0 | A1 | T | C | A1 ÷ C |
|---|---|---|---|---|---|
| 2 | 0.123 | 0.118 | 0.133 | 0.034 | 3.4 |
| 3 | 0.057 | 0.0153 | 0.0149 | 0.0047 | 3.3 |
| 4 | 0.047 | 0.0043 | 0.0042 | 0.0014 | 3.0 |
| 5 | 0.0059 | 0.0016 | 0.0016 | 0.00057 | 2.9 |
| 6 | 0.00089 | 0.00048 | 0.00068 | 0.00036 | 1.3 |

C is best on this measure by construction; the finding is the size of the gap. At three
stations A1 measures everything to 3.0-3.9 µm; C measures the marginal cluster to
2.3-3.0 µm and lets the high-capability cluster go to 8.5-10.2 µm.

**Stations for a given risk:** ≤ 0.034: A0 5, A1 3, C 2. ≤ 0.0047: 6, 4, 3. ≤ 0.0014: 6, 6, 4.
≤ 0.00057: >6, 6, 5. C saves one to two stations against A1, two to three against A0.
TUR 4:1 (6.25 µm) is met at 3 stations (risk 0.015), 10:1 at 5.

**Decision rule and process knowledge** (4 stations, 5 builds): no measurement 0.049;
plan A1 with simple acceptance 0.027 (PFA 0.0069, PFR 0.0203); plan A1 with the 0.5
conformance-probability rule 0.019 (PFA 0.0139, PFR 0.0052); plan and rule by risk 0.017.
The 0.5 rule lowers the total by trading false rejections for false acceptances; a
threshold near 0.8 reproduces simple acceptance's balance here. At two stations simple
acceptance (0.160) is three times worse than not measuring. Value of a 4-station
measurement: 83% of unmeasured risk removed at 3 builds, 65% at 5, 37% at 10, 15% at 30,
none for a process known exactly.

**Sensitivity** (A1 ÷ C at 2-5 stations): baseline 2.9-3.4; 1 µm floor 2.4-3.8; 2 µm
1.9-3.0; 4 µm 1.15-1.9; ±25 µm 1.3-2.9; ±100 µm 1.7-5.8; centred marginal 1.3-1.4. C
reaches every risk target with at least one station fewer in every case.

**Where the objective does not matter:** without the network, A1 and C choose the same
4-station plan. Objective B (linearised weights) fails in this cell: 7 and 21 times C's
risk at 3 and 4 stations.

**Anchor frame** (`results/cell_results.json`, the earlier convention, anchor at the −x
tip): with the network, 2 added stations give risk 0.22-0.26 against 1.05 for 4 stations
without it; A1 is 18%, 9%, 6%, 2.5% above C at 2-5 added stations. The frame matters: the
same four loop stations read 14-35 µm in the anchor's frame and 8-14 µm against the datum
targets.

## 4. What it supports, and what it does not

* Supported on this cell: the layout (reference network) matters most; given a layout with
  freedom, planning for risk needs fewer stations than planning for uncertainty, by a margin
  that survives the sensitivity cases but depends strongly on unmodelled uncertainty; the
  decision rule and the state of process knowledge matter as much as the plan.
* Not supported: any absolute number for a real cell. Station count is a proxy for time
  (one published timing, Hughes et al. 2011: 5 stations, 126 observations, 55 min).
* The earlier null results stand and are part of the picture (`docs/conditions_sweep.md`,
  `docs/station_track_experiment.md` §6b).

## 5. Known gaps

Environmental terms (refraction, thermal drift) and nest repeatability are an isotropic
floor only, and the advantage of C over A1 is sensitive to it; nests have no calibration
uncertainty or drift; datum targets are assumed at nominal on a rigid part; single-point
profile characteristics only; PFA and PFR weighted equally; correlation between decisions
ignored when summing; nothing validated on hardware.

## 6. A-frame note for CLAUDE.md §4

Nothing here contradicts §3 or §4: inputs are the §4a values, unchanged; the §4b gate is
reproduced (1.22 / 5.88 / 8.41 µm at 2.5 m). The floor cases are a sensitivity study, not a
calibrated atmospheric model (§4c remains a later step).

## 7. Placeholders needing real data

Part and track dimensions; station heights; characteristic positions and tolerances;
which features belong to which cluster, and the clusters' means, spreads and number of
builds; datum target positions; reference nest count, positions and heights
(`planning.cell.CELL_ERS_*`); incidence limit (80°) and range (30 m); the uncertainty floor.
