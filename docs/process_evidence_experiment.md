# Accepting characteristics on process evidence: the station track

**Status: first complete version, placeholder cell, placeholder priors.** The
earlier experiments (`docs/station_track_experiment.md`, `docs/conditions_sweep.md`)
measured every characteristic, and found that risk-based planning saves almost
nothing against a planner that works to uncertainty along each tolerance (A1).
The wing-box experiment found where a saving can come from: *not measuring* a
characteristic whose process is capable, and pricing it at the process's own
escape risk. This experiment lets the track plan do that.

## 1. The model

`planning.process_evidence`. A plan now chooses, with its layout, which
characteristics to measure; the rest are accepted on process evidence.

* **Unmeasured price.** Accept every part: no false rejections, a false accept
  whenever the true value is out of tolerance, i.e. the cluster's nonconformance
  rate: **3.2e-5** for the marginal cluster (mean 38 µm, σ 3 µm, ±50 µm,
  Cpk 1.33) and **2.0e-9** for the high-capability one (Cpk 2.0).
* **Measured price.** The usual PFA + PFR at the characteristic's projected
  uncertainty, *simple acceptance* (acceptance limits = tolerance limits), as
  everywhere else in this repo.
* **Total risk** is the sum, in expected wrong decisions, directly comparable with the
  measure-everything plans. For a given layout the planner measures every
  measurable characteristic that lowers risk, then tops up to the policy floor
  with the cheapest ones.
* **Two explicit controls**, because the priors are placeholders with no data and
  an unconstrained risk-minimiser would otherwise decide everything by them:
  `min_measured`, a sampling-policy floor (as the wing box's `min_task_coverage`), and
  `escape_price_multiplier`, the unmeasured price as a multiple of the elicited
  prior's (how much worse than elicited the real process would have to be).
* **Tie points** either required (the track scene's rule: every target, ties
  included, must be observed, which pins the station floor at 3) or relaxed (a plan
  observes only what it needs; every station still has to share ≥ 3 targets with a chain
  to the anchor).
* **Check.** With a floor of all six and ties required the evaluator reproduces the earlier
  measure-everything GA results (risk 1.710, 1.051, 1.010, 0.9766 at 3-6 stations
  against 1.7105, 1.0511, 1.0098, 0.9767 before) and objective C on any layout within 2%
  (tabulated risk), pinned by `tests/test_process_evidence.py` (10 tests).

Search: the same differential-evolution driver over `(s_m, height_m)` as every
other track search, seeded with the known measure-everything layouts at 3-6 stations,
anchor at the −x tip, ±50 µm. `python -m viz.run_station_track_process_evidence`
(about 70 min); data `results/station_track_process_evidence.json`.

## 2. Results

Best total risk (expected wrong decisions over six characteristics) by number of
added stations. **H** = high-capability characteristics measured, **M** = marginal
characteristics measured; the rest accepted on evidence. "inf" = no feasible layout.
Reference: the A1 plan that measures everything needs **4 stations, risk 1.0511**
(`docs/station_track_experiment.md` §6b).

| Configuration | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **Measure all six** (floor 6), ties required | inf | inf | 1.710 | **1.051** | 1.010 | 0.977 |
| Measure all six, ties relaxed | inf | 1.785 | 1.348 | **1.051** | 1.010 | 0.977 |
| **Floor 3** (price 1: H measured, M accepted), ties required | inf | inf | 0.168 | 0.0132 | 0.0046 | 0.0032 |
| Floor 3, ties relaxed | 0.228 | 0.0183 | 0.0088 | 0.0058 | 0.0046 | 0.0039 |
| Floor 0, price ×1e2 / ×1e3 (marginal price 0.003 / 0.03) | | | measure nothing: 0.0095 / 0.095 at any count | | | |
| Floor 0, price ×1e4 (marginal price 0.32) | | | measure nothing: 0.950 | | | 0.948 [M] |
| **Floor 0, price ×1e5** (marginal price capped at 1: marginal must be measured), ties required | inf | inf | 1.297 [M] | 1.038 | 1.000 | 0.973 |
| Same, ties relaxed | 1.723 [M] | 1.283 | 1.153 | **1.038** | 1.000 | 0.973 |
| Floor 0, price ×1 (the model's prior) | measure nothing: risk 9.5e-5, with no station at all | | | | | |

**Stations needed to match the A1 measure-everything plan's risk (1.0511):**

| Plan | Stations | vs A1 (4) |
|---|---|---|
| Measure all, either tie rule | 4 | none saved |
| Marginal cluster must be measured (price ×1e5), either tie rule | 4 (1.038) | none saved |
| Floor 3, ties required | 3 (0.168; 3 is the network floor) | 1 saved |
| Floor 3, ties relaxed | **1** (0.228) | 3 saved |
| No floor, model prior | **0** (9.5e-5) | all four |

## 3. What this says

1. **A station saving exists, and it comes entirely from not measuring.** Measuring
   everything saves nothing against A1 whichever way ties are handled (relaxing ties
   makes 2 and 3 stations feasible, but at 1.79 and 1.35 they are far riskier than 4).
   Accepting the marginal cluster on its process evidence and measuring only the three
   capable characteristics reaches a risk 5-80 times below A1's with one station (ties
   relaxed) and 3-4 stations (ties required).
2. **It is a statement about the prior, not about the cell.** Accepting the marginal
   cluster is "right" only if the process really is as elicited. The **break-even
   multiplier** (`break_even_multiplier`) says how wrong it could be: measuring a
   marginal characteristic at the 28-32 µm the 4-station A1 plan achieves (±50 µm) costs
   0.34 expected wrong decisions, **about 1.1 × 10⁴ times** the 3.2e-5 of accepting it. The
   real process would have to be failing the tolerance on about a third of parts, not
   0.003%, before measuring it from this cell reduces wrong decisions. The ×1e2-×1e5
   rows are that sensitivity: the planner flips from "measure nothing" to "measure the
   marginal cluster" between a marginal nonconformance of 3% and 30-100%.
3. **If policy insists that the marginal characteristics be measured** (price ×1e5), the saving
   disappears (4 stations either way) and the risk stays near 1.0: measuring the marginal cluster
   as well as this cell can (28-32 µm against ±50 µm) generates ~0.34 wrong decisions each, almost
   all false rejections. Only more stations (and a cell that gets nearer the part) reduce it.
4. **The "measure the capable cluster" outcome at floor 3 is the arithmetic's choice, not an
   engineering recommendation.** Among the characteristics a floor forces the plan to measure,
   the capable ones are cheapest (lowest measured risk); a metrologist would sooner pick the
   marginal ones. The two rows above bracket the choice.

## 4. The finding underneath: the decision rule and the prior

Everything above uses **simple acceptance**. That rule is not risk-optimal when the process prior is
tighter than the measurement, and here it is (σ 3 µm and 8.3 µm against u = 13-32 µm: the measurement is 1.6-10 times broader than the process spread).
`best_acceptance_risk` searches symmetric acceptance limits for the smallest PFA + PFR:

| u (µm) | marginal: simple | marginal: best rule | high-cap.: simple | high-cap.: best rule |
|---|---|---|---|---|
| 3 | 0.0023 | 3.17e-5 | 1.6e-8 | 2.0e-9 |
| 8 | 0.080 | 3.17e-5 | 1.5e-5 | 2.0e-9 |
| 13 | 0.184 | 3.17e-5 | 0.0012 | 2.0e-9 |
| 30 | 0.347 | 3.17e-5 | 0.108 | 2.0e-9 |

(unmeasured price: 3.17e-5 and 1.97e-9). The best rule accepts nearly everything (limits at 1.3-6 tolerance half-widths);
its risk equals the unmeasured price. **With these priors a measurement at u ≳ 2-3 µm tells the decision-maker
almost nothing that the process evidence did not, whatever the rule**, and simple acceptance turns that
uninformative measurement into 0.1-0.35 wrong decisions per characteristic. So:

* The 1.0-1.7 expected wrong decisions of every measure-everything plan on the track is largely a
  property of simple acceptance combined with tight priors, not of tracker placement. Placement
  matters for how much *that* costs; it does not change that a better rule would erase most of it.
* The case for measuring rests on **doubt about the prior**: the 10% chance a feature belongs
  to a worse process (the repo's `ClusterAssignment.confidence` of 0.9, "reported, not propagated"), a
  drifting process, a first article. That doubt is currently *not in the model*. Propagating it (a mixture prior
  with a stated worse component) is the principled version of the `escape_price_multiplier` sweep here, and
  needs a source for the worse component (CLAUDE.md §6: not to be invented).
* This is the place the project's "risk from JCGM 106" contribution is strongest and most exposed: it says
  exactly when a measurement is worth taking, and for GKN's own Cpk targets the answer, on this evidence, is
  "rarely, at tracker-on-a-track uncertainties", which is either a finding or a sign the priors are too tight to
  be believed. That needs discussing with the supervisor and GKN before it goes in a paper.

## 5. What was not done

* A mixture or doubt-propagating prior (needs a sourced worse component).
* One-sided or asymmetric acceptance windows (the biased marginal cluster might gain slightly).
* A tolerance sweep of this experiment (the conditions sweep covers the measure-everything case).
* GA noise: searches are single runs (seeded); values agree with the earlier exact GA winners where they
  overlap (§1), differences between configurations here are orders of magnitude, not percent.

## 6. Reproduce

```
python -m pytest tests/test_process_evidence.py
python -m viz.run_station_track_process_evidence       # about 70 min on 4 cores; resumable
```
