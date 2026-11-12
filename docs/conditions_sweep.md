# When does the objective change the plan? A conditions sweep

**Question.** On the headline scenes, A1 (uniform weights on uncertainty along each
tolerance), B (risk-derived weights) and C (direct expected risk) chose nearly the
same plans. Is that a property of those particular scenes, or general? This sweep
moves one condition at a time and records how far each objective's best plan lands
above C's in total risk, at a **fixed** number of stations.

Scene: the headline ring (`default_two_cluster_scenario`: three high-capability
characteristics at y ≈ +2 m, three marginal at y ≈ −2 m, all profile tolerances
along +z), the 24-pose ring, **exhaustive** grid search at 2 and 3 added stations,
exact risk throughout (no GA noise). Run: `python -m viz.run_conditions_sweep`
(about 45 min on 4 cores), figure `python -m viz.make_conditions_sweep_figure`
→ `results/panel12_conditions_sweep.png`, data `results/conditions_sweep.json`.

| Condition | Values | What it changes |
|---|---|---|
| (a) tolerance | ±25 … ±250 µm, priors scaled (Cpk fixed) | uncertainty relative to tolerance |
| (b) marginal cluster mean | 20 … 44 µm at σ 3 µm, ±50 µm (Cpk 3.3 … 0.67) | how different the two processes are |
| (c) separation | clusters at y = ±0.25 … ±3 m | how much serving one cluster costs the other |
| (d) systematic floor | 0 … 20 µm isotropic per point | uncertainty no placement can reduce |

Baseline (the headline experiment): ±50 µm, mean 38 µm, d = 2 m, no floor.

## Results

Risk of each objective's best plan above C's best (percent), extremes over the sweep:

| | 2 added stations | 3 added stations |
|---|---|---|
| **B** (risk weights at 8 µm) | ≤ 0.6% | ≤ 0.8% |
| **A1** (uniform, along the tolerance) | **≤ 7.8%** (mean 30 µm) | ≤ 2.3% |
| **A0** (trace) | ≤ 7.8% (same plan as A1 except at d = 3 m) | ≤ 16.3% (mean 20 µm) |

* **A1 and C separate most when the two processes differ but neither is hopeless,
  and the uncertainty sits where the marginal cluster's risk is still steep and the
  capable cluster's is flat**: peak 7.8% at marginal mean 30 µm (Cpk 2.2), and
  5.7-5.8% at ±100-150 µm (2 stations). At very tight tolerances, or very marginal
  processes, every characteristic's risk is steep, so "less uncertainty everywhere"
  (A1) and "less risk" (C) agree; at very wide tolerances or very capable
  processes all risk is tiny and nothing matters.
* **The clusters have to be far enough apart to compete for stations**: at or below
  d = 1 m one plan serves both and A1, B and C agree (A1 ≤ 0.2%; A0 still differs at 3 stations).
* **An irreducible floor erases the difference**: with a 10-20 µm isotropic term
  per point, placement barely changes risk and all objectives agree within 0.7%.
* **A third station shrinks A1's gap** (≤ 2.3%) but widens A0's: with more
  stations to place, the tolerance direction matters more than the weights.
* **B tracks C almost exactly** on this scene (≤ 0.8%). The one case seen where B
  departed from C is the station track with the anchor at the +x tip, where the
  plan pushed uncertainties far from B's 8 µm reference (`docs/station_track_experiment.md`).
* **In absolute terms the gaps are small**: the largest A1 − C difference is 0.008
  expected wrong decisions (mean 41 µm, 2 stations).

## What this means for the project

At a fixed station count, **risk weighting buys at most a few percent over
tolerance-aware uncertainty planning (A1)**, under every condition tried here, and
nothing on the station track. The direction of each tolerance is what makes the
difference against the literature's trace criterion (A0); the risk weights add
little on top. If industry already plans along the tolerance, the case for risk-
based planning cannot rest on reweighting a fixed number of stations.

Where it can rest, given everything run so far:

1. **Which characteristics to measure, and how well.** The wing-box saving against
   an A1-like plan (9 → 3 stations) came from leaving capable features unmeasured
   or loosely measured at a stated price; neither the ring nor the track lets a
   plan do that.
2. **A stopping rule that knows about the process.** A1's knee and the TUR rule
   ignore capability; TUR 4:1 could not be met at all on the track below ±250 µm.
3. **Pricing a plan.** Risk says when measuring is worse than not measuring
   (on the track at ±50 µm a marginal characteristic measured at ~30 µm carries
   ~0.34 expected wrong decisions against ~3e-5 for accepting it on process
   evidence), which no uncertainty criterion can express.

Caveats: one synthetic scene, two clusters, one tolerance type; each condition
moved alone (interactions not explored); B's reference fixed at 8 µm throughout.
