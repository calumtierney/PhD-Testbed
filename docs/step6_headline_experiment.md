# Step 6: the headline experiment

Companion to `src/planning/`. This is the experiment the project exists
to run (CLAUDE.md §1): does replacing the objective function -- same
optimiser, same scene, same budget, only the weighting changes -- pull
measurement effort away from high-capability characteristics and towards
marginal ones, the way the risk-based hypothesis predicts?

**This document was substantially revised after a supervisory review of a
paper drawing on this testbed.** That review found the original two-plan
("Plan A vs Plan B") framing overstated what was actually being compared,
found two metrologically invalid characteristic definitions, an
unrealistically incapable "marginal" process, and an unrealistic
candidate-placement envelope, among other things. Every substantive
finding that was a testbed issue (as opposed to a manuscript-writing
issue outside this repository) is fixed here; each section below says
what changed and why. The short answer is unchanged in direction: **the
objectives differ, in the direction the hypothesis predicts, modestly in
this scene** -- but several numbers changed, and the comparison itself is
now a proper four-way ablation rather than a two-plan one.

## The four objectives (was two)

A supervisory review pointed out that "Plan A" (uniform weights on
*projected* characteristic uncertainty) is not literally the field's
A-optimal criterion -- it already uses step 4's tolerance-direction
projection, which is itself a goal-oriented step existing A-optimal
practice doesn't take. The comparison needed a genuine A-optimal baseline,
and needed to ask why a *linearised* risk weight was used at all when a
single plan evaluation costs ~10-200 ms (cheap enough to search directly
under the real, nonlinear objective). `planning.objectives` now defines
four:

- **A0 -- trace of the full 3D point covariance.** The field's actual
  criterion (Schmitt et al.'s literal trace; Cai's and Wang-Forbes-
  Maropoulos's closely related summed point uncertainties). Blind to
  tolerance direction entirely.
- **A1 -- uniform weights on projected characteristic uncertainty.**
  What this codebase called "Plan A" before this revision. Already
  goal-oriented (it uses each tolerance's own constraint direction, step
  4), just unweighted.
- **B -- risk-derived weights, linearised.** What this codebase called
  "Plan B": each characteristic's weight is `d(risk)/d(uncertainty)` at a
  fixed reference point, computed once before the search.
- **C -- direct nonlinear risk minimisation.** No linearised weight at
  all: each candidate is scored by its own actual, achieved global risk.

All four share exactly the same `evaluate_plan` (one network solve, one
step 4 projection per characteristic); only the scoring function
(`ObjectiveFn`) differs -- CLAUDE.md §1's "replacing the objective
function, not the optimiser," now literally the architecture, not just
the framing. See `planning.objectives`'s module docstring for the full
reasoning, including why B is still worth reporting next to C (a
deployment argument -- B's weight can be dropped into an existing
weighted-sum tool that never needs to know about JCGM 106 at all; C
needs the risk model inside the search loop).

**Not built:** heuristic baselines (tolerance-normalised weights, 1/Cpk
weights, a minimum-TUR constraint as a *baseline* rather than a guard --
see "Guarding against unbounded degradation" below for the guard version
-- or an expert SpatialAnalyzer plan). A genuine comparison against
practitioner intuition needs a practitioner, not a heuristic Claude
invents; noted as a real gap for future work, not filled with a
placeholder.

## The scene: process capability corrected

The two feature clusters (`planning.experiment.default_two_cluster_scenario`)
are still one high-capability, one marginal, spatially separated group of
three profile characteristics each -- but the marginal cluster's process
parameters changed.

**What was wrong.** The original marginal cluster (mean 40 µm, std 6 µm
against a ±50 µm tolerance) has Cpk ≈ 0.56 -- roughly 4.6% baseline
nonconformance *before any measurement uncertainty is even considered*.
SAE AS13006 requires a minimum Cpk of 1.33 for a key characteristic in
production. At Cpk 0.56, the risk this experiment reports is mostly "this
process already makes bad parts", not "this process's risk is unusually
*sensitive to measurement uncertainty*" -- which is the actual
mechanism the experiment exists to demonstrate.

**The fix.** Marginal cluster: mean 38 µm, std 3 µm -- `Cpu = (50-38)/(3×3)
= 1.333`, exactly the AS13006 floor. "Marginal" now means *closer to its
limit at a still-legal capability*, not *out of control*. High-capability
cluster is unchanged (mean 0, std 5 µm, Cpk ≈ 3.33).

## The candidate envelope: floor-reachable, not full-hemisphere

**What was wrong.** `hemisphere_candidate_poses` allows any elevation
from 0° to 90°; a laser tracker is floor- or tripod-mounted and cannot
hover far above a part. The original default elevations (20°, 40°, 60°)
at a 4 m standoff put some candidates several metres above target height
-- unreachable by a real tripod for a part around head height.

**The fix.** `DEFAULT_ELEVATIONS_DEG = (0.0, 20.0)`: level with the
target, or up to ~1.4 m above it (a tripod on a low platform). Both
physically plausible for the scene's target heights (~0.8-1.5 m).
Reduced from three elevation levels to two, and combined with the
fix below, purely for **runtime**, not realism -- see next section.

## Runtime: why the grid is smaller than it could be

Objective C calls `risk.jcgm106.evaluate_conformity_risk` (several
`scipy.integrate.quad` calls each) once per characteristic per candidate:
measured at ~200 ms per full-scenario evaluation, against ~20 ms for a
network solve alone and ~5 µs for a linear weighted sum (A0/A1/B). At the
original 36-candidate (12 azimuth × 3 elevation) grid, `C(36, 2) = 630`
combinations, a C-objective search alone would take ~630 × 0.2 s ≈ 2
minutes; at 48 candidates (the floor-envelope fix's first draft, 12 × 4),
`C(48, 2) = 1128` pushed that past 4 minutes. Reduced to 12 azimuths × 2
elevations = 24 candidates, `C(24, 2) = 276`, keeping a full four-objective
run under two minutes without weakening the comparison itself (every
objective still searches the same candidate set). If a finer grid is
wanted later, the cheap route is speeding up objective C (an analytic
closed form exists for Gaussian-prior/Gaussian-noise conformity risk,
avoiding `quad` entirely -- not built here), not shrinking the candidate
set further.

## The result

At the default resolution (24 candidates, 2 searched stations, the
scenario's fixed anchor, `reference_uncertainty_m = 8 µm` for B's
linearisation):

| | A0 (trace) | A1 (uniform, projected) | B (linearised risk) | C (direct risk) |
|---|---|---|---|---|
| Stations | (4.88, 3.26, 2.37), (4.88, -3.26, 2.37) | *same as A0* | (-0.76, 0, 2.37), (4.88, -3.26, 2.37) | *same as B* |
| Summed uncertainty | 44.32 µm | 44.32 µm | 46.60 µm | 46.60 µm |
| Cluster 1 (high-cap) sum | 22.16 µm | 22.16 µm | 24.72 µm | 24.72 µm |
| Cluster 2 (marginal) sum | 22.16 µm | 22.16 µm | 21.88 µm | 21.88 µm |
| Global PFA | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Global PFR | 0.1989 | 0.1989 | 0.1926 | 0.1926 |
| **Global risk (PFA+PFR)** | **0.1990** | **0.1990** | **0.1926** | **0.1926** |
| Bounded part-level risk `1-∏(1-r_c)` | 0.1863 | 0.1863 | 0.1807 | 0.1807 |

Two things worth stating plainly rather than glossing over:

**A0 and A1 land on the identical plan here.** At this scene's exact
left-right symmetry and this candidate resolution, minimising the raw
covariance trace and minimising the projected characteristic uncertainty
happen to agree -- both pick the fully symmetric station pair. This is a
property of this scene (all six characteristics share one projected
direction, one tolerance, symmetric geometry), not a general equivalence;
§4b's own reasoning for why A0 ≠ A1 in general still holds; a scene with
mixed tolerance types (position, parallelism, profile together) would be
the natural next check.

**B and C also land on the identical plan here.** The linearised weight
(B) and the direct nonlinear objective (C) agree at this resolution: the
linearisation cost nothing in this particular case. That's a real,
reportable finding, not a foregone conclusion -- it's exactly the
question the ablation exists to answer, and here the answer is "no
difference detected", which is itself informative (a finer grid or a
scene where uncertainty swings more between candidates could in
principle separate them; not observed here).

**The result the earlier framing reported still holds, restated
correctly**: risk-based weighting (B/C) moves effort from the
high-capability cluster (+11.6%, 22.16 → 24.72 µm) to the marginal one
(-1.3%, 22.16 → 21.88 µm), at a cost of +5.1% traditional "summed
uncertainty" (44.32 → 46.60 µm) but a gain of -3.2% global risk (0.1990 →
0.1926). Smaller in magnitude than the pre-revision numbers (which used
the unrealistic Cpk 0.56 cluster and a coarser accounting) but the same
direction, for the reason already documented below (headroom and scene
symmetry) plus the new one above (a milder, more realistic marginal
process is, correctly, less dramatic to fix).

## Guarding against unbounded degradation (M9)

**What was wrong.** B's linearised weight for the high-capability cluster
here is tiny (see "The weights" below) -- nothing in the unconstrained
objective stops the high-capability cluster's uncertainty from growing
arbitrarily large in exchange for an arbitrarily small gain to the
marginal cluster, well past where the small-perturbation linearisation
the weight was computed under stops being a good approximation.

**The fix.** `objectives.constrained_objective` wraps any objective with
a hard per-characteristic uncertainty cap; a candidate violating it for
*any* characteristic scores `+inf` (disqualified), applied identically
across all four objectives via `run_headline_experiment(max_uncertainty_m=...)`.
Demonstrated at `max_uncertainty_m = 8.5 µm` (feasible -- A1's own
unconstrained maximum is 8.11 µm; binding on B, whose unconstrained
maximum is 9.16 µm):

| | B unconstrained | B capped at 8.5 µm |
|---|---|---|
| Max per-characteristic uncertainty | 9.16 µm | 8.39 µm |
| Summed uncertainty | 46.60 µm | 44.55 µm |
| Global risk | 0.1926 | 0.1983 |

The cap pulls B back towards A1's station arrangement and most of A1's
risk level -- exactly the intended trade: a quality system's own
uncertainty ceiling (a minimum TUR, an FAI requirement) can be imposed
without abandoning risk-weighting, at the cost of giving up some of its
benefit. Whether that trade is worth it in a given programme is a policy
question this codebase doesn't answer; it now at least makes the trade
possible to compute.

## The iso-risk comparison (M11, M14)

**What was wrong.** The pre-revision reporting compared objectives at one
fixed station count and quoted a risk *percentage* difference. That
doesn't answer the industrially meaningful question: does risk-based
weighting let you reach an *acceptable* risk level with *fewer*
resources? `planning.experiment.iso_risk_station_counts` answers this
directly -- for a target total risk, the smallest station count (1 to
`max_stations`) each objective's best plan actually reaches it at.
Restricted to the two cheap objectives (A1, B; objective C's per-
evaluation cost makes a multi-station-count sweep of it impractical at
this session's time budget -- a fast, closed-form risk calculation would
remove that restriction, see "Runtime" above):

| Target total risk | A1 (uniform) | B (risk-weighted) |
|---|---|---|
| ≤ 0.30 | 2 stations | **1 station** |
| ≤ 0.25 | 2 stations | **1 station** |
| ≤ 0.20 | 2 stations | 2 stations |
| ≤ 0.19 | 3 stations | 3 stations |

At a loose target, risk-weighted planning reaches it with half the
stations. At tighter targets the two tie in this scene. This is exactly
the "how many stations does each objective actually need" question a
supervisory review asked for, and the honest answer is "sometimes fewer,
not always" -- not the unconditional win a station-count-agnostic
percentage comparison could be read to imply.

**Still not built:** this is a station-*count* iso-risk test, not a
station-*time* one. `PlanReport.summed_characteristic_uncertainty_m` is
named for exactly what it is now (a sum of uncertainty), not "effort" or
"time" -- station count and target set are identical within any one
comparison here, so neither actually varies, and nothing above should be
read as a time-reduction claim. A real time model (setup time per
station, move time between them) is a separate, uncalibrated piece of
work, not attempted here.

## Reporting risk correctly (M12)

**What was wrong.** Only the summed PFA+PFR was reported per plan; the
summed value is an expected *count* of erroneous decisions (valid, since
expectation is additive regardless of correlation) but is not itself a
bounded probability, and could exceed 1 for a large enough characteristic
set.

**The fix.** `PlanReport` now exposes `global_pfa` and `global_pfr`
*separately* (both were already computed, just summed together before
being reported) and a new `bounded_part_level_risk = 1 - ∏(1 - r_c)`,
which is a genuine probability. Its own docstring states plainly what it
assumes (each characteristic's error is independent of the others') and
why that's not exactly true here (every characteristic in a plan shares
the same stations, so their errors share the same station-pose
uncertainty) -- reported as an upper-bound-style summary, not an exact
joint part-level risk. An exact version needs the joint distribution over
every characteristic's decision, not built here.

## The weights, and why the ratio didn't get much smaller

At the corrected Cpk (marginal 1.33, high-capability 3.33), the risk-
derived weight ratio is still roughly **62,000:1** -- barely smaller than
the pre-revision figure computed from the unrealistic Cpk 0.56 cluster.
This is worth understanding, not just noting: the weight is a *local
slope* (`d(risk)/d(uncertainty)` at the reference point), not the
absolute risk level. A tighter, more capable process (std 3 µm vs the
old 6 µm) sitting the same relative distance from its limit transitions
*more sharply* as uncertainty grows past the point where it starts to
matter -- a smaller distribution "sees" the limit more suddenly than a
wider, smeared-out one does. So fixing the unrealistic *baseline risk*
(M9's and minor issue 14's complaint) did not, and should not have been
expected to, fix the weight ratio's extremity on its own; the guard
constraint above is the mechanism that actually bounds its consequence.

## The uncertainty budget: how much is even reallocatable (M6)

**What was wrong.** Every number above comes from `LaserTracker`'s
sensor-noise-only model (~7-10 µm from Table 4 inputs at this scene's
ranges). Real shop-floor uncertainty also includes calibration-parameter
uncertainty in the NPL geometric model, thermal drift, refraction, SMR/
nest repeatability, gravity and fixturing -- none of which shrink when a
station moves to a better spot. Reporting a placement effect without
saying what fraction of the total budget placement can even influence
overstates what station optimisation alone can achieve.

**The fix (partial -- flagged as such).** `LaserTracker.systematic_std_m`
adds an isotropic, placement-independent variance term (default 0.0,
so every existing result above is unaffected), and
`instruments.laser_tracker.reallocatable_fraction` reports what share of
the total variance placement can actually move:

| Systematic std | Total RSS uncertainty | Reallocatable fraction |
|---|---|---|
| 0 µm (sensor noise only, this doc's numbers above) | 10.33 µm | 100% |
| 4 µm (≈ SMR nest repeatability alone) | 12.44 µm | 69% |
| 10 µm (nest + a share of thermal/refraction) | 20.17 µm | 26% |
| 15 µm | 27.96 µm | 14% |

At a plausible combined systematic term of ~10 µm, only about a quarter
of the total uncertainty budget is reallocatable by placement at all --
meaning the headline percentages above, computed at 0% systematic
contribution, are an upper bound on what placement alone can achieve, not
a realistic shop-floor prediction. Re-running the full four-objective
ablation with a nonzero systematic term is the natural next step; not
done in this revision (it does not change which candidate any objective
picks in a way this document has verified, only how much of the resulting
number is real) -- flagged honestly as future work rather than silently
assumed away. The isotropic form of the term is itself a simplification
(real systematic effects have their own directional and cross-station
correlation structure a single per-point isotropic number cannot
capture) -- see `LaserTracker.systematic_std_m`'s docstring.

## Datum dependence is expected, not a defect (M7)

**What was wrong (in framing, not in the fix itself).** An earlier
version of this document described the finding that swapping which of
two stations is treated as the network solve's frame-of-reference anchor
changes the resulting target covariance by an amount the *same order of
magnitude* as the covariance itself as a "methodological trap this
experiment caught" -- worded as a surprising anomaly this codebase
discovered.

**The correction.** This is not a surprise; it is exactly what free-
network (geodetic datum) theory predicts. A network built purely from
relative range/angle observations determines the *shape* of the
station/target configuration but not where that shape sits or how it's
oriented in space -- a classic gauge freedom (Baarda's S-transformations;
inner-constraint/free-network solutions in the geodetic network-design
literature this codebase's laser-tracker-network setting descends from).
Coordinate covariance is inherently datum-dependent: fixing a different
station as the exactly-known reference is choosing a different datum,
and different datums legitimately give different coordinate covariances
for the same physical network. What *was* a genuine implementation
mistake -- and what the fix actually addresses -- is letting "whichever
candidate happens to be evaluated first" silently become that datum
during a grid search: since the choice matters, letting it vary
uncontrolled between candidates would bias the search in a way unrelated
to genuine placement quality. The fix (an external, fixed anchor shared
by every candidate under every objective) is unchanged by this
correction; only the description of *why* it's needed is corrected here.
A more complete treatment would evaluate characteristics in a
datum-invariant way (e.g. relative to measured datum features, not an
arbitrary external station) -- not built here, noted as the more
principled long-term fix.

## The "exact observations" shortcut, and why it's valid (unchanged)

`evaluate_plan` scores a candidate by solving the network from *exact*
(noiseless) observations at the true scene geometry
(`network.solve.exact_observations`) rather than simulating random noise
and fitting it: for this linear-Gaussian measurement model, the resulting
covariance is -- to numerical precision -- identical to what a real noisy
solve's covariance would be, verified directly
(`tests/test_planning.py::test_evaluate_plan_design_covariance_matches_a_noisy_solve`,
agreement within 5%, limited by how many noisy samples were compared
against, not a systematic difference). This makes the grid search
deterministic (no RNG state to manage across hundreds of candidates) and
roughly three orders of magnitude cheaper per evaluation than a full
noisy nonlinear fit, since the solver converges in a single function
evaluation (it starts exactly at its own answer).

## What step 6 still deliberately doesn't do

Per CLAUDE.md §3 ("do not invent an optimiser") and §9 ("if a requested
feature would take the project outside the ladder, say so"):

- No GA. Grid search over the reduced 2-DOF-per-station space is fast
  enough for A0/A1/B; only C's per-evaluation cost is a real constraint,
  addressed by a smaller candidate grid rather than a different search
  method (CLAUDE.md §3's own fallback condition, "if grid is too slow",
  is about the *search*, and a smaller grid resolves it without
  introducing a new search algorithm).
- No occlusion (step 2) in the search itself -- the scene here has no
  obstacles, so every station sees every target.
- Only one instrument type, one tolerance type (profile) across every
  characteristic in the demonstration scene -- both are
  `PlanningScenario` fields a caller can vary; nothing in `planning`
  assumes either is fixed. A scene mixing tolerance types is the natural
  next check for whether A0 and A1 keep agreeing (see "The result" above).
- No heuristic baselines (1/Cpk weights, a practitioner's plan) in the
  ablation -- flagged, not filled with an invented placeholder.
- No fast, closed-form replacement for objective C's `scipy.integrate.quad`
  calls, which would remove this document's own runtime constraint on
  candidate resolution and the C-objective iso-risk sweep.
