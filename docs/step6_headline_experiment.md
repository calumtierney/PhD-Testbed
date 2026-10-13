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
three profile characteristics each -- but both clusters' process
parameters have now been changed, in two separate corrections.

**What was wrong (marginal cluster).** The original marginal cluster
(mean 40 µm, std 6 µm against a ±50 µm tolerance) has Cpk ≈ 0.56 --
roughly 4.6% baseline nonconformance *before any measurement uncertainty
is even considered*. SAE AS13006 requires a minimum Cpk of 1.33 for a key
characteristic in production. At Cpk 0.56, the risk this experiment
reports is mostly "this process already makes bad parts", not "this
process's risk is unusually *sensitive to measurement uncertainty*" --
which is the actual mechanism the experiment exists to demonstrate.

**The fix.** Marginal cluster: mean 38 µm, std 3 µm -- `Cpu = (50-38)/(3×3)
= 1.333`, exactly the AS13006 floor. "Marginal" now means *closer to its
limit at a still-legal capability*, not *out of control*.

**What was wrong (high-capability cluster, a second, separate
correction).** The original fix above left the high-capability cluster at
mean 0, std 5 µm, Cpk ≈ 3.33 -- unchanged from the very first draft, and
unsourced: "comfortably clears the AS13006 floor" is true but doesn't say
*how far* above 1.33 is actually representative of anything, and CLAUDE.md
§6 ("never fabricate a reference number") applies to this input exactly as
much as to any physics constant. GKN Aerospace -- this project's named
industrial partner (CLAUDE.md §1) -- publishes two capability tiers: a
minimum Cpk of 1.33 for standard manufacturing and quality requirements,
and Cpk 2.0 (their "zero-defect" standard) for critical or
high-performance programme requirements. The marginal cluster already
sits exactly on GKN's own standard-process floor; the high-capability
cluster had no equivalent anchor at all.

**The fix.** High-capability cluster: mean 0, std 8.33 µm -- `Cpu =
50/(3×8.33) = 2.0`, GKN's own critical/high-performance programme target,
replacing the unsourced 3.33. Both clusters are now anchored to a real,
citable GKN capability tier rather than one sourced value next to one
arbitrary one.

**This changed the headline weight ratio substantially -- see "The
weights" below.** The two process-capability fixes affect the results
table differently: every *uncertainty* number in "The result" is
unchanged (uncertainty depends only on measurement geometry, which
neither fix touches), but the *risk* numbers and the risk-derived weight
ratio both move, because both depend on exactly how tight the process
prior is assumed to be.

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
| **Global risk (PFA+PFR)** | **0.1990** | **0.1990** | **0.1927** | **0.1927** |
| Bounded part-level risk `1-∏(1-r_c)` | 0.1863 | 0.1863 | 0.1808 | 0.1808 |

(Station positions and every uncertainty number are unchanged from the
unsourced-Cpk version of this table -- they depend only on measurement
geometry. Global risk moved slightly, in the fourth decimal place: at
Cpk 2.0 the high-capability cluster carries a little more risk than it
did at the old, unsourced Cpk 3.33, still utterly negligible next to the
marginal cluster's contribution -- see "The weights" below for the one
number that *did* move substantially.)

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
0.1927). Smaller in magnitude than the pre-revision numbers (which used
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
| Global risk | 0.1927 | 0.1984 |

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

## The weights, and why the ratio changed by 50x when the Cpk source did

With the marginal cluster at AS13006's floor (Cpk 1.33) and the
high-capability cluster at the old, *unsourced* Cpk 3.33, the risk-
derived weight ratio was roughly **62,000:1** -- barely smaller than the
pre-revision figure computed from the unrealistic Cpk 0.56 marginal
cluster. With the high-capability cluster corrected to GKN's own cited
Cpk 2.0 (critical/high-performance programme target -- "The scene" above),
the same calculation gives roughly **1,239:1** instead -- a **50x drop**,
from one single-input correction.

This is worth understanding precisely, not just noting the new number. A
weight here is a *local slope*, `d(risk)/d(uncertainty)` evaluated at the
shared 8 µm reference point -- not the absolute risk level, and not the
process's own Cpk directly. Computing that slope for each cluster in
isolation:

| Cluster | Cpk | std | Slope at 8 µm |
|---|---|---|---|
| Marginal (unchanged) | 1.33 | 3 µm | 22,851 |
| High-capability, old (unsourced) | 3.33 | 5 µm | 0.37 |
| High-capability, new (GKN critical target) | 2.0 | 8.33 µm | 18.4 |

The marginal cluster's own slope never moved -- only the high-capability
cluster's did, by roughly 50x, which is exactly where the ratio's 50x
drop comes from (the marginal slope is the numerator in both ratios, so
it cancels). The old Cpk 3.33 process was so tight (std 5 µm against a 50
µm tolerance -- essentially all of its probability mass sits more than
three tolerance-widths from the limit) that an 8 µm measurement
uncertainty barely reaches any of it: the risk curve is close to flat
all the way out past 8 µm, so its local slope there is close to zero
almost by construction. GKN's real Cpk 2.0 process (std 8.33 µm) is still
comfortably, legally capable, but it has enough more spread that a
noticeably larger share of its probability mass sits within reach of an
8 µm measurement uncertainty -- so its risk curve has already started
climbing by that point, and its slope there is correspondingly larger.

The pre-revision fix (Cpk 0.56 → 1.33 for the *marginal* cluster, the
first "What was wrong" above) barely moved this ratio, because it never
touched the number actually driving the extremity -- the high-capability
cluster's own, unsourced tightness. That the ratio dropped 50x the moment
a *real, cited* number replaced the unsourced one is itself the
illustration of why CLAUDE.md §6 ("never fabricate a reference number")
matters beyond the physics constants in §4: an invented input was
silently inflating a headline comparison figure by more than an order of
magnitude, and the inflation wasn't visible from the code or the maths --
only from checking where each number actually came from. The guard
constraint above remains the mechanism that bounds what a large ratio can
*do* to a candidate's own uncertainty, regardless of how large the ratio
itself turns out to be.

## The uncertainty budget: how much is even reallocatable (M6)

**What was wrong.** Every number above comes from `LaserTracker`'s
sensor-noise-only model (~7-10 µm from Table 4 inputs at this scene's
ranges). Real shop-floor uncertainty also includes calibration-parameter
uncertainty in the NPL geometric model, thermal drift, refraction, SMR/
nest repeatability, gravity and fixturing -- none of which shrink when a
station moves to a better spot. Reporting a placement effect without
saying what fraction of the total budget placement can even influence
overstates what station optimisation alone can achieve.

**The fix.** `LaserTracker.systematic_std_m` adds an isotropic,
placement-independent variance term (default 0.0, so every existing
result above is unaffected), and `instruments.laser_tracker.
reallocatable_fraction` reports what share of the total variance
placement can actually move:

| Systematic std | Total RSS uncertainty | Reallocatable fraction |
|---|---|---|
| 0 µm (sensor noise only, this doc's numbers above) | 10.33 µm | 100% |
| 4 µm (≈ SMR nest repeatability alone) | 12.44 µm | 69% |
| 10 µm (nest + a share of thermal/refraction) | 20.17 µm | 26% |
| 15 µm | 27.96 µm | 14% |

**A bug this fix initially had, since corrected.** The table above comes
from `LaserTracker.covariance_local`/`covariance_global` -- the
single-station model -- and was, and still is, correct on its own. But
`network.solve.solve_network`, which every `planning` search actually
calls, weights its residuals purely by `sigma_d`/`sigma_theta`/
`sigma_phi` (spherical sensor noise) and originally never added
`systematic_std_m` to the fitted covariance at all. The consequence: a
`PlanningScenario` built with a nonzero-systematic tracker
(`default_two_cluster_scenario(tracker=...)`) changed *nothing* about a
multi-station plan's result -- the term was silently dropped the moment
more than one station was involved, exactly the case this section exists
to talk about. It was caught (not by the original supervisory review)
after this document had already been written and described as "not yet
re-run" below; the actual bug was worse than "not yet re-run" -- it was
"would have shown no effect if it had been". Fixed by adding the same
isotropic term, once per target, to `solve_network`'s fitted covariance
(see that function's comment for why once-per-target rather than
once-per-observation: it is modelled as a per-point floor, not a
spherical-observation noise source, so it never belongs in the
residual weighting). `tests/test_network_solve.py`'s
`test_systematic_std_survives_a_multi_station_network_solve` and
`test_systematic_std_does_not_shrink_with_more_stations` are the
regression tests.

**The full ablation, re-run with a nonzero systematic term.** At the
plausible combined value of 10 µm (nest + a share of thermal/refraction),
running the identical four-objective comparison from the "Results"
section above through this scenario's tracker:

| | A0 = A1 | B = C | effort change | risk change |
|---|---|---|---|---|
| systematic = 0 µm (as above) | summed 44.320 µm, total risk 0.198980 | summed 46.597 µm, total risk 0.192680 | +5.14% | -3.17% |
| systematic = 10 µm | summed 74.644 µm, total risk 0.524915 | summed 76.055 µm, total risk 0.522890 | +1.89% | -0.39% |

(Both rows use the GKN-sourced high-capability cluster -- "The scene"
above -- not the earlier unsourced-Cpk numbers; every uncertainty figure
is identical to the unsourced-Cpk version of this table since uncertainty
depends only on geometry, and every risk figure moved by less than a
percentage point, since the high-capability cluster's own contribution to
total risk stays negligible at either Cpk.)

Both objectives pick the *same* station arrangements as the
systematic-free case (A0/A1 and B/C each land on the identical pair of
candidates) -- the systematic term shifts every candidate's uncertainty
by the same fixed amount, so it does not change *which* arrangement wins,
only how much of the resulting number reallocation can take credit for.
That's the qualitative confirmation the reallocatable-fraction framing
predicts: at 10 µm systematic, `reallocatable_fraction` says only ~26%
of the total variance is placement's to move, and the risk-weighting
benefit shrinks correspondingly (-3.17% -> -0.39%) rather than
disappearing outright or growing -- station placement is doing the same
job, just on a smaller share of a larger total. The isotropic form of
the systematic term is itself a simplification (real systematic effects
have their own directional and cross-station correlation structure a
single per-point isotropic number cannot capture) -- see
`LaserTracker.systematic_std_m`'s docstring.

`results/panel6a_stations_systematic10um.png` and `results/
panel6b_comparison_systematic10um.png` show this 10 µm-systematic run in
the same station-layout / per-characteristic-bar form as the zero-
systematic figures referenced above, with a shared ellipsoid exaggeration
scale across both objective groups so the two panels stay visually
comparable. `results/panel7_sensitivity_systematic10um.png` overlays two
bands onto the flat-vs-steep risk-vs-uncertainty curves: where this
experiment's *achieved* uncertainty actually lands with the systematic
term off (sensor noise only, 6.7-9.2 µm -- not 0 µm; sensor noise alone
is never zero) and with it on (+10 µm systematic, 12.1-13.6 µm). The
local risk-sensitivity ratio at the fixed 8 µm reference is unaffected by
the systematic term (~1,239:1 at the now GKN-sourced Cpk 2.0 -- see "The
weights" above for why this is 50x smaller than the figure computed at
the old, unsourced Cpk 3.33, a separate and much larger effect than
anything the systematic term does here) -- what the systematic term moves
is where the achieved uncertainty sits on the curve, and how much of that
band is reallocatable by placement.

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

## Mixed tolerance types: the check "A0 = A1, B = C" invited

**Why this check exists.** "The result" above found A0 identical to A1,
and B identical to C, and said plainly that this was a property of that
scene's symmetry -- every one of the six characteristics is a
`ProfileTolerance` against the same surface normal -- not a general
equivalence, and that "a scene with mixed tolerance types (position,
parallelism, profile together) would be the natural next check." This
section is that check, not a hypothetical anymore.

**The scene.** `planning.experiment.mixed_tolerance_two_cluster_scenario`
keeps the same two clusters, the same six target points, the same
process priors -- only the *tolerances* change. Each cluster now gets one
of each type on CLAUDE.md §3's own example pairing: a `PositionTolerance`
(spherical -- A1 sees the *whole* covariance here, the one point where it
should agree with A0 most closely), a `ProfileTolerance` (one direction,
the surface's own normal, z), and a `ParallelismTolerance` (one
direction, an *external* datum normal, x -- unrelated to the surface and
unrelated to the profile point's own direction).

**At the default 24-candidate grid, A0 = A1 and B = C still hold.**

| | A0 (trace) | A1 (uniform, projected) | B (linearised risk) | C (direct risk) |
|---|---|---|---|---|
| Stations | (4.88, 3.26, 2.37), (4.88, -3.26, 2.37) | *same as A0* | (3.00, -4.00, 1.00), (4.88, -3.26, 2.37) | *same as B* |
| Summed uncertainty | 37.88 µm | 37.88 µm | 39.69 µm | 39.69 µm |
| Global risk (PFA+PFR) | 0.1741 | 0.1741 | 0.1687 | 0.1687 |

Effort +4.8%, risk -3.1% -- the same direction, similar magnitude, to the
single-tolerance-type scene's +5.1%/-3.2% (`results/
panel6a_stations_mixedtol.png`, `results/panel6b_comparison_mixedtol.png`).
Mixing tolerance types *alone*, at this grid, was not enough to separate
either pair. Worth understanding why, not just reporting it: within one
cluster, the three points sit 0.2-0.4 m apart -- close enough that
whichever station pair is best for the cluster overall (mainly a function
of range and viewing angle to the *cluster*, not to any one point) is
close to best for every point in it, regardless of which single direction
that point's own tolerance happens to keep. The `ParallelismTolerance`
point's own projected uncertainty is also small in absolute terms (~2.4
µm vs ~8 µm for position/profile), so even where it *does* prefer a
different station, it isn't large enough to outvote the other two points
in A1's unweighted sum. Both effects point the same way: this scene's
raw geometry (how close three tightly-clustered points are to a
candidate), not tolerance direction, is still the dominant signal at this
resolution.

**At a coarser candidate grid, A0 and A1 do separate.** Reducing to 4
azimuths x 2 elevations (8 candidates instead of 24 -- the same reduction
`tests/test_planning.py::test_mixed_tolerance_scenario_can_separate_a0_from_a1`
uses, and `results/panel8c_mixedtol_divergence.png`):

| | A0 (trace) | A1 (uniform, projected) | B = C (risk-based) |
|---|---|---|---|
| Stations | (3.00, 3.46, 3.00), (3.00, -4.00, 1.00) | (3.00, 3.46, 3.00), (3.00, -3.46, 3.00) | (3.00, -4.00, 1.00), (3.00, -3.46, 3.00) |
| Summed uncertainty | 38.51 µm | 38.39 µm | 38.81 µm |
| Global risk (PFA+PFR) | 0.1919 | 0.1761 | 0.1708 |

A0 and A1 now pick genuinely different candidates -- not a numerical
rounding difference; their global risk differs by nearly 9% relative
(0.1919 vs 0.1761). With fewer alternatives on the table, a
close-but-not-identical ranking (position's direction-blind trace vs the
mixed scene's direction-aware projections) has fewer competing candidates
to lose to before it actually changes the winner. B and C still coincide
at this resolution too -- not asserted to hold in general, simply what
was observed here; `tests/test_planning.py`'s own test deliberately
doesn't assert either way on that pair, only on A0 != A1.

**The honest takeaway.** "A0 = A1" and "B = C" were never claimed as
general equivalences (§4e's own standard: never adjust an input to force
an output, and never claim more than what was actually checked) -- they
were reported as a property of one symmetric scene, with the prediction
that breaking the symmetry was the test of that claim. Breaking it one
way (mixed tolerance types, same resolution) didn't move the needle; a
second, independent change (coarser candidate grid) did. That both
matters -- resolution and tolerance diversity interacting, not either
alone -- is itself the finding this check was built to surface, not a
single headline number.

## The wing scenario: occlusion, a finer search space, and a GA

**Why this exists.** Every scene above stands candidates on a ring at one
fixed radius around a compact, few-metres-across target cluster, with
nothing in the scene to block a sightline. That's a reasonable proxy for
"which side, how high" when a part is small enough that distance barely
varies -- it says nothing about a part where distance genuinely is part of
the placement decision, or a factory floor with tooling in the way.
`planning.experiment.wing_four_point_scenario` is the harder scene built
to check that: four named fittings (root/mid-span/near-tip/tip) across a
real ~30 m span (CLAUDE.md §1's own upper range bound), plus four network
tie points, with a support-tower obstacle (`geometry.mesh.box_mesh`) under
mid-span that measurably reduces visibility across a representative
candidate set without ever reducing any single target to zero coverage
(`tests/test_planning.py::test_wing_scenario_support_tower_reduces_but_does_not_eliminate_visibility`).

A candidate grid fine enough to let station *distance* vary, not just
angle, multiplies the candidate count; `grid_search_station_placement`'s
`C(n_candidates, n_stations)` cost multiplies combinatorially on top of
that. CLAUDE.md §3's own named fallback is a standard GA, so this scene
is searched with `planning.search.genetic_algorithm_station_placement`
(`scipy.optimize.differential_evolution`, searching each station's
continuous (x, y, z) position directly) instead of a discretised grid.

**Getting a trustworthy GA comparison took three attempts, and that's
worth reporting as plainly as the numbers themselves.** A GA is not an
exhaustive search -- CLAUDE.md §4e's "never adjust an input to force an
output" applies just as much to accepting an under-converged search
result as it does to a physics constant, so each attempt was checked, not
assumed:

1. **First attempt** (40 generations, population 15, ~5500-6000
   evaluations per run): every one of the 8 runs (4 objectives x
   tower/no-tower) reported `converged=False`. Cross-checking directly
   -- evaluating objective C's own risk formula at *objective A1's*
   winning stations from the same search -- showed C's own search had
   converged to a total risk of 0.86 while a configuration its own
   formula scores at 0.56 already existed, found by a different run. A
   clear search failure, not a finding about the objectives.
2. **Second attempt**: `genetic_algorithm_station_placement` parallelised
   (`workers=-1`, ~3.5x measured speedup once every BLAS library was
   pinned to one thread per process -- see that function's own
   docstring) and the budget raised ~6x (150 generations, population 25,
   ~34,000 evaluations per run). Still every run reported
   `converged=False`, and the same cross-check now caught objective B:
   its own search converged to 0.000174, while objective C's winning
   stations scored 0.0000463 under B's own formula -- a ~3.75x miss.
   More raw budget alone was not fixing this.
3. **Third attempt**: initial-population seeding
   (`seed_station_positions_m`). Each scenario's objectives were run in
   sequence -- A0 first from a blind random initial population, then
   each subsequent objective's search seeded with every previous
   objective's winning stations, on top of the same random fill for the
   rest of the population. Verified directly before trusting it: seeding
   objective B with A0/A1/C's stations reproduced the 0.0000463 value C
   had found, with 1/6th the evaluations the second attempt used.

The numbers below are from this third attempt (60 generations, population
20, ~11,000 evaluations per run, `workers=-1`). `converged=False` is still
reported by every run -- DE's own convergence criterion is about
population spread, not really reachable at all on a landscape with a hard
feasibility penalty cliff (an infeasible candidate scores a fixed
`INFEASIBLE_PENALTY` regardless of how infeasible, so some population
spread survives indefinitely even near a true optimum) -- so it is not,
by itself, a reliability signal either way here. What *is* checked:
cross-validating every objective's winning stations against every other
objective's own formula. The residual gaps are now a few percent (e.g.
objective A1's own search landed ~2.7% worse, by its own metric, than
objective C's stations did) rather than the earlier 4-10x misses --
consistent with ordinary GA imprecision, not a repeat of the same
failure. Still worth stating directly: this is a looser guarantee than
the main scene's exhaustive grid search, which is provably the best
*in the searched candidate set*. A GA result is a good plan found, not a
certified optimum.

**The result**, `n_stations=3`, `bounds_m` spanning the reachable floor
area, `reference_uncertainty_m = 8 µm` for B exactly as in the main scene:

| | A0 (trace) | A1 (uniform, projected) | B (linearised risk) | C (direct risk) |
|---|---|---|---|---|
| **With the support tower** | | | | |
| Global risk (PFA+PFR) | 0.3731 | 0.3731 | 0.3731 | **0.3485** |
| **Without it** | | | | |
| Global risk (PFA+PFR) | 0.3617 | 0.3337 | 0.3202 | **0.2939** |

Two things, stated as plainly as the main scene's table states its own:

**Without the tower, risk falls monotonically from A0 to C** (0.362 →
0.334 → 0.320 → 0.294, a 19% reduction from A0 to C) -- the clean,
textbook version of the hypothesis this project exists to test, on a
scene an order of magnitude larger and considerably less symmetric than
the original. Cross-validated directly (every objective's winning
stations scored under every other objective's own formula): C's stations
are at or within a few percent of best under every one of the four
metrics simultaneously here, not just its own -- this scene did not turn
out to pit the objectives sharply against each other the way the main
scene's ablation does, which is itself worth saying rather than quietly
picking the more dramatic-looking number to report.

**With the tower, A0, A1 and B land on (within GA noise of) the same
plan, and C alone finds something measurably better** (6.6% lower risk
than the other three). This is the same mechanism already documented
above for the main scene and in "Mixed tolerance types" below: every one
of this scene's four characteristics is a `ProfileTolerance` against the
same surface normal, so A0 (trace-blind to direction) and A1 (projected,
but with nothing to disagree about direction-wise) have no reason to
diverge, and B's risk-derived weights here are dominated almost entirely
by the two marginal-cluster characteristics (near-tip/tip), which
happen, in this scene, to want nearly the same station arrangement A0
already finds. C's gain comes specifically from the occlusion
interacting with the direct risk objective in a way the linearised/
trace-based proxies don't capture at this resolution -- not a new
mechanism, the same "linearisation can cost something, it just usually
doesn't in this project's scenes" finding B/C's comparison already
makes, now actually observed rather than reported as a null result.

**Occlusion costs every objective risk, and costs the risk-directed one
more in relative terms.** Comparing each objective's own "with tower" row
to its "without" row: A0 degrades by 3.2% (0.362 → 0.373), C by 18.6%
(0.294 → 0.349). The tower genuinely narrows C's options more than it
narrows A0's, in relative terms -- plausible since C is normally free to
seek out a specific risk-minimising arrangement that the occlusion then
forecloses, while A0 was never aiming at anything that precise in the
first place. Not asserted as a general property of occlusion, only as
what this one scene showed.

## What step 6 still deliberately doesn't do

Per CLAUDE.md §3 ("do not invent an optimiser") and §9 ("if a requested
feature would take the project outside the ladder, say so"):

- Grid search, not a GA, over the main scene's reduced 2-DOF-per-station
  space -- fast enough for A0/A1/B; only C's per-evaluation cost is a
  real constraint, addressed by a smaller candidate grid rather than a
  different search method there. A GA *is* now built and used
  (`planning.search.genetic_algorithm_station_placement`) for the wing
  scenario below, where a finer/larger candidate space makes grid
  search's combinatorics intractable -- CLAUDE.md §3's own named
  fallback, not a new optimiser.
- Occlusion (step 2) *is* now wired into the search itself
  (`objectives.evaluate_plan`, `PlanningScenario.obstacles`/
  `visibility_limits`) and exercised by the wing scenario below; the
  main scene above still has no obstacles, so every station there sees
  every target, by choice (keeping that scene's own comparison
  unchanged).
- Only one instrument type across the demonstration scene -- a
  `PlanningScenario` field a caller can vary; nothing in `planning`
  assumes it's fixed, not done here. (Tolerance type *is* now varied --
  see "Mixed tolerance types" below, the check this bullet used to flag
  as outstanding.)
- No heuristic baselines (1/Cpk weights, a practitioner's plan) in the
  ablation -- flagged, not filled with an invented placeholder.
- No fast, closed-form replacement for objective C's `scipy.integrate.quad`
  calls, which would remove this document's own runtime constraint on
  candidate resolution and the C-objective iso-risk sweep.
