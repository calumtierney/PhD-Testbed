# The wing-box inspection experiment: uncertainty-driven vs risk-driven station plans

**Status: first complete version. The part is a synthetic stand-in; every
number below is a statement about this testbed on that stand-in, not about the
real wing box. §9 lists what is needed from the author to make it about the
real part.**

This is a separate experiment from step 6's headline comparison
(`docs/step6_headline_experiment.md`). That one asks, on small scenes,
whether four *objective functions* choose different station layouts. This
one asks the question the project exists for, on a part with the structure
of a real inspection: **how many instrument stations does a ~500-point
verification need, if the plan is set by measurement uncertainty, and how
many if it is set by decision risk?** Station count is the cost, because
every station is a move, a set-up and a network tie. It is outside the
six-step ladder of CLAUDE.md §5 (it is an application of steps 1-5 inside a
new step-6-style comparison), which is why it has its own document.

## 1. What was asked, and what was built

The part (Figure 9.13 of the source description): an assembly wing box about
1.5 x 2.0 x 1.5 m on a fixture, ~500 pre-defined inspection points, six tasks

| Task | What | Placeholder points |
|---|---|---|
| T1 | profile of the leading edge, right end | 60 |
| T2 | profile of the leading edge, left end | 60 |
| T3 | one free-form leading-edge surface | 120 |
| T4 | the flat top surface | 150 |
| T5 | profile of the flap, right end | 55 |
| T6 | profile of the flap, left end | 55 |

Each task acquires 3D coordinates and compares them with CAD: a *profile*
(distance along the surface normal), no GD&T. The framework therefore starts
at instrument selection -- a single laser tracker (the step-1 model, inputs
fixed by CLAUDE.md §4a), whose stations are what the plan chooses.

Two plans, searched identically (§5):

* **Plan U -- uncertainty-driven.** Reduce uncertainty while saving
  stations. The conventional planner's view: every point is measured and the
  layout minimises the mean variance of the points' uncertainty (the
  A-optimal, trace-style criterion of CLAUDE.md §1). A trace never says when
  to stop, so the station count is taken from the **knee** of its
  stations-versus-uncertainty curve: the smallest K where one more station
  lowers mean variance by less than 10% (reported for 20%, 10%, 5%, 2%).
  It is blind to the tolerance and to which features are hard to hold.
* **Plan R -- risk-driven.** Minimise station count subject to decision
  *risk* staying within a limit (§4). Points whose process is highly capable
  need not be measured tightly, or at all; points in a marginal cluster must.

The chain is the project's own, unchanged: station pose -> visibility (range,
incidence, ray-cast occlusion) -> instrument error (step 1) -> coordinate
covariance (`network.design`, the block-diagonal fast form of step 3) ->
uncertainty along the surface normal (step 4's projection, one direction per
point) -> JCGM 106 decision risk (step 5) -> plan objective.

## 2. The part is a stand-in

`geometry.wing_box.build_wing_box_part()` builds a parametric wing section --
an elliptical nose (with a 6 mm free-form bump for T3), a flat top skin, a
tapered flap, three fixture plates -- with the task point counts above. **No
CAD or inspection-point list was available**, so the outline, the counts and
the fixture are representative inventions, chosen only to give the right
*kind* of geometry (a rounded edge, a thin trailing edge, a flat skin, a
free-form patch, a fixture that blocks low sightlines). To use the real
part, pass its points, unit normals and an STL (`geometry.mesh.load_stl`) in
`WingBoxPart`'s place; nothing else changes.

One consequence matters for reading the results. The synthetic part has a
region -- the **lower flap skins** (T5, T6, 54 points) and the underside of
the nose -- that a tracker can see only from a low, close station standing
off the end of the part, at a grazing angle, tied to the rest of the network
by few shared points. That is geometry the real part may or may not share;
it is what makes some points expensive, and the experiment's finding is
partly about *whether the expensive points are the ones that matter*. §7
tests exactly that.

## 3. The risk model, and the one structural fact it rests on

Every point carries a JCGM 106 decision risk (`risk.jcgm106`) from its
uncertainty *along its own normal* and its process prior. Two process
clusters, assigned by task (**a placeholder, TODO(source)**): T1-T3, the
compound-curvature leading-edge work, are the *marginal* cluster (Cpk 4/3,
the AS13006 floor / GKN standard tier already used in step 6), and T4-T6,
the flat skin and flap, are *high-capability* (Cpk 2, zero-defect tier). The
repo's established marginal prior is a process biased towards one limit
(mean 38 µm, std 3 µm against ±50 µm); priors scale with the tolerance so
Cpk means the same at every tolerance. The tolerance itself is the repo's
±50 µm reference scale (CLAUDE.md §4d) and is **swept** (§7), because no
wing-box tolerance was available.

**Cluster assignment is itself uncertain** (CLAUDE.md §3), so a point's
process is its labelled cluster with probability *c* (default 0.9, the
repo's own value) and the other cluster otherwise. This is not a footnote:
it decides whether skipping a high-capability point is cheap.

**An unmeasured point is priced, not forbidden.** It is never falsely
rejected, and it escapes with its process's own nonconformance probability:
3.17×10⁻⁵ for the marginal prior, 1.97×10⁻⁹ for the high-capability one,
mixed by *c* (so an unmeasured "high-capability" point carries
0.9 × 2×10⁻⁹ + 0.1 × 3.2×10⁻⁵ ≈ 3.2×10⁻⁶, not 2×10⁻⁹).

**The structural fact.** A point's false-accept probability can never exceed
its process's own nonconformance rate, and for any tracker-scale uncertainty
(≳ 2 µm) it is already within a factor of ~3 of that ceiling. So the
consumer's risk is set mostly by the *process*, not by the plan: for the
whole part, the probability that at least one point is falsely accepted
(`P_escape`) is 7.6×10⁻³ with nothing measured, 3.5×10⁻³ with every point
measured at the TUR 4:1 limit (u = 6.25 µm), and 2.2×10⁻³ with every point at
1 µm -- a factor of ~3.5 between measuring nothing and measuring perfectly. It also sits
~600× below Z540.3's 2% per-result bound for every plan, so that bound never
binds. What a plan *can* change is (a) which points are measured at all and
(b) the **false-rejection** risk of the points that are, which grows steeply
with uncertainty for a marginal process (expected falsely rejected points per
part: 0.56 at u = 3 µm, 10 at 6.25 µm, 30 at 10 µm). Plan R is therefore held
to two numbers, both over the whole part:

* `P_escape`, the safety metric (probability at least one point is falsely
  accepted, points taken as independent -- an upper-bound-style summary,
  since errors sharing a station-pose error are correlated); and
* `E_FR`, the expected number of falsely rejected points per part (the
  producer's yield metric).

## 4. What "acceptable risk" is -- and what was not invented

The brief says this is a safety-critical part and to research the limit. The
sourced anchors are:

* **ANSI/NCSL Z540.3-2006 §5.3**: probability of false accept ≤ 2% per
  result; a test uncertainty ratio ≥ 4:1 is its fallback only where PFA
  cannot be estimated.
* **JCGM 106**, ILAC-G8 and NASA-HDBK-8739.19-4 (Annex 4): how to compute
  and control decision risk and set guard bands. None sets a universal
  number.

**No published acceptable false-accept limit for safety-critical *aerospace*
inspection was found.** Z540.3's 2% is not aerospace-specific and, by §3, is
~600× looser than anything these plans produce, so it constrains nothing.
Rather than invent a limit, plan R is held to two anchored ones, and the
absolute limit is swept:

* **R-match** -- *no riskier than the conventional plan U actually is*:
  `P_escape` and `E_FR` no worse than plan U's chosen layout realises. The
  strictest fair comparison. If plan R meets it with fewer stations, no one
  can say it is the less safe plan.
* **R** -- *no riskier than the TUR rule guarantees*: the risk if every
  point sat exactly at the TUR 4:1 limit. An externally anchored ceiling
  that does not depend on plan U. It is looser than R-match here, since
  plan U's layouts beat the ceiling.
* a sweep of the risk limit (×0.25 to ×2) and a **sampling-policy floor**
  (every task at least 25-100% measured). The floor is a *policy* knob, not
  a risk quantity: a risk calculation resting on placeholder process priors
  should not be able to switch a whole inspection task off, and no source
  gives the right floor, so it is swept and never defaulted.

A programme or customer limit, if one exists, drops into `PlanSpec`
unchanged (`TODO(source)`).

## 5. How the plans are searched -- and held constant

The experiment varies the *objective*; the optimiser is the project's
standard GA (`scipy.optimize.differential_evolution` through
`planning.search.run_station_search`, the one driver step 6 also uses), with
identical settings for every plan: population 14 x (3K), 80 generations,
seed 0, stations free in a box around the part (x -3.5..5 m, y -3.5..5.5 m,
z 0.8..3 m, outside the part and fixture), every station boresighted at the
part's centre. What a plan adds is only its cost function (a large weight on
the constraint violation plus a small secondary term, so among layouts that
meet the constraint the better one is kept).

Getting a search that could find these layouts at all took four changes,
each reported because each could have flattered one plan:

1. **A fast evaluator.** `network.design` computes the same covariance as the
   step-3 network solve (pinned to it to <1e-4 by
   `tests/test_design_covariance.py`) from the block-arrow structure of the
   normal equations, and `geometry.visibility.visibility_matrix` ray-casts all
   500 points per station in one batch. One layout (visibility + covariance +
   risk for 500 points) costs ~5 ms (CLAUDE.md §3: exploit the block
   structure before any surrogate), so the baseline any surrogate speed-up is
   measured against is fast.
2. **Rigid networks only.** A layout is feasible only if every station is
   tied to the anchor by shared points *and* the station-pose information
   matrix has full rank. (This is the rule whose absence invalidated the
   earlier wing-scenario numbers -- see `docs/step6_headline_experiment.md`.)
3. **Seeds, applied to every plan alike.** A random start almost never lands
   in the narrow regions that see the underside of the nose and the lower
   flap skin, so every plan's population is seeded with coverage-greedy
   covers (new stations must share ≥6 or ≥3 points with the cover so far)
   and with a forward-greedy layout built with *that plan's own cost*; each
   K+1 search is also seeded with the K layout plus the best-tied extra
   station. Candidate positions include 4 per point placed along its own
   surface normal, because uniform random candidates missed two nose-underside
   points entirely (`tests/test_inspection.py` pins this).
4. **A backward-elimination pass** over each finished sweep: every K-station
   layout is also seeded with the best layout got by dropping one station from
   the (K+1)-station answer. It removed steps in plan U's curve that were
   search luck, not physics (its knee fell from 12 to 10 stations).

A GA finds *a* layout, not a proof that none with fewer stations exists:
"no layout found" and every station count below are what this search found.

## 6. Result, base case

Synthetic wing box, tolerance ±50 µm, offset priors (marginal T1-T3, high-
capability T4-T6), assignment confidence 0.9, one tracker, up to 14 stations
for plan U. Figures: `results/panel10a_wingbox_layouts.png` (layouts),
`panel10b_wingbox_uncertainty.png` (per-point u), `panel10c_wingbox_kcurves.png`
(stations versus criterion and risk). Raw record: `results/wingbox_results.json`.

| | Stations | Points measured | Shots | P(escape) | E[false rejects] | Mean / max u (µm) |
|---|---|---|---|---|---|---|
| **U** uncertainty-driven (A-optimal, knee 10%) | **9** | 500 | 1151 | 3.03×10⁻³ | 8.70 | 4.8 / 38.8 |
| **R-match** risk-driven, no riskier than U | **3** | 444 | 679 | 2.89×10⁻³ | 0.30 | 2.2 / 4.5 |
| **R** risk-driven, TUR 4:1 risk ceiling | **2** | 380 | 444 | 3.17×10⁻³ | 0.60 | 2.9 / 5.1 |
| R-all (ceiling, every point measured) | 8 | 500 | 1214 | 2.75×10⁻³ | 8.13 | 5.1 / 46.8 |
| T (hard TUR 4:1 on every point) | none found ≤ 14 | | | | | |

Read across the rows:

* **Plan R-match meets or beats plan U's own risk on both metrics with a
  third of the stations** (3 against 9) and 59% of the shots (679 against
  1151): `P_escape` is 5% lower, and expected false rejections are 29×
  lower (0.30 against 8.70). Stations and shots both fall, so R-match is
  cheaper for *any* positive set-up time per station and measurement time
  per point -- no cost model had to be assumed.
* **Where the stations go.** U must measure every point, and its mean
  variance is dominated by the few worst: the lower flap skins, seen only by
  one weakly tied low station each (T6's mean u is 19.6 µm, with 5.3 of U's
  8.7 expected false rejections). Much of U's station budget goes to giving
  those points a second, better view (its worst point falls from 60 µm at 5
  stations to 39 µm at 9). R-match measures the marginal leading-edge tasks
  (T1-T3) at 1.4-2.5 µm, measures the flat skin and the *upper* flap skins,
  and leaves only the 56 lower-flap points unmeasured, each carrying
  ~3×10⁻⁶ of escape risk (1.8×10⁻⁴ expected escapes over all 56) --
  already counted in the 2.89×10⁻³ above.
* **R-all** (every point measured, risk limit at the TUR ceiling) needs 8:
  almost all of plan U's cost is the obligation to measure everything, not
  the uncertainty it delivers. **R** (the looser ceiling) needs 2.
* **The conventional TUR rule cannot be met at all** (plan T): the best
  layout found at 14 stations still has points at 33 µm against a 6.25 µm
  limit. A handful of grazing, single-station points defeat it. This is why
  plan U is the A-optimal planner with a knee rule and not a hard TUR rule.
* **Plan U's count is a judgement.** The knee is K = 9 for "stops paying" at
  20% or 10% per station, 11 at 5%, 12 at 2%; its curve
  (`panel10c`, left) falls in steps. Repeat searches differ by about one
  station (§7), so "9" is "9 ± 1", and R-match's "3" is "3 ± 1".

What the result is *not*: it is not that risk-driven planning measures less
because it is careless. It measures the points whose process can fail
tightly, and skips points whose process is very unlikely to have failed --
and that last clause rests entirely on the placeholder priors and on the
0.9 confidence in the cluster assignment, which is why §7 exists.
