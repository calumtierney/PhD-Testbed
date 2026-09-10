# Step 6: the headline experiment

Companion to `src/planning/`. This is the experiment the project exists
to run (CLAUDE.md §1): does replacing the objective function -- same
optimiser, same scene, same budget, only the weighting changes -- pull
measurement effort away from high-capability characteristics and towards
marginal ones, the way the risk-based hypothesis predicts?

**Short answer: yes, consistently, but modestly in this scene, and the
size of the effect is itself explained by the scene's geometry, not
mysterious.** Full numbers below.

## The two objectives

- **A (uniform weights, A-optimal).** Minimise the plain sum of
  characteristic uncertainties. CLAUDE.md §1: "minimising the trace of
  the coordinate variance matrix" -- applied here at the characteristic
  level (post step 4 projection), which is what existing planning tools
  do.
- **B (risk-derived weights).** Minimise a weighted sum of the same
  uncertainties, where each characteristic's weight is
  `d(PFA + PFR)/d(uncertainty)` -- the local slope of its own risk curve
  (step 5), evaluated at a shared reference uncertainty and normalised to
  mean 1 (`planning.objectives.risk_derived_weights`; see that function's
  docstring for why a *shared, fixed* reference point rather than each
  candidate's own achieved uncertainty).

Both run through `planning.objectives.evaluate_plan` -- same network
solve, same characteristic projection, same risk model -- and the same
grid search (`planning.search.grid_search_station_placement`, CLAUDE.md
§3: the optimiser is held conventional). The only difference between A
and B is which weight vector is passed in.

## The scene

Two spatially separated groups of three targets each (`planning.
experiment.default_two_cluster_scenario`), all flatness characteristics
(against the same nominal surface normal, for simplicity -- position and
parallelism would work the same way, flatness was chosen because step 5's
risk model is exact for it, not approximated):

- **Cluster 1** (y ~ +2): a high-capability process, reused exactly from
  step 5's validated scenario (mean 0, std 5 um, Cpk ~ 3.3).
- **Cluster 2** (y ~ -2): a marginal process, also reused from step 5
  (mean 40 um, std 6 um, Cpk ~ 0.56), against a shared +-50 um tolerance.

The spatial separation matters physically: it means different station
placements genuinely serve the two clusters differently (mainly through
range -- a station azimuthally closer to one cluster is more precise for
it and less precise for the other), giving the optimiser a real trade-off
to make. Reusing step 5's exact process numbers means the risk weights
aren't a new, untested set of constants -- they're already known to
produce sane risk values.

## The weights, computed

At a reference uncertainty of 8 um (`planning.experiment.
run_headline_experiment`'s default -- squarely inside the 5-9 um range
CLAUDE.md §4b predicts for a single station), the risk-derived weights
come out at:

| Cluster | Raw sensitivity `d(risk)/du` | Normalised weight |
|---|---|---|
| High-capability | ~0.37 per metre | **0.0000358** |
| Marginal | ~20,574 per metre | **2.00** |

A ratio of about **56,000:1**. This is not a tuning choice -- it falls
directly out of step 5's own risk curves (see
`docs/step5_risk_layer_physics.md`): a Cpk ~3.3 process sits so far from
its tolerance limit that its risk curve is, for all practical purposes,
flat across realistic uncertainty; a Cpk ~0.56 process sits on the steep
part of its own curve. The weight ratio is a direct, quantitative
statement of that flat-vs-steep contrast.

## The candidate set and search

Stations stand on a hemisphere of radius 4 m around the target centroid,
boresight aimed at it, at 12 azimuths (30 deg apart) x 3 elevations (20,
40, 60 deg) = 36 candidate poses (`planning.candidates.
hemisphere_candidate_poses` -- see that module's docstring for why
station placement is reduced to this 2-DOF-per-station search space in
the first place). A third station -- the frame-of-reference anchor
CLAUDE.md §5 step 3 requires -- is held fixed at `(0, 0, 1)` m,
equidistant in y from both clusters, for *every* candidate evaluated by
*both* objectives (see "A methodological trap" below for why this has to
be fixed and shared, not one of the searched candidates).

Grid search over 2 additional stations: `C(36, 2) = 630` combinations,
evaluated for each objective (~13 seconds total on this machine). CLAUDE.md
§3's documented fallback, a GA for when grid search is too slow, isn't
needed at this scale.

## The result

| | Plan A (uniform) | Plan B (risk-weighted) | Change |
|---|---|---|---|
| Total inspection effort (sum of uncertainty) | 44.07 µm | 49.06 µm | **+11.3%** |
| Global PFA (sum) | 0.0536 | 0.0533 | -0.5% |
| Global PFR (sum) | 0.3466 | 0.3392 | -2.1% |
| **Global risk (PFA+PFR)** | **0.4002** | **0.3925** | **-1.9%** |
| Cluster 1 (high-capability) uncertainty sum | 22.09 µm | 27.44 µm | **+24.2%** |
| Cluster 2 (marginal) uncertainty sum | 21.99 µm | 21.62 µm | **-1.7%** |

**The two plans differ, and in the direction the hypothesis predicts.**
Plan B measures cluster 1 (the high-capability process) noticeably worse
-- uncertainty up almost a quarter -- in exchange for a small
improvement to cluster 2 (the marginal process). By the traditional
A-optimal metric (total uncertainty), B looks *worse* -- 11.3% higher.
By the metric that actually reflects decision cost (global risk), B is
*better* -- both PFA and PFR are lower, total risk down 1.9%. That's the
whole thesis in one number: a plan an existing A-optimal tool would
reject as inferior is, on the metric an inspection actually exists to
serve, the better plan.

**But the effect size is modest, and that's a real, explainable finding,
not a caveat to bury.** Two follow-up checks establish why:

1. **Cluster 2's headroom is nearly used up already under Plan A.**
   Running the grid search with weight *entirely* on cluster 2 (ignoring
   cluster 1) finds a best-case sum of 21.62 µm -- and Plan B already
   achieves 21.62 µm. Plan B isn't leaving anything on the table for
   cluster 2; it's already at that cluster's best achievable precision
   within this candidate set. Plan A's cluster 2 result (21.99 µm) was
   already within ~1.7% of that ceiling. There simply isn't much more
   precision available to fight over, in this particular geometry.
2. **The scene's own left-right symmetry (cluster 1 at y=+2, cluster 2 at
   y=-2, otherwise identical) means a "balanced" 2-station placement is
   already close to what a placement dedicated to either cluster alone
   would achieve** (cluster-1-only best case: 21.62 µm -- numerically the
   same as cluster-2-only's). A uniform-weight optimiser doesn't need to
   choose between the clusters very hard here, because compromising is
   nearly free. A less symmetric scene -- one cluster genuinely harder to
   reach than the other -- would very likely show a bigger effect.
3. **More stations shrink the effect further, not just don't help it.**
   Re-run with 3 searched stations instead of 2 (a coarser grid, 8
   azimuths x 3 elevations = 24 candidates, `C(24,3) = 2024`
   combinations, ~52 s): effort change drops to **+1.4%**, risk change to
   **-0.9%**. More stations give the network enough redundancy to serve
   both clusters nearly optimally regardless of weighting, so there's
   even less need to trade one off against the other. The 2-station
   result above is the more informative one precisely *because* it's
   resource-constrained -- that's where the choice of objective matters
   most, and real LVM planning is regularly resource-constrained in
   exactly this way (CLAUDE.md §1: "deciding which instruments to use,
   where to put them" is the slow expert task this project exists to
   help with).

**If they had not differed, or differed in the wrong direction, that
would have been reported exactly this plainly** (per the standing
instruction not to be talked out of a null result). They differ, in the
right direction, by an amount this scene's own geometry explains. A
scene with more clusters, more asymmetry, or a tighter station budget is
the natural next experiment to see how large the effect can get -- not
built here, since step 6's own gate is about the *comparison*
methodology working correctly, which it does.

## Design decisions worth flagging

**A methodological trap this experiment caught.** `network.solve.
solve_network`'s frame-of-reference constraint (CLAUDE.md §5 step 3)
holds one station's pose exactly fixed to remove gauge freedom. Direct
check during this step's development: swapping *which* of two stations
is treated as the anchor, keeping the geometry otherwise identical,
changes the resulting target covariance by an amount the *same order of
magnitude* as the covariance itself -- not numerical noise, a real
difference. This makes sense in hindsight (the anchor is modelled as
having exactly zero pose uncertainty, and the *other* station's pose
uncertainty does propagate into target uncertainty), but it means
letting "whichever candidate happens to be evaluated first" silently
become the anchor -- the first version of this experiment did exactly
that -- would have biased the grid search in a way unrelated to genuine
placement quality: some candidate pairs would look artificially better or
worse purely because of which one got the free pass to zero uncertainty.
Fixed by giving every candidate combination, in both objectives, the
*same*, externally-fixed anchor station -- an anchor is now a genuine,
shared part of "the scene", not something that varies per candidate.

**The "exact observations" shortcut, and why it's valid here.**
`evaluate_plan` scores a candidate by solving the network from `network.
solve.exact_observations` (noiseless, at the true geometry) rather than
simulating random noise and fitting it. Verified directly
(`tests/test_planning.py::
test_evaluate_plan_design_covariance_matches_a_noisy_solve`): the
resulting covariance agrees with a real noisy solve's covariance to
within 5% (limited only by how many noisy samples were compared against,
not a systematic difference) -- because for this linear-Gaussian
measurement model, `(J^T J)^-1` at the converged point is the design
covariance regardless of which particular noise realisation was fit. This
makes the grid search deterministic (no RNG state to manage across
hundreds of candidates, no risk of a lucky/unlucky draw distorting one
candidate relative to another) and about 3 orders of magnitude cheaper
per evaluation than a full noisy nonlinear fit would be, since the solver
converges in a single function evaluation (it starts exactly at its own
answer).

**A candidate-generation bug caught by the first real run.** The initial
version of `hemisphere_candidate_poses` had elevation backwards -- a
"90 degree elevation" candidate came out *below* the target looking up,
not above it looking down. Caught immediately because the first
end-to-end run produced station z-coordinates around -1.5 m for a scene
with targets at z ~ 1 m, which is obviously wrong for a tracker mounted
on the shop floor. Fixed in `candidates.py`; `tests/test_planning.py`
pins both the zero-elevation ("level with the centre") and 90-degree
("directly above the centre") cases down directly.

## What step 6 deliberately doesn't do

Per CLAUDE.md §3 ("do not invent an optimiser") and §9 ("if a requested
feature would take the project outside the ladder, say so"):

- No GA. Grid search over the reduced 2-DOF-per-station space was fast
  enough (CLAUDE.md §3's own fallback condition, "if grid is too slow",
  wasn't met).
- No occlusion (step 2) in the search itself -- the scene here has no
  obstacles, so every station sees every target. Composing visibility
  filtering into the objective (only counting observations `geometry.
  visibility` reports VISIBLE) is a natural extension, not built, since
  it isn't needed to answer this step's actual question.
- Only one instrument type, one tolerance type (flatness) across every
  characteristic in the demonstration scene -- both are `PlanningScenario`
  fields a caller can vary; nothing in `planning` assumes either is fixed.
- The exploration above (cluster-only best-case, 3-station re-run) is
  reported as evidence for *why* the effect size is what it is, not
  packaged into a formal multi-scenario sweep. That's a reasonable next
  experiment, not required by this step's own gate.
