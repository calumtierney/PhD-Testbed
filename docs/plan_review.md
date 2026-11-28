# Plan review: comparing an engineer's plan with the algorithms' plans

An input and output route with no new physics, outside the six-step ladder of CLAUDE.md §5 (like the QIF route,
`docs/qif_integration.md`). It takes a measurement plan someone made, scores it on the same chain as the planners
(station pose, visibility, instrument error, datum-referenced uncertainty, conformity decision risk), searches what the
algorithms find for the same job, and shows in decision terms what each plan costs and what risk each carries. Every review
is logged, because that record is what later work, including any learning, is built on. This is also the thesis's
validation against practice (Ch2 §2.10.4, Ch5).

Code: `src/interchange/plan_file.py` (the plan format), `src/review/` (the core, pure Python), `viz/review_figures.py`,
`viz/review_plans.py` (command line), `viz/plan_review_tab.py` and `app.py` (the Streamlit tab).

## Use

```
streamlit run app.py                       # the "Plan review" tab
python -m viz.review_plans factory --demo --tracker-model noise_only --reviewer-role metrologist
python -m viz.review_plans part.qif plan_a.json plan_b.json --tracker-model npl_residual --reviewer-role planner \
    --counterparts --out ~/reviews/part_x
```

Choose a scene (the built-in synthetic factory cell, or a QIF file with an optional STL), one or more plan files (or the
synthetic demonstration plan), the instrument model and your role. Optionally search the algorithm counterparts of one plan
(slow; the tab and the command line say how many searches first). Read the table, the plan view and the per-characteristic risk
(and, for any plan, "Explain the numbers": the chain behind every figure, with independent checks);
record which plan you would sign off, and why. **The instrument model has no default**: the Run button stays disabled and the
command line refuses until one is chosen, because which model is used changes the numbers materially
(`docs/step1b_npl_geometric_model.md`), and it is printed on every table, figure and log record.

## The plan file

`src/interchange/plan_file.py`; schema version 1; metres only.

```json
{"schema_version": 1, "name": "...", "author_role": "engineer", "source": "free text", "created": "2025-11-27T10:00:00Z",
 "frame": "cell", "units": "m",
 "stations": [{"position_m": [x, y, z], "instrument_id": 0, "faces": ["front"],
               "track": {"s_m": 12.0, "height_m": 1.25}, "targets": ["name", "..."]}]}
```

| Field | Rule |
|---|---|
| `author_role` | `engineer`, `algorithm` or `synthetic` |
| `frame` | `cell` (the testbed cell's coordinates) or `file` (the QIF document's; converted with the scene's part-to-cell placement; the built-in scene has no file frame) |
| `units` | `"m"`, anything else refused |
| `position_m` | any point: engineers do not stand on our track |
| `instrument_id` | default 0 (one tracker for every station); stations with the same id share their tracker's parameter errors |
| `faces` | default `["front"]`; both faces double the readings |
| `track` | optional label; when given, the position is what is scored |
| `targets` | optional; **if given, the station measures exactly these (if it can see them)**; a listed target it cannot see is reported with its reason (range, elevation, incidence, occlusion) and not counted; an unknown name makes the plan infeasible with that reason |

Everything else is required; unknown keys, non-finite numbers and empty lists are refused. **The first station is the anchor** the
others are tied to: list it first. Datum-referenced uncertainties do not depend on the choice; whether the others can be tied
to it does.

**Orientation.** The file has none. Each station is a levelled tracker (standing axis vertical) turned to face the centroid of the
scene's targets (`planning.cell.layout_poses`'s convention). With the sensor-noise model the turn about the vertical changes
nothing: the azimuth encoder has the same noise at every angle. With the NPL geometric parameters it does matter, by a few tenths of
a percent on the factory-cell scene, because the encoder harmonics and axis tilts are fixed in the instrument's frame. So the
convention is fixed and printed, and a real tracker's heading, if it matters, would have to come from the plan source.

## What a review reports

`review.core.review_plan` / `review_plans` (one scorer and one risk model for a whole comparison; `compare_plans` refuses reviews
from different contexts):

  * stations, shots and readings; time as two readings of one published timing (Hughes et al. 2011: 5 stations, 126 readings,
    55 minutes), proportional to stations or to readings, neither a measured time for this cell;
  * per characteristic: projected uncertainty (datum-referenced, `planning.cell.PartFrameScorer`), TUR (tolerance half-width / 2u),
    PFA, PFR and their total, and the acceptance interval under the decision rule;
  * characteristics not measured, priced at their process prior as the better of accepting or rejecting them unseen
    (`risk.decision.unmeasured_risk`: equal to `planning.process_evidence`'s price whenever accepting unseen is the better decision);
  * totals: expected wrong decisions, expected false accepts and rejects, and escape probability (1 - prod(1 - PFA), independence
    assumed, as `planning.inspection` defines it);
  * feasibility, with reasons; **an infeasible plan is reported and never repaired** (no station added, no target dropped);
  * provenance: the instrument model and parameter-set hash, the floor, each cluster's prior and whether it is a placeholder, the
    decision rule, whether uncertainties are datum-referenced, the elevation-limit setting, whether the placement was assumed,
    the commit and a scene hash.

**Risk, three ways.** The headline `expected_wrong_decisions` is the tabulated risk curve at each uncertainty, summed: the quantity
the planners minimise (log-log interpolation on a 60-point grid). PFA and PFR are the exact JCGM 106 integrals at the same
uncertainty; their sum differs from the curve by the interpolation, reported as `risk_model_vs_exact_relative_gap` (a test bounds it).

**There is no "risk if nothing is measured" line.** It was in the first design as a reference and was removed: under placeholder
priors it beats every plan (the synthetic cell's marginal-cluster statistics make measurement create false rejections faster than it
prevents false acceptances; `planning/process_evidence.py` explains why), so it misleads more than it informs. Read plans against each
other. A characteristic a plan leaves unmeasured is still priced at its own process prior and reported (`risk_from_unmeasured`).

### Feasibility findings

`review.feasibility`, pinned to `PartFrameScorer` by test (feasible if and only if there is no blocking finding): a station that
measures fewer than 3 targets; a station sharing fewer than 3 targets with the stations tied to station 1; fewer than 3 datum
targets measured, or in a line; a singular network; a characteristic no station measures (a warning, blocking only if the
`require_all_characteristics` setting is on); unknown target names. Stations are numbered from 1 in the order listed.

## Algorithm counterparts

`review.counterparts.algorithm_counterparts`: the existing search (`planning.cell.search_cell_layout`, differential evolution over
arc length and height per station) with the same settings and seed for every objective (CLAUDE.md §3: the optimiser is held
constant, the objective varies), for (a) the plan's station count under A0, A1, TUR and risk, (b) the fewest stations whose expected
wrong decisions are no worse than the plan's (iso-risk), (c) the lowest-risk plan at one station fewer. The given plan is never
re-optimised. Algorithm plans always measure every characteristic and stand on the scene's track; each is then reviewed by the
same context as the given plan. The search finds a good plan, not a provably best one, and a counterpart that does not beat the
plan is shown as such.

## Explaining and checking the numbers

`python -m viz.review_plans factory --demo --tracker-model noise_only --no-log --explain` (or the tab's "Explain the numbers") prints,
for one plan, how every figure is obtained: **inputs** (each with where it comes from and whether it is a placeholder), **the chain**
(visibility, one reading, the network, the part frame, the decision: the numbers at each link, for every characteristic and in full for
one), **outputs**, **what drives the risk** (the same characteristic with one input changed at a time) and **independent checks**
(`review/explain.py`, `review/verify.py`). `--verify-trials N` adds the whole-chain Monte Carlo below.

**What the headline figure is.** `expected wrong decisions` is a *sum over characteristics* of each one's PFA + PFR: the expected
number of wrong accept/reject decisions per part inspected, not the probability that a part is wrongly judged. Each PFA and PFR is a
joint probability over the production population (PFR: the part conforms *and* its measurement falls outside the acceptance limits).
A characteristic's risk is large only when many parts sit where a measurement of uncertainty u can push them across a limit.

**The independent checks** recompute a figure a different way:

| Check | What it shares with the review | Result on the built-in cell, demo plan (stress case unless stated) |
|---|---|---|
| PFA, PFR in closed form (adaptive quadrature of the one-dimensional integral) | the process prior and u | agrees with `risk.decision` to 1e-6 relative, all six characteristics |
| PFA, PFR by simulating parts (draw true deviation from the prior, add measurement error, apply the limits, count) | the process prior and u | agrees within 4 standard errors (stress case, 4 million parts: 0.1168 against 0.1170 for `cluster2_profile_0`) |
| u by simulating the *whole measurement chain*: tracker readings with the sensor noise (and, for the NPL model, one draw of the parameter errors per tracker per trial), nonlinear network solve, an explicit best-fit rigid datum frame (SVD), deviation along the tolerance direction | the visibility matrix, the sensor and parameter definitions and the nonlinear solver; **not** the design covariance, the S-transformation or the scorer | sensor noise, 3000 trials: u within 0.98 to 1.01 of the review's (standard error 1.3%), no bias; NPL model, 4500 trials: 0.99 to 1.01 (1.1%) |

So the arithmetic from uncertainty to risk, and the uncertainties themselves under both instrument models, hold up. They are checked
against the *model's own physics* (the sensor noise of Hughes et al. 2011 and the NPL parameter uncertainties), not against a real
tracker: nothing here validates the model against hardware.

### The built-in cell's placeholder process, and why the demo plan once read 0.4

The built-in cell's three "marginal" characteristics come from a placeholder process, and which one matters enormously. Both choices are
Cpk 1.33 (the standard-process floor quoted from GKN) and both are placeholders (`ReviewScene.from_factory_cell(marginal_process=...)`,
`--marginal-process`, a switch in the tab):

| | process mean | std | demo plan, expected wrong decisions (sensor noise / NPL) |
|---|---|---|---|
| **typical** (the default) | +10 um | 10 um | **0.0094** / 0.0126 |
| stress case | +38 um | 3 um | 0.40 / 0.44 |

The stress case is `planning.experiment.MARGINAL_CLUSTER`, which the headline, track and cell experiments and the paper's saved results
use deliberately (to show that a process near its limit makes measurement uncertainty matter), and **it is untouched**: only the review's
built-in scene has the choice. In a review the stress case makes every plan look alarming, and it was the demo plan's first reading
of about 0.4 that prompted the explanation and checks above. Why it is large:

  * the stress process is centred **38 um off nominal** with a 3 um spread, against a tolerance of +-50 um: Cpk 1.33, but the margin is lost
    to *offset*, so it runs 12 um inside the limit;
  * the demo plan measures each of those characteristics from one station (the other stations meet its surface edge-on or from behind),
    at 4.6 to 5.3 m, with a projected uncertainty u of about 10 um (a single reading's lateral error is about 2.35 to 3.4 um per metre of
    range, and referring the point to the datum frame adds a little);
  * measured values therefore spread by sqrt(3^2 + 9.6^2) = 10.1 um around a mean that sits 12 um from the limit, and about 12% are
    measured outside it: PFR = 1 - Phi(12 / 10.1) = 0.116 against the tool's 0.117.

With the typical process the same measurement leaves the mean 40 um from the limit and the spread is 10 um: measured values spread by
sqrt(10^2 + 9.6^2) = 13.9 um, and PFR falls to 0.002 per characteristic. The demo plan's total is then 0.009, almost all of it still
false rejection of the marginal characteristics (the escape probability, about 4e-5, is the process's own nonconformance rate, which no
plan changes). The independent checks pass on both priors (typical: PFR 0.001966 closed form against 0.001978 from 4 million simulated
parts; the uncertainties u do not depend on the process at all).

The risk is extraordinarily sensitive to *where* the process sits. One characteristic, +-50 um, process std 3 um (the stress case's), by
process mean and by measurement uncertainty (total risk, PFA + PFR):

| process mean | Cpk | u = 3 um | 5 um | 10 um | 15 um |
|---|---|---|---|---|---|
| 0 um | 5.56 | 2.6e-32 | 9.6e-18 | 1.7e-06 | 1.1e-03 |
| 20 um | 3.33 | 7.7e-13 | 1.3e-07 | 2.0e-03 | 2.5e-02 |
| 30 um | 2.22 | 1.2e-06 | 3.0e-04 | 2.8e-02 | 9.6e-02 |
| **38 um** | **1.33** | 2.3e-03 | 2.0e-02 | **1.25e-01** | 2.2e-01 |

and a capability index does not say where the margin went: at the same Cpk of 1.33, a process *centred* with a 12.5 um spread
(`planning.experiment.CENTRED_MARGINAL_CLUSTER`) has a risk of 1.8e-03 at u = 10 um, **about 70 times lower** than the offset one. A
real aerospace process with its mean 76% of the way to a limit would normally be re-centred, which is the "calibrated machines" point:
the stress figure is large because the placeholder asks "what if a process sits near its limit", not because the measurement is poor.
With real process data for the cluster (the QIF route takes it from earlier inspections in the file) these numbers would change by
orders of magnitude in either direction, so they should be read as statements about the placeholder. `docs/cell_experiment.md` and the
`centred_cpk1` and `residual_offset_10um` sensitivity cases of `viz/run_cell_sensitivity.py` study exactly this dependence.

Two smaller points the explanation makes visible: PFA and PFR are weighted equally, though a false rejection (a re-measurement) and a
false acceptance (an escape) rarely cost the same; and the total is the tabulated risk curve the planners minimise, with the exact
integrals 0.3 to 0.4% apart.

## Settings

`review.settings.ReviewSettings`: `tracker_model` (required: `noise_only` or `npl_residual`), `systematic_floor_um`,
`decision_rule` (`simple`, or `best`, whose risk curves take about 70 s to tabulate on the built-in cell and about four minutes
for the sheet-metal sample), `n_builds` (built-in cell only), `require_all_characteristics`, `elevation_limits` (`none`, which is what
`plan_qif_cell` assumes, or `api_t3`: -60 to +77 degrees, TODO(source) for GKN's tracker), and the search settings. A QIF scene's
placement is assumed unless supplied, and its station track and reference network are laid out by rule: **a plan is then scored
against an assumed reference network, not the real cell's**. The output says so.

## The log

`review.log`: append-only JSON Lines in `LVM_TESTBED_LOG_DIR`, else the user's data directory; **a directory inside the repository's
working tree is refused** (a log may hold a customer's plans). Every record has `schema_version`, `event`, `timestamp_utc`,
`record_id`, `reviewer_role` (and `reviewer_name` only if given), `scene_hash`, `plan_hashes`, `settings`, `provenance`, `payload`.
Events: `scene_loaded`, `plan_reviewed` (the full `PlanReview`), `comparison`, `sign_off` (the chosen plan hash, or none, and the
reason) and `execution_result` (the Loop 2 input: network residuals, measured deviation, decision and outcome per characteristic by
index; defined and readable now, with no user interface). `load_log` refuses an unknown schema version, event or key, naming the
line. Scene names, the file name and free text are left out by default (characteristics are `char_<k>`); `log_names` keeps them.
The `review.log` docstring says what `planning.surrogate` would need from these records and what is missing.

## Confidentiality and IP

  * The tool runs fully offline: no network calls, no telemetry. Uploads are read in memory (the STL included) and never copied
    into the repository. Exports are written only where the reviewer says, never inside the repository.
  * **No customer data** is in the repository: tests, fixtures, `results/`, `docs/` and examples use only synthetic scenes, the
    synthetic plan and the public QIF sample in `scenes/qif/`.
  * **The repository has no licence file, and none was added or changed.** Licensing is being decided under the IP and publication
    protocol with GKN.
  * No new dependency: the app already depended on Streamlit (optional, `pyproject.toml`); nothing else was added.

## What is still needed before a real GKN plan can be loaded

An adapter (`interchange.plan_file.read_plan_from_spatialanalyzer`, a stub with `TODO(source)`) cannot be written until these are
known, from a real export of a real plan:

  * **the export format** and version, and where it records instrument stations and which are planned and which only history;
  * **the coordinate frame** the positions are in (job, part, aircraft), the length unit, and the transform to the CAD / QIF frame the
    scene was built from;
  * **target naming**: what a station's measured targets are called and how those names map to QIF features and characteristics (the
    review recognises a characteristic's name, its feature's name, `nest_<k>` and `part_target_<i>`);
  * the instrument's identity per station (one tracker or several), the face of each reading, and the measurement order;
  * from the cell: the real reference-network nest positions, the part's placement, and the process data for the clusters
    (without it the statistics are placeholders and the risk figures illustrative).

## Porting to Qt

Not built, by decision. The core is written so a Qt front end calls it unchanged.

**What a front end needs from the core.**

| Need | Call |
|---|---|
| Build a scene | `ReviewScene.from_factory_cell()`, `ReviewScene.from_qif_bytes(qif, label, stl, ...)`, `qif_scopes(qif)` |
| Read plans | `read_plan_file(bytes or path)` -> `PlanFile` (or `PlanFileError`); `demo_plan(scene)` |
| Settings | `ReviewSettings(tracker_model=..., ...)` (JSON via `to_dict`/`from_dict`) |
| Review | `build_context(scene, settings)` once (slow for the `best` rule), then `review_with_context(context, plan)`, or `review_plans` |
| Counterparts | `algorithm_counterparts(scene, plan, settings, context, progress)` with `progress(done, total, label)` |
| Compare | `compare_plans(reviews)` -> dict; `format_comparison` for text |
| Export | `comparison_bundle`, `write_bundle` |
| Log | `ReviewLog(...).log_session(...)`, `.log_sign_off(...)`, `load_log` |

Everything returned is plain data (dicts, lists, numbers, text), JSON-serialisable; no Streamlit, Qt or matplotlib import in
`src/`. Figures are drawn by `viz/review_figures.py` from that data.

**What would change.** Only `viz/plan_review_tab.py` is replaced: widgets and layout; the required-choice rules (no run before the
instrument model and role are chosen); the slow-case warnings; the sign-off form. Specific to Qt: the searches and the `best`-rule
tabulation block, so they run on a worker thread or process, reporting through the `progress` callback (differential evolution
here cannot be cancelled mid-search: a cancel would take effect between searches); the figures embed through a matplotlib Qt
canvas or are redrawn natively from the same data; Streamlit's `AppTest` gate becomes a pytest-qt test; scene and context caching
(`st.cache_resource` here) becomes an object the window owns.

**Licensing.** PySide6 (Qt for Python) is LGPL (also GPL and commercial); PyQt is GPL or a commercial licence. That matters for
installing the tool at GKN, since a GPL dependency changes what may be distributed with it. **It is decided under the IP and
publication protocol, not here**, and the current terms should be checked at the time.

## Gates (each has a test)

1 regression: the suite passes and `python -m viz.make_figures` leaves `git diff results/` empty. 2 round trip: a `plan_qif_cell` plan
reviewed reproduces the planner's numbers (`tests/test_review_core.py`). 3 frame equivalence. 4 same scorer and risk model.
5 infeasibility reported, not repaired (`tests/test_review_core.py`, `tests/test_review_feasibility.py`). 6 unmeasured
characteristics priced as `process_evidence` and `unmeasured_risk` price them. 7 logging (`tests/test_review_log.py`). 8 the Streamlit
tab under `AppTest` (`tests/test_app_plan_review.py`). Also `tests/test_plan_file.py` and `tests/test_review_cli.py`.
