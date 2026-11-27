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
(slow; the tab and the command line say how many searches first). Read the table, the plan view and the per-characteristic risk;
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
  * totals: expected wrong decisions, expected false accepts and rejects, escape probability (1 - prod(1 - PFA), independence
    assumed, as `planning.inspection` defines it), and the risk if nothing is measured, as a reference line;
  * feasibility, with reasons; **an infeasible plan is reported and never repaired** (no station added, no target dropped);
  * provenance: the instrument model and parameter-set hash, the floor, each cluster's prior and whether it is a placeholder, the
    decision rule, whether uncertainties are datum-referenced, the elevation-limit setting, whether the placement was assumed,
    the commit and a scene hash.

**Risk, three ways.** The headline `expected_wrong_decisions` is the tabulated risk curve at each uncertainty, summed: the quantity
the planners minimise (log-log interpolation on a 60-point grid). PFA and PFR are the exact JCGM 106 integrals at the same
uncertainty; their sum differs from the curve by the interpolation, reported as `risk_model_vs_exact_relative_gap` (a test bounds it).

**Under placeholder priors, every plan can be worse than measuring nothing.** The synthetic cell's marginal-cluster statistics make
measurement create false rejections faster than it prevents false acceptances (`planning/process_evidence.py` explains why). The
comparison says so in a note rather than leaving it a puzzle; read plans against each other, not against that line.

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
