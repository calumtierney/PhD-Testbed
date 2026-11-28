"""The Plan review tab of the Streamlit explorer (`app.py`): a thin layer over `src/review`.

Choose a scene (the built-in synthetic cell, or a QIF file with an optional STL), upload one or more plan files (or add the synthetic
demonstration plan), optionally search the algorithm counterparts of one plan, and read the comparison: a side-by-side table, a true-scale plan
view of every plan's stations, the risk of each characteristic under each plan, and the reasons any plan is infeasible. A sign-off panel records
which plan the reviewer would accept and why. **Every review is logged** (`review.log`) to a directory outside the repository, and the
comparison can be downloaded as JSON and as PNG and SVG figures.

Nothing is computed here. This file chooses widgets, calls `review.core`, `review.counterparts`, `review.log`, `review.export` and
`viz.review_figures`, and shows what they return; a front end other than Streamlit would call the same functions (docs/plan_review.md,
"Porting to Qt"). Runs fully offline; uploaded files are read in memory and never written to disk.

The instrument model has no default: the Run button stays disabled until one is chosen, and the choice is printed above every result.
"""
import json
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import streamlit as st

from interchange.plan_file import PlanFile, PlanFileError, read_plan_file
from interchange.qif import QifError
from review.core import build_context, compare_plans, format_comparison, review_with_context
from review.counterparts import OBJECTIVES, algorithm_counterparts
from review.demo import demo_plan
from review.explain import explain_plan
from review.export import bundle_json, comparison_bundle
from review.log import LOG_FILE_NAME, LogError, ReviewLog, resolve_log_dir
from review.scene import ReviewScene, qif_scopes
from review.settings import DECISION_RULES, ELEVATION_LIMITS, TRACKER_MODELS, ReviewSettings
from viz.review_figures import figure_bytes, plan_view_figure, risk_by_characteristic_figure

FACTORY, UPLOAD = "Built-in factory cell (synthetic)", "Upload a QIF file"
TRACKER_LABELS = {"noise_only": "Sensor noise only (the step-1 instrument)",
                  "npl_residual": "Sensor noise + NPL geometric parameters (Hughes 2011 residual), consider mode"}
ROLES = ("engineer", "metrologist", "planner", "researcher", "other")
NONE_ACCEPTABLE = "__none__"


# -- cached, because they are slow and the same for every rerun ----------------------------------
@st.cache_resource(show_spinner="Building the built-in cell ...")
def _factory_scene() -> ReviewScene:
    return ReviewScene.from_factory_cell()


@st.cache_resource(show_spinner="Reading the QIF file and building the cell ...")
def _qif_scene(qif: bytes, stl: Optional[bytes], stl_scale: Optional[float], scope: Optional[str], up_axis: str, alignment: str,
               label: str) -> ReviewScene:
    return ReviewScene.from_qif_bytes(qif, label, stl, stl_scale, scope, up_axis, alignment)


@st.cache_resource(show_spinner="Building the scorer and the risk model (the 'best' rule takes minutes) ...")
def _context(_scene: ReviewScene, scene_hash: str, settings_json: str):
    return build_context(_scene, ReviewSettings.from_dict(json.loads(settings_json)))


# -- sections ------------------------------------------------------------------------------------------
def _scene_section() -> Optional[ReviewScene]:
    st.subheader("1. Scene")
    source = st.radio("Scene", [FACTORY, UPLOAD], horizontal=True, key="pr_scene_source", label_visibility="collapsed")
    if source == FACTORY:
        scene = _factory_scene()
        st.caption("A synthetic long part on a station track with a reference network. Every dimension, tolerance and process statistic is a "
                   "placeholder (docs/cell_experiment.md); use it to see how the tool works, not to judge a real plan.")
    else:
        qif = st.file_uploader("QIF file (read in memory, never saved)", type=["qif", "xml"], key="pr_qif")
        stl = st.file_uploader("Part shape, STL (optional; in the QIF file's coordinates; blocks sight lines)", type=["stl"], key="pr_stl")
        columns = st.columns(4)
        up_axis = columns[0].selectbox("Which file axis is vertical", ["z", "y", "x"], key="pr_up_axis")
        alignment = columns[1].selectbox("Part frame", ["auto", "datums", "best_fit"], key="pr_alignment")
        scale = columns[2].number_input("STL metres per unit (0 = the QIF file's unit)", min_value=0.0, value=0.0, format="%g", key="pr_stl_scale")
        if qif is None:
            st.info("Upload a QIF file to review plans against its part, or switch to the built-in cell.")
            return None
        qif_bytes = qif.getvalue()
        try:
            scopes = qif_scopes(qif_bytes)
        except (QifError, ValueError) as error:
            st.error(f"Cannot read {qif.name}: {error}")
            return None
        scope = None
        if len(scopes) > 1:
            scope = columns[3].selectbox("Part or assembly", list(scopes), format_func=lambda k: scopes[k], key="pr_scope")
        try:
            scene = _qif_scene(qif_bytes, stl.getvalue() if stl else None, scale or None, scope, up_axis, alignment, qif.name)
        except (QifError, ValueError) as error:
            st.error(f"Cannot build a scene from {qif.name}: {error}")
            return None
    summary = scene.summary()
    st.write(f"**{scene.label}**: {summary['n_characteristics']} characteristics, {summary['n_targets']} targets "
             f"({summary['n_datum_targets']} datum targets), scene `{summary['scene_hash'][:10]}`")
    if scene.placement_assumed:
        st.warning("The part's placement in the cell was not supplied and is ASSUMED (the file's up axis vertical, the longer side along the cell). "
                   "The station track and the reference network are laid out by rule round it. Plans given in `file` coordinates are placed with "
                   "that assumption.")
    if scene.kind == "qif":
        st.warning("The reference network is assumed (laid out by rule round the part), not the real cell's: a plan's ties to it are judged against that.")
    with st.expander("Scene notes and what was left out"):
        for note in scene.notes:
            st.write("- " + note)
    return scene


def _settings_section(scene: ReviewScene):
    st.subheader("2. Settings")
    columns = st.columns(2)
    tracker_model = columns[0].selectbox(
        "Instrument model (required: there is no default)", list(TRACKER_MODELS), index=None, placeholder="Choose the instrument model",
        format_func=lambda k: TRACKER_LABELS[k], key="pr_tracker_model",
        help="Which model the risk is computed with changes the numbers materially (docs/step1b_npl_geometric_model.md). It is printed on every output.")
    rule = columns[1].radio("Decision rule", list(DECISION_RULES), horizontal=True, key="pr_rule",
                            format_func=lambda r: {"simple": "simple acceptance", "best": "conformance probability (slow)"}[r])
    if rule == "best":
        st.warning("The conformance-probability rule tabulates its risk curves first: about 70 s for the built-in cell, about four minutes for the "
                   "sheet-metal sample (docs/qif_integration.md). Simple acceptance takes about a second.")
    columns = st.columns(4)
    floor = columns[0].number_input("Systematic floor [µm]", min_value=0.0, max_value=50.0, value=0.0, step=0.5, key="pr_floor_um",
                                    help="An isotropic placement-irreducible term added once per characteristic.")
    n_builds = None
    if scene.kind == "factory_cell":
        if columns[1].checkbox("Read the cell's process statistics as from n builds", key="pr_use_builds"):
            n_builds = int(columns[1].number_input("n builds", min_value=2, value=5, step=1, key="pr_n_builds"))
    require_all = columns[2].checkbox("A characteristic no station measures makes the plan infeasible", value=False, key="pr_require_all",
                                      help="Off: it is priced at its process prior and reported.")
    elevation = columns[3].selectbox("Tracker elevation limits", list(ELEVATION_LIMITS), key="pr_elevation",
                                     format_func=lambda e: {"none": "none", "api_t3": "API T3: -60 to +77 deg"}[e])
    st.markdown("**Who is reviewing** (logged; a role, not a name)")
    columns = st.columns(3)
    role = columns[0].selectbox("Reviewer role (required)", list(ROLES), index=None, placeholder="Choose your role", key="pr_role")
    name = columns[1].text_input("Name (optional; recorded only if you give one)", key="pr_name")
    log_names = columns[2].checkbox("Keep the scene's names in the log", value=False, key="pr_log_names",
                                    help="Off: characteristic and target names, the file name and free text are left out of the log.")
    settings = None
    if tracker_model is not None:
        try:
            settings = ReviewSettings(tracker_model, floor, rule, n_builds, require_all, elevation)
        except ValueError as error:
            st.error(str(error))
    return settings, role, (name.strip() or None), log_names


def _plans_section(scene: ReviewScene) -> List[PlanFile]:
    st.subheader("3. Plans")
    uploaded = st.file_uploader("Plan files (JSON; see docs/plan_review.md for the format)", type=["json"], accept_multiple_files=True, key="pr_plan_files")
    plans: List[PlanFile] = []
    for upload in uploaded or []:
        try:
            plans.append(read_plan_file(upload.getvalue()))
        except PlanFileError as error:
            st.error(f"{upload.name}: {error}")
    if st.checkbox("Add the synthetic demonstration plan (4 stations evenly spaced round the track; NOT an engineer's plan)", key="pr_add_demo"):
        plans.append(demo_plan(scene))
    if plans:
        st.dataframe([{"#": i + 1, "plan": p.name, "role": p.author_role, "stations": p.n_stations, "frame": p.frame,
                       "source": p.source[:90]} for i, p in enumerate(plans)], hide_index=True, width="stretch")
    else:
        st.info("Upload one or more plan files, or add the demonstration plan.")
    return plans


def _counterparts_section(plans: List[PlanFile], settings: Optional[ReviewSettings]):
    st.subheader("4. Algorithm counterparts (optional, slow)")
    if not st.checkbox("Search algorithm plans for one of the plans above", key="pr_counterparts",
                       help="Same station count under A0, A1, TUR and risk; the fewest stations with no more risk; the lowest-risk plan at one "
                            "station fewer. The one optimiser, the same settings and seed for every objective."):
        return None
    if not plans:
        st.info("Add a plan first.")
        return None
    columns = st.columns(2)
    target = columns[0].selectbox("Counterparts of", list(range(len(plans))), format_func=lambda i: f"#{i + 1} {plans[i].name}", key="pr_counterpart_target")
    objectives = columns[1].multiselect("Objectives", list(OBJECTIVES), default=list(OBJECTIVES), key="pr_objectives")
    with st.expander("Search settings (the one optimiser: differential evolution over arc length and height per station)"):
        cols = st.columns(3)
        maxiter = int(cols[0].number_input("Generations", min_value=1, value=60, step=1, key="pr_maxiter"))
        popsize = int(cols[1].number_input("Population", min_value=2, value=15, step=1, key="pr_popsize"))
        seed = int(cols[2].number_input("Seed", min_value=0, value=0, step=1, key="pr_seed"))
    k = plans[target].n_stations
    st.warning(f"This runs up to {len(objectives) + k + 1} differential-evolution searches ({len(objectives)} for the same station count, up to {k} for the "
               f"iso-risk search, one for one station fewer), each {maxiter} generations x population {popsize}. A search on the built-in cell took "
               "tens of seconds at 20 x 8 and about 150 s for the whole set at 60 x 15; a larger scene takes longer. The progress bar shows each search finishing.")
    return {"target": target, "objectives": objectives, "maxiter": maxiter, "popsize": popsize, "seed": seed}


# -- running ---------------------------------------------------------------------------------------------
def _figures(scene: ReviewScene, reviews) -> Dict[str, bytes]:
    out = {}
    for stem, fig in (("plan_view", plan_view_figure(scene, reviews)), ("risk_by_characteristic", risk_by_characteristic_figure(reviews))):
        for fmt in ("png", "svg"):
            out[f"{stem}.{fmt}"] = figure_bytes(fig, fmt)
        plt.close(fig)
    return out


def _run(scene, settings, plans, role, name, log_names, counterpart_cfg) -> Dict:
    bar = st.progress(0.0, text="Building the scorer and the risk model ...")
    if counterpart_cfg is not None:
        settings = ReviewSettings.from_dict({**settings.to_dict(), "de_maxiter": counterpart_cfg["maxiter"], "de_popsize": counterpart_cfg["popsize"],
                                             "seed": counterpart_cfg["seed"]})
    log = ReviewLog(None, role, name, log_names)  # fails first if the log directory is not allowed
    context = _context(scene, scene.scene_hash(), json.dumps(settings.to_dict(), sort_keys=True))
    bar.progress(0.05, text="Reviewing the plans ...")
    reviews = [review_with_context(context, p) for p in plans]
    counterparts, not_found, search = (), (), {}
    if counterpart_cfg is not None:
        def progress(done, total, label):
            bar.progress(0.1 + 0.85 * min(done / total, 1.0), text=f"Searching algorithm plans: {label} ({done}/{total})")
        result = algorithm_counterparts(scene, plans[counterpart_cfg["target"]], settings, context=context, objectives=counterpart_cfg["objectives"],
                                        progress=progress)
        counterparts, not_found, search = result.counterparts, result.not_found, result.search
        reviews += [c.review for c in counterparts]
    bar.progress(0.96, text="Drawing the figures and writing the log ...")
    comparison = compare_plans(reviews)
    figures = _figures(scene, reviews)
    records = log.log_session(scene, settings, reviews, comparison, counterparts, search)
    bar.progress(1.0, text="Done")
    bundle = comparison_bundle(comparison, reviews, counterparts, search, not_found)
    return {"comparison": comparison, "text": format_comparison(comparison), "bundle_json": bundle_json(bundle), "figures": figures,
            "not_found": list(not_found), "log_path": str(log.path), "n_logged": len(records), "role": role, "name": name, "log_names": log_names,
            "labels": {row["plan_hash"]: f"#{row['label']} {row['name']}" for row in comparison["rows"]},
            "plans_all": list(plans) + [c.plan for c in counterparts], "reviews_all": reviews, "settings_json": json.dumps(settings.to_dict(), sort_keys=True)}


# -- results -----------------------------------------------------------------------------------------------
def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def _show_result(result: Dict, scene: ReviewScene) -> None:
    comparison = result["comparison"]
    prov = comparison["provenance"]
    st.subheader("Result")
    st.markdown(f"**Instrument model:** {comparison['tracker_label']}  \n**Decision rule:** {prov['decision_rule']}  \n"
                f"**Uncertainty:** {prov['uncertainty_reference']['method']}  \n**Elevation limits:** {prov['elevation_limits']}  \n"
                f"**Station 1 is the anchor:** {prov['first_station']}.")
    if prov["placement_assumed"]:
        st.warning("PLACEMENT ASSUMED: the part's position in the cell was not supplied.")
    if prov["process_priors_include_placeholder"]:
        st.warning("PLACEHOLDER process statistics are in use: the risk figures are illustrative only.")
    st.caption(f"Reference network: {prov['reference_network']}. Time: {prov['time_basis']}. Commit {prov['git']['commit'][:10]}"
               f"{' (uncommitted changes)' if prov['git']['dirty'] else ''}; scene {prov['scene']['scene_hash'][:10]}; context {comparison['context_id'][:10]}.")
    rows = []
    for r in comparison["rows"]:
        rows.append({
            "#": r["label"], "Plan": r["name"], "Role": r["author_role"], "Stations": r["n_stations"], "Shots": r["n_shots"],
            "Minutes (per station | per reading)": f"{r['time_per_station_model_min']:.0f} | {r['time_per_observation_model_min']:.0f}",
            "Expected wrong decisions (sum)": _fmt(r["expected_wrong_decisions"], ".4g"), "PFA": _fmt(r["expected_false_accepts"], ".2e"),
            "PFR": _fmt(r["expected_false_rejects"], ".2e"), "Escape probability": _fmt(r["escape_probability"], ".2e"),
            "Worst TUR": "-" if r["worst_tur"] is None else f"{r['worst_tur']:.1f}:1", "Largest u [µm]": _fmt(r["largest_u_um"], ".1f"),
            "Unmeasured": _fmt(r["n_unmeasured"], "d"), "Status": "feasible" if r["feasible"] else "INFEASIBLE - not scored"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Expected wrong decisions = the expected number of wrong accept/reject decisions per part inspected, summed over the characteristics "
               "(not a probability); PFA and PFR are its false-accept and false-reject parts. Use \"Explain the numbers\" below to see how each figure "
               f"is obtained. {comparison['caveat']}.")
    for r in comparison["rows"]:
        if r["blocking_findings"] or r["warnings"]:
            with st.expander(f"#{r['label']} {r['name']}: {len(r['blocking_findings'])} blocking, {len(r['warnings'])} warning(s)", expanded=bool(r["blocking_findings"])):
                for text in r["blocking_findings"]:
                    st.error("BLOCKING: " + text)
                for text in r["warnings"]:
                    st.warning(text)
    for line in result["not_found"]:
        st.info("Not found: " + line)

    st.markdown("#### Plan view")
    st.image(result["figures"]["plan_view.png"], width="stretch")
    st.markdown("#### Risk of each characteristic under each plan")
    st.image(result["figures"]["risk_by_characteristic.png"], width="stretch")
    if comparison["characteristics"]:
        table = []
        for name, entry in comparison["characteristics"].items():
            row = {"Characteristic": name, "Tolerance ± [µm]": f"{entry['tolerance_half_um']:.0f}"}
            for plan_entry in entry["plans"]:
                tag = f"#{plan_entry['label']}"
                row[f"{tag} u [µm]"] = "-" if not plan_entry["feasible"] else _fmt(plan_entry["u_um"], ".1f")
                row[f"{tag} risk"] = "-" if not plan_entry["feasible"] else format(plan_entry["risk"], ".2e") + ("" if plan_entry["measured"] else " *")
            table.append(row)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("\\* not measured: priced at its process prior as accepted or rejected unseen.")

    _explain_section(result, scene)

    st.markdown("#### Export")
    figures = result["figures"]
    columns = st.columns(5)
    columns[0].download_button("Comparison (JSON)", result["bundle_json"], "comparison.json", "application/json", key="pr_dl_json")
    columns[1].download_button("Plan view (PNG)", figures["plan_view.png"], "plan_view.png", "image/png", key="pr_dl_plan_png")
    columns[2].download_button("Plan view (SVG)", figures["plan_view.svg"], "plan_view.svg", "image/svg+xml", key="pr_dl_plan_svg")
    columns[3].download_button("Risk figure (PNG)", figures["risk_by_characteristic.png"], "risk_by_characteristic.png", "image/png", key="pr_dl_risk_png")
    columns[4].download_button("Risk figure (SVG)", figures["risk_by_characteristic.svg"], "risk_by_characteristic.svg", "image/svg+xml", key="pr_dl_risk_svg")
    st.caption(f"Logged {result['n_logged']} records to {result['log_path']}.")
    _signoff(result)


def _explain_section(result: Dict, scene: ReviewScene) -> None:
    st.markdown("#### Explain the numbers")
    st.caption("How each figure of one plan is obtained: the inputs (and which are placeholders), each link of the chain with its numbers, the outputs, "
               "what drives the risk, and independent checks that recompute the risk and (optionally) the whole measurement chain a different way.")
    reviews, plans = result["reviews_all"], result["plans_all"]
    options = [i for i, r in enumerate(reviews) if r.feasible]
    if not options:
        st.info("No plan is feasible, so there is no chain to explain; the reasons are above.")
        return
    columns = st.columns(3)
    plan_index = columns[0].selectbox("Plan", options, format_func=lambda i: f"#{i + 1} {reviews[i].plan['name']}", key="pr_explain_plan")
    names = [c["name"] for c in reviews[plan_index].characteristics if c["measured"]]
    choice = columns[1].selectbox("Follow this characteristic through every link", ["(the one with the most risk)"] + names, key="pr_explain_char")
    trials = int(columns[2].number_input("Whole-chain Monte Carlo trials (0 = skip; about 0.2 s each on the built-in cell)", min_value=0, value=0, step=100,
                                         key="pr_explain_trials"))
    if trials:
        st.warning(f"The whole-chain check simulates {trials} sets of tracker readings and solves the network for each: about {trials * 0.2:.0f} s on the built-in cell, "
                   "longer for a larger scene.")
    if st.button("Explain", key="pr_explain_run"):
        context = _context(scene, scene.scene_hash(), result["settings_json"])
        bar = st.progress(0.0, text="Working through the chain ...") if trials else None
        progress = (lambda done, total: bar.progress(done / total, text=f"Whole-chain Monte Carlo: trial {done} of {total}")) if trials else None
        explanation = explain_plan(context, plans[plan_index], reviews[plan_index], None if choice.startswith("(") else choice, 500_000, trials, 0, progress)
        st.session_state["pr_explanation"] = explanation.to_text()
    if "pr_explanation" in st.session_state:
        st.code(st.session_state["pr_explanation"], language=None)
        st.download_button("Explanation (text)", st.session_state["pr_explanation"], "explanation.txt", "text/plain", key="pr_dl_explanation")


def _signoff(result: Dict) -> None:
    st.markdown("#### Sign-off")
    labels = result["labels"]
    with st.form("pr_signoff_form"):
        choice = st.selectbox("Which plan would you sign off?", [NONE_ACCEPTABLE] + list(labels), index=None, placeholder="Choose a plan",
                              format_func=lambda h: "None of these is acceptable" if h == NONE_ACCEPTABLE else labels[h], key="pr_signoff_choice")
        reason = st.text_area("Why (free text; logged as written)", key="pr_signoff_reason")
        submitted = st.form_submit_button("Record the sign-off", key="pr_signoff_submit")
    if submitted:
        if choice is None or not reason.strip():
            st.error("Choose a plan (or 'none of these') and give a reason.")
            return
        try:
            record = ReviewLog(None, result["role"], result["name"], result["log_names"]).log_sign_off(
                result["comparison"], None if choice == NONE_ACCEPTABLE else choice, reason)
        except LogError as error:
            st.error(str(error))
            return
        st.success(f"Recorded (record {record['record_id'][:8]}) in {result['log_path']}.")


# -- the tab ---------------------------------------------------------------------------------------------------
def render() -> None:
    st.markdown(
        "Compare measurement plans on the testbed's chain -- stations, visibility, instrument error, datum-referenced uncertainty, conformity "
        "decision risk -- and see in decision terms what each costs. An engineer's plan and the algorithms' plans are scored by the same "
        "scorer and risk model. An infeasible plan is reported with its reasons and **never repaired**. Runs offline; uploaded files stay in memory.")
    try:
        log_dir = resolve_log_dir()
    except LogError as error:
        st.error(f"The review log cannot be written: {error}")
        return
    st.caption(f"Every review is logged to `{log_dir / LOG_FILE_NAME}` (on this machine only; no telemetry). Scene names are left out of the log unless you ask.")
    scene = _scene_section()
    if scene is None:
        return
    settings_pack = _settings_section(scene)
    settings, role, name, log_names = settings_pack
    plans = _plans_section(scene)
    counterpart_cfg = _counterparts_section(plans, settings)
    missing = []
    if settings is None:
        missing.append("choose the instrument model")
    if role is None:
        missing.append("choose your role")
    if not plans:
        missing.append("add a plan")
    if counterpart_cfg is not None and not counterpart_cfg["objectives"]:
        missing.append("choose at least one objective")
    if missing:
        st.caption("To run: " + ", ".join(missing) + ".")
    if st.button("Review the plans", type="primary", disabled=bool(missing), key="pr_run"):
        try:
            st.session_state["pr_result"] = _run(scene, settings, plans, role, name, log_names, counterpart_cfg)
        except LogError as error:
            st.error(str(error))
        except ValueError as error:
            st.error(f"The review could not run: {error}")
    if "pr_result" in st.session_state:
        _show_result(st.session_state["pr_result"], scene)
