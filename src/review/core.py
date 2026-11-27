"""Review a measurement plan on the testbed's chain, and compare plans in decision terms.

    station pose -> visibility -> instrument error -> coordinate covariance
        -> characteristic uncertainty (datum-referenced) -> conformity decision risk

`review_plan` takes a plan (an engineer's, an algorithm's, a synthetic one: `interchange.plan_file`)
and a scene (`review.scene`) and reports what the plan delivers: the stations and shots, the time
(two readings of one published timing), each characteristic's uncertainty, test uncertainty ratio and
decision risk, the characteristics left unmeasured and what that costs, totals, and whether the plan is
feasible at all, with reasons. **An infeasible plan is reported and never repaired.**

**One scorer, one risk model.** A comparison is only meaningful if every plan was scored the same way.
`build_context` makes the scorer and the risk model once; `review_plans` reviews every plan with that
context, and `compare_plans` refuses reviews from different contexts (`context_id`).

**What is a station.** Each station is a levelled tracker (standing axis vertical) turned to face the
centroid of the scene's targets, at the position the plan gives. See `interchange.plan_file` for why that
is a convention and when it matters. A station that lists its targets measures exactly those it can see;
one that does not measures every target it can see. The first station is the anchor of the network.

**Risk, three ways, and how they relate.** The headline `expected_wrong_decisions` is the sum, over the
characteristics, of the risk model's tabulated curve at each projected uncertainty (the quantity the planners
minimise; a log-log interpolation on a 60-point grid). `pfa` and `pfr` are computed directly at the same
uncertainty by the JCGM 106 integrals (`risk.decision`), exactly; their sum differs from the curve by the
interpolation, reported as `risk_model_vs_exact_relative_gap`. A characteristic no station measures is priced
at its process prior as the better of accepting or rejecting it unseen (`risk.decision.unmeasured_risk`):
accepted unseen it costs its nonconformance rate (a false accept), rejected unseen its conformance rate
(false rejects). `escape_probability` is P(at least one characteristic falsely accepted) = 1 - prod(1 - PFA),
assuming independence, as `planning.inspection` defines it (an upper-bound-style summary: errors that share a
station-pose error are correlated).

**Time.** `measurement_time_minutes`: two readings of one timing (Hughes et al. 2011: 5 stations, 126 readings,
55 minutes), proportional to stations or to readings. A reading is one target on one face. These are
neither measured times for this cell; the truth lies between.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.pose import InstrumentPose
from geometry.visibility import API_T3_MAX_ELEVATION_RAD, API_T3_MIN_ELEVATION_RAD
from interchange.plan_file import PlanFile
from planning.candidates import levelled_orientation
from planning.cell import CellRiskModel, PartFrameScorer, measurement_time_minutes
from review.feasibility import BLOCKING, Finding, diagnose, listed_target_findings
from review.provenance import digest, git_state, settings_digest, tracker_description
from review.scene import ReviewScene, SceneError
from review.settings import ReviewSettings
from risk.decision import ProcessKnowledge, best_rule_risk, decision_risk

REVIEW_SCHEMA_VERSION = 1
ORIENTATION_CONVENTION = ("each station is a levelled tracker (standing axis vertical) turned to face the centroid of the "
                          "scene's targets; with the sensor-noise model the turn changes nothing, with the NPL geometric "
                          "parameters it changes uncertainties by a few tenths of a percent")
TIME_BASIS = ("two readings of one published timing (Hughes et al. 2011: 5 stations, 126 readings, 55 minutes): proportional "
              "to the number of stations, or to the number of readings; neither is a measured time for this cell")
FIRST_STATION_NOTE = "station 1, the first listed, is the anchor the other stations are tied to"


@dataclass(frozen=True)
class ReviewContext:
    """Everything one comparison shares: the scene, the settings, the scenario with the settings' instrument,
    the scorer, the risk model and the process knowledge. Build once with `build_context`."""

    scene: ReviewScene
    settings: ReviewSettings
    scenario: Any                         # PlanningScenario with the settings' tracker and visibility limits
    scorer: PartFrameScorer
    risk_model: CellRiskModel
    knowledge: Tuple[ProcessKnowledge, ...]
    context_id: str
    provenance: Dict[str, Any] = field(default_factory=dict)


def build_context(scene: ReviewScene, settings: ReviewSettings, require_all_characteristics: Optional[bool] = None) -> ReviewContext:
    """The scorer and risk model every plan in a comparison is reviewed with. `require_all_characteristics` overrides
    the setting (used by `algorithm_counterparts`, whose plans always measure everything)."""
    require_all = settings.require_all_characteristics if require_all_characteristics is None else require_all_characteristics
    tracker = settings.build_tracker()
    limits = scene.scenario.visibility_limits
    if settings.elevation_limits == "api_t3":
        limits = replace(limits, min_elevation_rad=API_T3_MIN_ELEVATION_RAD, max_elevation_rad=API_T3_MAX_ELEVATION_RAD)
    scenario = replace(scene.scenario, tracker=tracker, visibility_limits=limits)
    scorer = PartFrameScorer(scenario, tuple(scene.datum_indices), require_all)
    model = scene.risk_model(settings.decision_rule, settings.n_builds)
    knowledge = scene.process_knowledge(settings.n_builds)
    summary = scene.summary()
    provenance = {
        "tracker_model": settings.tracker_model, "tracker_label": settings.tracker_label(),
        "tracker": tracker_description(tracker), "settings": settings.to_dict(),
        "decision_rule": ("simple acceptance (acceptance limits = tolerance limits)" if settings.decision_rule == "simple" else
                          "conformance-probability rule (accept when P(conforming | measured) >= 0.5)"),
        "uncertainty_reference": {
            "datum_referenced": True,
            "method": f"deviation of each point from a rigid best fit of {len(scene.datum_indices)} datum targets (S-transformation; "
                      "planning.cell.PartFrameScorer)"},
        "elevation_limits": ("none" if settings.elevation_limits == "none" else
                             "tracker mechanical elevation range -60 to +77 degrees (API T3, Hughes et al. 2011; TODO(source): GKN's tracker)"),
        "process_priors": summary["clusters"],
        "process_priors_include_placeholder": any(c["placeholder"] for c in summary["clusters"]),
        "scene": {k: summary[k] for k in ("label", "kind", "scene_hash", "n_targets", "n_characteristics", "n_datum_targets")},
        "placement_assumed": scene.placement_assumed,
        "reference_network": ("assumed: laid out by rule round the part (planning.qif_cell.CellRules), not the real cell's"
                              if scene.kind == "qif" else "synthetic: the built-in cell's placeholder reference nests"),
        "scene_notes": list(scene.notes),
        "orientation_convention": ORIENTATION_CONVENTION,
        "first_station": FIRST_STATION_NOTE,
        "time_basis": TIME_BASIS,
        "git": git_state(),
    }
    context_id = digest(summary["scene_hash"], settings.to_dict(), bool(require_all))
    return ReviewContext(scene, settings, scenario, scorer, model, knowledge, context_id, provenance)


# -- the review ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanReview:
    """What one plan delivers. Plain data: `to_dict()` is JSON-serialisable and `from_dict()` its inverse."""

    plan: Dict[str, Any]
    context_id: str
    stations: List[Dict[str, Any]]
    n_stations: int
    n_shots: int
    n_readings: int
    time_min: Dict[str, float]
    feasible: bool
    findings: List[Dict[str, Any]]
    characteristics: Optional[List[Dict[str, Any]]]
    totals: Optional[Dict[str, Any]]
    provenance: Dict[str, Any]
    schema_version: int = REVIEW_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanReview":
        if data.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unknown PlanReview schema_version {data.get('schema_version')!r}")
        return cls(**data)

    @property
    def name(self) -> str:
        return self.plan["name"]

    @property
    def blocking(self) -> List[Dict[str, Any]]:
        return [f for f in self.findings if f["severity"] == BLOCKING]


def _none_if_not_finite(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def _escape_probability(pfa: np.ndarray) -> float:
    """P(at least one characteristic falsely accepted) = 1 - prod(1 - PFA), independence assumed
    (the definition of `planning.inspection._escape_probability`; a test checks they agree)."""
    return float(-np.expm1(np.log1p(-np.minimum(pfa, 1.0 - 1e-16)).sum()))


def _unmeasured_split(prior: ProcessKnowledge, lower_m: float, upper_m: float) -> Tuple[float, float, str]:
    """(PFA, PFR, decision) of the better of the two decisions available without a measurement; its total is
    `risk.decision.unmeasured_risk`."""
    nonconformance = prior.nonconformance(lower_m, upper_m)
    if nonconformance <= 1.0 - nonconformance:
        return nonconformance, 0.0, "accepted unseen"
    return 0.0, 1.0 - nonconformance, "rejected unseen"


def review_with_context(context: ReviewContext, plan: PlanFile) -> PlanReview:
    """Review `plan` with `context`'s scorer and risk model. See the module docstring."""
    scene, scenario, settings = context.scene, context.scenario, context.settings
    positions = np.array([s.position_m for s in plan.stations], dtype=float)
    cell_positions = scene.to_cell(positions) if plan.frame == "file" else positions
    file_positions = scene.to_file(cell_positions)
    aim = scene.aim_point_m
    poses = [InstrumentPose(position_m=p, orientation=levelled_orientation(p, aim)) for p in cell_positions]

    visible_full = context.scorer.visible(poses)
    visible = visible_full.copy()
    findings: List[Finding] = []
    listed: Dict[int, List[int]] = {}
    for s, station in enumerate(plan.stations):
        if station.targets is None:
            continue
        indices: List[int] = []
        for name in station.targets:
            try:
                indices.append(scene.target_index(name))
            except SceneError as error:
                findings.append(Finding("unknown_target", BLOCKING, f"station {s + 1}: {error}", station=s + 1, target=name))
        listed[s] = indices
        keep = np.zeros(visible.shape[1], dtype=bool)
        keep[indices] = True
        visible[s] = visible_full[s] & keep
    findings += listed_target_findings(poses, scene, scenario, listed, visible_full)

    instrument_ids = None if all(s.instrument_id == 0 for s in plan.stations) else [s.instrument_id for s in plan.stations]
    faces = None if all(s.faces == ("front",) for s in plan.stations) else [tuple(s.faces) for s in plan.stations]
    n_shots = int(visible.sum())
    n_readings = int(sum(int(visible[s].sum()) * len(station.faces) for s, station in enumerate(plan.stations)))

    findings += diagnose(scenario, scene.datum_indices, scene.characteristic_names, visible, settings.require_all_characteristics)
    score = None
    if not any(f.kind == "unknown_target" for f in findings):
        score = context.scorer.score_visibility(poses, visible, instrument_ids, faces)
        if not score.feasible and not any(f.severity == BLOCKING for f in findings):
            findings.append(Finding(
                "network_not_rigid", BLOCKING, "the network solve is singular or ill-conditioned (stations tied through "
                "targets too nearly collinear, or too few independent shared targets), so no covariance can be computed"))
    feasible = score is not None and score.feasible and not any(f.severity == BLOCKING for f in findings)

    stations = []
    for s, station in enumerate(plan.stations):
        entry = {"index": s + 1, "position_cell_m": cell_positions[s].tolist(),
                 "position_file_m": None if file_positions is None else file_positions[s].tolist(),
                 "instrument_id": station.instrument_id, "faces": list(station.faces),
                 "n_targets_measured": int(visible[s].sum()), "n_targets_visible": int(visible_full[s].sum()),
                 "targets_listed": station.targets is not None}
        if station.s_m is not None:
            entry["track"] = {"s_m": station.s_m, "height_m": station.height_m}
        stations.append(entry)

    characteristics = totals = None
    if feasible:
        characteristics, totals = _characteristics_and_totals(context, score.u_m)
    review_provenance = dict(context.provenance)
    # what identifies the physical plan, whatever frame it was written in: cell-frame positions to the nanometre,
    # the instruments, faces and (resolved) target lists, in station order
    plan_hash = digest(np.round(cell_positions, 9), [s.instrument_id for s in plan.stations], [list(s.faces) for s in plan.stations],
                       [None if s.targets is None else sorted(s.targets) for s in plan.stations])
    return PlanReview(
        plan={"name": plan.name, "author_role": plan.author_role, "source": plan.source, "created": plan.created,
              "frame": plan.frame, "plan_hash": plan_hash, "file_hash": plan.content_hash(), "n_stations": plan.n_stations},
        context_id=context.context_id, stations=stations, n_stations=plan.n_stations, n_shots=n_shots, n_readings=n_readings,
        time_min=measurement_time_minutes(plan.n_stations, n_readings), feasible=feasible,
        findings=[f.to_dict() for f in findings], characteristics=characteristics, totals=totals, provenance=review_provenance,
    )


def _characteristics_and_totals(context: ReviewContext, u_m: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scene, settings = context.scene, context.settings
    limits = scene.tolerance_limits_m()
    curve_risk = context.risk_model.per_characteristic(u_m)  # NaN (unmeasured) costs its unmeasured price
    rows, pfas, pfrs, exact, unmeasured_cost = [], [], [], [], 0.0
    for k, name in enumerate(scene.characteristic_names):
        lower, upper = float(limits[k, 0]), float(limits[k, 1])
        half = 0.5 * (upper - lower)
        prior = context.knowledge[k]
        measured = bool(np.isfinite(u_m[k]))
        if measured:
            if settings.decision_rule == "simple":
                pfa, pfr = decision_risk(prior, lower, upper, float(u_m[k]))
                acceptance = (lower, upper)
            else:
                pfa, pfr, acceptance = best_rule_risk(prior, lower, upper, float(u_m[k]), 0.5)
            decision = "measured"
        else:
            pfa, pfr, decision = _unmeasured_split(prior, lower, upper)
            acceptance = (float("nan"), float("nan"))
            unmeasured_cost += float(curve_risk[k])
        pfas.append(pfa)
        pfrs.append(pfr)
        exact.append(pfa + pfr)
        rows.append({
            "name": name, "kind": scene.characteristic_kinds[k], "cluster": scene.clusters[k], "measured": measured,
            "decision": decision, "tolerance_half_um": half * 1e6,
            "u_um": _none_if_not_finite(u_m[k] * 1e6) if measured else None,
            "tur": float(half / (2.0 * u_m[k])) if measured else None,
            "pfa": float(pfa), "pfr": float(pfr), "risk_exact": float(pfa + pfr), "risk": float(curve_risk[k]),
            "acceptance_um": None if not measured or acceptance[0] != acceptance[0] else
            [_none_if_not_finite(acceptance[0] * 1e6), _none_if_not_finite(acceptance[1] * 1e6)],
            "acceptance_note": None if not measured else
            ("acceptance limits = tolerance limits" if settings.decision_rule == "simple" else
             "a side without a limit (null) accepts every measured value on that side; both null: no measured value is rejected"),
        })
    pfa_arr, exact_arr = np.array(pfas), np.array(exact)
    total_curve, total_exact = float(curve_risk.sum()), float(exact_arr.sum())
    measured_u = np.array([r["u_um"] for r in rows if r["measured"]])
    totals = {
        "expected_wrong_decisions": total_curve, "expected_false_accepts": float(pfa_arr.sum()),
        "expected_false_rejects": float(np.sum(pfrs)), "expected_wrong_decisions_exact": total_exact,
        "risk_model_vs_exact_relative_gap": float(abs(total_curve - total_exact) / total_exact) if total_exact > 0 else 0.0,
        "escape_probability": _escape_probability(pfa_arr),
        "risk_if_nothing_is_measured": context.risk_model.unmeasured_total(),
        "n_characteristics": len(rows), "n_measured": int(sum(r["measured"] for r in rows)),
        "n_unmeasured": int(sum(not r["measured"] for r in rows)), "risk_from_unmeasured": unmeasured_cost,
        "worst_tur": float(min(r["tur"] for r in rows if r["measured"])) if measured_u.size else None,
        "largest_u_um": float(measured_u.max()) if measured_u.size else None,
    }
    return rows, totals


def review_plan(scene: ReviewScene, plan: PlanFile, settings: ReviewSettings, context: Optional[ReviewContext] = None) -> PlanReview:
    """`review_with_context` for one plan. Pass `context` to share a scorer and risk model across calls; with
    none, one is built for this call alone (so two such reviews are NOT comparable: use `review_plans`)."""
    return review_with_context(build_context(scene, settings) if context is None else context, plan)


def review_plans(scene: ReviewScene, plans: Sequence[PlanFile], settings: ReviewSettings) -> Tuple[ReviewContext, List[PlanReview]]:
    """Review every plan with one context: the one scorer and one risk model a comparison needs."""
    context = build_context(scene, settings)
    return context, [review_with_context(context, plan) for plan in plans]


# -- comparison -----------------------------------------------------------------------------
def compare_plans(reviews: Sequence[PlanReview], reference: int = 0) -> Dict[str, Any]:
    """A side-by-side table of the plans and the per-characteristic differences from `reference` (an index into
    `reviews`). Refuses reviews that were not scored with one context (scene, settings, scorer). JSON-serialisable."""
    if not reviews:
        raise ValueError("nothing to compare")
    contexts = {r.context_id for r in reviews}
    if len(contexts) != 1:
        raise ValueError("these reviews were not scored with one scorer, risk model and settings (different context_id): "
                         "review the plans together with review_plans")
    ref = reviews[reference]
    rows = []
    for i, r in enumerate(reviews):
        t = r.totals
        rows.append({
            "label": f"{i + 1}", "name": r.plan["name"], "author_role": r.plan["author_role"], "plan_hash": r.plan["plan_hash"],
            "feasible": r.feasible, "n_stations": r.n_stations, "n_shots": r.n_shots, "n_readings": r.n_readings,
            "time_per_station_model_min": r.time_min["per_station_model_min"],
            "time_per_observation_model_min": r.time_min["per_observation_model_min"],
            "expected_wrong_decisions": None if t is None else t["expected_wrong_decisions"],
            "expected_false_accepts": None if t is None else t["expected_false_accepts"],
            "expected_false_rejects": None if t is None else t["expected_false_rejects"],
            "escape_probability": None if t is None else t["escape_probability"],
            "worst_tur": None if t is None else t["worst_tur"],
            "largest_u_um": None if t is None else t["largest_u_um"],
            "n_unmeasured": None if t is None else t["n_unmeasured"],
            "blocking_findings": [f["detail"] for f in r.findings if f["severity"] == BLOCKING],
            "warnings": [f["detail"] for f in r.findings if f["severity"] != BLOCKING],
        })
    by_characteristic: Dict[str, Any] = {}
    if any(r.feasible for r in reviews):
        names = [c["name"] for c in next(r for r in reviews if r.feasible).characteristics]
        for k, name in enumerate(names):
            entry = []
            ref_row = ref.characteristics[k] if ref.feasible else None
            for i, r in enumerate(reviews):
                if not r.feasible:
                    entry.append({"label": f"{i + 1}", "feasible": False})
                    continue
                c = r.characteristics[k]
                item = {"label": f"{i + 1}", "feasible": True, "measured": c["measured"], "u_um": c["u_um"], "tur": c["tur"],
                        "pfa": c["pfa"], "pfr": c["pfr"], "risk": c["risk"]}
                if ref_row is not None:
                    item["risk_minus_reference"] = c["risk"] - ref_row["risk"]
                    item["u_um_minus_reference"] = None if c["u_um"] is None or ref_row["u_um"] is None else c["u_um"] - ref_row["u_um"]
                entry.append(item)
            by_characteristic[name] = {"tolerance_half_um": next(r for r in reviews if r.feasible).characteristics[k]["tolerance_half_um"],
                                       "plans": entry}
    nothing = next((r.totals["risk_if_nothing_is_measured"] for r in reviews if r.totals), None)
    feasible_rows = [row for row in rows if row["feasible"]]
    ranking = [row["label"] for row in sorted(feasible_rows, key=lambda row: row["expected_wrong_decisions"])]
    statements: List[str] = []
    if nothing is not None and feasible_rows and min(r["expected_wrong_decisions"] for r in feasible_rows) > nothing:
        statements.append(
            f"Every feasible plan's expected wrong decisions exceed the {nothing:.3g} of measuring nothing: under these process statistics "
            "the model prices measurement as creating false rejections faster than it prevents false acceptances "
            "(planning.process_evidence explains why, and the statistics here may be placeholders); read the plans against each other, "
            "not against the nothing-measured line")
    return {
        "statements": statements,
        "context_id": ref.context_id, "reference": ref.plan["name"], "reference_label": f"{reference + 1}",
        "tracker_label": ref.provenance["tracker_label"], "settings": ref.provenance["settings"],
        "provenance": ref.provenance, "risk_if_nothing_is_measured": nothing,
        "rows": rows, "characteristics": by_characteristic, "ranking_by_expected_wrong_decisions": ranking,
        "caveat": ("An algorithm plan is the best a differential-evolution search found, not provably the best; a plan with "
                   "lower risk is not thereby better on anything this model does not contain"),
    }


def _fmt(value, spec: str, none: str = "-") -> str:
    return none if value is None else format(value, spec)


def format_comparison(comparison: Dict[str, Any]) -> str:
    """The comparison as plain text: the header (instrument model, decision rule, placeholders), the plan table,
    each infeasible plan's reasons and each characteristic's risk per plan."""
    prov = comparison["provenance"]
    lines = [f"Plan review: {prov['scene']['label']}", f"  instrument model : {comparison['tracker_label']}",
             f"  decision rule    : {prov['decision_rule']}",
             f"  uncertainty      : {prov['uncertainty_reference']['method']}",
             f"  orientation      : {prov['orientation_convention']}",
             f"  {prov['first_station']}"]
    if prov["placement_assumed"]:
        lines.append("  PLACEMENT ASSUMED: the part's position in the cell was not supplied (see the scene notes)")
    lines.append(f"  reference network: {prov['reference_network']}")
    if prov["process_priors_include_placeholder"]:
        lines.append("  PLACEHOLDER process statistics in use: risk figures are illustrative only")
    lines.append(f"  time             : {prov['time_basis']}")
    lines.append(f"  risk if nothing is measured: {_fmt(comparison['risk_if_nothing_is_measured'], '.4g')} expected wrong decisions")
    lines.append(f"  commit {prov['git']['commit'][:10]}{' (uncommitted changes)' if prov['git']['dirty'] else ''}; "
                 f"scene {prov['scene']['scene_hash'][:10]}; context {comparison['context_id'][:10]}")
    lines.append("")
    head = f"{'#':>2} {'plan':<30} {'role':<9} {'stn':>3} {'shots':>5} {'min (stn|obs)':>14} {'wrong dec.':>10} {'PFA':>9} {'PFR':>9} {'escape':>9} {'worst TUR':>9} {'max u um':>8} {'unmeas':>6}"
    lines.append(head)
    lines.append("-" * len(head))
    for row in comparison["rows"]:
        times = f"{row['time_per_station_model_min']:.0f}|{row['time_per_observation_model_min']:.0f}"
        if row["feasible"]:
            lines.append(f"{row['label']:>2} {row['name'][:30]:<30} {row['author_role']:<9} {row['n_stations']:>3} {row['n_shots']:>5} {times:>14} "
                         f"{row['expected_wrong_decisions']:>10.4g} {row['expected_false_accepts']:>9.2e} {row['expected_false_rejects']:>9.2e} "
                         f"{row['escape_probability']:>9.2e} {row['worst_tur']:>8.1f}:1 {row['largest_u_um']:>8.1f} {row['n_unmeasured']:>6}")
        else:
            lines.append(f"{row['label']:>2} {row['name'][:30]:<30} {row['author_role']:<9} {row['n_stations']:>3} {row['n_shots']:>5} {times:>14} "
                         f"{'INFEASIBLE - not scored':>52}")
    for row in comparison["rows"]:
        for reason in row["blocking_findings"]:
            lines.append(f"  [{row['label']}] BLOCKING: {reason}")
        for reason in row["warnings"]:
            lines.append(f"  [{row['label']}] warning: {reason}")
    if comparison["characteristics"]:
        lines += ["", "Risk per characteristic (expected wrong decisions; u in um; '*' = not measured, priced at its prior)"]
        labels = [row["label"] for row in comparison["rows"]]
        lines.append(f"{'characteristic':<26} {'tol +-um':>8}  " + "  ".join(f"{('#' + lab + ' u'):>8} {('#' + lab + ' risk'):>10}" for lab in labels))
        for name, entry in comparison["characteristics"].items():
            cells = []
            for plan_entry in entry["plans"]:
                if not plan_entry["feasible"]:
                    cells.append(f"{'-':>8} {'-':>10}")
                else:
                    mark = "" if plan_entry["measured"] else "*"
                    cells.append(f"{_fmt(plan_entry['u_um'], '.1f'):>8} {format(plan_entry['risk'], '.3e') + mark:>10}")
            lines.append(f"{name[:26]:<26} {entry['tolerance_half_um']:>8.0f}  " + "  ".join(cells))
    for statement in comparison["statements"]:
        lines += ["", "NOTE: " + statement]
    lines += ["", f"Ranking by expected wrong decisions (feasible plans): {' < '.join('#' + x for x in comparison['ranking_by_expected_wrong_decisions']) or 'none feasible'}",
              comparison["caveat"]]
    return "\n".join(lines)
