"""Explain a review: its inputs, the chain that turns them into every figure, and the outputs, with the numbers at each link.

    INPUTS                  what was assumed, where each value comes from, and whether it is a placeholder
    THE CHAIN (per characteristic)
      1 visibility          which stations see its target, from how far, at what angle
      2 one shot            what a single reading's error is, resolved along the tolerance direction
      3 the network         what combining the stations does; the error in the anchor station's frame
      4 the part frame      what referring the point to the datum targets does; the projected uncertainty u
      5 the decision        PFA and PFR from the process prior, the tolerance limits, the acceptance rule and u
    OUTPUTS                 per-characteristic table, totals, shares
    WHAT DRIVES THE RISK    the same characteristic's risk as u, the process offset and the capability index vary
    INDEPENDENT CHECKS      `review.verify`: closed form, Monte Carlo of the decision step, optionally the whole chain

Nothing here is a second implementation of the review: every number is either read from the review or recomputed from the same
library calls the scorer uses, and `tests/test_review_explain.py` pins the explanation's projected uncertainty to the review's. What is
independent is in `review.verify`.

**What the headline figure is.** `expected_wrong_decisions` is a *sum over characteristics* of each one's PFA + PFR: the expected
number of wrong accept/reject decisions per part inspected, not the probability that a part is wrongly judged. Each PFA and PFR is a
*joint* probability over the production population: PFR is P(the part conforms on this characteristic AND its measurement falls outside
the acceptance limits), PFA is P(it does not conform AND it is accepted). So a characteristic's risk is large only if many parts sit
where a measurement of uncertainty u can push them across a limit: i.e. when the process is close to a limit compared with u.
"""
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from geometry.visibility import check_target_visibility
from network.design import datum_referenced_functional, design_joint_covariance
from review.core import PlanReview, ReviewContext, measurement_pattern, review_with_context
from review.verify import Check, check_review, risk_closed_form
from risk.decision import PriorComponent, ProcessKnowledge, decision_risk

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0
CPK_NOTE = "Cpk = min(upper limit - mean, mean - lower limit) / (3 std): the distance from the process mean to the nearer limit, in units of 3 std"


@dataclass(frozen=True)
class Explanation:
    """The explanation of one plan's review. `to_dict()` is JSON-serialisable; `to_text(focus)` renders it."""

    plan: Dict[str, Any]
    feasible: bool
    inputs: List[Dict[str, Any]]
    chain: List[Dict[str, Any]]                # one entry per measured characteristic
    outputs: Optional[Dict[str, Any]]
    drivers: Dict[str, Any]                    # by characteristic name
    checks: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    focus: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_text(self, focus: Optional[str] = None) -> str:
        return render_text(self, focus or self.focus)


# -- inputs ---------------------------------------------------------------------------------------------
def _inputs(context: ReviewContext, plan, review: PlanReview) -> List[Dict[str, Any]]:
    scene, scenario, settings = context.scene, context.scenario, context.settings
    tracker = scenario.tracker
    prov = review.provenance
    rows: List[Dict[str, Any]] = []

    def add(group, item, value, source, placeholder=False):
        rows.append({"group": group, "item": item, "value": value, "source": source, "placeholder": bool(placeholder)})

    add("instrument", "range noise, sigma_d", f"{tracker.sigma_d_m * 1e6:.3f} um (does not grow with range)",
        "Hughes et al. 2011 Table 4, a posteriori noise of an API T3 (CLAUDE.md §4a)")
    add("instrument", "azimuth noise, sigma_az", f"{np.degrees(tracker.sigma_theta_rad) * 3600:.3f} arcsec = {tracker.sigma_theta_rad * 1e6:.2f} urad "
        f"(a lateral error of {tracker.sigma_theta_rad * 1e6:.2f} um per metre of range)", "Hughes et al. 2011 Table 4 (CLAUDE.md §4a)")
    add("instrument", "elevation noise, sigma_el", f"{np.degrees(tracker.sigma_phi_rad) * 3600:.3f} arcsec = {tracker.sigma_phi_rad * 1e6:.2f} urad "
        f"(a lateral error of {tracker.sigma_phi_rad * 1e6:.2f} um per metre of range)", "Hughes et al. 2011 Table 4 (CLAUDE.md §4a)")
    if settings.tracker_model == "npl_residual":
        add("instrument", "geometric parameter errors", "NPL 16-parameter model, zero mean, covariance from the paper's Tables 2-3 uncertainties; "
            "shared by every station that uses the same tracker", "Hughes et al. 2011 Tables 2-3: one API T3 in IFM mode, 2011. TODO(source): GKN's tracker", True)
    add("instrument", "systematic floor", f"{settings.systematic_floor_um:g} um" + (" (none)" if settings.systematic_floor_um == 0 else ", isotropic, once per characteristic"),
        "review setting")
    for s in review.stations:
        x, y, z = s["position_cell_m"]
        add("plan", f"station {s['index']}" + (" (anchor)" if s["index"] == 1 else ""),
            f"({x:.2f}, {y:.2f}, {z:.2f}) m cell; instrument {s['instrument_id']}; faces {'+'.join(s['faces'])}; measures {s['n_targets_measured']} targets",
            f"plan file '{plan.name}' ({plan.author_role})")
    add("plan", "orientation", prov["orientation_convention"], "review convention (interchange/plan_file.py)")
    limits = scene.tolerance_limits_m()
    for k, name in enumerate(scene.characteristic_names):
        placeholder = scene.kind == "factory_cell"
        add("scene", f"{name}: tolerance", f"{limits[k, 0] * 1e6:+.0f} to {limits[k, 1] * 1e6:+.0f} um, judged along its tolerance direction",
            "the built-in cell: CLAUDE.md §4d's reference scale, a placeholder" if placeholder else "the QIF file", placeholder)
    add("scene", "targets / datum targets", f"{prov['scene']['n_targets']} targets, {prov['scene']['n_datum_targets']} of them datum targets",
        "the built-in cell's placeholder layout" if scene.kind == "factory_cell" else "the QIF file; reference nests laid out by rule", True)
    lim = scenario.visibility_limits
    add("scene", "visibility limits", f"range <= {lim.max_range_m:g} m, incidence <= {np.degrees(lim.max_incidence_angle_rad):.0f} deg"
        + ("" if lim.min_elevation_rad is None else f", elevation {np.degrees(lim.min_elevation_rad):.0f} to {np.degrees(lim.max_elevation_rad):.0f} deg"),
        "the repo's convention, not a literature value", True)
    for c in prov["process_priors"]:
        mean, std = c["mean_um"], c["std_um"]
        half = float(np.mean((limits[:, 1] - limits[:, 0]) / 2) * 1e6)
        cpk = (half - abs(mean)) / (3 * std) if std > 0 else float("inf")
        add("process", f"cluster '{c['name'][:40]}'", f"true deviation ~ N(mean {mean:+.1f} um, std {std:.1f} um): Cpk {cpk:.2f}"
            + ("" if c["n_builds"] is None else f" (read as {c['n_builds']} builds: Student-t)"), c["source"], c["placeholder"])
    add("decision", "rule", prov["decision_rule"], "review setting")
    return rows


# -- the chain ---------------------------------------------------------------------------------------------
def _prior_text(prior: ProcessKnowledge) -> str:
    parts = []
    for c in prior.components:
        kind = "normal" if c.dof is None else f"Student-t ({c.dof:g} dof)"
        parts.append(f"{c.weight:.2g} x {kind}(mean {c.mean_m * 1e6:+.1f} um, scale {c.scale_m * 1e6:.1f} um)")
    return " + ".join(parts)


def _largest_direction(matrix: np.ndarray):
    values, vectors = np.linalg.eigh(np.atleast_2d(matrix))
    return float(values[-1]), vectors[:, -1]


def _chain_for(context: ReviewContext, plan, review: PlanReview, pattern, k: int) -> Dict[str, Any]:
    scene, scenario = context.scene, context.scenario
    tracker = scenario.tracker
    noise_tracker = replace(tracker, geometric_params=None, systematic_std_m=0.0, floor_excludes_geometric_parameters=False)
    characteristic = scenario.characteristics[k]
    t = characteristic.target_indices[0]
    point = scenario.target_points_m[t]
    directions = np.atleast_2d(np.asarray(characteristic.tolerance.constraint_directions(), dtype=float))
    row = review.characteristics[k]
    poses, visible = pattern.poses, pattern.visible
    normals = scenario.target_normals_m
    out: Dict[str, Any] = {"name": scene.characteristic_names[k], "kind": scene.characteristic_kinds[k], "cluster": scene.clusters[k],
                           "target_index": int(t), "tolerance_direction": directions[0].tolist() if directions.shape[0] == 1 else "several directions"}

    # 1. visibility, and 2. one shot
    seeing, shots = [], []
    for s, pose in enumerate(poses):
        beam = point - pose.position_m
        distance = float(np.linalg.norm(beam))
        local = pose.to_local(point)
        elevation = float(np.degrees(np.arcsin(np.clip(local[2] / distance, -1, 1))))
        incidence = float(np.degrees(np.arccos(np.clip(-beam @ normals[t] / distance, -1, 1)))) if normals is not None else None
        measured = bool(visible[s, t])
        if not measured and pattern.visible_full[s, t]:
            reason = "visible, but this station's target list does not include it"
        elif not measured:
            reason = check_target_visibility(pose.position_m, point, scenario.visibility_limits, None if normals is None else normals[t],
                                             scenario.obstacles, pose.orientation).reason.value
        else:
            reason = "measured"
        seeing.append({"station": s + 1, "measured": measured, "reason": reason, "range_m": distance, "incidence_deg": incidence, "elevation_deg": elevation,
                       "n_targets_measured": int(visible[s].sum())})
        if measured:
            cov = noise_tracker.covariance_global(point, pose)
            projected = directions @ cov @ directions.T
            shots.append({"station": s + 1, "range_m": distance, "radial_um": tracker.sigma_d_m * 1e6,
                          "lateral_azimuth_um": distance * np.cos(np.radians(elevation)) * tracker.sigma_theta_rad * 1e6,
                          "lateral_elevation_um": distance * tracker.sigma_phi_rad * 1e6,
                          "along_tolerance_um": float(np.sqrt(np.linalg.eigvalsh(np.atleast_2d(projected)).max()) * 1e6)})
    out["visibility"] = {"stations": seeing, "n_measuring": len(shots),
                         "datum_targets_measured": int(len([i for i in scene.datum_indices if visible[:, i].any()])),
                         "datum_targets_total": len(scene.datum_indices)}
    out["one_shot"] = shots

    # 3. the network (anchor frame), 4. the part frame
    seen = visible.any(axis=0)
    datum = [i for i in scene.datum_indices if seen[i]]
    subset = list(dict.fromkeys([t] + datum))
    place = {i: n for n, i in enumerate(subset)}
    points = scenario.target_points_m
    joints = {}
    for label, trk in (("noise", noise_tracker), ("total", tracker)):
        joints[label], _, _ = design_joint_covariance(points, poses, trk, visible, subset, 0, pattern.instrument_ids, pattern.faces)
    own = slice(3 * place[t], 3 * place[t] + 3)
    anchor = {label: directions @ joints[label][own, own] @ directions.T for label in joints}
    functional = datum_referenced_functional(points[subset], place[t], [place[i] for i in datum])
    referenced = {label: directions @ (functional @ joints[label] @ functional.T) @ directions.T for label in joints}
    floor = tracker.systematic_std_m
    top, direction = _largest_direction(referenced["total"])
    noise_var = float(direction @ referenced["noise"] @ direction)
    u_um = float(np.sqrt(top + floor ** 2) * 1e6)
    anchor_total = float(np.sqrt(_largest_direction(anchor["total"])[0]) * 1e6)
    out["network"] = {"anchor_frame_sigma_along_tolerance_um": anchor_total,
                      "anchor_frame_sensor_noise_um": float(np.sqrt(_largest_direction(anchor["noise"])[0]) * 1e6),
                      "n_stations_measuring": len(shots), "n_shots_in_network": int(visible.sum())}
    out["part_frame"] = {"n_datum_targets_used": len(datum), "referenced_sigma_along_tolerance_um": float(np.sqrt(top) * 1e6),
                         "sensor_noise_variance_share": noise_var / top if top > 0 else 1.0,
                         "geometric_parameter_variance_share": 1.0 - noise_var / top if top > 0 else 0.0,
                         "systematic_floor_um": floor * 1e6, "u_um": u_um, "u_reviewed_um": row["u_um"],
                         "reconciles_with_review": bool(abs(u_um - row["u_um"]) <= 1e-9 * max(u_um, 1e-30) + 1e-9)}

    # 5. the decision
    limits = scene.tolerance_limits_m()[k]
    prior = context.knowledge[k]
    comp = prior.components[0]
    half = float((limits[1] - limits[0]) / 2)
    out["decision"] = {
        "prior": _prior_text(prior), "tolerance_um": [limits[0] * 1e6, limits[1] * 1e6], "acceptance_um": row["acceptance_um"],
        "u_um": row["u_um"], "tur": row["tur"], "pfa": row["pfa"], "pfr": row["pfr"], "risk": row["risk"], "risk_exact": row["risk_exact"],
        "normal_prior": len(prior.components) == 1 and comp.dof is None,
        "cpk": (half - abs(comp.mean_m)) / (3 * comp.scale_m) if len(prior.components) == 1 and comp.dof is None else None,
        "margin_to_nearer_limit_um": float(min(limits[1] - comp.mean_m, comp.mean_m - limits[0]) * 1e6) if len(prior.components) == 1 else None,
        "nonconformance": float(prior.nonconformance(limits[0], limits[1])),
    }
    return out


# -- what drives the risk ------------------------------------------------------------------------------------
def _with_mean(prior: ProcessKnowledge, mean_m: float) -> ProcessKnowledge:
    c = prior.components[0]
    return ProcessKnowledge((PriorComponent(1.0, mean_m, c.scale_m, c.dof),))


def _drivers(context: ReviewContext, review: PlanReview, k: int) -> Dict[str, Any]:
    prior = context.knowledge[k]
    limits = context.scene.tolerance_limits_m()[k]
    lower, upper = float(limits[0]), float(limits[1])
    row = review.characteristics[k]
    u = row["u_um"] * 1e-6
    if len(prior.components) != 1:
        return {"available": False, "note": "the process knowledge is a mixture: the what-if tables are shown for a single normal or Student-t prior only"}
    comp = prior.components[0]

    def total(p, uu):
        pfa, pfr = decision_risk(p, lower, upper, uu)
        return pfa + pfr

    out: Dict[str, Any] = {"available": True, "measured_u_um": row["u_um"], "prior_mean_um": comp.mean_m * 1e6, "prior_scale_um": comp.scale_m * 1e6,
                           "risk_now": total(prior, u)}
    out["vs_u"] = [{"u_factor": f, "u_um": f * row["u_um"], "risk": total(prior, f * u)} for f in (0.25, 0.5, 1.0, 2.0)]
    out["vs_offset"] = [{"mean_um": f * comp.mean_m * 1e6, "fraction_of_prior_mean": f, "risk": total(_with_mean(prior, f * comp.mean_m), u)}
                        for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if comp.dof is None and comp.mean_m != 0.0:
        half = 0.5 * (upper - lower)
        cpk = (half - abs(comp.mean_m)) / (3 * comp.scale_m)
        if cpk > 0:
            centred_scale = half / (3 * cpk)
            centred = ProcessKnowledge.known(0.0, centred_scale)
            out["same_cpk_centred"] = {"cpk": cpk, "mean_um": 0.0, "std_um": centred_scale * 1e6, "risk": total(centred, u),
                                       "ratio_actual_to_centred": total(prior, u) / max(total(centred, u), 1e-300)}
    return out


# -- putting it together --------------------------------------------------------------------------------------------
def explain_plan(context: ReviewContext, plan, review: Optional[PlanReview] = None, characteristic: Optional[str] = None,
                 n_parts: int = 1_000_000, verify_trials: int = 0, seed: int = 0, progress: Optional[Callable[[int, int], None]] = None) -> Explanation:
    """Explain `plan`'s review: see the module docstring. `characteristic` is the one explained in full (default: the one with the most risk);
    every measured characteristic gets its chain. `n_parts` sizes the decision-step Monte Carlo; `verify_trials` > 0 also runs the
    whole-chain Monte Carlo (slow: about 0.2 s per trial on the built-in cell)."""
    review = review_with_context(context, plan) if review is None else review
    inputs = _inputs(context, plan, review)
    plan_info = {"name": plan.name, "author_role": plan.author_role, "plan_hash": review.plan["plan_hash"], "context_id": review.context_id}
    if not review.feasible:
        return Explanation(plan_info, False, inputs, [], None, {}, [], review.findings, None)
    pattern = measurement_pattern(context, plan)
    chain, drivers = [], {}
    for k, row in enumerate(review.characteristics):
        if not row["measured"]:
            continue
        chain.append(_chain_for(context, plan, review, pattern, k))
        drivers[row["name"]] = _drivers(context, review, k)
    measured = [r for r in review.characteristics if r["measured"]]
    names = [r["name"] for r in measured]
    if characteristic is not None and characteristic not in names:
        raise ValueError(f"{characteristic!r} is not a measured characteristic of this plan; choose from {names}")
    focus = characteristic or (max(measured, key=lambda r: r["risk"])["name"] if measured else None)
    total = review.totals["expected_wrong_decisions"]
    outputs = {
        "rows": [{"name": r["name"], "u_um": r["u_um"], "tur": r["tur"], "pfa": r["pfa"], "pfr": r["pfr"], "risk": r["risk"], "measured": r["measured"],
                  "share_of_total": r["risk"] / total if total > 0 else 0.0} for r in review.characteristics],
        "totals": review.totals, "n_stations": review.n_stations, "n_shots": review.n_shots, "n_readings": review.n_readings, "time_min": review.time_min}
    checks = [c.to_dict() for c in check_review(context, plan, review, n_parts, verify_trials, seed, progress=progress)]
    return Explanation(plan_info, True, inputs, chain, outputs, drivers, checks, review.findings, focus)


# -- text -------------------------------------------------------------------------------------------------------------------
def _g(value, spec=".3g"):
    return "-" if value is None else format(value, spec)


def render_text(e: Explanation, focus: Optional[str] = None) -> str:
    L: List[str] = []
    rule = "=" * 100
    L += [rule, f"HOW THE NUMBERS FOR  '{e.plan['name']}'  ARE OBTAINED", rule, ""]
    L += ["WHAT THE HEADLINE FIGURE IS",
          "  'Expected wrong decisions' is the SUM over the characteristics of each one's PFA + PFR: the expected NUMBER of wrong accept/reject decisions",
          "  per part inspected. It is not the probability that a part is wrongly judged. Each PFA / PFR is a JOINT probability over the production",
          "  population: PFR = P(the part conforms on this characteristic AND its measured value falls outside the acceptance limits); PFA = P(it does",
          "  not conform AND it is accepted). A characteristic's risk is large only when many parts sit where a measurement of uncertainty u can push",
          "  them across a limit, i.e. when the process runs close to a limit compared with u.", ""]
    L += ["1. INPUTS  (placeholder = a stand-in value, not a measured or sourced one)", ""]
    group = None
    for r in e.inputs:
        if r["group"] != group:
            group = r["group"]
            L.append(f"  [{group}]")
        L.append(f"    {r['item']:<34s} {r['value']}")
        L.append(f"    {'':<34s}   source: {r['source']}{'   <-- PLACEHOLDER' if r['placeholder'] else ''}")
    L.append("")
    if not e.feasible:
        L += ["THE PLAN IS INFEASIBLE, so there is no chain to show:"] + [f"  - {f['detail']}" for f in e.findings if f["severity"] == "blocking"]
        return "\n".join(L)

    # compact chain table
    L += ["2. THE CHAIN, ALL CHARACTERISTICS (each column is one link; the next section follows one characteristic through them in full)", ""]
    head = (f"  {'characteristic':<22s} {'stations':>8s} {'nearest':>8s} {'best 1-shot':>11s} {'anchor frame':>12s} {'part frame u':>12s} "
            f"{'prior':>26s} {'PFA':>9s} {'PFR':>9s}")
    L += [head, "  " + "-" * (len(head) - 2)]
    for c in e.chain:
        d, shots = c["decision"], c["one_shot"]
        prior = c["decision"]["prior"]
        short = prior.replace("1 x normal", "N").replace("(mean ", "(").replace(" um, scale ", ",").replace(" um)", ")")
        L.append(f"  {c['name']:<22s} {c['visibility']['n_measuring']:>8d} {min(s['range_m'] for s in shots):>7.1f}m "
                 f"{min(s['along_tolerance_um'] for s in shots):>9.1f}um {c['network']['anchor_frame_sigma_along_tolerance_um']:>10.1f}um "
                 f"{c['part_frame']['u_um']:>10.1f}um {short[:26]:>26s} {d['pfa']:>9.2e} {d['pfr']:>9.2e}")
    L += ["", "  stations = how many stations measure it; nearest = shortest range to it; best 1-shot = the smallest single-reading error along its tolerance",
          "  direction; anchor frame = the error of its position in station 1's frame after combining all stations; part frame u = after referring it to",
          "  the datum targets (the figure used for the risk); prior = the process the part came from (units um).", ""]

    focus = focus or e.focus
    c = next(x for x in e.chain if x["name"] == focus)
    d = c["decision"]
    L += [f"3. ONE CHARACTERISTIC IN FULL:  {c['name']}  ({c['kind']}, cluster '{c['cluster'][:40]}')", ""]
    L += ["  Link 1 - visibility. Which stations measure its target, and from how far:"]
    for s in c["visibility"]["stations"]:
        inc = "" if s["incidence_deg"] is None else f", incidence {s['incidence_deg']:.0f} deg"
        L.append(f"    station {s['station']}: range {s['range_m']:.2f} m, elevation {s['elevation_deg']:+.0f} deg{inc}: {s['reason']}")
    L += [f"    {c['visibility']['n_measuring']} stations measure it; {c['visibility']['datum_targets_measured']} of {c['visibility']['datum_targets_total']} datum targets are measured "
          "(they define the part frame).", ""]
    L += ["  Link 2 - one reading. A tracker reports range d (error sigma_d, whatever the range) and two angles (errors sigma_az, sigma_el, each turning into a",
          "  lateral position error equal to range x angle error, so growing with range). Resolved along the tolerance direction:"]
    for s in c["one_shot"]:
        L.append(f"    station {s['station']}: range {s['range_m']:5.2f} m -> radial {s['radial_um']:.2f} um, lateral {s['lateral_azimuth_um']:.1f} um (azimuth) and "
                 f"{s['lateral_elevation_um']:.1f} um (elevation) => {s['along_tolerance_um']:.1f} um along the tolerance direction")
    L += [""]
    n = c["network"]
    only_anchor = n["n_stations_measuring"] == 1 and c["one_shot"][0]["station"] == 1
    if only_anchor:
        L += ["  Link 3 - the network. Only station 1, the anchor, measures this target, so the network adds nothing to it: its error in station 1's frame is a",
              f"  single reading's, {n['anchor_frame_sigma_along_tolerance_um']:.1f} um along the tolerance direction (sensor noise alone: {n['anchor_frame_sensor_noise_um']:.1f} um). "
              f"The other stations' {n['n_shots_in_network'] - c['visibility']['n_measuring']} readings serve the datum targets and the ties between stations, not this point.", ""]
    else:
        L += ["  Link 3 - the network. The stations' readings are combined in one least-squares solve (all targets and station poses together, station 1 held",
              f"  fixed): {n['n_shots_in_network']} readings, {c['visibility']['n_measuring']} of them on this target. Its error in station 1's frame is "
              f"{n['anchor_frame_sigma_along_tolerance_um']:.1f} um along the tolerance direction (sensor noise alone: {n['anchor_frame_sensor_noise_um']:.1f} um); "
              "that includes the uncertainty of where the measuring stations are relative to station 1.", ""]
    pf = c["part_frame"]
    moved = pf["referenced_sigma_along_tolerance_um"] - n["anchor_frame_sigma_along_tolerance_um"]
    L += ["  Link 4 - the part frame. A deviation is judged against the part's datum targets, not against a tracker. The datum targets' measured positions are",
          f"  best-fitted to their nominal positions ({pf['n_datum_targets_used']} targets) and the point is referred to that frame. That does two things: it removes the errors the",
          "  point shares with the datum targets (station poses, a tracker's parameter errors move the frame with it), and it adds the frame's own uncertainty, since the",
          f"  datum targets are measured too. Here the net effect is {'+' if moved >= 0 else '-'}{abs(moved):.2f} um ({n['anchor_frame_sigma_along_tolerance_um']:.2f} -> {pf['referenced_sigma_along_tolerance_um']:.2f} um)"
          + (f"; of the variance, {100 * pf['sensor_noise_variance_share']:.0f}% is sensor noise and {100 * pf['geometric_parameter_variance_share']:.0f}% tracker geometric parameters"
             if pf["geometric_parameter_variance_share"] > 1e-6 else "; all of it sensor noise") + ".",
          f"  Adding the systematic floor {pf['systematic_floor_um']:.2f} um in quadrature:  u = {pf['u_um']:.2f} um"
          + ("   [matches the review]" if pf["reconciles_with_review"] else f"   [DOES NOT match the review's {pf['u_reviewed_um']:.2f} um]") + "."]
    L += [f"  Test uncertainty ratio = tolerance half-width / (2 u) = {d['tolerance_um'][1]:.0f} / (2 x {d['u_um']:.2f}) = {d['tur']:.2f} : 1.", ""]
    L += ["  Link 5 - the decision. A part's true deviation x comes from the process; the measurement gives m = x + e with e ~ N(0, u); the part is accepted if m",
          f"  lies in [{d['acceptance_um'][0]:+.0f}, {d['acceptance_um'][1]:+.0f}] um (acceptance limits) and conforms if x lies in [{d['tolerance_um'][0]:+.0f}, {d['tolerance_um'][1]:+.0f}] um.",
          f"    process: {d['prior']}",
          f"    PFR = P(x conforms, m rejected) = integral over the tolerance of f(x) [1 - P(accept | x)] dx = {d['pfr']:.4e}",
          f"    PFA = P(x does not conform, m accepted)                                              = {d['pfa']:.4e}",
          f"    risk = PFA + PFR = {d['risk_exact']:.4e} (exact); the planners' tabulated risk curve gives {d['risk']:.4e}"]
    if d["normal_prior"]:
        L.append(f"    margin: the process mean sits {d['margin_to_nearer_limit_um']:.1f} um inside the nearer limit; Cpk {d['cpk']:.2f}. {CPK_NOTE}.")
        sd = float(d["prior"].split("scale ")[1].split(" um")[0])
        L.append(f"    reading it: measured values spread by sqrt(std^2 + u^2) = {np.hypot(sd, d['u_um']):.1f} um around the process mean, against a margin of {d['margin_to_nearer_limit_um']:.1f} um.")
    L.append("")

    o = e.outputs
    L += ["4. OUTPUTS", "", f"  {'characteristic':<22s} {'u um':>7s} {'TUR':>7s} {'PFA':>10s} {'PFR':>10s} {'risk':>10s} {'share of total':>15s}", "  " + "-" * 85]
    for r in o["rows"]:
        if r["measured"]:
            L.append(f"  {r['name']:<22s} {r['u_um']:>7.2f} {r['tur']:>6.1f}: {r['pfa']:>10.2e} {r['pfr']:>10.2e} {r['risk']:>10.3e} {100 * r['share_of_total']:>13.1f} %")
        else:
            L.append(f"  {r['name']:<22s} {'not measured':>7s}  priced at its process prior: {r['risk']:.3e}  ({100 * r['share_of_total']:.1f} % of total)")
    t = o["totals"]
    L += ["  " + "-" * 85, f"  expected wrong decisions = sum of the risks = {t['expected_wrong_decisions']:.4f}  (from the planners' tabulated risk curve; the exact integrals sum to "
          f"{t['expected_wrong_decisions_exact']:.4f}, {100 * t['risk_model_vs_exact_relative_gap']:.1f}% apart)",
          f"    split (exact integrals): false rejections {t['expected_false_rejects']:.4f}, false accepts {t['expected_false_accepts']:.2e}",
          f"  escape probability = P(at least one false accept) = 1 - prod(1 - PFA) = {t['escape_probability']:.3e}",
          f"  {o['n_stations']} stations, {o['n_shots']} shots ({o['n_readings']} readings); time {o['time_min']['per_station_model_min']:.0f} min (proportional to stations) / "
          f"{o['time_min']['per_observation_model_min']:.0f} min (proportional to readings): two readings of one published timing, not a measured time", ""]

    dr = e.drivers.get(focus, {"available": False})
    L += [f"5. WHAT DRIVES {focus}'S RISK  (same plan, same measurement; one input changed at a time)", ""]
    if not dr.get("available"):
        L += ["  " + dr.get("note", "not available"), ""]
    else:
        L += [f"  as it is: risk {dr['risk_now']:.3e}  (u {dr['measured_u_um']:.1f} um; process mean {dr['prior_mean_um']:+.1f} um, scale {dr['prior_scale_um']:.1f} um)", "",
              "  a) a better (or worse) measurement:   " + "   ".join(f"u x{r['u_factor']:g} = {r['u_um']:.1f} um -> {r['risk']:.2e}" for r in dr["vs_u"]),
              "  b) the process mean moved toward nominal:   " + "   ".join(f"mean {r['mean_um']:+.4g} um -> {r['risk']:.2e}" for r in dr["vs_offset"])]
        if "same_cpk_centred" in dr:
            s = dr["same_cpk_centred"]
            L += ["  c) the SAME capability index, with the margin lost to spread instead of offset:",
                  f"       centred, mean 0, std {s['std_um']:.1f} um (Cpk {s['cpk']:.2f}) -> risk {s['risk']:.2e}; "
                  + ("the two are about the same at this u" if 0.95 < s["ratio_actual_to_centred"] < 1.05 else
                     f"the offset process is {s['ratio_actual_to_centred']:.{0 if s['ratio_actual_to_centred'] >= 10 else 1}f} times "
                     + ("worse" if s["ratio_actual_to_centred"] > 1 else "better") + " at this u")]
        L.append("")

    L += ["6. INDEPENDENT CHECKS  (each recomputes a figure a different way; see review/verify.py)", ""]
    if not e.checks:
        L += ["  none run"]
    for ch in e.checks:
        flag = "ok  " if ch["passed"] else "FAIL"
        note = "   (too rare to see in this many parts; the closed-form check covers it)" if ch["detail"].startswith("0 events") else ""
        L.append(f"  [{flag}] {ch['name']}: reviewed {ch['reviewed']:.4g}, independent {ch['independent']:.4g}  (within {ch['tolerance']}){note}")
    n_fail = sum(not ch["passed"] for ch in e.checks)
    L += ["", f"  {len(e.checks) - n_fail} of {len(e.checks)} checks agree." if e.checks else ""]
    return "\n".join(L)
