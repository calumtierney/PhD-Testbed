"""Noise-only against the full NPL model, on the existing experiments.

    python -m viz.run_npl_side_by_side        # about 6 minutes

The full model here is *consider mode* with `hughes_2011_residual()`: one
tracker, compensated, its remaining parameter errors (Tables 2-3
uncertainties, no correlations) shared by every station. The characteristic
layer is the per-point one as it stands (`characteristics.characteristic`):
each characteristic reads only its own point's 3 x 3 covariance. Nothing is
adopted as a default; this reports.

  A  how much of each characteristic's uncertainty the parameters add, per
     point, on the existing scenes (and, for the factory cell, what the
     datum-referenced scoring -- which uses the joint covariance -- gives
     instead);
  B  the headline experiment (`planning.experiment.run_headline_experiment`,
     A0 / A1 / B / C on the two-cluster scene): does any chosen plan change,
     and what happens to sum u, PFA, PFR and total risk;
  C  the same for the station-track experiment's grid stage
     (`viz.run_station_track_experiment`: 1.0 m spacing, 3 added stations,
     exhaustive), the stage whose result does not depend on a GA seed;
  D  milliseconds per plan evaluation over 200 evaluations per objective.

Writes results/npl_side_by_side.json.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from instruments.laser_tracker import LaserTracker
from instruments.npl_model import NPLErrorParameters
from planning.candidates import track_candidate_poses
from planning.cell import CellLayoutScorer, DatumCellScorer, factory_cell_scenario, layout_poses
from planning.experiment import (
    LONG_PART_GRID_HEIGHTS_M, _plan_report, default_two_cluster_scenario, long_part_track, long_part_two_cluster_scenario, run_headline_experiment,
)
from planning.objectives import (
    evaluate_plan, make_weighted_uncertainty_objective, objective_a0_trace_covariance, objective_c_direct_risk, risk_derived_weights,
    uniform_weights,
)
from planning.track_experiment import grid_all_objectives, summarise_plan, time_evaluations, track_objectives

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "npl_side_by_side.json"
RESIDUAL = NPLErrorParameters.hughes_2011_residual()
UM = 1e6


def with_model(scenario):
    return replace(scenario, tracker=LaserTracker(geometric_params=RESIDUAL))


def station_positions(evaluation):
    return np.array([p.position_m for p in evaluation.station_poses])


def summarise_ring_plan(evaluation, scenario):
    """`planning.track_experiment.summarise_plan` for plans of plain poses (the
    headline's hemisphere ring), which have no track coordinate."""
    report = _plan_report(evaluation, scenario)
    return {
        "stations_xyz_m": [p.position_m.tolist() for p in evaluation.station_poses],
        "sum_u_um": report.summed_characteristic_uncertainty_m * UM, "pfa": report.global_pfa, "pfr": report.global_pfr,
        "risk": report.global_pfa + report.global_pfr,
        "per_characteristic_u_um": {k: v * UM for k, v in report.per_characteristic_uncertainty_m.items()},
    }


def compare_plans(plan_noise, plan_full, scenario_noise, scenario_full, summarise=summarise_plan):
    """What changes for one objective: do the plans coincide; the plan chosen
    under each model scored under its own model; and the noise-only plan scored
    under the full model (what the full model says about the plan the
    noise-only one would have built)."""
    same = bool(plan_noise.station_poses and len(plan_noise.station_poses) == len(plan_full.station_poses)
                and np.allclose(station_positions(plan_noise), station_positions(plan_full), atol=1e-9))
    rescored = evaluate_plan(plan_noise.station_poses, scenario_full)
    return {
        "same_plan": same,
        "noise_only_plan_under_noise_only": summarise(plan_noise, scenario_noise),
        "noise_only_plan_under_full": summarise(rescored, scenario_full),
        "full_plan_under_full": summarise(plan_full, scenario_full),
    }


def per_point_contribution(u_noise, u_full):
    """Per characteristic: parameter-induced share of the variance, 1 - (u_noise/u_full)^2, and the
    parameter contribution in quadrature, sqrt(u_full^2 - u_noise^2), micrometres."""
    u_noise, u_full = np.asarray(u_noise, float), np.asarray(u_full, float)
    share = 1.0 - (u_noise / u_full) ** 2
    return {"u_noise_um": (u_noise * UM).tolist(), "u_full_um": (u_full * UM).tolist(), "variance_share": share.tolist(),
            "parameter_contribution_um": (np.sqrt(np.maximum(u_full**2 - u_noise**2, 0.0)) * UM).tolist(),
            "share_min_median_max": [float(share.min()), float(np.median(share)), float(share.max())]}


def main():
    results = {"model": "consider mode, hughes_2011_residual(), one tracker, per-point characteristic layer"}

    # -- B: the headline experiment --------------------------------------------------------------
    scenario_noise = default_two_cluster_scenario()
    scenario_full = with_model(scenario_noise)
    print("headline: noise-only ...", flush=True)
    start = time.time()
    headline_noise = run_headline_experiment(scenario_noise)
    t_noise = time.time() - start
    start = time.time()
    headline_full = run_headline_experiment(scenario_full)
    t_full = time.time() - start
    names = {"A0": "search_a0", "A1": "search_a1", "B": "search_b", "C": "search_c"}
    results["headline"] = {"seconds_noise_only": t_noise, "seconds_full": t_full, "objectives": {}}
    for name, attribute in names.items():
        results["headline"]["objectives"][name] = compare_plans(
            getattr(headline_noise, attribute).best, getattr(headline_full, attribute).best, scenario_noise, scenario_full,
            summarise_ring_plan)
        entry = results["headline"]["objectives"][name]
        print(f"  {name}: same plan {entry['same_plan']}; risk noise-only {entry['noise_only_plan_under_noise_only']['risk']:.5f} "
              f"-> full (same plan) {entry['noise_only_plan_under_full']['risk']:.5f} -> full's own plan {entry['full_plan_under_full']['risk']:.5f}", flush=True)

    # -- C: the station-track experiment, grid stage ---------------------------------------------------
    track = long_part_track()
    track_noise = long_part_two_cluster_scenario(track=track)
    track_full = with_model(track_noise)
    coarse = track_candidate_poses(track, 1.0, LONG_PART_GRID_HEIGHTS_M)
    print("track grid: noise-only ...", flush=True)
    grid_noise = grid_all_objectives(coarse, 3, track_noise, track_objectives(track_noise))
    print(f"  {grid_noise.seconds:.0f} s; full ...", flush=True)
    grid_full = grid_all_objectives(coarse, 3, track_full, track_objectives(track_full))
    results["station_track_grid_1.0m_3_stations"] = {
        "seconds_noise_only": grid_noise.seconds, "seconds_full": grid_full.seconds, "n_feasible": [grid_noise.n_feasible, grid_full.n_feasible],
        "objectives": {}}
    for name in grid_noise.best:
        entry = compare_plans(grid_noise.best[name], grid_full.best[name], track_noise, track_full)
        results["station_track_grid_1.0m_3_stations"]["objectives"][name] = entry
        print(f"  {name}: same plan {entry['same_plan']}; risk noise-only {entry['noise_only_plan_under_noise_only']['risk']:.4f} "
              f"-> full (same plan) {entry['noise_only_plan_under_full']['risk']:.4f} -> full's own plan {entry['full_plan_under_full']['risk']:.4f}", flush=True)

    # -- A: per-point parameter contribution on the existing scenes -----------------------------------------
    contribution = {}
    plan_h = headline_noise.search_a1.best
    contribution["headline_two_cluster, A1 plan, 2 stations"] = per_point_contribution(
        [evaluate_plan(plan_h.station_poses, scenario_noise).per_characteristic_uncertainty_m[c.name] for c in scenario_noise.characteristics],
        [evaluate_plan(plan_h.station_poses, scenario_full).per_characteristic_uncertainty_m[c.name] for c in scenario_full.characteristics])
    plan_t = grid_noise.best["A1"]
    contribution["long_part_on_track (anchor frame), A1 plan, 3 added stations"] = per_point_contribution(
        [evaluate_plan(plan_t.station_poses, track_noise).per_characteristic_uncertainty_m[c.name] for c in track_noise.characteristics],
        [evaluate_plan(plan_t.station_poses, track_full).per_characteristic_uncertainty_m[c.name] for c in track_full.characteristics])
    cell_noise = factory_cell_scenario(track=track)
    cell_full = with_model(cell_noise)
    cell_layouts = json.loads((RESULTS_PATH.parent / "cell_results_datum.json").read_text())["layouts"]
    for key in ("C|3", "C|4"):
        stations = cell_layouts[key]["stations"]
        poses_noise = layout_poses(stations, cell_noise, track)
        poses_full = layout_poses(stations, cell_full, track)
        per_point_n = CellLayoutScorer(cell_noise).score(poses_noise)
        per_point_f = CellLayoutScorer(cell_full).score(poses_full)
        datum_n = DatumCellScorer(cell_noise).score(poses_noise)
        datum_f = DatumCellScorer(cell_full).score(poses_full)
        if per_point_n.feasible and datum_n.feasible:
            contribution[f"factory_cell, risk-planned layout {key}, per-point in the anchor frame"] = per_point_contribution(per_point_n.u_m, per_point_f.u_m)
            contribution[f"factory_cell, risk-planned layout {key}, datum-referenced (joint covariance)"] = per_point_contribution(datum_n.u_m, datum_f.u_m)
    results["per_point_parameter_contribution"] = contribution
    for label, c in contribution.items():
        print(f"  {label}: parameter share of variance min/median/max {[round(v, 4) for v in c['share_min_median_max']]}", flush=True)

    # -- D: timing --------------------------------------------------------------------------------------
    timing = {}
    # the headline's grid: 2-station plans from the 24-pose ring
    from planning.candidates import hemisphere_candidate_poses
    from planning.experiment import DEFAULT_AZIMUTHS_DEG, DEFAULT_ELEVATIONS_DEG

    def headline_timing(scenario):
        centre = scenario.target_points_m.mean(axis=0)
        ring = hemisphere_candidate_poses(centre, 4.0, DEFAULT_AZIMUTHS_DEG, DEFAULT_ELEVATIONS_DEG)
        objectives = {"A0": objective_a0_trace_covariance, "A1": make_weighted_uncertainty_objective(uniform_weights(scenario)),
                      "B": make_weighted_uncertainty_objective(risk_derived_weights(scenario, 8e-6)), "C": objective_c_direct_risk}
        rng = np.random.default_rng(0)
        combos = [tuple(sorted(rng.choice(len(ring), 2, replace=False))) for _ in range(200)]
        start = time.time()
        evaluations = [evaluate_plan([ring[i] for i in combo], scenario) for combo in combos]
        evaluate_ms = 1e3 * (time.time() - start) / len(combos)
        evaluations = [e for e in evaluations if e is not None]
        out = {"evaluate_plan_ms": evaluate_ms, "n": len(evaluations)}
        for name, fn in objectives.items():
            start = time.time()
            for e in evaluations:
                fn(e, scenario)
            out[f"{name}_total_ms"] = evaluate_ms + 1e3 * (time.time() - start) / len(evaluations)
        return out

    timing["headline_evaluate_plan_path"] = {"noise_only": headline_timing(scenario_noise), "full": headline_timing(scenario_full)}
    t_n = time_evaluations(track_noise, coarse, track_objectives(track_noise), 3, 200)
    t_f = time_evaluations(track_full, coarse, track_objectives(track_full), 3, 200)
    timing["track_evaluate_plan_path"] = {"noise_only": t_n, "full": t_f}

    # the fast design path the cell experiments use (200 random 4-station layouts)
    rng = np.random.default_rng(1)
    layouts = [[(float(rng.uniform(0, track.perimeter_m)), float(rng.uniform(0.8, 3.0))) for _ in range(4)] for _ in range(400)]
    fast = {}
    for label, cell in (("noise_only", cell_noise), ("full", cell_full)):
        for scorer_name, scorer in (("PartFrameScorer-like DatumCellScorer", DatumCellScorer(cell)), ("CellLayoutScorer (per point, anchor frame)", CellLayoutScorer(cell))):
            poses = [layout_poses(s, cell, track) for s in layouts]
            feasible = [p for p in poses if scorer.score(p).feasible][:200]
            start = time.time()
            for p in feasible:
                scorer.score(p)
            fast.setdefault(scorer_name, {})[label] = {"ms_per_score": 1e3 * (time.time() - start) / len(feasible), "n": len(feasible)}
    timing["fast_design_path_cell_scorers"] = fast
    results["timing"] = timing
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))
    print("timing:", json.dumps(timing, indent=1, default=float)[:3000])
    print("wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
