"""Run the long-part-on-a-station-track experiment and write
results/station_track_results.json (re-written after every stage, so a long run
that is interrupted keeps what it finished).

    python -m viz.run_station_track_experiment

Stages, in order (each one is reported in docs/station_track_experiment.md):

  timing       ms per plan evaluation and per objective call on feasible plans
               of the scene (200 plans), which fixes the grid spacing;
  grid         exhaustive search over the track candidates, 3 added stations,
               every feasible plan scored under all four objectives. 1.0 m
               spacing first; 0.5 m as well if the extrapolated time is under
               30 minutes. (1 and 2 added stations are infeasible on this
               scene -- no combination covers every target and ties every
               station to the anchor -- so 3 is the smallest search there is.)
  ga           the (s, height) GA per objective at 3 and at 4 added stations,
               the same settings and seed for every objective, checked against
               the grid (3 stations) and against each other's winners;
  sensitivity  the 3-station GA again with the anchor at the opposite (+x) tip;
  iso-risk     the smallest station count whose best plan reaches a target risk.

The pinning of BLAS threads below must come before numpy is imported:
`workers=-1` starts one process per core and each would otherwise start its own
BLAS thread pool and oversubscribe the machine (see
`planning.search.genetic_algorithm_station_placement`).
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import time
from pathlib import Path

import numpy as np

from planning.candidates import track_candidate_poses
from planning.experiment import (
    LONG_PART_GA_HEIGHT_BOUNDS_M,
    LONG_PART_GRID_HEIGHTS_M,
    long_part_track,
    long_part_two_cluster_scenario,
    total_expected_risk,
)
from planning.track_experiment import (
    OBJECTIVE_NAMES,
    cross_score,
    grid_all_objectives,
    run_track_ga,
    summarise_plan,
    time_evaluations,
    track_objectives,
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "station_track_results.json"
DE_KWARGS = dict(seed=0, maxiter=120, popsize=25, tol=1e-8, workers=-1, updating="deferred")
GRID_TIME_BUDGET_S = 30 * 60
GA_VS_GRID_TOLERANCE = 1.05  # the GA's value may be at most 5% above the grid's (it can be, and is, lower)


def save(results):
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))


def ga_stage(scenario, track, objectives, n_stations, label, results):
    """GA for every objective at `n_stations`; if some objective's own GA is
    beaten, under its own metric, by another objective's winner (it got stuck:
    a GA finds a good plan, not provably the best), re-run that objective
    seeded with every winner so far and say so."""
    print(f"[{label}] GA, {n_stations} added stations", flush=True)
    start = time.time()
    runs = run_track_ga(scenario, track, n_stations, objectives, LONG_PART_GA_HEIGHT_BOUNDS_M, DE_KWARGS)
    reseeded = []
    for _ in range(2):
        winners = {n: r.best for n, r in runs.items() if r.best is not None}
        table = cross_score(winners, scenario, objectives)
        stuck = [o for o in objectives if o in winners and min(table[w][o] for w in winners) < table[o][o] * (1 - 1e-9)]
        if not stuck:
            break
        seeds = [[(p.s_m, p.height_m) for p in evaluation.station_poses] for evaluation in winners.values()]
        for name in stuck:
            print(f"[{label}]   {name}'s GA was beaten under its own metric by another winner: re-running seeded", flush=True)
            runs[name] = run_track_ga(scenario, track, n_stations, {name: objectives[name]}, LONG_PART_GA_HEIGHT_BOUNDS_M,
                                      DE_KWARGS, seed_stations={name: seeds})[name]
            reseeded.append(name)
    winners = {n: r.best for n, r in runs.items() if r.best is not None}
    out = {
        "n_stations": n_stations,
        "seconds": time.time() - start,
        "reseeded_objectives": sorted(set(reseeded)),
        "plans": {n: summarise_plan(e, scenario) for n, e in winners.items()},
        "ga_diagnostics": {n: {"nfev": r.n_function_evals, "generations": r.n_generations, "converged": r.converged,
                               "objective_value": r.best_objective_value} for n, r in runs.items()},
        "cross_score": cross_score(winners, scenario, objectives),
    }
    results[label] = out
    save(results)
    for name, plan in out["plans"].items():
        print(f"[{label}]   {name}: stations {[(round(s, 2), round(h, 2), seg) for s, h, seg in plan['stations']]} "
              f"sum u {plan['sum_u_um']:.2f} um  PFA {plan['pfa']:.4f} PFR {plan['pfr']:.4f} risk {plan['risk']:.4f}", flush=True)
    return winners


def main():
    track = long_part_track()
    scenario = long_part_two_cluster_scenario(track=track)
    objectives = track_objectives(scenario)
    results = {
        "scene": {
            "perimeter_m": track.perimeter_m, "loop_radius_m": track.loop_radius_m, "n_targets": len(scenario.target_points_m),
            "anchor_s_m": track.tip_minus_x_s_m, "anchor_height_m": float(scenario.anchor_pose.position_m[2]),
            "grid_heights_m": list(LONG_PART_GRID_HEIGHTS_M), "ga_height_bounds_m": list(LONG_PART_GA_HEIGHT_BOUNDS_M),
            "de_kwargs": {k: v for k, v in DE_KWARGS.items()},
        }
    }

    # -- timing, before choosing the spacing ---------------------------------------------------
    coarse = track_candidate_poses(track, 1.0, LONG_PART_GRID_HEIGHTS_M)
    timing = time_evaluations(scenario, coarse, objectives, n_stations=3, n_plans=200)
    results["timing"] = timing
    save(results)
    print("timing over", timing["n_plans"], "feasible 3-station plans: evaluate_plan %.1f ms, objectives ms %s"
          % (timing["evaluate_plan_ms"], {k: round(v, 3) for k, v in timing["objective_ms"].items()}), flush=True)

    # -- grid, 3 added stations --------------------------------------------------------------------
    grid_results = {}
    spacings = [1.0]
    grid_1m = grid_all_objectives(coarse, 3, scenario, objectives)
    grid_results["1.0"] = grid_1m
    est_half_m = grid_1m.seconds * 8.0  # combinations scale as n^3
    results["grid_spacing_decision"] = {
        "time_at_1.0m_s": grid_1m.seconds, "extrapolated_time_at_0.5m_s": est_half_m, "budget_s": GRID_TIME_BUDGET_S,
        "chosen_spacing_m": 0.5 if est_half_m < GRID_TIME_BUDGET_S else 1.0,
    }
    if est_half_m < GRID_TIME_BUDGET_S:
        fine = track_candidate_poses(track, 0.5, LONG_PART_GRID_HEIGHTS_M)
        grid_results["0.5"] = grid_all_objectives(fine, 3, scenario, objectives)
        spacings.append(0.5)
    chosen = results["grid_spacing_decision"]["chosen_spacing_m"]
    results["grid"] = {}
    for key, grid in grid_results.items():
        results["grid"][key] = {
            "n_combinations": grid.n_combinations, "n_feasible": grid.n_feasible, "seconds": grid.seconds,
            "best_value": grid.best_value,
            "plans": {n: summarise_plan(e, scenario) for n, e in grid.best.items() if e is not None},
            "cross_score": cross_score({n: e for n, e in grid.best.items() if e is not None}, scenario, objectives),
        }
        print(f"[grid {key} m] {grid.n_feasible} feasible of {grid.n_combinations} combinations, {grid.seconds:.0f} s", flush=True)
        for name, plan in results["grid"][key]["plans"].items():
            print(f"   {name}: {[(round(s, 2), round(h, 2), seg) for s, h, seg in plan['stations']]} risk {plan['risk']:.4f}", flush=True)
    save(results)
    grid_best_value = grid_results[str(chosen)].best_value

    # -- GA at 3 and 4 added stations; cross-check the 3-station GA against the grid ------------------
    ga3 = ga_stage(scenario, track, objectives, 3, "ga_3", results)
    check = {}
    for name in OBJECTIVE_NAMES:
        ga_value, grid_value = results["ga_3"]["ga_diagnostics"][name]["objective_value"], grid_best_value[name]
        check[name] = {"ga_value": ga_value, "grid_value": grid_value, "ratio": ga_value / grid_value,
                       "within_tolerance": ga_value <= GA_VS_GRID_TOLERANCE * grid_value}
        print(f"[cross-check] {name}: GA {ga_value:.6g} vs grid {grid_value:.6g} (ratio {ga_value / grid_value:.3f}, "
              f"{'ok' if check[name]['within_tolerance'] else 'OUTSIDE TOLERANCE'})", flush=True)
    results["ga_vs_grid_3_stations"] = {"tolerance": GA_VS_GRID_TOLERANCE, "grid_spacing_m": chosen, "objectives": check}
    save(results)
    ga4 = ga_stage(scenario, track, objectives, 4, "ga_4", results)

    # -- iso-risk station counts: smallest n whose best plan reaches a target risk ---------------------------
    risks = {"3": {n: total_expected_risk(e, scenario) for n, e in ga3.items()},
             "4": {n: total_expected_risk(e, scenario) for n, e in ga4.items()}}
    best4 = min(risks["4"].values())
    # targets rounded *up* to 6 decimals, so that a target equal to a plan's own risk is reachable by it
    targets = [float(np.ceil(best4 * f * 1e6) / 1e6) for f in (1.0, 1.02, 1.05, 1.10, 1.25)] + [float(np.ceil(risks["3"]["C"] * 1e6) / 1e6)]
    counts = {}
    for target in sorted(set(targets)):
        counts[f"{target:g}"] = {
            name: next((int(n) for n in ("3", "4") if risks[n].get(name, np.inf) <= target * (1 + 1e-12)), None)
            for name in OBJECTIVE_NAMES
        }
    results["iso_risk"] = {"best_total_risk_by_station_count": risks, "station_count_reaching_target": counts,
                           "note": "1 and 2 added stations are infeasible on this scene; counts come from the GA winners at 3 and 4."}
    save(results)
    print("[iso-risk]", json.dumps(results["iso_risk"], default=float), flush=True)

    # -- anchor sensitivity: the anchor at the opposite (+x) tip ---------------------------------------------
    other = long_part_two_cluster_scenario(anchor_s_m=track.tip_plus_x_s_m, track=track)
    ga_stage(other, track, track_objectives(other), 3, "anchor_plus_x_tip_ga_3", results)
    results["anchor_sensitivity"] = {"anchor_s_m": track.tip_plus_x_s_m}
    save(results)
    print("done; wrote", RESULTS_PATH, flush=True)


if __name__ == "__main__":
    main()
