"""Station count on the station track: risk-based planning against the plans
industry plausibly uses now (uncertainty along the tolerance, and the TUR rule),
each choosing its own number of stations, across tolerances.

    python -m viz.run_station_track_station_count

The fixed-count experiment (`viz.run_station_track_experiment`) gives every
objective the same stations, so it cannot say anything about measurement time.
Here each plan decides how many stations to use (3-6 added; 1 and 2 are
infeasible on this scene, more than 6 is not searched), every characteristic is
measured, and the station count is the result. The design mirrors the wing-box
experiment (`planning.inspection`, docs/wingbox_experiment.md):

  A1-knee  minimise summed uncertainty along each tolerance (objective A1, the
           tolerance-aware uncertainty planner); a trace or sum never says when
           to stop, so the count is the knee of its station-count curve: the
           smallest K where one more station improves A1 by less than 10%
           (also reported at 20% and 5%). Tolerance-blind, so the same layout
           and count at every tolerance.
  A0-knee  the same with the trace of the covariance (the literature criterion).
  T        the TUR 4:1 rule: every characteristic's u <= tolerance half-width / 8
           (Z540.3: half-width over U = 2u). The layout minimising the worst u at
           each count is searched once (it does not depend on the tolerance).
  R        the fewest stations whose risk-optimal layout (objective C) is no
           riskier than a reference: R-A1 = the A1-knee plan's own realised risk;
           R-TUR = what the TUR rule guarantees (every characteristic exactly at
           the limit); R-T = the TUR plan's realised risk, where T is achievable.

Process priors scale with the tolerance so Cpk keeps its meaning
(`planning.track_experiment.with_tolerance`). At +-50 um the A0/A1/C layouts are
the existing GA winners (results/station_track_results.json, `ga_3`-`ga_6`); at
other tolerances C is re-searched with the same GA, seeded with every known
layout at that count, using tabulated risk inside the search
(`TabulatedRiskObjective`, within 3% of the integral) and exact risk for every
reported number. Writes `station_count` into results/station_track_results.json.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"  # before numpy is imported, or every worker oversubscribes the cores

import json
import time
from pathlib import Path

import numpy as np

from planning.experiment import LONG_PART_GA_HEIGHT_BOUNDS_M, long_part_track, long_part_two_cluster_scenario
from planning.inspection import knee_station_count
from planning.objectives import evaluate_plan, objective_c_direct_risk
from planning.search import TrackStationCost, track_genetic_algorithm_station_placement
from planning.track_experiment import (
    TabulatedRiskObjective,
    objective_max_projected_uncertainty,
    risk_at_uniform_uncertainty,
    stations_of,
    summarise_plan,
    track_objectives,
    with_tolerance,
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "station_track_results.json"
COUNTS = (3, 4, 5, 6)
TOLERANCES_UM = (25.0, 50.0, 100.0, 250.0)
TUR = 4.0
KNEE_GAINS = (0.20, 0.10, 0.05)
DE_SEEDED = dict(seed=0, maxiter=60, popsize=25, tol=1e-8, workers=-1, updating="deferred")


def layout_from(stations, scenario, track):
    flat = np.array([[s, h] for s, h, *_ in stations], dtype=float).reshape(-1)
    return TrackStationCost(scenario, None, track, len(stations), scenario.target_points_m.mean(axis=0)).station_poses(flat)


def evaluate(stations, scenario, track):
    return evaluate_plan(layout_from(stations, scenario, track), scenario)


def save(results):
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))


def main():
    results = json.loads(RESULTS_PATH.read_text())
    track = long_part_track()
    base = long_part_two_cluster_scenario(track=track)
    objectives = track_objectives(base)
    known = {k: {name: [tuple(s) for s in results[f"ga_{k}"]["plans"][name]["stations"]] for name in ("A0", "A1", "B", "C")}
             for k in COUNTS}
    out = results.setdefault("station_count", {})
    out["settings"] = {"counts": list(COUNTS), "tur": TUR, "knee_gains": list(KNEE_GAINS), "de_seeded": DE_SEEDED,
                       "tolerances_um": list(TOLERANCES_UM)}

    # -- tolerance-blind plans: A0, A1 (existing winners) and T's minimax-u layouts -------------
    a_values = {name: {k: float(objectives[name](evaluate(known[k][name], base, track), base)) for k in COUNTS}
                for name in ("A0", "A1")}
    knees = {name: {f"{g:g}": knee_station_count(a_values[name], g) for g in KNEE_GAINS} for name in ("A0", "A1")}
    out["uncertainty_plans"] = {"criterion_by_count": a_values, "knee_station_count": knees}
    print("[knee] A0/A1 criterion by count:", a_values, "knees:", knees, flush=True)

    t_layouts = out.get("tur_layouts", {})
    for k in COUNTS:
        if str(k) in t_layouts:
            continue
        start = time.time()
        seeds = [[(s, h) for s, h, *_ in known[k][name]] for name in ("A0", "A1", "B", "C")]
        ga = track_genetic_algorithm_station_placement(
            base, track, k, objective_max_projected_uncertainty, height_bounds_m=LONG_PART_GA_HEIGHT_BOUNDS_M,
            seed_stations=seeds, **DE_SEEDED)
        t_layouts[str(k)] = {"stations": [list(s) for s in stations_of(ga.best)], "max_u_um": ga.best_objective_value * 1e6,
                             "seconds": time.time() - start}
        out["tur_layouts"] = t_layouts
        save(results)
        print(f"[T] {k} stations: worst u {ga.best_objective_value * 1e6:.2f} um ({time.time() - start:.0f} s)", flush=True)

    # -- per tolerance: realised risk of each plan, and the fewest stations R needs --------------
    by_tol = out.setdefault("by_tolerance", {})
    for tol_um in TOLERANCES_UM:
        key = f"{tol_um:g}"
        tol = tol_um * 1e-6
        scenario = with_tolerance(base, tol)
        table = TabulatedRiskObjective.build(scenario)
        exact_risk = lambda stations: float(objective_c_direct_risk(evaluate(stations, scenario, track), scenario))
        entry = by_tol.get(key, {})
        u_limit = tol / (2.0 * TUR)
        entry["tur_limit_um"] = u_limit * 1e6

        # plans that do not look at risk: their station count and realised risk here
        k_a1, k_a0 = knees["A1"]["0.1"], knees["A0"]["0.1"]
        entry["A1_knee"] = {"stations": k_a1, "risk": exact_risk(known[k_a1]["A1"]),
                            "risk_by_knee": {g: exact_risk(known[kk]["A1"]) for g, kk in knees["A1"].items()}}
        entry["A0_knee"] = {"stations": k_a0, "risk": exact_risk(known[k_a0]["A0"])}
        k_t = next((k for k in COUNTS if t_layouts[str(k)]["max_u_um"] * 1e-6 <= u_limit), None)
        entry["T"] = {"stations": k_t, "risk": exact_risk(t_layouts[str(k_t)]["stations"]) if k_t else None,
                      "best_worst_u_um": min(t_layouts[str(k)]["max_u_um"] for k in COUNTS)}
        tur_guarantee = risk_at_uniform_uncertainty(scenario, u_limit)
        limits = {"R-A1": entry["A1_knee"]["risk"], "R-A0": entry["A0_knee"]["risk"], "R-TUR": tur_guarantee}
        if entry["T"]["risk"] is not None:
            limits["R-T"] = entry["T"]["risk"]
        entry["limits"] = limits

        # the risk-optimal layout at each count (searched only as far as needed)
        c_by_k = entry.get("C_by_count", {})
        r_counts = {name: None for name in limits}
        for k in COUNTS:
            if str(k) not in c_by_k:
                start = time.time()
                candidates = [known[k][n] for n in ("A0", "A1", "B", "C")] + [[tuple(s) for s in t_layouts[str(k)]["stations"]]]
                if tol_um == 50.0:
                    best = known[k]["C"]  # the exact-C GA winner already searched at +-50 um
                else:
                    ga = track_genetic_algorithm_station_placement(
                        scenario, track, k, table, height_bounds_m=LONG_PART_GA_HEIGHT_BOUNDS_M,
                        seed_stations=[[(s, h) for s, h, *_ in c] for c in candidates], **DE_SEEDED)
                    best = [tuple(s) for s in stations_of(ga.best)]
                # cross-check: the winner must not be beaten, in exact risk, by any known layout at this count
                scored = sorted((exact_risk(c), i, c) for i, c in enumerate([best] + candidates))
                c_by_k[str(k)] = {"stations": [list(s) for s in scored[0][2]], "risk": scored[0][0],
                                  "search_winner_was_best": scored[0][1] == 0, "seconds": time.time() - start,
                                  "summary": summarise_plan(evaluate(scored[0][2], scenario, track), scenario)}
                entry["C_by_count"] = c_by_k
                by_tol[key] = entry
                save(results)
                print(f"[+-{key} um] C at {k} stations: risk {scored[0][0]:.5g} "
                      f"({'GA winner' if scored[0][1] == 0 else 'a known layout beat the GA'}, {time.time() - start:.0f} s)", flush=True)
            for name, limit in limits.items():
                if r_counts[name] is None and c_by_k[str(k)]["risk"] <= limit * (1 + 1e-9):
                    r_counts[name] = k
            if all(v is not None for v in r_counts.values()):
                break
        entry["R_station_count"] = r_counts
        by_tol[key] = entry
        save(results)
        print(f"[+-{key} um] TUR limit u <= {u_limit * 1e6:.2f} um; A1-knee {k_a1} st. risk {entry['A1_knee']['risk']:.4g}; "
              f"A0-knee {k_a0} st. risk {entry['A0_knee']['risk']:.4g}; T {k_t} st.; R counts {r_counts}; "
              f"TUR-guaranteed risk {tur_guarantee:.4g}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
