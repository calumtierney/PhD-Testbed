"""When does risk weighting change a plan? A one-condition-at-a-time sweep of the
headline ring scene.

    python -m viz.run_conditions_sweep

Scene: `planning.experiment.default_two_cluster_scenario` (two clusters of three
profile characteristics, high-capability at y ~ +2 m, marginal at y ~ -2 m), the
24-pose ring, exhaustive grid search at 2 and 3 added stations, every feasible
plan scored under all four objectives (A0 trace, A1 uniform projected, B
risk-weighted at 8 um, C direct risk; `planning.track_experiment.
grid_all_objectives`, exact risk throughout). Each sweep moves one condition
from the baseline and keeps the rest:

  tolerance      +-25 ... +-250 um, priors scaled so Cpk is unchanged
                 (`with_tolerance`): only uncertainty-to-tolerance changes;
  capability     the marginal cluster's mean 20 ... 44 um at std 3 um, +-50 um
                 (Cpk 3.33 ... 0.67): how different the two clusters are;
  separation     how far apart the two clusters are across the ring (y = +-d):
                 how much a station that serves one costs the other;
  systematic     an isotropic per-point floor added to every target
                 (`LaserTracker.systematic_std_m`, 0 ... 20 um): uncertainty
                 that placement cannot reduce.

For every point: each objective's winning plan, its exact total risk, how far
above C's it is, and whether its stations differ from C's. Writes
results/conditions_sweep.json. The baseline (tolerance +-50 um, mean 38 um,
d = 2 m, no systematic floor) is the headline experiment itself.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from instruments.laser_tracker import LaserTracker
from planning.candidates import hemisphere_candidate_poses
from planning.experiment import DEFAULT_AZIMUTHS_DEG, DEFAULT_ELEVATIONS_DEG, default_two_cluster_scenario, with_marginal_prior
from planning.objectives import objective_c_direct_risk
from planning.track_experiment import OBJECTIVE_NAMES, grid_all_objectives, track_objectives, with_tolerance
from risk.jcgm106 import ProcessPrior

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "conditions_sweep.json"
STATION_COUNTS = (2, 3)
SWEEPS = {
    "tolerance_um": [25, 35, 50, 75, 100, 150, 250],
    "marginal_mean_um": [20, 30, 35, 38, 41, 44],
    "separation_m": [0.25, 0.5, 1.0, 2.0, 3.0],
    "systematic_um": [0, 2, 5, 10, 20],
}
BASELINE = {"tolerance_um": 50, "marginal_mean_um": 38, "separation_m": 2.0, "systematic_um": 0}


def with_separation(scenario, d_m):
    """Move the two clusters to y = +-d (each keeps its internal layout)."""
    points = scenario.target_points_m.copy()
    for i in range(len(points)):
        sign = 1.0 if i < 3 else -1.0
        points[i, 1] = sign * (d_m + (abs(points[i, 1]) - 2.0))
    return replace(scenario, target_points_m=points)


def build(axis, value):
    scenario = default_two_cluster_scenario()
    if axis == "separation_m":
        scenario = with_separation(scenario, value)
    if axis == "systematic_um":
        scenario = replace(scenario, tracker=LaserTracker(systematic_std_m=value * 1e-6))
    if axis == "marginal_mean_um":
        scenario = with_marginal_prior(scenario, ProcessPrior(mean_m=value * 1e-6, std_m=3e-6))
    if axis == "tolerance_um":
        scenario = with_tolerance(scenario, value * 1e-6)
    return scenario


def run_point(job):
    axis, value, n_stations = job
    start = time.time()
    scenario = build(axis, value)
    candidates = hemisphere_candidate_poses(scenario.target_points_m.mean(axis=0), 4.0, DEFAULT_AZIMUTHS_DEG, DEFAULT_ELEVATIONS_DEG)
    objectives = track_objectives(scenario)
    grid = grid_all_objectives(candidates, n_stations, scenario, objectives)
    index = {id(p): i for i, p in enumerate(candidates)}
    plans = {}
    for name in OBJECTIVE_NAMES:
        evaluation = grid.best[name]
        plans[name] = {
            "candidates": sorted(index[id(p)] for p in evaluation.station_poses),
            "risk": float(objective_c_direct_risk(evaluation, scenario)),
            "sum_u_um": 1e6 * float(sum(evaluation.per_characteristic_uncertainty_m.values())),
            "u_um": {k: v * 1e6 for k, v in evaluation.per_characteristic_uncertainty_m.items()},
        }
    c = plans["C"]
    for name in OBJECTIVE_NAMES:
        plans[name]["percent_above_C"] = 100.0 * (plans[name]["risk"] / c["risk"] - 1.0) if c["risk"] > 0 else 0.0
        plans[name]["same_stations_as_C"] = plans[name]["candidates"] == c["candidates"]
    return {"axis": axis, "value": value, "n_stations": n_stations, "seconds": time.time() - start,
            "n_combinations": grid.n_combinations, "plans": plans}


def main():
    jobs = [(axis, value, n) for n in STATION_COUNTS for axis, values in SWEEPS.items() for value in values]
    results = {"baseline": BASELINE, "sweeps": SWEEPS, "station_counts": list(STATION_COUNTS), "points": []}
    with Pool(4) as pool:
        for point in pool.imap_unordered(run_point, jobs):
            results["points"].append(point)
            RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))
            p = point["plans"]
            print(f"[{point['axis']} = {point['value']}, {point['n_stations']} st.] C risk {p['C']['risk']:.4g}; "
                  + "; ".join(f"{n} +{p[n]['percent_above_C']:.1f}%{'' if p[n]['same_stations_as_C'] else ' (moved)'}"
                              for n in ("A0", "A1", "B"))
                  + f"  ({point['seconds']:.0f} s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
