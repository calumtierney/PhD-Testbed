"""Does the factory-cell result survive changes to what was assumed?

    python -m viz.run_cell_sensitivity        # about 40 min on 4 cores

The main experiment (`viz.run_cell_experiment --frame datum`) found that, with a
reference network and uncertainty referenced to the part's datums, the plan that
minimises decision risk (objective C) carries about a third of the risk of the
plan that minimises summed uncertainty along the tolerances (A1) at the same
number of stations. That is one configuration. This repeats the comparison of
A0, A1 and C at 2-5 stations with one assumption changed at a time:

  floor_1um, floor_2um, floor_4um   an isotropic uncertainty no placement can
        reduce, added to every point (`LaserTracker.systematic_std_m`). The
        sensor model has none; CLAUDE.md §4c records an SMR nest repeatability
        of about 4 um as a *distribution width*, so the standard deviation it
        implies is not known and a range is run.
  tol_25um, tol_100um               the tolerance, priors scaled so Cpk is kept.
  centred_marginal                  the marginal cluster centred with a wide
        spread (mean 0, std 12.5 um: the same Cpk 1.33 as the offset one).
  anchor_frame                      is not here: see results/cell_results.json.

For each case and station count it records each objective's best layout, its
uncertainties and its total risk (simple acceptance, process known: objective
C's own measure), and the station count each objective needs to reach a set of
risk targets. Writes results/cell_sensitivity.json.
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
from planning.cell import DatumCellScorer, factory_cell_scenario, layout_poses, search_cell_layout, standard_objectives
from planning.experiment import LONG_PART_GA_HEIGHT_BOUNDS_M, long_part_track, with_marginal_prior
from planning.track_experiment import with_tolerance
from risk.jcgm106 import ProcessPrior

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "cell_sensitivity.json"
COUNTS = (2, 3, 4, 5)
NAMES = ("A0", "A1", "C")
DE = dict(seed=0, maxiter=150, popsize=25, tol=1e-8, workers=1)
CASES = ("baseline", "floor_1um", "floor_2um", "floor_4um", "tol_25um", "tol_100um", "centred_marginal")
_CACHE = {}


def build(case):
    if case not in _CACHE:
        track = long_part_track()
        scenario = factory_cell_scenario(track=track)
        if case.startswith("floor_"):
            scenario = replace(scenario, tracker=LaserTracker(systematic_std_m=float(case[6:-2]) * 1e-6))
        elif case.startswith("tol_"):
            scenario = with_tolerance(scenario, float(case[4:-2]) * 1e-6)
        elif case == "centred_marginal":
            scenario = with_marginal_prior(scenario, ProcessPrior(mean_m=0.0, std_m=12.5e-6))
        _CACHE[case] = (track, scenario, standard_objectives(scenario))
    return _CACHE[case]


def run_job(job):
    case, name, k, seeds = job
    track, scenario, objectives = build(case)
    start = time.time()
    result = search_cell_layout(DatumCellScorer(scenario), objectives[name], track, k, LONG_PART_GA_HEIGHT_BOUNDS_M,
                                seed_stations=seeds, **DE)
    return case, name, k, [list(s) for s in result.stations], result.value, time.time() - start


def describe(case, stations):
    track, scenario, objectives = build(case)
    score = DatumCellScorer(scenario).score(layout_poses(stations, scenario, track))
    return {"stations": stations, "u_um": (score.u_m * 1e6).tolist(), "n_shots": score.n_shots,
            "objective_values": {n: float(objectives[n](score)) for n in NAMES}}


def main():
    layouts = {}
    with Pool(4) as pool:
        jobs = [(case, name, k, None) for case in CASES for k in COUNTS for name in NAMES]
        for case, name, k, stations, value, seconds in pool.imap_unordered(run_job, jobs):
            layouts[(case, name, k)] = stations
            print(f"[{case} {name} k={k}] {value:.4g} ({seconds:.0f} s)", flush=True)
        # one seeded re-run wherever another objective's layout beats an objective under its own measure
        again = []
        for case in CASES:
            for k in COUNTS:
                known = [layouts[(case, n, k)] for n in NAMES if layouts.get((case, n, k))]
                values = [describe(case, s)["objective_values"] for s in known]
                for name in NAMES:
                    own = describe(case, layouts[(case, name, k)])["objective_values"][name]
                    if min(v[name] for v in values) < own * (1 - 1e-3):
                        again.append((case, name, k, known))
        for case, name, k, stations, value, seconds in pool.imap_unordered(run_job, again):
            candidates = [stations] + [layouts[(case, n, k)] for n in NAMES]
            layouts[(case, name, k)] = min(candidates, key=lambda s: describe(case, s)["objective_values"][name])
            print(f"[{case} {name} k={k}] re-run seeded ({seconds:.0f} s)", flush=True)

    results = {"de": DE, "counts": list(COUNTS), "cases": {}}
    for case in CASES:
        entry = {"layouts": {f"{name}|{k}": describe(case, layouts[(case, name, k)]) for name in NAMES for k in COUNTS}}
        risk = {name: {k: entry["layouts"][f"{name}|{k}"]["objective_values"]["C"] for k in COUNTS} for name in NAMES}
        entry["risk_by_objective_and_count"] = risk
        entry["risk_ratio_to_C"] = {name: {k: risk[name][k] / risk["C"][k] for k in COUNTS} for name in ("A0", "A1")}
        targets = sorted({risk["C"][k] for k in COUNTS[:-1]}, reverse=True)  # what C achieves with 2, 3 and 4 stations
        entry["stations_to_reach"] = {
            f"{t:.4g}": {name: next((k for k in COUNTS if risk[name][k] <= t * (1 + 1e-9)), None) for name in NAMES} for t in targets
        }
        results["cases"][case] = entry
        print(case, "risk ratio A1/C:", {k: round(v, 2) for k, v in entry["risk_ratio_to_C"]["A1"].items()},
              "stations to reach:", entry["stations_to_reach"], flush=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", RESULTS_PATH, flush=True)


if __name__ == "__main__":
    main()
