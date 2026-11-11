"""Extend the station-track iso-risk comparison to 5 and 6 added stations.

    python -m viz.run_station_track_iso_risk

`viz.run_station_track_experiment` stops at 4 added stations, where A0 (the
field's criterion) reached 1.2475 expected wrong decisions and A1/B/C 1.0511.
This runs the identical GA (same settings, seed, cross-check and seeded re-run
rule, `ga_stage`) at 5 and 6 added stations for all four objectives, loads and
re-writes results/station_track_results.json, and adds `iso_risk_extended`: for
a ladder of target risks, the smallest station count (3-6) whose best plan
reaches it, per objective. One and two added stations are infeasible on this
scene. BLAS threads are pinned to one before numpy is imported (see the main runner).
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"  # before numpy is imported, or every worker oversubscribes the cores

import json

import numpy as np

from planning.experiment import long_part_track, long_part_two_cluster_scenario, total_expected_risk
from planning.track_experiment import OBJECTIVE_NAMES, track_objectives
from viz.run_station_track_experiment import RESULTS_PATH, ga_stage, save


def main():
    results = json.loads(RESULTS_PATH.read_text())
    track = long_part_track()
    scenario = long_part_two_cluster_scenario(track=track)
    objectives = track_objectives(scenario)
    for n_stations in (5, 6):
        ga_stage(scenario, track, objectives, n_stations, f"ga_{n_stations}", results)

    risks = {str(n): {name: results[f"ga_{n}"]["plans"][name]["risk"] for name in OBJECTIVE_NAMES} for n in (3, 4, 5, 6)}
    best = {n: min(v.values()) for n, v in risks.items()}
    targets = sorted({float(np.ceil(best[n] * f * 1e4) / 1e4) for n in ("4", "5", "6") for f in (1.0, 1.1, 1.25)})
    counts = {
        f"{t:g}": {name: next((int(n) for n in ("3", "4", "5", "6") if risks[n][name] <= t * (1 + 1e-12)), None)
                   for name in OBJECTIVE_NAMES}
        for t in targets
    }
    results["iso_risk_extended"] = {"risk_by_station_count": risks, "station_count_reaching_target": counts,
                                    "a0_excess_over_best_percent": {n: 100 * (risks[n]["A0"] / best[n] - 1) for n in risks}}
    save(results)
    print(json.dumps(results["iso_risk_extended"], indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
