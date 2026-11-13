"""The process-evidence experiment on the station track: what if some
characteristics are accepted on what is known about their process instead of
being measured?

    python -m viz.run_station_track_process_evidence

See `planning.process_evidence` for the model (what an unmeasured characteristic
costs, the two controls, which targets must be observed). Here, at +-50 um and
with the anchor at the -x tip (the existing track scene):

  * for every station count 1-6, the risk-optimal layout *and* choice of which
    characteristics to measure, found by the same differential-evolution driver
    over `(s_m, height_m)` as every other track search;
  * under a policy floor on the number measured (0, 3 of 6, 6 of 6) and, at floor
    0, under a range of escape-price multipliers (how much worse than the
    elicited process prior the real process would have to be);
  * with the tie points required (the track scene's rule: every target observed)
    and relaxed (a plan observes only the ties it needs);
  * against the reference plans of the earlier experiments: the A1 plan that
    measures everything (the knee, 4 stations, total risk 1.0511) and the
    measure-everything optimum at each count (objective C).

A floor of 6 with ties required is the existing measure-everything experiment;
it is re-evaluated here with the new evaluator as a check (it reproduces the
GA's exact objective-C values within 0.1%). Writes
results/station_track_process_evidence.json after every configuration.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"  # before numpy is imported, or every worker oversubscribes the cores

import json
import time
from pathlib import Path

import numpy as np

from planning.experiment import LONG_PART_GA_HEIGHT_BOUNDS_M, long_part_track, long_part_two_cluster_scenario
from planning.process_evidence import (
    ProcessEvidenceEvaluator,
    break_even_multiplier,
    exact_plan_risk,
    search_evidence_layout,
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "station_track_process_evidence.json"
MAIN_RESULTS = Path(__file__).resolve().parent.parent / "results" / "station_track_results.json"
COUNTS = (1, 2, 3, 4, 5, 6)
FLOORS = (3, 6)  # at price 1; floor 0 at price 1 is "measure nothing", computed directly
PRICES = (1e2, 1e3, 1e4, 1e5)  # at floor 0: the unmeasured price = nonconformance rate x this
DE = dict(seed=0, maxiter=60, popsize=15, tol=1e-8, workers=-1, updating="deferred")


def main():
    main_results = json.loads(MAIN_RESULTS.read_text())
    track = long_part_track()
    scenario = long_part_two_cluster_scenario(track=track)
    known = {k: [[(s, h) for s, h, _ in main_results[f"ga_{k}"]["plans"][n]["stations"]] for n in ("A0", "A1", "B", "C")]
             for k in (3, 4, 5, 6)}
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results["settings"] = {"de": DE, "counts": list(COUNTS), "floors": list(FLOORS), "prices": list(PRICES),
                           "reference_a1_knee": {"stations": 4, "risk": main_results["ga_4"]["plans"]["A1"]["risk"]}}
    configs = results.setdefault("configs", {})

    base = ProcessEvidenceEvaluator(scenario, min_measured=0)
    results["measure_nothing_risk"] = base.baseline_risk()
    results["escape_price_model"] = {c.name: base.escape_price(c.name) for c in scenario.characteristics}

    # -- break-even: how wrong would the prior have to be before measuring pays? ------------------
    plan4 = main_results["ga_4"]["plans"]["A1"]
    u = {k: v * 1e-6 for k, v in plan4["per_characteristic_u_um"].items()}
    results["break_even_multiplier_a1_4_stations"] = break_even_multiplier(scenario, u)
    print("[break-even] A1 plan, 4 stations:", {k[-9:]: f"{v:.3g}" for k, v in results["break_even_multiplier_a1_4_stations"].items()}, flush=True)

    jobs = []
    for ties_required in (True, False):
        for floor in FLOORS:
            jobs.append((ties_required, floor, 1.0))
        for price in PRICES:
            jobs.append((ties_required, 0, price))
    for ties_required, floor, price in jobs:
        key = f"ties_{'required' if ties_required else 'relaxed'}|floor_{floor}|price_x{price:g}"
        evaluator = ProcessEvidenceEvaluator(scenario, min_measured=floor, ties_required=ties_required,
                                             escape_price_multiplier=price)
        entry = configs.setdefault(key, {"ties_required": ties_required, "floor": floor, "price_multiplier": price,
                                         "measure_nothing_risk": evaluator.baseline_risk(), "by_count": {}})
        for k in COUNTS:
            if str(k) in entry["by_count"]:
                continue
            start = time.time()
            seeds = known.get(k)  # feasible measure-everything layouts (3-6 stations); none exist for 1-2
            plan, poses, de = search_evidence_layout(evaluator, track, k, LONG_PART_GA_HEIGHT_BOUNDS_M,
                                                     seed_stations=seeds, **DE)
            if plan is None:
                entry["by_count"][str(k)] = {"feasible": False, "seconds": time.time() - start}
                print(f"[{key}] k={k}: no feasible layout found ({time.time() - start:.0f} s)", flush=True)
            else:
                entry["by_count"][str(k)] = {
                    "feasible": True, "risk": plan.risk, "risk_exact": exact_plan_risk(scenario, plan, evaluator),
                    "risk_measured": plan.risk_measured, "risk_unmeasured": plan.risk_unmeasured,
                    "measured": list(plan.measured), "accepted_on_evidence": list(plan.accepted_on_evidence),
                    "u_um": {n: v * 1e6 for n, v in plan.u_m.items()},
                    "stations": [[p.s_m, p.height_m, p.segment] for p in poses], "seconds": time.time() - start,
                }
                print(f"[{key}] k={k}: risk {plan.risk:.4g} measured {len(plan.measured)}/6 "
                      f"({[n[-9:] for n in plan.accepted_on_evidence]} accepted)  ({time.time() - start:.0f} s)", flush=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))
    print("done", flush=True)


if __name__ == "__main__":
    main()
