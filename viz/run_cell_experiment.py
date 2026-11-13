"""The factory-cell experiment: the long part on its station track with a
reference network (`planning.cell`), planned by uncertainty and by decision risk.

    python -m viz.run_cell_experiment --frame datum    # the main experiment (about 20 min on 4 cores)
    python -m viz.run_cell_experiment --frame datum --no-network   # the same cell without the reference nests
    python -m viz.run_cell_experiment                  # anchor-frame uncertainty, anchor at the -x tip
    python -m viz.run_cell_experiment --anchor plus_x_tip

`--frame datum` scores each characteristic relative to the part's datum frame and
plans every station (`planning.cell.DatumCellScorer`; station counts are totals).
`--frame anchor` (the default, and the convention of the earlier track experiments)
scores in the frame of a fixed anchor station and plans the added stations.

For every station count 1-6 it searches the layout that is best under each
objective, with the same differential-evolution settings and seed for all:

  A0, A1, B   the uncertainty objectives of the headline experiment
  T           the worst characteristic's uncertainty (the TUR plan's criterion)
  C           total risk, simple acceptance, process known exactly (as before)
  R<n>        total risk under the conformance-probability rule (accept when the
              part is more likely conforming than not), process known from n
              builds (n = 3, 5, 10, 30)
  R<n>free    the same, but a characteristic may go unobserved (it is then
              decided on process knowledge alone)

Each layout is then re-scored under every objective and every risk model; if
some other layout at the same station count beats an objective's own winner
under that objective, the search is re-run seeded with all known layouts (a GA
finds a good plan, not provably the best) and the better of the two is kept.
Writes results/cell_results.json, with `_datum`, `_<anchor>` and `_no_network`
suffixes for the variants.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from planning.cell import (
    CellLayoutScorer, CellRiskModel, DatumCellScorer, MaxUncertaintyObjective, RiskObjective, factory_cell_scenario,
    layout_poses, measurement_time_minutes, search_cell_layout, standard_objectives,
)
from planning.experiment import LONG_PART_GA_HEIGHT_BOUNDS_M, long_part_track

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COUNTS = (1, 2, 3, 4, 5, 6)
BUILDS = (3, 5, 10, 30)
DE = dict(seed=0, maxiter=150, popsize=25, tol=1e-8, workers=1)
OBJECTIVES = ["A0", "A1", "B", "T", "C"] + [f"R{n}" for n in BUILDS] + [f"R{n}free" for n in BUILDS]
RISK_MODELS = {"simple|known": (None, "simple"), "best|known": (None, "best")}
RISK_MODELS.update({f"simple|n{n}": (n, "simple") for n in BUILDS})
RISK_MODELS.update({f"best|n{n}": (n, "best") for n in BUILDS})

_CACHE = {}


def context(config):
    """`config` = (anchor, frame, with_network). Built once in the parent and handed
    to the workers (`_init`), because the risk tables take minutes to build."""
    if config not in _CACHE:
        anchor, frame, with_network = config
        track = long_part_track()
        anchor_s = None if anchor == "minus_x_tip" else track.tip_plus_x_s_m
        scenario = factory_cell_scenario(anchor_s_m=anchor_s, track=track, with_reference_network=with_network)
        objectives = dict(standard_objectives(scenario))
        objectives["T"] = MaxUncertaintyObjective()
        models = {name: CellRiskModel.build(scenario, n, rule) for name, (n, rule) in RISK_MODELS.items()}
        for n in BUILDS:
            objectives[f"R{n}"] = RiskObjective(models[f"best|n{n}"])
            objectives[f"R{n}free"] = RiskObjective(models[f"best|n{n}"])
        _CACHE[config] = (track, scenario, objectives, models)
    return _CACHE[config]


def _init(cache):
    _CACHE.update(cache)


def make_scorer(config, require_all):
    scenario = context(config)[1]
    cls = DatumCellScorer if config[1] == "datum" else CellLayoutScorer
    return cls(scenario, require_all_characteristics=require_all)


def run_job(job):
    config, name, k, seeds = job
    track, scenario, objectives, _ = context(config)
    scorer = make_scorer(config, not name.endswith("free"))
    start = time.time()
    result = search_cell_layout(scorer, objectives[name], track, k, LONG_PART_GA_HEIGHT_BOUNDS_M, seed_stations=seeds, **DE)
    return name, k, [list(s) for s in result.stations], result.value, time.time() - start


def describe(stations, name, config):
    """Everything reported for one layout: uncertainties, shots, time, and its
    risk under every risk model (per characteristic and total)."""
    track, scenario, objectives, models = context(config)
    score = make_scorer(config, False).score(layout_poses(stations, scenario, track))
    names = [c.name for c in scenario.characteristics]
    n_total = len(stations) + (0 if config[1] == "datum" else 1)  # the anchor-frame model has a fixed anchor station too
    out = {
        "stations": stations, "u_um": dict(zip(names, (score.u_m * 1e6).tolist())), "n_shots": score.n_shots,
        "n_stations_total": n_total, "time_min": measurement_time_minutes(n_total, score.n_shots),
        "risk": {m: model.total(score.u_m) for m, model in models.items()},
        "risk_by_characteristic": {m: dict(zip(names, model.per_characteristic(score.u_m).tolist())) for m, model in models.items()},
        "objective_values": {},
    }
    complete = bool(np.all(np.isfinite(score.u_m)))
    for o, fn in objectives.items():
        if not complete and not o.endswith("free"):
            continue  # a layout that leaves a characteristic unobserved is not a candidate for a measure-all objective
        out["objective_values"][o] = float(fn(score))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", choices=("minus_x_tip", "plus_x_tip"), default="minus_x_tip")
    parser.add_argument("--frame", choices=("anchor", "datum"), default="anchor")
    parser.add_argument("--no-network", action="store_true", help="the same cell without the reference nests")
    args = parser.parse_args()
    anchor = args.anchor
    config = (args.anchor, args.frame, not args.no_network)
    suffix = ("_datum" if args.frame == "datum" else "") + ("" if anchor == "minus_x_tip" or args.frame == "datum" else f"_{anchor}") \
        + ("_no_network" if args.no_network else "")
    path = RESULTS_DIR / f"cell_results{suffix}.json"
    track, scenario, objectives, models = context(config)
    results = {"anchor": anchor if args.frame == "anchor" else None, "frame": args.frame, "reference_network": not args.no_network,
               "de": DE, "counts": list(COUNTS), "builds": list(BUILDS),
               "n_targets": int(len(scenario.target_points_m)), "risk_models": list(RISK_MODELS),
               "unmeasured_risk": {m: model.unmeasured_total() for m, model in models.items()}, "layouts": {}, "reseeded": []}
    layouts = {}
    anchor = config  # every job and description below is keyed by the full configuration

    with Pool(4, initializer=_init, initargs=(dict(_CACHE),)) as pool:
        jobs = [(anchor, name, k, None) for k in COUNTS for name in OBJECTIVES]
        for name, k, stations, value, seconds in pool.imap_unordered(run_job, jobs):
            layouts[(name, k)] = (stations, value)
            print(f"[{name} k={k}] {'infeasible' if value is None else f'{value:.5g}'} ({seconds:.0f} s)", flush=True)

        # cross-check and seeded re-runs
        for _ in range(2):
            again = []
            for k in COUNTS:
                known = [s for (n, kk), (s, v) in layouts.items() if kk == k and v is not None]
                if not known:
                    continue
                described = [describe(s, "x", anchor) for s in known]
                for name in OBJECTIVES:
                    own = layouts[(name, k)][1]
                    best_known = min((d["objective_values"].get(name, np.inf) for d in described), default=np.inf)
                    if np.isfinite(best_known) and (own is None or best_known < own * (1 - 1e-3)):
                        again.append((anchor, name, k, known))
            if not again:
                break
            for name, k, stations, value, seconds in pool.imap_unordered(run_job, again):
                seeds = [s for (n, kk), (s, v) in layouts.items() if kk == k and v is not None]
                candidates = [(value, stations)] if value is not None else []
                for s in seeds:  # the seeded search can only improve on its seeds, but keep the best seen anyway
                    v = describe(s, "x", anchor)["objective_values"].get(name)
                    if v is not None:
                        candidates.append((v, s))
                best = min(candidates, key=lambda c: c[0])
                layouts[(name, k)] = (best[1], best[0])
                results["reseeded"].append(f"{name}|{k}")
                print(f"[{name} k={k}] re-run seeded: {best[0]:.5g} ({seconds:.0f} s)", flush=True)

    for (name, k), (stations, value) in sorted(layouts.items()):
        key = f"{name}|{k}"
        if value is None:
            results["layouts"][key] = {"feasible": False}
        else:
            results["layouts"][key] = {"feasible": True, "value": value, **describe(stations, name, anchor)}
    path.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
