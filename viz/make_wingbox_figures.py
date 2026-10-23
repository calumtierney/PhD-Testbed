"""Run the wing-box inspection experiment (plan U vs plan R), its sensitivity
sweeps and the surrogate study, and draw their figures:

    python -m viz.make_wingbox_figures --stage experiment   # ~2 h, writes results/wingbox_results.json
    python -m viz.make_wingbox_figures --stage seeds        # ~15 min, adds GA-repeatability runs to that file
    python -m viz.make_wingbox_figures --stage surrogate    # ~15 min, writes results/wingbox_surrogate.json
    python -m viz.make_wingbox_figures --stage figures      # seconds, writes results/panel10*.png

Stages are separate because the first two are slow and the third should be
cheap to redo. The JSON files are the experiment's record -- every plan's
layout, per-task summary, K-curve and cross-evaluation -- and the figures
stage rebuilds every plotted number from them with the exact evaluator, so a
figure cannot drift from the recorded layouts.

Pin the BLAS libraries to one thread per process (as `viz.make_wing_figures`
explains) or the parallel searches run slower than serial:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        python -m viz.make_wingbox_figures --stage experiment

Like the wing scenario's panels this is **not bit-for-bit reproducible**: the
search is a genetic algorithm (scipy differential evolution, seeded -- so
repeatable on one machine, but a GA finds a layout, not a proof of the
fewest stations). The synthetic part (`geometry.wing_box`) stands in for the
real one; nothing here is a statement about the real wing box.

The worker processes `differential_evolution(workers=-1)` starts re-import
the launching module, which is why every entry point below is behind
`if __name__ == "__main__"`.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

from planning.inspection import (
    build_candidate_pool, build_inspection_scenario, evaluate_layout, iso_risk_plan, task_summary,
)
from planning.inspection_experiment import cross_evaluate, run_inspection_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EXPERIMENT_JSON = RESULTS_DIR / "wingbox_results.json"
SURROGATE_JSON = RESULTS_DIR / "wingbox_surrogate.json"

BASE = dict(tolerance_half_m=50e-6, prior_style="offset", assignment_confidence=0.9, marginal_tasks=(1, 2, 3))
DE = dict(maxiter=80, popsize=14, tol=1e-8, workers=-1)

# One change at a time from BASE. Plan U's layouts do not depend on any of
# these (its cost has no tolerance or priors in it), so they are searched
# once, in the base run, and re-scored here.
VARIANTS = {
    "tolerance +-100 um": dict(scenario=dict(tolerance_half_m=100e-6)),
    "tolerance +-250 um": dict(scenario=dict(tolerance_half_m=250e-6)),
    "tolerance +-25 um": dict(scenario=dict(tolerance_half_m=25e-6)),
    "centred priors": dict(scenario=dict(prior_style="centred")),
    "assignment confidence 1.0": dict(scenario=dict(assignment_confidence=1.0)),
    "assignment confidence 0.5": dict(scenario=dict(assignment_confidence=0.5)),
    "flaps marginal (T5, T6)": dict(scenario=dict(marginal_tasks=(5, 6))),
    "top skin marginal (T4)": dict(scenario=dict(marginal_tasks=(4,))),
    "all tasks marginal": dict(scenario=dict(marginal_tasks=(1, 2, 3, 4, 5, 6))),
    "no marginal tasks": dict(scenario=dict(marginal_tasks=())),
    "risk limit x0.5": dict(run=dict(risk_slack=0.5)),
    "risk limit x0.25": dict(run=dict(risk_slack=0.25)),
    "risk limit x2": dict(run=dict(risk_slack=2.0)),
    "task floor 25%": dict(run=dict(min_task_coverage=0.25)),
    "task floor 50%": dict(run=dict(min_task_coverage=0.5)),
    "task floor 75%": dict(run=dict(min_task_coverage=0.75)),
    "task floor 100%": dict(run=dict(min_task_coverage=1.0)),
}


def summarise(scenario, result) -> dict:
    """JSON-ready record of one experiment: per plan, the answer, layout,
    cross-evaluation, task summary, spec and sweep."""
    plans = {}
    for name, run in result.runs.items():
        ans = run.answer
        plans[name] = dict(
            answer_k=ans.n_stations if ans else None,
            spec=dict(kind=run.spec.kind, u_max_m=None if not np.isfinite(run.spec.u_max_m) else run.spec.u_max_m,
                      max_p_escape=None if not np.isfinite(run.spec.max_p_escape) else run.spec.max_p_escape,
                      max_e_fr=None if not np.isfinite(run.spec.max_e_fr) else run.spec.max_e_fr,
                      allow_unmeasured=run.spec.allow_unmeasured, min_task_coverage=run.spec.min_task_coverage),
            layout=ans.positions_m.tolist() if ans else None,
            cross=cross_evaluate(scenario, ans.positions_m) if ans else None,
            tasks=task_summary(scenario, ans.outcome) if ans else None,
            seconds=run.seconds,
            sweep=[dict(
                k=r.n_stations, satisfied=r.satisfied, violation=float(r.violation) if np.isfinite(r.violation) else None,
                feasible=r.outcome.feasible, nfev=r.n_function_evals,
                n_measured=r.outcome.n_measured if r.outcome.feasible else None,
                p_escape=r.outcome.p_escape if r.outcome.feasible else None,
                e_fr=r.outcome.e_fr if r.outcome.feasible else None,
                mean_u_um=float(np.nanmean(r.outcome.u_m) * 1e6) if r.outcome.feasible else None,
                max_u_um=float(np.nanmax(r.outcome.u_m) * 1e6) if r.outcome.feasible else None,
                layout=r.positions_m.tolist()) for r in run.sweep],
        )
    return dict(u_curve={str(k): v for k, v in result.u_curve.items()},
                knee_by_gain={str(k): v for k, v in result.knee_by_gain.items()}, plans=plans, timing_s=result.timing_s)


def experiment_stage(quick: bool = False):
    de = dict(DE, maxiter=20, popsize=8) if quick else DE
    pool_scenario = build_inspection_scenario(**BASE)
    pool = build_candidate_pool(pool_scenario, n_candidates=4000 if quick else 20000, seed=0)
    record = dict(base=dict(args={k: (list(v) if isinstance(v, tuple) else v) for k, v in BASE.items()}), variants={})
    t0 = time.time()
    base = run_inspection_experiment(pool_scenario, de_kwargs=de, pool=pool, include_r_all=True, include_tur_rule=True,
                                     k_values_u=range(3, 9 if quick else 15), verbose=True)
    record["base"].update(summarise(pool_scenario, base))
    record["base"]["pool_size"] = base.pool_size
    EXPERIMENT_JSON.write_text(json.dumps(record, indent=1))  # the surrogate stage needs only the base
    print(f"base done in {time.time() - t0:.0f}s", flush=True)
    for name, change in VARIANTS.items():
        t1 = time.time()
        sc = build_inspection_scenario(**dict(BASE, **change.get("scenario", {})))
        res = run_inspection_experiment(sc, de_kwargs=de, pool=pool, u_run=base.runs["U"], **change.get("run", {}),
                                        k_values_u=range(3, 9 if quick else 15))
        record["variants"][name] = dict(change={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                                for k, v in change.items()}, **summarise(sc, res))
        print(f"  variant {name}: " + ", ".join(f"{n}={p['answer_k']}" for n, p in record["variants"][name]["plans"].items())
              + f" ({time.time() - t1:.0f}s)", flush=True)
        EXPERIMENT_JSON.write_text(json.dumps(record, indent=1))
    EXPERIMENT_JSON.write_text(json.dumps(record, indent=1))
    print("wrote", EXPERIMENT_JSON)


def surrogate_stage(quick: bool = False):
    """Train the surrogate on simulated layouts, test it on held-out groups
    and a held-out region, and compare a surrogate-assisted search against
    the plain one (see `planning.surrogate`)."""
    from planning.surrogate import LayoutSurrogate, run_surrogate_study, surrogate_assisted_search
    from planning.inspection import search_station_counts

    record = json.loads(EXPERIMENT_JSON.read_text())
    scenario = build_inspection_scenario(**BASE)
    spec = iso_risk_plan(scenario)
    pool = build_candidate_pool(scenario, n_candidates=3000 if quick else 12000, seed=1)
    bases = [np.array(r["layout"]) for plan in record["base"]["plans"].values() for r in plan["sweep"]
             if r["feasible"] and r["k"] <= 6]
    gen = dict(n_random_per_k=100, n_greedy_bases_per_k=10, jitters_per_base=6) if quick else \
        dict(n_random_per_k=800, n_greedy_bases_per_k=100, jitters_per_base=12)
    study = run_surrogate_study(scenario, pool, spec, k_values=(1, 2, 3, 4, 5, 6), search_bases=bases,
                                generate_kwargs=gen, seed=0)

    # surrogate-assisted vs plain search at each count, same DE settings and seeds
    de = dict(maxiter=20, popsize=8) if quick else dict(maxiter=80, popsize=14)
    assisted = {}
    plain = {}
    for k in (1, 2, 3, 4):
        t0 = time.time()
        plain_results = search_station_counts(scenario, spec, [k], de_kwargs=dict(de, tol=1e-8, workers=-1), seed=0,
                                              stop_at_first_satisfied=False, seed_layouts=bases)
        pr = plain_results[0]
        plain[k] = dict(violation=float(pr.violation), satisfied=bool(pr.satisfied), nfev=pr.n_function_evals,
                        seconds=time.time() - t0)
        ar = surrogate_assisted_search(scenario, spec, study.model, k, seed_layouts=bases, n_verify=60,
                                       de_kwargs=dict(de, tol=1e-8), seed=0)
        assisted[k] = dict(violation=float(ar.best.violation), satisfied=bool(ar.best.satisfied),
                           n_surrogate_evals=ar.n_surrogate_evals, n_exact_evals=ar.n_exact_evals,
                           shortlist_satisfied=ar.n_shortlist_satisfied, seconds_search=ar.seconds_search,
                           seconds_verify=ar.seconds_verify)
        print(f"  K={k}: plain violation {plain[k]['violation']:.4f} ({plain[k]['seconds']:.0f}s, {plain[k]['nfev']} exact evals); "
              f"assisted {assisted[k]['violation']:.4f} ({assisted[k]['seconds_search'] + assisted[k]['seconds_verify']:.1f}s, "
              f"{assisted[k]['n_exact_evals']} exact evals, {assisted[k]['shortlist_satisfied']}/60 shortlist satisfied)", flush=True)

    def scores(report):
        return {name: {k: float(v) for k, v in r.items()} for name, r in report.items()}

    out = dict(
        n_records=study.n_records, n_feasible=study.n_feasible, n_train=study.n_train, n_val=study.n_val,
        n_test_groups=study.n_test_groups, n_test_region=study.n_test_region,
        n_groups=dict(train=len(study.train_groups), val=len(study.val_groups), test=len(study.test_groups)),
        groups_disjoint=bool(set(study.train_groups).isdisjoint(study.val_groups)
                             and set(study.train_groups).isdisjoint(study.test_groups)
                             and set(study.val_groups).isdisjoint(study.test_groups)),
        report_groups=scores(study.report_groups), report_region=scores(study.report_region),
        feasibility_accuracy=dict(groups=study.feasibility_accuracy_groups, region=study.feasibility_accuracy_region),
        boundary_groups=study.boundary_groups,
        seconds=dict(generate=study.seconds_generate, train=study.seconds_train),
        ms_per_layout=dict(exact=study.exact_ms_per_layout, surrogate_batch=study.surrogate_ms_per_layout_batch),
        plain=plain, assisted=assisted,
        # arrays for the parity figure
        parity=dict(
            groups={k: study.truth_groups[k][study.truth_groups["feasible"]].tolist() for k in ("coverage", "log10_p_escape", "log10_e_fr")},
            groups_pred={k: study.prediction_groups[k][study.truth_groups["feasible"]].tolist() for k in ("coverage", "log10_p_escape", "log10_e_fr")},
            region={k: study.truth_region[k][study.truth_region["feasible"]].tolist() for k in ("coverage", "log10_p_escape", "log10_e_fr")},
            region_pred={k: study.prediction_region[k][study.truth_region["feasible"]].tolist() for k in ("coverage", "log10_p_escape", "log10_e_fr")},
        ),
    )
    SURROGATE_JSON.write_text(json.dumps(out, indent=1))
    print("wrote", SURROGATE_JSON)


def figures_stage():
    """Draw panel 10a-e from the recorded JSON (exact evaluator re-run on
    the recorded layouts, so nothing plotted is a stored number)."""
    import matplotlib.pyplot as plt
    from visualization.wingbox_plots import (
        plot_plan_layouts, plot_sensitivity, plot_station_curves, plot_surrogate_parity, plot_uncertainty_maps,
    )

    record = json.loads(EXPERIMENT_JSON.read_text())
    base = record["base"]
    scenario = build_inspection_scenario(**BASE)
    tol_um = BASE["tolerance_half_m"] * 1e6
    plans = {n: p for n, p in base["plans"].items() if n in ("U", "R-match", "R") and p["layout"] is not None}
    positions = {n: np.array(p["layout"]) for n, p in plans.items()}
    outcomes = {n: evaluate_layout(scenario, positions[n], allow_unmeasured=plans[n]["spec"]["allow_unmeasured"]) for n in plans}
    label = {"U": "U (uncertainty-driven)", "R-match": "R-match (risk-driven, no riskier than U)",
             "R": "R (risk-driven, TUR-rule risk ceiling)"}

    fig, _ = plot_plan_layouts(scenario.part, outcomes, positions,
                               f"Station layouts, tolerance +-{tol_um:.0f} um. Synthetic wing box (not the real part); geometry at true scale.")
    fig.savefig(RESULTS_DIR / "panel10a_wingbox_layouts.png", dpi=150)
    fig, _ = plot_uncertainty_maps(scenario.part, outcomes, tol_um, scenario.uniform_uncertainty_limit_m() * 1e6)
    fig.savefig(RESULTS_DIR / "panel10b_wingbox_uncertainty.png", dpi=150)

    sweeps = {n: base["plans"][n]["sweep"] for n in ("U", "R-match", "R") if n in base["plans"]}
    limits = {n: dict(p_escape=base["plans"][n]["spec"]["max_p_escape"], e_fr=base["plans"][n]["spec"]["max_e_fr"])
              for n in ("R-match", "R") if n in base["plans"]}
    fig, _ = plot_station_curves({int(k): v for k, v in base["u_curve"].items()},
                                 {float(k): v for k, v in base["knee_by_gain"].items()}, sweeps, limits)
    fig.savefig(RESULTS_DIR / "panel10c_wingbox_kcurves.png", dpi=150)

    answers = lambda r: {n: r["plans"][n]["answer_k"] for n in ("U", "R-match", "R") if n in r["plans"]}
    fig, _ = plot_sensitivity({n: answers(v) for n, v in record["variants"].items()}, answers(base))
    fig.savefig(RESULTS_DIR / "panel10d_wingbox_sensitivity.png", dpi=150)

    if SURROGATE_JSON.exists():
        sur = json.loads(SURROGATE_JSON.read_text())
        for tag, title in (("groups", "held-out base layouts (interpolation)"), ("region", "held-out region of the station box (extrapolation)")):
            truth = {k: np.array(v) for k, v in sur["parity"][tag].items()}
            pred = {k: np.array(v) for k, v in sur["parity"][tag + "_pred"].items()}
            truth["feasible"] = np.ones(len(truth["coverage"]), bool)
            fig, _ = plot_surrogate_parity(truth, pred, sur["report_" + tag], f"Surrogate vs exact evaluator on {title}")
            fig.savefig(RESULTS_DIR / f"panel10e_wingbox_surrogate_{tag}.png", dpi=150)
    plt.close("all")
    print("wrote panel10a-e to", RESULTS_DIR)


def load_u_run(scenario, record):
    """Rebuild plan U's `PlanRun` from the recorded base layouts (re-scored
    by the exact evaluator), so later stages can reuse its layouts without
    repeating its ~30 minute search."""
    from planning.inspection import KResult, plan_violation, uncertainty_plan
    from planning.inspection_experiment import PlanRun

    spec = uncertainty_plan()
    plan = record["base"]["plans"]["U"]
    sweep = []
    for row in plan["sweep"]:
        positions = np.array(row["layout"])
        outcome = evaluate_layout(scenario, positions, spec.allow_unmeasured)
        sweep.append(KResult(row["k"], positions, outcome, plan_violation(spec, outcome), row["nfev"]))
    chosen = next(r for r in sweep if r.n_stations == plan["answer_k"])
    return PlanRun("U", spec, sweep, plan["seconds"], chosen)


def seeds_stage(n_seeds: int = 5):
    """How much of the station count is search noise? Repeat plans R-match and
    R with different GA seeds against the *same* plan U layouts (searching U
    again would take ~30 minutes per seed); the spread of the answers is the
    GA's own repeatability on this problem."""
    record = json.loads(EXPERIMENT_JSON.read_text())
    scenario = build_inspection_scenario(**BASE)
    pool = build_candidate_pool(scenario, n_candidates=20000, seed=0)
    u_run = load_u_run(scenario, record)
    repeats = []
    for seed in range(1, n_seeds + 1):
        res = run_inspection_experiment(scenario, de_kwargs=DE, pool=pool, u_run=u_run, seed=seed)
        row = dict(seed=seed, answers={n: r.answer.n_stations if r.answer else None for n, r in res.runs.items()})
        repeats.append(row)
        print("  seed", seed, row["answers"], flush=True)
        record["seed_repeats"] = repeats
        EXPERIMENT_JSON.write_text(json.dumps(record, indent=1))
    print("wrote", EXPERIMENT_JSON)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["experiment", "seeds", "surrogate", "figures"], required=True)
    parser.add_argument("--quick", action="store_true", help="small budgets: a smoke test, not a result")
    args = parser.parse_args()
    if args.stage == "experiment":
        experiment_stage(args.quick)
    elif args.stage == "seeds":
        seeds_stage(2 if args.quick else 5)
    elif args.stage == "surrogate":
        surrogate_stage(args.quick)
    else:
        figures_stage()


if __name__ == "__main__":
    main()
