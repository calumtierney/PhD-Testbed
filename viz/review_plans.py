"""Review measurement plans and compare them: the command-line twin of the app's Plan review tab.

    python -m viz.review_plans factory --demo --tracker-model noise_only --reviewer-role metrologist
    python -m viz.review_plans part.qif plan_a.json plan_b.json --tracker-model npl_residual --reviewer-role planner
    python -m viz.review_plans part.qif plan.json --counterparts --out ~/reviews/part_x --tracker-model noise_only \\
        --reviewer-role metrologist

`scene` is a QIF file or the word `factory` (the built-in synthetic cell). Plans are plan files (`interchange.plan_file`); `--demo` adds
the synthetic demonstration plan (not an engineer's plan). `--tracker-model` is required: there is no default (docs/plan_review.md).
`--counterparts` searches the algorithm counterparts of one plan (`--counterparts-of N`, default the first), which is slow; the
`best` decision rule's tabulation is slow too, and both warn first.

**Every review is logged** (`review.log`, a directory outside the repository: `LVM_TESTBED_LOG_DIR`, else the user's data directory)
unless `--no-log` is given for a dry run. `--out DIR` writes `comparison.json` and the two figures (PNG and SVG) to a directory outside
the repository. Nothing here computes: it reads, calls `src/review`, prints and draws.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import sys
import time
from pathlib import Path

from interchange.plan_file import PlanFileError, read_plan_file
from review.core import build_context, compare_plans, format_comparison, review_with_context
from review.counterparts import OBJECTIVES, algorithm_counterparts
from review.demo import demo_plan
from review.export import comparison_bundle, write_bundle
from review.log import LogError, ReviewLog
from review.scene import ReviewScene
from review.settings import DECISION_RULES, ELEVATION_LIMITS, TRACKER_MODELS, ReviewSettings
from viz.review_figures import figure_bytes, plan_view_figure, risk_by_characteristic_figure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("scene", help="a QIF file, or 'factory' for the built-in synthetic cell")
    p.add_argument("plans", nargs="*", type=Path, help="plan files (JSON)")
    p.add_argument("--demo", action="store_true", help="add the synthetic demonstration plan (4 stations evenly spaced; NOT an engineer's plan)")
    p.add_argument("--tracker-model", required=True, choices=TRACKER_MODELS, help="the instrument model: required, no default")
    p.add_argument("--rule", choices=DECISION_RULES, default="simple", help="decision rule ('best' takes minutes to tabulate)")
    p.add_argument("--floor-um", type=float, default=0.0, help="isotropic placement-irreducible floor, um")
    p.add_argument("--n-builds", type=int, default=None, help="factory cell only: read its process statistics as from this many builds")
    p.add_argument("--require-all", action="store_true", help="a characteristic no station measures makes the plan infeasible (default: priced)")
    p.add_argument("--elevation-limits", choices=ELEVATION_LIMITS, default="none")
    p.add_argument("--stl", type=Path, default=None, help="the part's shape (STL, in the QIF file's coordinates) to block sight lines")
    p.add_argument("--stl-metres-per-unit", type=float, default=None)
    p.add_argument("--up-axis", choices=["x", "y", "z"], default="z")
    p.add_argument("--alignment", choices=["auto", "datums", "best_fit"], default="auto")
    p.add_argument("--scope", default=None, help="QIF part or assembly id to review against (default: the root)")
    p.add_argument("--counterparts", action="store_true", help="search the algorithm counterparts of one plan (slow)")
    p.add_argument("--counterparts-of", type=int, default=1, help="which plan (1-based, in the order given) to find counterparts for")
    p.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--popsize", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="directory for comparison.json and the figures (outside the repository)")
    p.add_argument("--reviewer-role", default=None, help="your role (e.g. metrologist); required unless --no-log")
    p.add_argument("--reviewer-name", default=None, help="optional; a name is recorded only if you give one")
    p.add_argument("--log-names", action="store_true", help="keep the scene's names in the log (default: left out)")
    p.add_argument("--no-log", action="store_true", help="a dry run: do not write the review log")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.no_log and not args.reviewer_role:
        print("error: --reviewer-role is required (every review is logged); use --no-log for a dry run", file=sys.stderr)
        return 2
    try:
        log = None if args.no_log else ReviewLog(None, args.reviewer_role, args.reviewer_name, args.log_names)
        settings = ReviewSettings(args.tracker_model, args.floor_um, args.rule, args.n_builds, args.require_all, args.elevation_limits,
                                  args.maxiter, args.popsize, 1e-8, args.seed)
    except (LogError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.scene == "factory":
        scene = ReviewScene.from_factory_cell()
    else:
        qif = Path(args.scene)
        stl = args.stl.read_bytes() if args.stl else None
        try:
            scene = ReviewScene.from_qif_bytes(qif.read_bytes(), qif.name, stl, args.stl_metres_per_unit, args.scope, args.up_axis, args.alignment)
        except (OSError, ValueError) as error:
            print(f"error: cannot build a scene from {qif}: {error}", file=sys.stderr)
            return 2
    plans = []
    try:
        plans = [read_plan_file(path) for path in args.plans]
    except (OSError, PlanFileError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.demo:
        plans.append(demo_plan(scene))
    if not plans:
        print("error: no plans: give plan files and/or --demo", file=sys.stderr)
        return 2
    if args.counterparts and not 1 <= args.counterparts_of <= len(plans):
        print(f"error: --counterparts-of {args.counterparts_of} but there are {len(plans)} plan(s)", file=sys.stderr)
        return 2

    if args.rule == "best":
        print("note: the 'best' decision rule tabulates its risk curves first: about a minute for the built-in cell, minutes for a QIF file", flush=True)
    start = time.time()
    context = build_context(scene, settings)
    reviews = [review_with_context(context, plan) for plan in plans]
    counterparts, not_found, search = (), (), {}
    if args.counterparts:
        k = plans[args.counterparts_of - 1].n_stations
        print(f"searching counterparts of plan {args.counterparts_of} ({k} stations): up to {len(args.objectives) + k + 1} differential-evolution "
              f"searches, each maxiter {args.maxiter} x popsize {args.popsize}; this is slow", flush=True)
        result = algorithm_counterparts(scene, plans[args.counterparts_of - 1], settings, context=context, objectives=args.objectives,
                                        progress=lambda done, total, label: print(f"  [{time.time() - start:5.0f} s] {done}/{total}  {label}", flush=True))
        counterparts, not_found, search = result.counterparts, result.not_found, result.search
        reviews += [c.review for c in counterparts]
    comparison = compare_plans(reviews)
    print(format_comparison(comparison))
    for line in not_found:
        print("not found:", line)

    if args.out is not None:
        figures = {}
        for stem, fig in (("plan_view", plan_view_figure(scene, reviews)), ("risk_by_characteristic", risk_by_characteristic_figure(reviews))):
            for fmt in ("png", "svg"):
                figures[f"{stem}.{fmt}"] = figure_bytes(fig, fmt)
        try:
            for path in write_bundle(args.out, comparison_bundle(comparison, reviews, counterparts, search, not_found), figures):
                print("wrote", path)
        except LogError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    if log is not None:
        log.log_session(scene, settings, reviews, comparison, counterparts, search)
        print("logged to", log.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
