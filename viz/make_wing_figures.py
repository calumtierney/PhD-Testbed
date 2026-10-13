"""Regenerate the wing-scenario figures (docs/step6_headline_experiment.md's
"The wing scenario" section):

    python -m viz.make_wing_figures

Separate from `viz.make_figures` (which stays at its documented ~2-3
minutes) because this one is genuinely slow: `planning.experiment.
run_wing_experiment` runs a GA search four times per scenario (A0/A1/B/C,
cascaded-seeded) over two scenarios (with and without the support-tower
obstacle). Expect roughly 15-25 minutes on a 4-core machine with
`workers=-1` (the default `run_wing_experiment` uses) -- *and only that
fast* if every BLAS library is pinned to one thread per process first:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        python -m viz.make_wing_figures

Without pinning them, each worker process's own multi-threaded BLAS calls
oversubscribe the machine's cores against each other and the run can come
out *slower* than running serially -- confirmed directly while building
this, not a theoretical caveat (see `planning.search.
genetic_algorithm_station_placement`'s own docstring). This command is
not run by CI or by `viz.make_figures`'s own `main()` for exactly this
reason; regenerate it deliberately, when the wing scenario itself
changes.

Unlike `viz.make_figures`'s panels, this is **not bit-for-bit
reproducible**: `scipy.optimize.differential_evolution` is a genetic
algorithm, not an exhaustive search, and `docs/step6_headline_experiment.md`'s
own "Getting a trustworthy GA comparison took three attempts" section
documents the real, measured search-quality caveats that come with that.
Re-running this command may find slightly different station positions
and objective values than the ones committed in `results/` and quoted in
that doc; the same fixed `seed=0` and cascaded seeding
(`planning.experiment.run_wing_experiment`) make it *repeatable* on the
same machine, not independent of the search method's own limitations.
"""
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planning.experiment import run_wing_experiment
from visualization.plotting import _auto_ellipsoid_scale, plot_network_result, plot_plan_comparison

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _shared_ellipsoid_scale(net_a, net_b, true_targets_m):
    """Same convention `viz.make_figures` uses: one exaggeration factor
    shared by both panels of a station-layout comparison, so a reader
    never mistakes a scale difference for an uncertainty difference."""

    def scale_for(net_result):
        points_m = net_result.target_points_m
        station_positions_m = np.array([p.position_m for p in net_result.station_poses])
        all_points_m = np.vstack([points_m, station_positions_m, true_targets_m])
        centre_m = all_points_m.mean(axis=0)
        scene_radius_m = max(float(np.max(np.linalg.norm(all_points_m - centre_m, axis=1))), 1e-6)
        max_semi_axis_m = max(
            (np.sqrt(np.linalg.eigvalsh(net_result.target_point_covariance(i)).max()) for i in range(len(points_m))),
            default=0.0,
        )
        return _auto_ellipsoid_scale(max_semi_axis_m, scene_radius_m)

    return min(scale_for(net_a), scale_for(net_b))


def _make_wing_panels(result, suffix: str, subtitle: str):
    """panel9a_wing_stations{suffix}.png and panel9b_wing_comparison{suffix}.png
    for one `WingExperimentResult` -- A0 (the field's own criterion)
    against C (direct risk minimisation), the two ends of this project's
    comparison. A1 and B's own plans are reported in the doc's table
    alongside these (cross-validated there, not just asserted), but a
    two-panel figure keeps CLAUDE.md §8's "the target view" comparison
    legible; a four-way station-layout plot earns its place only once
    all four genuinely diverge, which this scene's own results (see the
    doc) show is not the case here.
    """
    scenario = result.scenario
    true_targets_m = scenario.target_points_m
    plan_a0 = result.plan_a0  # PlanReport -- per-characteristic uncertainty/risk, not station poses
    plan_c = result.plan_c

    network_a0 = result.search_a0.best.network_solve
    network_c = result.search_c.best.network_solve
    shared_scale = _shared_ellipsoid_scale(network_a0, network_c, true_targets_m)

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_network_result(
        network_a0, ax=ax1, ellipsoid_scale=shared_scale, true_target_points_m=true_targets_m,
        title="A0: trace of raw covariance\n(the field's own criterion)",
    )
    plot_network_result(
        network_c, ax=ax2, ellipsoid_scale=shared_scale, true_target_points_m=true_targets_m,
        title="C: direct risk minimisation",
    )
    fig.suptitle(subtitle, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"panel9a_wing_stations{suffix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / f'panel9a_wing_stations{suffix}.png'}")

    labels = list(plan_a0.per_characteristic_uncertainty_m.keys())
    uncertainty_a0 = [plan_a0.per_characteristic_uncertainty_m[l] for l in labels]
    uncertainty_c = [plan_c.per_characteristic_uncertainty_m[l] for l in labels]
    risk_a0 = plan_a0.global_pfa + plan_a0.global_pfr
    risk_c = plan_c.global_pfa + plan_c.global_pfr
    risk_change_pct = (risk_c - risk_a0) / risk_a0 * 100

    fig2, ax = plt.subplots(figsize=(8, 5))
    plot_plan_comparison(
        labels, uncertainty_a0, uncertainty_c, label_a="A0 (trace)", label_b="C (direct risk)", ax=ax,
        title=f"Wing scenario: per-characteristic uncertainty\n(global risk {risk_change_pct:+.1f}% A0 to C)",
    )
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / f"panel9b_wing_comparison{suffix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {RESULTS_DIR / f'panel9b_wing_comparison{suffix}.png'}")


def make_panel9_wing():
    """The wing-scenario comparison, with and without the support-tower
    occluder (docs/step6_headline_experiment.md's "The wing scenario"
    section)."""
    for tower, suffix, label in [(True, "_tower", "with the support-tower obstacle"), (False, "", "without it")]:
        print(f"Running wing experiment, {label} ...")
        t0 = time.time()
        result = run_wing_experiment(include_support_tower=tower)
        print(f"  done in {time.time() - t0:.0f}s")
        _make_wing_panels(
            result, suffix=suffix,
            subtitle=f"Wing scenario (~30 m span, 4 characteristics), {label}",
        )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    make_panel9_wing()


if __name__ == "__main__":
    main()
