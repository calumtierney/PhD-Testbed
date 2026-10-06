"""Reproduce the step 6 headline-experiment numbers and figures reported
in docs/step6_headline_experiment.md's "uncertainty budget" section --
including the 10 um systematic-term re-run and the three figures
committed under results/ (panel6a_stations_systematic10um.png,
panel6b_comparison_systematic10um.png, panel7_sensitivity_systematic10um.png).

This is the independent-verification entry point CLAUDE.md's repo layout
implies for results/ ("experiment outputs, never edited by hand"): the
committed PNGs there are this script's output, not hand-authored, so
running it should reproduce them exactly -- planning.objectives.
evaluate_plan solves from `network.solve.exact_observations` (noiseless,
deterministic observations at the true geometry), so there is no RNG in
this path and the numbers/figures below should match the doc and the
committed PNGs bit-for-bit (up to matplotlib/BLAS backend differences).

Run from the repository root, after `pip install -e ".[dev]"`:

    python scripts/reproduce_headline_experiment.py

Takes roughly 2-3 minutes: each `run_headline_experiment` call is ~60-90s
(objective C's several `scipy.integrate.quad` calls per characteristic,
per candidate pair, dominate -- see `planning.experiment.
run_headline_experiment`'s own docstring for why the candidate grid is
sized the way it is), and this script runs it twice (systematic = 0 and
10 um).

For the underlying physics gates themselves (steps 1-5, and the two
`test_network_solve.py` regression tests for this exact systematic-term
fix), the fast, RNG-seeded, sub-minute way to verify is the test suite:

    pytest
"""
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from instruments.laser_tracker import LaserTracker
from planning.experiment import default_two_cluster_scenario, run_headline_experiment
from visualization.plotting import _auto_ellipsoid_scale, plot_network_result, plot_plan_comparison
from risk.jcgm106 import ClusterAssignment, DecisionRule, evaluate_conformity_risk
from planning.experiment import HIGH_CAPABILITY_CLUSTER, MARGINAL_CLUSTER, TOLERANCE_ZONE_WIDTH_M

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SYSTEMATIC_STD_M = 10e-6  # the "plausible combined value" the doc uses


def _run(systematic_std_m: float):
    tracker = LaserTracker(systematic_std_m=systematic_std_m)
    scenario = default_two_cluster_scenario(tracker=tracker)
    t0 = time.time()
    result = run_headline_experiment(scenario)
    elapsed = time.time() - t0
    print(f"  ({elapsed:.0f}s)")
    return result


def _summary_row(result):
    plan_a1 = result.plan_a1
    plan_b = result.plan_b
    risk_a1 = plan_a1.global_pfa + plan_a1.global_pfr
    risk_b = plan_b.global_pfa + plan_b.global_pfr
    effort_change_pct = (
        (plan_b.summed_characteristic_uncertainty_m - plan_a1.summed_characteristic_uncertainty_m)
        / plan_a1.summed_characteristic_uncertainty_m
        * 100
    )
    risk_change_pct = (risk_b - risk_a1) / risk_a1 * 100
    return plan_a1, plan_b, effort_change_pct, risk_change_pct


def reproduce_numbers():
    """Print the same summary table as docs/step6_headline_experiment.md's
    "The full ablation, re-run with a nonzero systematic term" section."""
    print("Running headline experiment, systematic_std_m = 0 um ...")
    result_0 = _run(0.0)
    print("Running headline experiment, systematic_std_m = 10 um ...")
    result_10 = _run(SYSTEMATIC_STD_M)

    print()
    print(f"{'':22s} {'A0=A1 summed':>16s} {'B=C summed':>14s} {'A0=A1 risk':>12s} {'B=C risk':>10s} {'effort d':>10s} {'risk d':>8s}")
    for label, result in [("systematic = 0 um", result_0), ("systematic = 10 um", result_10)]:
        plan_a1, plan_b, effort_change_pct, risk_change_pct = _summary_row(result)
        risk_a1 = plan_a1.global_pfa + plan_a1.global_pfr
        risk_b = plan_b.global_pfa + plan_b.global_pfr
        print(
            f"{label:22s} {plan_a1.summed_characteristic_uncertainty_m * 1e6:14.3f}um "
            f"{plan_b.summed_characteristic_uncertainty_m * 1e6:12.3f}um "
            f"{risk_a1:12.6f} {risk_b:10.6f} {effort_change_pct:+9.2f}% {risk_change_pct:+7.2f}%"
        )
    print()
    print("Expected (docs/step6_headline_experiment.md):")
    print("  systematic = 0 um : summed 44.320 / 46.597 um, risk 0.198953 / 0.192605, effort +5.14%, risk -3.19%")
    print("  systematic = 10 um: summed 74.644 / 76.055 um, risk 0.522947 / 0.520279, effort +1.89%, risk -0.51%")
    return result_0, result_10


def _shared_ellipsoid_scale(net_a, net_b, true_targets_m):
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

    # min(), not each panel's own scale: the two subplots must share one
    # exaggeration factor to be visually comparable (see visualization.
    # plotting's module docstring) -- using each panel's own auto-scale
    # independently was a real mistake caught earlier in this project's
    # history (see git log for "ellipsoid-exaggeration label placement").
    return min(scale_for(net_a), scale_for(net_b))


def reproduce_figures(result_10):
    """Regenerate the three committed results/*_systematic10um.png files
    from a systematic_std_m=10e-6 HeadlineExperimentResult."""
    scenario = result_10.scenario
    true_targets_m = scenario.target_points_m

    net_a = result_10.search_a1.best.network_solve
    net_b = result_10.search_b.best.network_solve
    shared_scale = _shared_ellipsoid_scale(net_a, net_b, true_targets_m)

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_network_result(
        net_a, ax=ax1, ellipsoid_scale=shared_scale, true_target_points_m=true_targets_m,
        title="A0 = A1: trace / uniform-projected\n(aggregate-uncertainty objectives)",
    )
    plot_network_result(
        net_b, ax=ax2, ellipsoid_scale=shared_scale, true_target_points_m=true_targets_m,
        title="B = C: linearised / direct risk\n(risk-based objectives)",
    )
    fig.suptitle("With a 10 µm placement-irreducible systematic term", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "panel6a_stations_systematic10um.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel6a_stations_systematic10um.png'}")

    plan_a1, plan_b, effort_change_pct, risk_change_pct = _summary_row(result_10)
    labels = list(plan_a1.per_characteristic_uncertainty_m.keys())
    uncertainty_a = [plan_a1.per_characteristic_uncertainty_m[l] for l in labels]
    uncertainty_b = [plan_b.per_characteristic_uncertainty_m[l] for l in labels]

    fig2, ax = plt.subplots(figsize=(8, 5))
    plot_plan_comparison(
        labels, uncertainty_a, uncertainty_b,
        label_a="A0=A1 (aggregate uncertainty)", label_b="B=C (risk-based)",
        ax=ax,
        title=(
            f"Step 6: per-characteristic uncertainty, 10 µm systematic term\n"
            f"(effort {effort_change_pct:+.1f}%, risk {risk_change_pct:+.1f}%)"
        ),
    )
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "panel6b_comparison_systematic10um.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {RESULTS_DIR / 'panel6b_comparison_systematic10um.png'}")

    _reproduce_sensitivity_figure(plan_a1, plan_b)


def _reproduce_sensitivity_figure(plan_a1_10um, plan_b_10um):
    """panel7_sensitivity_systematic10um.png: the flat-vs-steep risk-vs-
    uncertainty curves, with the achieved-uncertainty band from a
    systematic=0 run (hardcoded from the committed doc/figure -- run
    reproduce_numbers()'s systematic=0 branch again to recompute from
    scratch) and from this systematic=10um run overlaid."""
    decision_rule = DecisionRule.from_zone_width(TOLERANCE_ZONE_WIDTH_M)
    assignment_high = ClusterAssignment("flat (high-capability)", HIGH_CAPABILITY_CLUSTER)
    assignment_marginal = ClusterAssignment("steep (marginal)", MARGINAL_CLUSTER)

    from visualization.plotting import plot_risk_vs_uncertainty

    uncertainties_um_wide = np.linspace(1, 60, 60)
    uncertainties_m_wide = uncertainties_um_wide * 1e-6
    results_high_wide = [evaluate_conformity_risk(assignment_high, u, decision_rule) for u in uncertainties_m_wide]
    results_marginal_wide = [evaluate_conformity_risk(assignment_marginal, u, decision_rule) for u in uncertainties_m_wide]

    uncertainties_um_zoom = np.linspace(1, 20, 80)
    uncertainties_m_zoom = uncertainties_um_zoom * 1e-6
    results_high_zoom = [evaluate_conformity_risk(assignment_high, u, decision_rule) for u in uncertainties_m_zoom]
    results_marginal_zoom = [evaluate_conformity_risk(assignment_marginal, u, decision_rule) for u in uncertainties_m_zoom]

    reference_uncertainty_m = 8e-6
    epsilon_m = 0.5e-6

    def slope_at(assignment):
        r_plus = evaluate_conformity_risk(assignment, reference_uncertainty_m + epsilon_m, decision_rule)
        r_minus = evaluate_conformity_risk(assignment, reference_uncertainty_m - epsilon_m, decision_rule)
        total_plus = r_plus.probability_false_acceptance + r_plus.probability_false_rejection
        total_minus = r_minus.probability_false_acceptance + r_minus.probability_false_rejection
        return (total_plus - total_minus) / (2 * epsilon_m)

    ratio = slope_at(assignment_marginal) / slope_at(assignment_high)
    r_high_ref = evaluate_conformity_risk(assignment_high, reference_uncertainty_m, decision_rule)
    r_marginal_ref = evaluate_conformity_risk(assignment_marginal, reference_uncertainty_m, decision_rule)

    # These bands mark where the step 6 experiment's *achieved*
    # uncertainty actually lands on the x-axis -- not the systematic
    # term's own value (which would be a single point, not a range, and
    # is 0 or 10 um respectively). "with_systematic"/"sensor_noise_only"
    # names which of the two LaserTracker.systematic_std_m settings
    # produced the achieved-uncertainty range, since even with the
    # systematic term off, sensor noise alone still gives a nonzero
    # achieved uncertainty (6.7-9.2 um here) -- a band labelled "0 um"
    # sitting well away from x=0 is exactly the confusing part a reader
    # flagged, hence this naming.
    band_with_systematic_um = (
        min(plan_a1_10um.per_characteristic_uncertainty_m.values()) * 1e6,
        max(plan_b_10um.per_characteristic_uncertainty_m.values()) * 1e6,
    )
    # Doc-reported sensor-noise-only band (docs/step6_headline_experiment.
    # md's "Results" section) -- rerun reproduce_numbers()'s
    # systematic=0 branch to recompute this live instead of using this
    # fixed reference.
    band_sensor_noise_only_um = (6.690814087350281, 9.156836281823471)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    plot_risk_vs_uncertainty(
        uncertainties_m_wide,
        {
            f"flat: high-capability (Cpk~{r_high_ref.process_cpk:.2f})": results_high_wide,
            f"steep: marginal (Cpk~{r_marginal_ref.process_cpk:.2f}, AS13006 floor)": results_marginal_wide,
        },
        ax=ax1, title="Overview (log scale)", log_scale=True,
    )
    ax1.axvspan(*band_sensor_noise_only_um, color="tab:green", alpha=0.15)
    ax1.axvspan(*band_with_systematic_um, color="tab:red", alpha=0.15)

    plot_risk_vs_uncertainty(
        uncertainties_m_zoom,
        {
            f"flat: high-capability (Cpk~{r_high_ref.process_cpk:.2f})": results_high_zoom,
            f"steep: marginal (Cpk~{r_marginal_ref.process_cpk:.2f}, AS13006 floor)": results_marginal_zoom,
        },
        ax=ax2, title="Zoom near achieved uncertainty (linear scale)", log_scale=False,
    )
    ax2.axvspan(*band_sensor_noise_only_um, color="tab:green", alpha=0.15, label="achieved uncertainty: sensor noise only")
    ax2.axvspan(*band_with_systematic_um, color="tab:red", alpha=0.15, label="achieved uncertainty: +10 µm systematic")
    ax2.axvline(reference_uncertainty_m * 1e6, color="gray", linestyle="--", linewidth=1, label="B's weight reference (8 µm)")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Local risk sensitivity at the fixed 8 µm reference stays ~{ratio:.0f}:1 (marginal vs high-capability)\n"
        "regardless of the systematic term -- the bands below are NOT the systematic term's own value, they are\n"
        "where the experiment's achieved uncertainty actually lands; ~26% of the +10 µm-systematic band is\n"
        "reallocatable by placement, so the weighting has proportionally less of that band to work with",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(RESULTS_DIR / "panel7_sensitivity_systematic10um.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel7_sensitivity_systematic10um.png'}")


if __name__ == "__main__":
    result_0, result_10 = reproduce_numbers()
    reproduce_figures(result_10)
