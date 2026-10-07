"""Regenerate every figure under `results/` in one command (SETUP.md's
"daily use" section):

    python -m viz.make_figures

Each `panelN_*.png` corresponds to one of CLAUDE.md §5's six build-step
gates (steps 1-5) or the step 6 headline experiment; see each function's
docstring for which gate it illustrates and `README.md`'s "Visualizing
results" section for the underlying `visualization.plotting` calls.
Steps 1-5's panels use fixed RNG seeds, so they reproduce identically
run to run; the step 6 panels (`_make_headline_panels`) use
`network.solve.exact_observations` (noiseless, deterministic), so there
is no RNG in that path either -- the whole command is reproducible
end to end, which is the point: `git diff results/` after running this
should be empty.

Takes roughly 2-3 minutes in total, almost all of it the two
`run_headline_experiment` calls for the step 6 panels (objective C's
per-candidate risk integration dominates -- see
`planning.experiment.run_headline_experiment`'s own docstring).
"""
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from characteristics.characteristic import Characteristic, evaluate_characteristic
from characteristics.datum import Datum
from characteristics.tolerances import ParallelismTolerance, PositionTolerance, ProfileTolerance
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker
from network.solve import (
    exact_observations,
    mean_positional_uncertainty_m,
    perturbed_initial_guess,
    simulate_observations,
    solve_network,
)
from planning.experiment import (
    HIGH_CAPABILITY_CLUSTER,
    MARGINAL_CLUSTER,
    TOLERANCE_ZONE_WIDTH_M,
    default_two_cluster_scenario,
    run_headline_experiment,
)
from risk.jcgm106 import ClusterAssignment, DecisionRule, evaluate_conformity_risk
from visualization.plotting import (
    _auto_ellipsoid_scale,
    plot_characteristic_comparison,
    plot_network_result,
    plot_plan_comparison,
    plot_risk_vs_uncertainty,
    plot_scene,
    plot_uncertainty_vs_station_count,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Wang, Forbes & Maropoulos 2014's own mean-uncertainty-vs-station-count
# figures (CLAUDE.md §4d) -- a different scene and instrument, overlaid
# for *shape* comparison only (panel3), never a direct number match.
_WANG_FORBES_MAROPOULOS_REFERENCE_UM = {1: 26.5, 2: 16.6, 3: 13.7, 4: 10.7}


def _look_at_pose(position_m: np.ndarray, aim_point_m: np.ndarray) -> InstrumentPose:
    """A station pose whose local +x axis (this codebase's boresight
    direction) points from `position_m` at `aim_point_m` -- figure
    scaffolding only, not something the solver itself needs."""
    x_axis = aim_point_m - position_m
    x_axis = x_axis / np.linalg.norm(x_axis)
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference_up, x_axis)) > 0.99:
        reference_up = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference_up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    orientation = np.column_stack([x_axis, y_axis, z_axis])
    return InstrumentPose(position_m=position_m, orientation=orientation)


def make_panel1_scene():
    """Step 1 gate, illustrated: a single station's per-point error
    ellipsoids -- flattened discs, thin along the beam, wide across it."""
    rng = np.random.default_rng(0)
    target_points_m = rng.uniform(low=[-1, -1.5, -1], high=[1, 1.5, 1], size=(10, 3)) + np.array(
        [3.0, 0.0, 1.0]
    )
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=target_points_m, instrument_pose=pose)
    tracker = LaserTracker()
    fig, ax = plot_scene(
        scene, tracker=tracker,
        title="Step 1: per-point error ellipsoids\n(flattened across the beam, per CLAUDE.md §4b)",
    )
    fig.savefig(RESULTS_DIR / "panel1_scene.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel1_scene.png'}")


def make_panel2_network():
    """Step 3 gate, illustrated: a multi-station network solve's
    recovered targets and their full 3x3 covariance ellipsoids."""
    rng = np.random.default_rng(1)
    target_centre_m = np.array([4.0, 0.0, 1.0])
    true_targets_m = rng.uniform(low=[-1, -1, 0], high=[1, 1, 2], size=(12, 3)) + target_centre_m
    station_positions_m = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 4.0, 1.0]), np.array([0.0, -4.0, 1.0])]
    true_poses = [_look_at_pose(p, target_centre_m) for p in station_positions_m]
    tracker = LaserTracker()

    observations = simulate_observations(true_targets_m, true_poses, tracker, rng)
    initial_targets_m, initial_poses = perturbed_initial_guess(
        true_targets_m, true_poses, position_noise_m=0.05, rotation_noise_rad=0.01, rng=rng
    )
    result = solve_network(observations, initial_targets_m, initial_poses, tracker, anchor_index=0)

    fig, ax = plot_network_result(
        result, true_target_points_m=true_targets_m,
        title="Step 3: multi-station network solve\n(solved vs true target positions)",
    )
    fig.savefig(RESULTS_DIR / "panel2_network.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel2_network.png'}")


def make_panel3_uncertainty():
    """Step 3 gate, illustrated: mean positional uncertainty falls and
    flattens as stations are added."""
    rng = np.random.default_rng(1)
    target_centre_m = np.array([4.0, 0.0, 1.0])
    true_targets_m = rng.uniform(low=[-1, -1, 0], high=[1, 1, 2], size=(12, 3)) + target_centre_m
    candidate_positions_m = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 4.0, 1.0]),
        np.array([0.0, -4.0, 1.0]),
        np.array([-3.0, 0.0, 3.0]),
        np.array([-3.0, 3.0, -1.0]),
    ]
    tracker = LaserTracker()
    station_counts = list(range(1, 6))
    mean_uncertainties_um = []
    for n_stations in station_counts:
        poses = [_look_at_pose(p, target_centre_m) for p in candidate_positions_m[:n_stations]]
        observations = simulate_observations(true_targets_m, poses, tracker, rng)
        initial_targets_m, initial_poses = perturbed_initial_guess(
            true_targets_m, poses, position_noise_m=0.05, rotation_noise_rad=0.01, rng=rng
        )
        result = solve_network(observations, initial_targets_m, initial_poses, tracker, anchor_index=0)
        mean_uncertainties_um.append(mean_positional_uncertainty_m(result) * 1e6)

    fig, ax = plot_uncertainty_vs_station_count(
        station_counts, mean_uncertainties_um, reference_um=_WANG_FORBES_MAROPOULOS_REFERENCE_UM,
    )
    fig.savefig(RESULTS_DIR / "panel3_uncertainty.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel3_uncertainty.png'}")


def make_panel4_characteristics():
    """Step 4 gate, illustrated: the same point, the same station, three
    tolerance types -- parallelism (datum along the beam, picks up only
    the beam's small radial sigma), profile (datum-free, surface normal
    across the beam, picks up a large transverse sigma), position
    (spherical, picks up the whole ellipsoid's trace). Point placed at
    CLAUDE.md §4b's own 2.5 m reference range so the three bars land on
    that section's own lateral/radial figures (~1.2 / ~5.9 / ~10.3 µm)."""
    point_m = np.array([2.5, 0.0, 0.0])
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    covariance = tracker.covariance_global(point_m, pose)
    covariances = covariance[None, :, :]  # (1, 3, 3): _point_covariance's "raw array" branch

    datum_along_beam = Datum(name="A", normal=np.array([1.0, 0.0, 0.0]))
    characteristics = [
        Characteristic(
            name="parallelism (datum || beam)", target_indices=[0],
            tolerance=ParallelismTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M), datum=datum_along_beam,
        ),
        Characteristic(
            name="profile (normal=y)", target_indices=[0],
            tolerance=ProfileTolerance(zone_width_m=TOLERANCE_ZONE_WIDTH_M, surface_normal=np.array([0.0, 1.0, 0.0])),
        ),
        Characteristic(
            name="position (spherical)", target_indices=[0],
            tolerance=PositionTolerance(zone_diameter_m=TOLERANCE_ZONE_WIDTH_M),
        ),
    ]
    results = [evaluate_characteristic(c, covariances)[0] for c in characteristics]
    labels = [c.name.replace(" (", "\n(") for c in characteristics]

    fig, ax = plot_characteristic_comparison(
        labels, results, title="Step 4 gate: same point, same station, different tolerance types",
    )
    for bar, result in zip(ax.patches, results):
        ax.annotate(
            f"{result.uncertainty_m * 1e6:.2f} µm", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    fig.savefig(RESULTS_DIR / "panel4_characteristics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel4_characteristics.png'}")


def make_panel5_risk():
    """Step 5 gate, illustrated: a high-capability characteristic's risk
    stays near-zero across the whole uncertainty range; a marginal one
    (still AS13006-legal, Cpk~1.33) is sensitive to it throughout."""
    decision_rule = DecisionRule.from_zone_width(TOLERANCE_ZONE_WIDTH_M)
    assignment_high = ClusterAssignment("high-capability characteristic", HIGH_CAPABILITY_CLUSTER)
    assignment_marginal = ClusterAssignment("marginal characteristic", MARGINAL_CLUSTER)

    uncertainties_um = np.linspace(1, 60, 60)
    uncertainties_m = uncertainties_um * 1e-6
    results_high = [evaluate_conformity_risk(assignment_high, u, decision_rule) for u in uncertainties_m]
    results_marginal = [evaluate_conformity_risk(assignment_marginal, u, decision_rule) for u in uncertainties_m]
    r_high_ref = evaluate_conformity_risk(assignment_high, 8e-6, decision_rule)
    r_marginal_ref = evaluate_conformity_risk(assignment_marginal, 8e-6, decision_rule)

    fig, ax = plot_risk_vs_uncertainty(
        uncertainties_m,
        {
            f"high-capability (Cpk~{r_high_ref.process_cpk:.2f})": results_high,
            f"marginal process (Cpk~{r_marginal_ref.process_cpk:.2f}, AS13006 floor)": results_marginal,
        },
        title="Step 5 gate: risk stays near-zero for a high-capability\nprocess regardless of uncertainty",
    )
    fig.savefig(RESULTS_DIR / "panel5_risk.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'panel5_risk.png'}")


def _shared_ellipsoid_scale(net_a, net_b, true_targets_m):
    """One exaggeration factor shared by both panels of a station-layout
    comparison -- using each panel's own auto-scale independently reads
    as if one plan's uncertainty were larger than the other's just
    because of the scale, a real mistake caught earlier in this
    project's history (see git log for "ellipsoid-exaggeration label
    placement")."""

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


def _summary_row(result):
    plan_a1 = result.plan_a1
    plan_b = result.plan_b
    risk_a1 = plan_a1.global_pfa + plan_a1.global_pfr
    risk_b = plan_b.global_pfa + plan_b.global_pfr
    effort_change_pct = (
        (plan_b.summed_characteristic_uncertainty_m - plan_a1.summed_characteristic_uncertainty_m)
        / plan_a1.summed_characteristic_uncertainty_m * 100
    )
    risk_change_pct = (risk_b - risk_a1) / risk_a1 * 100
    return plan_a1, plan_b, effort_change_pct, risk_change_pct


def _make_station_and_bar_panels(result, suffix: str, subtitle: str = None):
    """panel6a_stations{suffix}.png and panel6b_comparison{suffix}.png
    for one `HeadlineExperimentResult` -- used for both the baseline
    (systematic_std_m=0, suffix="") and the 10 µm-systematic run
    (suffix="_systematic10um")."""
    scenario = result.scenario
    true_targets_m = scenario.target_points_m
    net_a = result.search_a1.best.network_solve
    net_b = result.search_b.best.network_solve
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
    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"panel6a_stations{suffix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / f'panel6a_stations{suffix}.png'}")

    plan_a1, plan_b, effort_change_pct, risk_change_pct = _summary_row(result)
    labels = list(plan_a1.per_characteristic_uncertainty_m.keys())
    uncertainty_a = [plan_a1.per_characteristic_uncertainty_m[l] for l in labels]
    uncertainty_b = [plan_b.per_characteristic_uncertainty_m[l] for l in labels]

    title_prefix = "Step 6: per-characteristic uncertainty"
    if subtitle:
        title_prefix += ", 10 µm systematic term"
    fig2, ax = plt.subplots(figsize=(8, 5))
    plot_plan_comparison(
        labels, uncertainty_a, uncertainty_b,
        label_a="A0=A1 (aggregate uncertainty)", label_b="B=C (risk-based)", ax=ax,
        title=f"{title_prefix}\n(effort {effort_change_pct:+.1f}%, risk {risk_change_pct:+.1f}%)",
    )
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / f"panel6b_comparison{suffix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {RESULTS_DIR / f'panel6b_comparison{suffix}.png'}")

    return plan_a1, plan_b


def _make_sensitivity_panel(band_sensor_noise_only_um, band_with_systematic_um=None):
    """panel7_sensitivity.png (baseline: just the curves and the 8 µm
    weight-reference line) or panel7_sensitivity_systematic10um.png
    (also overlays where this experiment's achieved uncertainty actually
    lands, with and without the systematic term -- see that figure's own
    suptitle for why these are NOT bands at x=0/x=10 µm)."""
    decision_rule = DecisionRule.from_zone_width(TOLERANCE_ZONE_WIDTH_M)
    assignment_high = ClusterAssignment("flat (high-capability)", HIGH_CAPABILITY_CLUSTER)
    assignment_marginal = ClusterAssignment("steep (marginal)", MARGINAL_CLUSTER)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    plot_risk_vs_uncertainty(
        uncertainties_m_wide,
        {
            f"flat: high-capability (Cpk~{r_high_ref.process_cpk:.2f})": results_high_wide,
            f"steep: marginal (Cpk~{r_marginal_ref.process_cpk:.2f}, AS13006 floor)": results_marginal_wide,
        },
        ax=ax1, title="Overview (log scale)", log_scale=True,
    )
    plot_risk_vs_uncertainty(
        uncertainties_m_zoom,
        {
            f"flat: high-capability (Cpk~{r_high_ref.process_cpk:.2f})": results_high_zoom,
            f"steep: marginal (Cpk~{r_marginal_ref.process_cpk:.2f}, AS13006 floor)": results_marginal_zoom,
        },
        ax=ax2,
        title=("Zoom near 8 µm reference (linear scale)" if band_with_systematic_um is None
               else "Zoom near achieved uncertainty (linear scale)"),
        log_scale=False,
    )

    if band_with_systematic_um is None:
        ax2.axvline(reference_uncertainty_m * 1e6, color="gray", linestyle="--", linewidth=1, label="B's weight reference (8 µm)")
        ax2.legend(loc="upper left", fontsize=8)
        fig.suptitle(
            f"Local risk sensitivity at the fixed 8 µm reference is ~{ratio:.0f}:1\n"
            "(marginal vs high-capability) -- this is where the step 6 weight ratio comes from",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        out_path = RESULTS_DIR / "panel7_sensitivity.png"
    else:
        ax1.axvspan(*band_sensor_noise_only_um, color="tab:green", alpha=0.15)
        ax1.axvspan(*band_with_systematic_um, color="tab:red", alpha=0.15)
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
        out_path = RESULTS_DIR / "panel7_sensitivity_systematic10um.png"

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_panel6_and_7():
    """The step 6 headline experiment's figures: station layouts and
    per-characteristic bars for the baseline (systematic_std_m=0) and
    10 µm-systematic runs, plus the flat-vs-steep sensitivity curves
    (docs/step6_headline_experiment.md's "uncertainty budget" section)."""
    print("Running headline experiment, systematic_std_m = 0 um ...")
    t0 = time.time()
    result_0 = run_headline_experiment(default_two_cluster_scenario(tracker=LaserTracker(systematic_std_m=0.0)))
    print(f"  ({time.time() - t0:.0f}s)")

    print("Running headline experiment, systematic_std_m = 10 um ...")
    t0 = time.time()
    result_10 = run_headline_experiment(default_two_cluster_scenario(tracker=LaserTracker(systematic_std_m=10e-6)))
    print(f"  ({time.time() - t0:.0f}s)")

    plan_a1_0, plan_b_0 = _make_station_and_bar_panels(result_0, suffix="")
    plan_a1_10, plan_b_10 = _make_station_and_bar_panels(
        result_10, suffix="_systematic10um", subtitle="With a 10 µm placement-irreducible systematic term",
    )

    band_sensor_noise_only_um = (
        min(plan_a1_0.per_characteristic_uncertainty_m.values()) * 1e6,
        max(plan_b_0.per_characteristic_uncertainty_m.values()) * 1e6,
    )
    band_with_systematic_um = (
        min(plan_a1_10.per_characteristic_uncertainty_m.values()) * 1e6,
        max(plan_b_10.per_characteristic_uncertainty_m.values()) * 1e6,
    )
    _make_sensitivity_panel(band_sensor_noise_only_um)
    _make_sensitivity_panel(band_sensor_noise_only_um, band_with_systematic_um)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    make_panel1_scene()
    make_panel2_network()
    make_panel3_uncertainty()
    make_panel4_characteristics()
    make_panel5_risk()
    make_panel6_and_7()


if __name__ == "__main__":
    main()
