"""Figures for the paper (docs/paper/manuscript.md), all from the testbed's own
code and result files; nothing is drawn by hand.

    python -m viz.make_paper_figures

Reads results/cell_results_datum.json, cell_results_datum_no_network.json and
cell_sensitivity.json (written by `viz.run_cell_experiment --frame datum`,
`... --no-network` and `viz.run_cell_sensitivity`) and writes
results/paper_fig1_verification ... paper_fig7_sensitivity as .png and .svg.

Every cell dimension and process prior is a placeholder (TODO(source)); the
figures say so where a reader could take a number for a real cell's.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker
from planning.cell import CellLayoutScorer, DatumCellScorer, ers_nests, factory_cell_scenario, layout_poses, part_datum_indices
from planning.experiment import long_part_track
from planning.objectives import evaluate_plan
from risk.decision import ProcessKnowledge, best_rule_risk, decision_risk, unmeasured_risk

RESULTS = Path(__file__).resolve().parent.parent / "results"
UM = 1e6
COLOURS = {"A0": "#555555", "A1": "#d55e00", "C": "#cc2f6a", "T": "#0072b2", "R5": "#009e73", "B": "#009e73"}
MARKERS = {"A0": "^", "A1": "s", "C": "o", "T": "v", "R5": "D", "B": "D"}
LABELS = {"A0": "A0: covariance trace (A-optimal)", "A1": "A1: uncertainty along the tolerances",
          "C": "C: decision risk", "T": "T: worst-case uncertainty (TUR plan)"}
SEGMENT_COLOURS = {"-y row": "#1b6ca8", "+x loop": "#e0a030", "+y row": "#c8443c", "-x loop": "#0e7c86"}
TOL = 50e-6
MARGINAL, HIGH = (38e-6, 3e-6), (0.0, 50e-6 / 6.0)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=180 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


# -- Figure 1: verification ---------------------------------------------------------------------
def figure_verification():
    tracker = LaserTracker()
    station = InstrumentPose(position_m=np.zeros(3))
    ranges = np.geomspace(1.0, 30.0, 40)
    radial, azimuthal, elevation = [], [], []
    for r in ranges:
        cov = tracker.covariance_global(np.array([r, 0.0, 0.0]), station)  # beam along x: y is azimuth-driven, z elevation-driven
        radial.append(np.sqrt(cov[0, 0]) * UM)
        azimuthal.append(np.sqrt(cov[1, 1]) * UM)
        elevation.append(np.sqrt(cov[2, 2]) * UM)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    ax.loglog(ranges, elevation, "-", color="#c8443c", linewidth=2, label="across the beam, elevation-driven")
    ax.loglog(ranges, azimuthal, "--", color="#d55e00", linewidth=2, label="across the beam, azimuth-driven")
    ax.loglog(ranges, radial, "-.", color="#0e7c86", linewidth=2, label="along the beam")
    for value, marker in ((8.4, "o"), (5.9, "s"), (1.2, "D")):
        ax.scatter([2.5], [value], s=70, marker=marker, facecolors="none", edgecolors="k", linewidths=1.6, zorder=5)
    ax.scatter([], [], s=70, marker="o", facecolors="none", edgecolors="k", linewidths=1.6, label="expected from the published sensor noise, 2.5 m")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("standard uncertainty of one shot (µm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    ax.set_title("(a) Instrument model: anisotropic, lateral error grows with range", fontsize=10, loc="left")

    # (b) fast design covariance against the full least-squares solve, every target of one layout
    track = long_part_track()
    bare = factory_cell_scenario(track=track, with_reference_network=False)
    layout = [(15.9, 1.28), (27.5, 1.45), (45.0, 1.17), (54.4, 1.18)]
    poses = layout_poses(layout, bare, track)
    solve = evaluate_plan(poses, bare).network_solve
    from network.design import design_covariances_for_poses

    visible = CellLayoutScorer(bare).visible(poses)
    design = design_covariances_for_poses(bare.target_points_m, [bare.anchor_pose] + poses, bare.tracker, visible, 0)
    full = np.array([np.sqrt(np.diag(solve.target_point_covariance(i))) for i in range(len(bare.target_points_m))]).ravel() * UM
    fast = np.sqrt(np.einsum("tii->ti", design.point_covariance_m2)).ravel() * UM
    ax = axes[1]
    ax.loglog(full, fast, "o", color="#0072b2", markersize=4, alpha=0.7)
    lim = [min(full.min(), fast.min()) * 0.8, max(full.max(), fast.max()) * 1.25]
    ax.plot(lim, lim, "k:", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("full least-squares network solve (µm)")
    ax.set_ylabel("block-elimination design covariance (µm)")
    ax.grid(alpha=0.25, which="both")
    ax.text(0.04, 0.9, f"78 coordinate uncertainties, 5 stations\nlargest relative difference {np.abs(fast / full - 1).max():.1e}",
            transform=ax.transAxes, fontsize=8.5, va="top")
    ax.set_title("(b) Network covariance: fast form against the full solve", fontsize=10, loc="left")

    # (c) decision risk against uncertainty for the two process clusters
    ax = axes[2]
    u = np.geomspace(0.5e-6, 60e-6, 50)
    for (mean, std), colour, name in ((MARGINAL, "#c8443c", "marginal (Cpk 1.33)"), (HIGH, "#0e7c86", "high-capability (Cpk 2.0)")):
        risk = [sum(decision_risk(ProcessKnowledge.known(mean, std), -TOL, TOL, x)) for x in u]
        ax.loglog(u * UM, risk, color=colour, linewidth=2, label=name)
    ax.set_xlabel("uncertainty along the tolerance, u (µm)")
    ax.set_ylabel("PFA + PFR, simple acceptance")
    ax.set_ylim(1e-10, 1.0)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(c) Decision risk: the same u, very different risk (±50 µm)", fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "paper_fig1_verification")


# -- Figure 2: the cell and two plans -------------------------------------------------------------
def draw_cell(ax, scenario, track, with_network=True):
    ax.add_patch(Rectangle((0, -1.25), 15.0, 2.5, facecolor="0.86", edgecolor="k", linewidth=1.3, zorder=2))
    s_dense = np.linspace(0, track.perimeter_m, 600)
    xy = np.array([track.position_xy_m(v) for v in s_dense])
    for name, colour in SEGMENT_COLOURS.items():
        sel = np.array([track.segment(v) == name for v in s_dense])
        ax.plot(np.where(sel, xy[:, 0], np.nan), np.where(sel, xy[:, 1], np.nan), color=colour, linewidth=2.2, zorder=1)
    points = scenario.target_points_m
    datum = part_datum_indices(scenario)
    ax.scatter(points[datum, 0], points[datum, 1], s=18, marker="x", color="0.25", linewidths=1.0, zorder=5)
    ax.scatter(points[:3, 0], points[:3, 1], s=40, color="#0e7c86", edgecolors="k", zorder=6)
    ax.scatter(points[3:6, 0], points[3:6, 1], s=40, color="#c8443c", edgecolors="k", zorder=6)
    if with_network:
        nests, _ = ers_nests()
        ax.scatter(nests[:, 0], nests[:, 1], s=34, marker="P", color="#7b3294", edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-9.5, 24.5)
    ax.set_ylim(-7.6, 7.6)
    ax.grid(alpha=0.2)


def figure_cell(datum_results):
    track = long_part_track()
    scenario = factory_cell_scenario(track=track)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    for ax, k in zip(axes, (3, 4)):
        draw_cell(ax, scenario, track)
        for name, dy in (("A1", 0.0), ("C", 0.0)):
            entry = datum_results["layouts"][f"{name}|{k}"]
            for pose in layout_poses(entry["stations"], scenario, track):
                ax.scatter([pose.position_m[0]], [pose.position_m[1]], s=230 if name == "A1" else 90, marker=MARKERS[name],
                           facecolors="none" if name == "A1" else COLOURS[name], edgecolors=COLOURS[name], linewidths=2.2, zorder=8)
        lines = []
        for name in ("A1", "C"):
            entry = datum_results["layouts"][f"{name}|{k}"]
            u = list(entry["u_um"].values())
            lines.append(f"{name}: u = {min(u[:3]):.1f}–{max(u[:3]):.1f} µm (high-cap.), {min(u[3:]):.1f}–{max(u[3:]):.1f} µm (marginal); "
                         f"risk {entry['risk']['simple|known']:.4f}")
        ax.set_title(f"({'ab'[k - 3]}) {k} stations\n" + "\n".join(lines), fontsize=9, loc="left")
        ax.set_xlabel("x, along the part (m)")
    axes[0].set_ylabel("y (m)")
    handles = [plt.Line2D([], [], color=c, linewidth=2.2, label=f"track: {n}") for n, c in SEGMENT_COLOURS.items()]
    handles += [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#0e7c86", markeredgecolor="k", markersize=7, label="high-capability characteristics (+y face)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#c8443c", markeredgecolor="k", markersize=7, label="marginal characteristics (−y face)"),
        plt.Line2D([], [], marker="x", color="0.25", linestyle="", label="datum targets on the part"),
        plt.Line2D([], [], marker="P", color="w", markerfacecolor="#7b3294", markeredgecolor="k", markersize=8, label="reference nests on the cell"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor="none", markeredgecolor=COLOURS["A1"], markeredgewidth=2.2, markersize=12, label="stations, plan A1"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=COLOURS["C"], markeredgecolor=COLOURS["C"], markersize=8, label="stations, plan C"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("The cell (placeholder dimensions) and where two objectives stand the tracker", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    save(fig, "paper_fig2_cell_and_plans")


# -- Figure 3: risk against station count ------------------------------------------------------
def figure_risk_vs_stations(datum_results, no_network):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    ax = axes[0]
    for name in ("A0", "A1", "T", "C"):
        ks, risks = [], []
        for k in range(1, 7):
            entry = datum_results["layouts"][f"{name}|{k}"]
            if entry["feasible"]:
                ks.append(k)
                risks.append(entry["risk"]["simple|known"])
        ax.semilogy(ks, risks, "-", color=COLOURS[name], marker=MARKERS[name], markersize=8, linewidth=2, label=LABELS[name])
    c = {k: datum_results["layouts"][f"C|{k}"]["risk"]["simple|known"] for k in range(2, 7)}
    for k in (3, 4):
        ax.axhline(c[k], color="0.6", linewidth=0.8, linestyle=":")
    ax.set_xlabel("stations")
    ax.set_ylabel("expected wrong decisions (PFA + PFR over 6 characteristics)")
    ax.set_xticks(range(2, 7))
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("(a) With the reference network: risk of each objective's best plan", fontsize=10, loc="left")

    ax = axes[1]
    for results, style, tag in ((datum_results, "-", "with reference network"), (no_network, "--", "without")):
        if results is None:
            continue
        for name in ("A1", "C"):
            ks, us = [], []
            for k in range(1, 7):
                entry = results["layouts"][f"{name}|{k}"]
                if entry["feasible"]:
                    ks.append(k)
                    us.append(np.mean(list(entry["u_um"].values())))
            ax.semilogy(ks, us, style, color=COLOURS[name], marker=MARKERS[name], markersize=7, linewidth=2, label=f"{name}, {tag}")
    ax.axhline(6.25, color="k", linewidth=1, linestyle=":")
    ax.text(6.05, 6.6, "TUR 4:1 (6.25 µm)", fontsize=8, ha="right")
    ax.set_yticks([2, 3, 5, 10, 20, 40])
    ax.set_yticklabels(["2", "3", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlabel("stations")
    ax.set_ylabel("mean uncertainty along the tolerance (µm)")
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5)
    ax.set_title("(b) What the reference network buys (datum-referenced uncertainty)", fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "paper_fig3_risk_vs_stations")


# -- Figure 4: where the precision goes ----------------------------------------------------------
def figure_allocation(datum_results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True)
    labels = ["HC 1", "HC 2", "HC 3", "M 1", "M 2", "M 3"]
    for col, k in enumerate((3, 4)):
        width = 0.26
        x = np.arange(6)
        for i, name in enumerate(("A0", "A1", "C")):
            entry = datum_results["layouts"][f"{name}|{k}"]
            u = list(entry["u_um"].values())
            risk = list(entry["risk_by_characteristic"]["simple|known"].values())
            hatch = ["", "//", ".."][i]
            axes[0, col].bar(x + (i - 1) * width, u, width, color=COLOURS[name], edgecolor="k", linewidth=0.5, hatch=hatch, label=LABELS[name])
            axes[1, col].bar(x + (i - 1) * width, risk, width, color=COLOURS[name], edgecolor="k", linewidth=0.5, hatch=hatch)
        axes[0, col].set_title(f"{k} stations", fontsize=10.5, loc="left")
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(1e-9, 1e-1)
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(labels)
        axes[1, col].set_xlabel("characteristic (HC: high-capability, +y face; M: marginal, −y face)")
        for row in (0, 1):
            axes[row, col].grid(alpha=0.25, axis="y")
    axes[0, 0].set_ylabel("uncertainty along the tolerance (µm)")
    axes[1, 0].set_ylabel("PFA + PFR per characteristic")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Where each plan spends its precision, and what that does to each decision", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "paper_fig4_allocation")


# -- Figure 5: decision rule and process knowledge -----------------------------------------------
def figure_decision_rule(datum_results):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    u = np.geomspace(0.5e-6, 40e-6, 36)
    for ax, (mean, std), title in ((axes[0], MARGINAL, "(a) Marginal cluster"), (axes[1], HIGH, "(b) High-capability cluster")):
        for n, colour in ((5, "#7b3294"), (None, "0.35")):
            prior = ProcessKnowledge.from_builds(mean, std, n)
            simple = [sum(decision_risk(prior, -TOL, TOL, x)) for x in u]
            best = [sum(best_rule_risk(prior, -TOL, TOL, x)[:2]) for x in u]
            tag = "5 builds" if n else "process known exactly"
            ax.loglog(u * UM, simple, "--", color=colour, linewidth=1.8, label=f"simple acceptance, {tag}")
            ax.loglog(u * UM, best, "-", color=colour, linewidth=2.2, label=f"conformance-probability rule, {tag}")
            ax.axhline(unmeasured_risk(prior, -TOL, TOL), color=colour, linewidth=1, linestyle=":")
        ax.set_xlabel("uncertainty along the tolerance, u (µm)")
        ax.set_ylabel("PFA + PFR")
        ax.set_ylim(1e-10 if mean == 0 else 1e-6, 1.0)
        ax.grid(alpha=0.25, which="both")
        ax.set_title(title + " (dotted: no measurement)", fontsize=10, loc="left")
    axes[0].legend(fontsize=7.8, loc="lower right")

    ax = axes[2]
    builds = datum_results["builds"]
    x = np.arange(len(builds))
    unmeasured = [datum_results["unmeasured_risk"][f"best|n{n}"] for n in builds]
    simple4 = [datum_results["layouts"]["A1|4"]["risk"][f"simple|n{n}"] for n in builds]
    best4 = [datum_results["layouts"]["A1|4"]["risk"][f"best|n{n}"] for n in builds]
    planned4 = [datum_results["layouts"][f"R{n}|4"]["risk"][f"best|n{n}"] for n in builds]
    width = 0.2
    ax.bar(x - 1.5 * width, unmeasured, width, color="0.75", edgecolor="k", linewidth=0.5, label="no measurement")
    ax.bar(x - 0.5 * width, simple4, width, color=COLOURS["A1"], edgecolor="k", linewidth=0.5, hatch="//", label="plan A1, simple acceptance")
    ax.bar(x + 0.5 * width, best4, width, color=COLOURS["A1"], edgecolor="k", linewidth=0.5, label="plan A1, conformance-probability rule")
    ax.bar(x + 1.5 * width, planned4, width, color=COLOURS["R5"], edgecolor="k", linewidth=0.5, hatch="..", label="plan and rule both set by risk")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} builds" for n in builds])
    ax.set_ylabel("expected wrong decisions, 4 stations")
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.legend(fontsize=7.8)
    ax.set_title("(c) The cell: what the rule and the plan are worth", fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "paper_fig5_decision_rule")


# -- Figure 6: stations needed ----------------------------------------------------------------------
def figure_stations_needed(datum_results):
    risks = {name: {k: datum_results["layouts"][f"{name}|{k}"]["risk"]["simple|known"] for k in range(2, 7)} for name in ("A0", "A1", "T", "C")}
    targets = [risks["C"][k] for k in (2, 3, 4, 5)]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width = 0.2
    x = np.arange(len(targets))
    for i, name in enumerate(("A0", "A1", "T", "C")):
        need = [next((k for k in range(2, 7) if risks[name][k] <= t * (1 + 1e-9)), None) for t in targets]
        heights = [n if n is not None else 6.6 for n in need]
        bars = ax.bar(x + (i - 1.5) * width, heights, width, color=COLOURS[name], edgecolor="k", linewidth=0.5,
                      hatch=["", "//", "xx", ".."][i], label=LABELS[name])
        for bar, n in zip(bars, need):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, str(n) if n is not None else ">6", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"≤ {t:.2g}" for t in targets])
    ax.set_xlabel("target risk (expected wrong decisions over 6 characteristics)")
    ax.set_ylabel("stations needed")
    ax.set_ylim(0, 7.6)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Stations each planning objective needs to reach a given decision risk", fontsize=10.5, loc="left")
    fig.tight_layout()
    save(fig, "paper_fig6_stations_needed")


# -- Figure 7: sensitivity ---------------------------------------------------------------------------
def figure_sensitivity(sensitivity):
    cases = [("baseline", "baseline"), ("floor_1um", "1 µm floor"), ("floor_2um", "2 µm floor"), ("floor_4um", "4 µm floor"),
             ("tol_25um", "±25 µm"), ("tol_100um", "±100 µm"), ("centred_marginal", "centred\nmarginal")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    counts = sensitivity["counts"]
    width = 0.8 / len(counts)
    x = np.arange(len(cases))
    for ax, name in zip(axes, ("A1", "A0")):
        for i, k in enumerate(counts):
            ratios = [sensitivity["cases"][c]["risk_ratio_to_C"][name][str(k)] for c, _ in cases]
            ax.bar(x + (i - (len(counts) - 1) / 2) * width, ratios, width, edgecolor="k", linewidth=0.5,
                   color=plt.cm.viridis(i / (len(counts) - 1) * 0.85), label=f"{k} stations")
        ax.axhline(1.0, color="k", linewidth=1)
        ax.set_yscale("log")
        ticks = [1, 1.5, 2, 3, 4, 6] if name == "A1" else [1, 2, 5, 10, 20, 40]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:g}" for v in ticks])
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in cases], fontsize=8.5)
        ax.set_ylabel(f"risk of plan {name} ÷ risk of plan C")
        ax.grid(alpha=0.25, axis="y", which="both")
        ax.set_title(f"({'ab'[name == 'A0']}) {LABELS[name]}", fontsize=10, loc="left")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does the result survive? One assumption changed at a time (1 = no difference from the risk-based plan)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "paper_fig7_sensitivity")


def main():
    datum_results = load("cell_results_datum.json")
    no_network = load("cell_results_datum_no_network.json")
    sensitivity = load("cell_sensitivity.json")
    figure_verification()
    if datum_results:
        figure_cell(datum_results)
        figure_risk_vs_stations(datum_results, no_network)
        figure_allocation(datum_results)
        figure_decision_rule(datum_results)
        figure_stations_needed(datum_results)
    if sensitivity:
        figure_sensitivity(sensitivity)


if __name__ == "__main__":
    main()
