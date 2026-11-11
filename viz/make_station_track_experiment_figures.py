"""Figures for the long-part-on-a-station-track experiment
(docs/station_track_experiment.md). Reads results/station_track_results.json
(written by `python -m viz.run_station_track_experiment`) and rebuilds each
chosen plan from its `(s_m, height_m)` stations, so nothing is plotted that was
not solved.

    python -m viz.make_station_track_experiment_figures

Writes, next to the existing panels 1-10 and without touching them:
  results/panel11a_track_plans.png    plan view: the cell, and where each objective put its stations
  results/panel11b_track_heatmap.png  the track as a heat map: score of one added station under A0 and C
  results/panel11c_track_uncertainty_risk.png  per-characteristic uncertainty and risk, by objective

**Every dimension of the cell is a placeholder** (TODO(source), GKN cell layout);
see planning.experiment's LONG_PART_* constants.

The heat map's base plan. No plan with fewer than three added stations is
feasible on this scene (every target must be seen and every station tied to the
anchor), so a single added station cannot be scored on its own. The base is
objective C's best 3-station plan (the same base for both maps, so they are
comparable); the added fourth station is scanned over every position on the
track and every height from 0.8 to 3.0 m. Adding a station to a feasible plan
cannot make it infeasible, so the whole track is scored. Objective values are
shown as fractions of each objective's own range over the scan (A0 is a sum of
variances in m^2, C an expected number of wrong decisions: different units),
with the actual minimum printed.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from geometry.station_track import SEGMENT_NAMES
from planning.experiment import (
    LONG_PART_LENGTH_M,
    LONG_PART_WIDTH_M,
    _plan_report,
    long_part_track,
    long_part_two_cluster_scenario,
)
from planning.objectives import evaluate_plan, objective_a0_trace_covariance, objective_c_direct_risk
from planning.search import TrackStationCost

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_PATH = RESULTS_DIR / "station_track_results.json"
UM = 1e6
SEGMENT_COLOURS = {"-y row": "#1b6ca8", "+x loop": "#e0a030", "+y row": "#c8443c", "-x loop": "#0e7c86"}
OBJECTIVE_STYLE = {  # colour-blind-safe hues, and a marker shape each so overlap and print stay readable
    "A0": ("#555555", "^", 15), "A1": ("#d55e00", "s", 12), "B": ("#009e73", "D", 9), "C": ("#cc2f6a", "o", 6),
}
GROUP_COLOURS = {"cluster1 (high-capability)": "#0e7c86", "cluster2 (marginal)": "#c8443c"}
HEAT_HEIGHTS_M = np.arange(0.8, 3.0001, 0.2)


def poses_from(stations, scenario, track):
    flat = np.array([[s, h] for s, h, _ in stations], dtype=float).reshape(-1)
    cost = TrackStationCost(scenario, objective_a0_trace_covariance, track, len(stations), scenario.target_points_m.mean(axis=0))
    return cost.station_poses(flat)


def _scan_one(args):
    scenario, base, s, h, track = args
    pose = poses_from([(s, h, "")], scenario, track)[0]
    evaluation = evaluate_plan(list(base) + [pose], scenario)
    if evaluation is None:
        return np.nan, np.nan
    return objective_a0_trace_covariance(evaluation, scenario), objective_c_direct_risk(evaluation, scenario)


def plan_view_panel(ax, results, scenario, track, key, title):
    ax.add_patch(Rectangle((0, -LONG_PART_WIDTH_M / 2), LONG_PART_LENGTH_M, LONG_PART_WIDTH_M, facecolor="0.85",
                           edgecolor="k", linewidth=1.5, zorder=2))
    s_dense = np.linspace(0, track.perimeter_m, 800)
    xy = np.array([track.position_xy_m(v) for v in s_dense])
    for name in SEGMENT_NAMES:
        sel = np.array([track.segment(v) == name for v in s_dense])
        ax.plot(np.where(sel, xy[:, 0], np.nan), np.where(sel, xy[:, 1], np.nan), color=SEGMENT_COLOURS[name], linewidth=3, zorder=1)
    for s_tick in np.arange(0, track.perimeter_m, 5.0):
        x, y = track.position_xy_m(s_tick)
        nearest = np.array([np.clip(x, 0.0, LONG_PART_LENGTH_M), np.clip(y, -LONG_PART_WIDTH_M / 2, LONG_PART_WIDTH_M / 2)])
        away = np.array([x, y]) - nearest
        away /= np.linalg.norm(away)
        ax.plot([x, x + 0.25 * away[0]], [y, y + 0.25 * away[1]], color="0.3", linewidth=1, zorder=5)
        ax.text(x + 0.7 * away[0], y + 0.7 * away[1], f"{s_tick:g}", ha="center", va="center", fontsize=7.5, color="0.25")
    x0, y0 = track.position_xy_m(0.0)
    ax.annotate("s = 0", (x0, y0), xytext=(x0 - 2.6, y0 - 1.1), fontsize=8.5, arrowprops=dict(arrowstyle="->"), ha="center")
    points, normals = scenario.target_points_m, scenario.target_normals_m
    ax.scatter(points[:3, 0], points[:3, 1], s=34, marker="o", color=GROUP_COLOURS["cluster1 (high-capability)"], edgecolors="k", zorder=6)
    ax.scatter(points[3:6, 0], points[3:6, 1], s=34, marker="o", color=GROUP_COLOURS["cluster2 (marginal)"], edgecolors="k", zorder=6)
    ax.scatter(points[6:, 0], points[6:, 1], s=16, marker="x", color="0.2", linewidths=1.0, zorder=6)
    ax.scatter([scenario.anchor_pose.position_m[0]], [scenario.anchor_pose.position_m[1]], s=170, marker="*", color="gold",
               edgecolors="k", zorder=8)
    ax.annotate("anchor\n(fixed, −x tip)", scenario.anchor_pose.position_m[:2], xytext=(-7.6, 4.6), fontsize=8.5,
                arrowprops=dict(arrowstyle="->"), ha="center")
    plans = results[key]["plans"]
    for name, plan in plans.items():
        colour, marker, size = OBJECTIVE_STYLE[name]
        stations = [tuple(s) for s in plan["stations"]]
        for pose in poses_from(stations, scenario, track):
            ax.scatter([pose.position_m[0]], [pose.position_m[1]], s=size ** 2 * 1.6, marker=marker, facecolors="none" if name != "C" else colour,
                       edgecolors=colour, linewidths=2.0, zorder=7 + (4 - list(OBJECTIVE_STYLE).index(name)))
    ax.set_aspect("equal")
    ax.set_xlim(-9.2, LONG_PART_LENGTH_M + 9.2)
    ax.set_ylim(-track.y_left_row_m - 1.6, track.y_left_row_m + 1.9)
    ax.grid(alpha=0.2)
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=10.5, loc="left")


def figure_plans(results, scenario, track):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 12.4))
    plan_view_panel(axes[0], results, scenario, track, "ga_3", "(a) Three added stations (plus the anchor): the least that can see and tie the whole part")
    plan_view_panel(axes[1], results, scenario, track, "ga_4", "(b) Four added stations")
    axes[1].set_xlabel("x, along the part (m)", labelpad=2)
    handles = [plt.Line2D([], [], color=c, linewidth=3, label=n) for n, c in SEGMENT_COLOURS.items()]
    handles += [plt.Line2D([], [], marker="o", color="w", markerfacecolor=c, markeredgecolor="k", markersize=8, label=f"targets: {n}")
                for n, c in GROUP_COLOURS.items()]
    handles += [plt.Line2D([], [], marker="x", color="0.2", linestyle="", label="tie points (no tolerance)")]
    for name, (colour, marker, size) in OBJECTIVE_STYLE.items():
        handles.append(plt.Line2D([], [], marker=marker, color="w", markerfacecolor="none" if name != "C" else colour,
                                  markeredgecolor=colour, markeredgewidth=2, markersize=size, label=f"stations chosen by {name}"))
    axes[0].legend(handles=handles, fontsize=7.8, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.04), frameon=False)
    for ax, key in zip(axes, ("ga_3", "ga_4")):
        lines = []
        for name, plan in results[key]["plans"].items():
            lines.append(f"{name}: " + "; ".join(f"s {s:.1f} m, h {h:.1f} m ({seg})" for s, h, seg in plan["stations"]))
        ax.text(0.5, -0.17 if key == "ga_4" else -0.36, "\n".join(lines), transform=ax.transAxes, fontsize=7.6, ha="center", va="top",
                family="monospace")
    fig.suptitle("Station track around a 15 × 2.5 m part: where each objective stands its trackers "
                 "(s ticks every 5 m; PLACEHOLDER cell, TODO(source))", fontsize=11.5, x=0.02, ha="left")
    fig.subplots_adjust(top=0.94, bottom=0.12, hspace=0.78, left=0.06, right=0.985)
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"panel11a_track_plans.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)


def figure_heatmap(results, scenario, track):
    base_stations = sorted(tuple(s) for s in results["ga_3"]["plans"]["C"]["stations"])
    base = poses_from(base_stations, scenario, track)
    s_values = np.arange(0.0, track.perimeter_m, 0.5)
    jobs = [(scenario, base, s, h, track) for h in HEAT_HEIGHTS_M for s in s_values]
    with Pool(4) as pool:
        values = np.array(pool.map(_scan_one, jobs, chunksize=16))
    a0 = values[:, 0].reshape(len(HEAT_HEIGHTS_M), len(s_values))
    c = values[:, 1].reshape(len(HEAT_HEIGHTS_M), len(s_values))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), sharex=True)
    minima = {}
    for ax, grid, name, label in ((axes[0], a0, "A0", "A0: trace of the 3D covariance (sum of variances, m²)"),
                                  (axes[1], c, "C", "C: expected wrong decisions (sum of PFA + PFR)")):
        feasible = np.isfinite(grid)
        low, high = np.nanmin(grid), np.nanmax(grid)
        fraction = np.where(feasible, (grid - low) / (high - low), np.nan)
        ds = s_values[1] - s_values[0]
        mesh = ax.pcolormesh(np.append(s_values, s_values[-1] + ds) - ds / 2, np.append(HEAT_HEIGHTS_M - 0.1, HEAT_HEIGHTS_M[-1] + 0.1),
                             fraction, cmap="viridis_r", vmin=0, vmax=1, shading="flat")
        ax.pcolormesh(np.append(s_values, s_values[-1] + ds) - ds / 2, np.append(HEAT_HEIGHTS_M - 0.1, HEAT_HEIGHTS_M[-1] + 0.1),
                      np.where(feasible, np.nan, 1.0), cmap="Greys", vmin=0, vmax=3, shading="flat", hatch="///", alpha=0.35)
        i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
        minima[name] = (float(s_values[j]), float(HEAT_HEIGHTS_M[i]), track.segment(s_values[j]), float(grid[i, j]))
        ax.scatter([s_values[j]], [HEAT_HEIGHTS_M[i]], marker="*", s=330, color="white", edgecolors="k", linewidths=1.4, zorder=5)
        side = -1 if s_values[j] > 0.7 * track.perimeter_m else 1
        ax.text(s_values[j] + side * 1.0, HEAT_HEIGHTS_M[i] + 0.12, f"minimum {grid[i, j]:.4g}\ns {s_values[j]:.1f} m, h {HEAT_HEIGHTS_M[i]:.1f} m ({minima[name][2]})",
                fontsize=8, color="k", ha="left" if side > 0 else "right",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5), zorder=6)
        edges = np.cumsum([0.0, track.row_length_m, np.pi * track.loop_radius_m, track.row_length_m, np.pi * track.loop_radius_m])
        for seg, a, b in zip(SEGMENT_NAMES, edges[:-1], edges[1:]):
            ax.axvline(a, color="k", linewidth=0.8)
            ax.text((a + b) / 2, 3.14, seg, ha="center", fontsize=8.5, color=SEGMENT_COLOURS[seg], fontweight="bold")
        for s_b, h_b, _ in base_stations:
            ax.scatter([s_b], [h_b], marker="v", s=90, color="white", edgecolors="k", zorder=5)
        ax.axvline(track.tip_minus_x_s_m, color="gold", linewidth=2, zorder=4)  # the anchor stands here
        ax.set_ylim(0.7, 3.25)
        ax.set_ylabel("station height (m)")
        ax.set_title(f"{label} — a fourth station scanned over the whole track", fontsize=9.5, loc="left")
        cb = fig.colorbar(mesh, ax=ax, pad=0.01, aspect=14)
        cb.set_label("fraction of range\n(0 = best)", fontsize=8)
    axes[1].set_xlabel("s along the track (m)")
    axes[1].set_xlim(0, track.perimeter_m)
    ds_min = abs(minima["A0"][0] - minima["C"][0])
    dh_min = abs(minima["A0"][1] - minima["C"][1])
    coincide = ds_min <= 0.6 and dh_min <= 0.11
    if coincide:
        verdict = "The A0 and C minima COINCIDE: the two objectives want the added station in the same place."
    elif minima["A0"][2] == minima["C"][2] and ds_min < 3.0:
        verdict = (f"The A0 and C minima are NOT the same cell but fall in the same stretch of the {minima['C'][2]} "
                   f"(Δs = {ds_min:.1f} m, Δheight = {dh_min:.1f} m).")
    else:
        verdict = "The A0 and C minima are in DIFFERENT places (see stars)."
    fig.suptitle("The track as a heat map: a fourth station added to objective C's best 3-station plan (▼), scored under A0 and under C\n" + verdict,
                 fontsize=10.5, x=0.02, ha="left")
    fig.subplots_adjust(top=0.88, hspace=0.18, left=0.06, right=0.97)
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"panel11b_track_heatmap.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)
    print("heat-map minima:", minima, "| coincide:", coincide)
    return minima, coincide, base_stations


def figure_bars(results, scenario, track):
    names = [c.name for c in scenario.characteristics]
    labels = [n.replace("cluster1_profile_", "HC ").replace("cluster2_profile_", "M ") for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True)
    for col, key in enumerate(("ga_3", "ga_4")):
        u = {o: [] for o in results[key]["plans"]}
        risk = {o: [] for o in results[key]["plans"]}
        for o, plan in results[key]["plans"].items():
            evaluation = evaluate_plan(poses_from([tuple(s) for s in plan["stations"]], scenario, track), scenario)
            report = _plan_report(evaluation, scenario)
            for n in names:
                u[o].append(report.per_characteristic_uncertainty_m[n] * UM)
                r = report.risk_by_characteristic[n]
                risk[o].append(r.probability_false_acceptance + r.probability_false_rejection)
        width = 0.8 / len(u)
        x = np.arange(len(names))
        for k, o in enumerate(u):
            colour = OBJECTIVE_STYLE[o][0]
            axes[0, col].bar(x + (k - 1.5) * width, u[o], width, color=colour, edgecolor="k", linewidth=0.5, label=o, hatch=["", "//", "..", "xx"][k])
            axes[1, col].bar(x + (k - 1.5) * width, risk[o], width, color=colour, edgecolor="k", linewidth=0.5, hatch=["", "//", "..", "xx"][k])
        axes[0, col].axhline(50, color="k", linestyle=":", linewidth=1)
        axes[0, col].text(len(names) - 0.5, 51, "tolerance half-width 50 µm", ha="right", fontsize=8)
        axes[0, col].set_title(f"{key[-1]} added stations", fontsize=10.5, loc="left")
        axes[0, col].set_ylabel("uncertainty along the tolerance, u (µm)")
        axes[1, col].set_ylabel("decision risk per characteristic, PFA + PFR")
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(labels)
        axes[1, col].set_xlabel("characteristic (HC = high-capability, +y face; M = marginal, −y face)")
        for row in (0, 1):
            axes[row, col].grid(alpha=0.25, axis="y")
    axes[0, 0].legend(title="plan chosen by", fontsize=8, ncol=4, loc="upper left")
    fig.suptitle("Per-characteristic uncertainty and risk of each objective's plan (PLACEHOLDER cell, TODO(source))", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"panel11c_track_uncertainty_risk.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)


def main():
    results = json.loads(RESULTS_PATH.read_text())
    track = long_part_track()
    scenario = long_part_two_cluster_scenario(track=track)
    figure_plans(results, scenario, track)
    figure_bars(results, scenario, track)
    minima, coincide, base = figure_heatmap(results, scenario, track)
    results["heatmap"] = {"base_stations": [list(b) for b in base], "minima": {k: list(v) for k, v in minima.items()}, "minima_coincide": bool(coincide)}
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=float))
    print("wrote panel11a/b/c to", RESULTS_DIR)


if __name__ == "__main__":
    main()
