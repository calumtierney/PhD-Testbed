"""Figures for the wing-box inspection experiment (`planning.inspection`,
`planning.inspection_experiment`, `planning.surrogate`).

Same ground rules as `visualization.plotting`: every function returns a
`(figure, axes)` pair and does no I/O; matplotlib only (CLAUDE.md §6, §8).
Nothing here draws a scaled covariance ellipsoid, so there is no exaggeration
factor to declare -- geometry is drawn at true size and uncertainty is shown
by colour with a colour bar in micrometres. Colour is never the only carrier
of meaning: tasks are distinguished by marker as well as hue, unmeasured
points by an open marker, stations by a numbered star (the anchor, station 1,
a filled star).
"""
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from geometry.wing_box import TASK_NAMES

TASK_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v", 6: "P"}
TASK_COLOURS = {1: "#1b6ca8", 2: "#59a5d8", 3: "#7b3fa0", 4: "#2e8b57", 5: "#d9822b", 6: "#a9561f"}
PLAN_COLOURS = {"U": "#444444", "R-all": "#7b3fa0", "R": "#c0392b"}


def _stations(ax, positions_m: np.ndarray, three_d: bool = False, dims=(0, 1)):
    for i, p in enumerate(positions_m):
        coords = p if three_d else p[list(dims)]
        ax.scatter(*coords, marker="*", s=230 if i == 0 else 150, c="k" if i == 0 else "w",
                   edgecolors="k", linewidths=1.2, zorder=10)
        offset = (6, 6) if not three_d else None
        if not three_d:
            ax.annotate(str(i + 1), coords, textcoords="offset points", xytext=offset, fontsize=8, zorder=11)


def plot_layout_views(part_points_m: np.ndarray, task_ids: np.ndarray, measured: np.ndarray,
                      stations_m: np.ndarray, title: str, axes=None):
    """Plan (x-y) and side (x-z) views of a layout: points by task (colour
    and marker), unmeasured points as open markers with a cross, stations
    as numbered stars."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    else:
        fig = axes[0].figure
    for ax, dims, labels in ((axes[0], (0, 1), ("x, chord (m)", "y, span (m)")),
                             (axes[1], (0, 2), ("x, chord (m)", "z, height (m)"))):
        for t, name in TASK_NAMES.items():
            sel = task_ids == t
            on = sel & measured
            off = sel & ~measured
            ax.scatter(part_points_m[on][:, dims[0]], part_points_m[on][:, dims[1]], s=9, marker=TASK_MARKERS[t],
                       c=TASK_COLOURS[t], alpha=0.8, linewidths=0)
            if off.any():
                ax.scatter(part_points_m[off][:, dims[0]], part_points_m[off][:, dims[1]], s=22, marker=TASK_MARKERS[t],
                           facecolors="none", edgecolors="k", linewidths=0.8)
        _stations(ax, stations_m, dims=dims)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    axes[0].set_title(title, fontsize=10, loc="left")
    return fig, axes


def task_legend_handles() -> Sequence[Line2D]:
    handles = [Line2D([], [], marker=TASK_MARKERS[t], ls="", color=TASK_COLOURS[t], label=TASK_NAMES[t]) for t in TASK_NAMES]
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="not measured"))
    handles.append(Line2D([], [], marker="*", ls="", mfc="k", mec="k", ms=11, label="station (1 = anchor)"))
    return handles


def plot_plan_layouts(scenario_part, outcomes: Dict[str, object], positions: Dict[str, np.ndarray], suptitle: str):
    """One row per plan: layout views, side by side. `outcomes[name]` is a
    `planning.inspection.LayoutOutcome`; `positions[name]` its stations."""
    names = list(outcomes)
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 4.4 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        o = outcomes[name]
        title = (f"Plan {name}: {len(positions[name])} stations, {o.n_measured}/{len(o.measured)} points measured, "
                 f"P(escape) {o.p_escape:.2e}, E[false rejects] {o.e_fr:.1f}")
        plot_layout_views(scenario_part.points_m, scenario_part.task_ids, o.measured, positions[name], title, axes[row])
    fig.legend(handles=task_legend_handles(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig, axes


def plot_uncertainty_maps(scenario_part, outcomes: Dict[str, object], tolerance_um: float, tur_limit_um: float):
    """Every point coloured by its standard uncertainty along the surface
    normal (um): plan view (x-y) on the top row, side view (x-z) below, one
    column per plan, one shared colour scale; unmeasured points are grey
    crosses. Points are drawn worst-last, so where the side view stacks many
    points along the span the highest uncertainty is the one visible. The
    TUR limit is a contour level marked on the colour bar. Stations are on
    the layout figure, not here, so the part can fill the panel."""
    names = list(outcomes)
    fig, axes = plt.subplots(2, len(names), figsize=(5.8 * len(names), 7.2), squeeze=False)
    all_u = np.concatenate([outcomes[n].u_m[outcomes[n].measured] * 1e6 for n in names])
    vmax = float(np.nanpercentile(all_u, 99.5))
    pts = scenario_part.points_m
    sc = None
    for col, name in enumerate(names):
        o = outcomes[name]
        order = np.argsort(np.nan_to_num(o.u_m, nan=-1.0))  # worst last
        for row, dims, labels in ((0, (0, 1), ("x, chord (m)", "y, span (m)")), (1, (0, 2), ("x, chord (m)", "z, height (m)"))):
            ax = axes[row, col]
            idx = order[o.measured[order]]
            sc = ax.scatter(pts[idx][:, dims[0]], pts[idx][:, dims[1]], c=o.u_m[idx] * 1e6, cmap="viridis", vmin=0,
                            vmax=vmax, s=16, linewidths=0)
            off = ~o.measured
            if off.any():
                ax.scatter(pts[off][:, dims[0]], pts[off][:, dims[1]], marker="x", c="0.4", s=18, linewidths=0.9)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.25)
        axes[0, col].set_title(f"Plan {name}: {len(o.visible)} stations, {o.n_measured} points measured", fontsize=10, loc="left")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
    cbar.set_label("standard uncertainty along the surface normal, u (um)")
    if tur_limit_um <= vmax:
        cbar.ax.axhline(tur_limit_um, c="r", lw=1.6)
        cbar.ax.text(1.2, tur_limit_um / vmax, f"TUR 4:1 limit\n{tur_limit_um:.2f} um", transform=cbar.ax.transAxes,
                     fontsize=7, color="r", va="center")
    fig.suptitle(f"Per-point uncertainty, tolerance +-{tolerance_um:.0f} um. Geometry at true scale; colour = u; x = not measured.",
                 fontsize=10)
    return fig, axes


def plot_k_curves(runs: Dict[str, object], max_p_escape: float, max_e_fr: float):
    """Left: constraint violation of each plan's best layout vs station
    count (0 = satisfied). Right/centre: the two risk metrics of plan R's
    best layout per K against their limits."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, run in runs.items():
        ks = [r.n_stations for r in run.sweep]
        viol = [min(r.violation, 10.0) if np.isfinite(r.violation) else 10.0 for r in run.sweep]
        axes[0].plot(ks, viol, "-o", c=PLAN_COLOURS.get(name, "k"), label=f"plan {name}")
        sat = [r for r in run.sweep if r.satisfied]
        if sat:
            axes[0].scatter([sat[0].n_stations], [0.0], s=220, facecolors="none", edgecolors=PLAN_COLOURS.get(name, "k"), lw=2)
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].set_xlabel("stations")
    axes[0].set_ylabel("constraint violation (0 = met; symlog)")
    axes[0].set_title("Best violation reached per station count\n(circle: fewest stations that meet the constraint)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    run = runs.get("R")
    if run is not None:
        feas = [r for r in run.sweep if r.outcome.feasible]
        ks = [r.n_stations for r in feas]
        axes[1].plot(ks, [r.outcome.p_escape for r in feas], "-o", c=PLAN_COLOURS["R"])
        axes[1].axhline(max_p_escape, c="k", ls="--", lw=1)
        axes[1].set_ylabel("P(at least one falsely accepted point)")
        axes[1].set_title("Plan R: escape probability vs limit (dashed)", fontsize=9)
        axes[2].plot(ks, [r.outcome.e_fr for r in feas], "-o", c=PLAN_COLOURS["R"])
        axes[2].axhline(max_e_fr, c="k", ls="--", lw=1)
        axes[2].set_ylabel("expected falsely rejected points per part")
        axes[2].set_title("Plan R: false rejections vs limit (dashed)", fontsize=9)
        for ax in axes[1:]:
            ax.set_xlabel("stations")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_limit_sweep(slacks: Sequence[float], stations_r: Sequence[Optional[int]], stations_r_all: Sequence[Optional[int]],
                     stations_u: Optional[int], p_escape_limits: Sequence[float], title: str):
    """Stations needed vs how strict the risk limit is (x: limit as a
    multiple of what the TUR rule guarantees; the top axis gives the
    corresponding P(escape) limit). None = no layout found in the sweep."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    top = max([k for k in list(stations_r) + list(stations_r_all) + [stations_u] if k] + [1]) + 1

    def series(values):
        return [v if v is not None else top for v in values]

    ax.plot(slacks, series(stations_r), "-o", c=PLAN_COLOURS["R"], label="plan R (unmeasured points allowed)")
    ax.plot(slacks, series(stations_r_all), "--s", c=PLAN_COLOURS["R-all"], label="plan R-all (every point measured)")
    if stations_u:
        ax.axhline(stations_u, c=PLAN_COLOURS["U"], ls=":", label=f"plan U (TUR 4:1): {stations_u} stations")
    ax.set_xscale("log")
    ax.set_xlabel("risk limit as a multiple of the TUR rule's own guarantee")
    ax.set_ylabel("stations needed")
    ax.set_yticks(range(1, top + 1))
    ax.set_yticklabels([str(i) for i in range(1, top)] + [f">{top - 1}"])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax2 = ax.secondary_xaxis("top", functions=(lambda s: s * p_escape_limits[0] / slacks[0], lambda p: p * slacks[0] / p_escape_limits[0]))
    ax2.set_xlabel("P(escape) limit")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_surrogate_parity(truth: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray], report: Dict[str, Dict[str, float]],
                          label: str):
    """Predicted vs exact for each regression target on held-out layouts."""
    names = [("coverage", "fraction of points measured"), ("log10_p_escape", "log10 P(escape)"),
             ("log10_e_fr", "log10 E[false rejects]")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    feasible = truth["feasible"]
    for ax, (key, text) in zip(axes, names):
        y, p = truth[key][feasible], prediction[key][feasible]
        ax.scatter(y, p, s=6, alpha=0.4, c="#1b6ca8", linewidths=0)
        lims = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lims, lims, "k--", lw=1)
        r = report[key]
        ax.set_title(f"{text}\nR$^2$ {r['r2']:.3f}, Spearman {r['spearman']:.3f}, MAE {r['mae']:.3g}, n={r['n']}", fontsize=9)
        ax.set_xlabel("exact evaluator")
        ax.set_ylabel("surrogate")
        ax.grid(alpha=0.25)
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    return fig, axes
