"""Figures for the plan review: a plan view of every plan's stations, and the risk of each characteristic under each plan.

`viz/` imports from `src/`, never the reverse (CLAUDE.md §8): everything drawn here is read from the review's own results
(`review.core.PlanReview`, `compare_plans`) and the scene; nothing is calculated.

Conventions (CLAUDE.md §8): units on every axis; no reliance on colour alone (every plan has its own marker shape as well as its
colour, an infeasible plan says INFEASIBLE in words); station positions marked and numbered; the plan view is true scale (equal
aspect, nothing exaggerated) and says so. The instrument model, decision rule and any placeholder or assumption are printed on the
figure, because a figure that is separated from its caption must still say what it is. Vector output (SVG) alongside PNG.

Colours: the categorical order of the reference palette (validated light-mode: adjacent colour-vision separation and lightness band
pass; four slots fall below 3:1 contrast on white, which the legend and the markers' shapes relieve), in fixed order by plan number.
"""
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from review.core import PlanReview

SERIES_COLOURS = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
CLUSTER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
INK, INK_SECONDARY, GRID = "#191919", "#666666", "#dcdcdc"
MAX_SERIES = len(SERIES_COLOURS)


def _caption(review: PlanReview) -> str:
    prov = review.provenance
    parts = [prov["tracker_label"], prov["decision_rule"].split(" (")[0], prov["uncertainty_reference"]["method"].split(" (")[0]]
    flags = []
    if prov["placement_assumed"]:
        flags.append("PLACEMENT ASSUMED")
    if prov["process_priors_include_placeholder"]:
        flags.append("PLACEHOLDER process statistics")
    flags.append("reference network " + ("assumed by rule" if prov["reference_network"].startswith("assumed") else "synthetic"))
    return " | ".join(parts) + "\n" + " | ".join(flags) + f" | scene {prov['scene']['scene_hash'][:8]} | commit {prov['git']['commit'][:8]}"


def plan_view_figure(scene, reviews: Sequence[PlanReview], max_columns: int = 3):
    """Plan view, true scale: the station track, the part's extent, reference nests, targets (marker shape = cluster) and, in one
    panel per plan, that plan's stations numbered in the order listed (S1 is the anchor)."""
    n = len(reviews)
    columns = min(max_columns, n)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 3.6 * rows + 1.6), squeeze=False)
    track = scene.track
    s = np.linspace(0.0, track.perimeter_m, 400)
    path = np.array([track.position_xy_m(v) for v in s])
    points = scene.scenario.target_points_m
    low, high = scene.part_extent_xy_m()
    clusters = list(dict.fromkeys(scene.clusters))
    char_targets = {c.target_indices[0]: scene.clusters[k] for k, c in enumerate(scene.scenario.characteristics)}
    others = [i for i in range(scene.n_part_targets) if i not in char_targets]
    nests = points[scene.n_part_targets:]
    handles: Dict[str, object] = {}
    # one drawing extent for every panel (the cell: track, nests and part, with a margin), so panels compare and the scale is true; a
    # station outside it is named in words rather than allowed to rescale its panel
    extent = np.vstack([path, points[:, :2], low[None, :], high[None, :]])
    margin = 1.5
    x_limits = (extent[:, 0].min() - margin, extent[:, 0].max() + margin)
    y_limits = (extent[:, 1].min() - margin, extent[:, 1].max() + margin)
    for index, ax in enumerate(axes.ravel()):
        if index >= n:
            ax.axis("off")
            continue
        review = reviews[index]
        ax.plot(path[:, 0], path[:, 1], color="0.6", lw=0.9, ls="--", label="station track", zorder=1)
        ax.add_patch(plt.Rectangle(low, *(high - low), fill=False, ec=INK, lw=1.1, label="part extent", zorder=2))
        if len(nests):
            ax.scatter(nests[:, 0], nests[:, 1], marker="s", s=16, facecolors="none", edgecolors="0.45", lw=0.8, label="reference nests", zorder=2)
        if others:
            ax.scatter(points[others, 0], points[others, 1], marker=".", s=10, color="0.45", label="datum / tie targets", zorder=2)
        for j, cluster in enumerate(clusters):
            idx = [t for t, c in char_targets.items() if c == cluster]
            ax.scatter(points[idx, 0], points[idx, 1], marker=CLUSTER_MARKERS[j % len(CLUSTER_MARKERS)], s=28, color=INK,
                       label=f"characteristics, cluster {j + 1}", zorder=3)
        off_drawing = []
        for st in review.stations:
            x, y, z = st["position_cell_m"]
            if not (x_limits[0] <= x <= x_limits[1] and y_limits[0] <= y <= y_limits[1]):
                off_drawing.append(f"S{st['index']} at ({x:.0f}, {y:.0f}, {z:.1f}) m")
                continue
            marker = ax.scatter([x], [y], marker="*", s=230, color=SERIES_COLOURS[index % MAX_SERIES], edgecolors=INK, linewidths=0.8, zorder=5,
                                label="tracker station (numbered)")
            ax.annotate(f"S{st['index']}\n{z:.1f} m", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=7.5, color=INK, zorder=6)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal")
        if off_drawing:
            ax.text(0.5, 0.04, "off the drawing: " + "; ".join(off_drawing), transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
                    color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK, lw=0.8), zorder=7)
        ax.set_xlabel("cell x (m)")
        ax.set_ylabel("cell y (m)")
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=8)
        header = f"#{index + 1}  {review.plan['name']}  [{review.plan['author_role']}]"
        if review.feasible:
            t = review.totals
            sub = (f"{review.n_stations} stations, {review.n_shots} shots | wrong decisions {t['expected_wrong_decisions']:.3g} | "
                   f"worst TUR {t['worst_tur']:.1f}:1 | largest u {t['largest_u_um']:.1f} um")
        else:
            reasons = "; ".join(f["detail"] for f in review.findings if f["severity"] == "blocking")
            sub = f"INFEASIBLE, not scored: {reasons[:150]}{'...' if len(reasons) > 150 else ''}"
        ax.set_title(header[:70] + "\n" + sub, fontsize=8, loc="left", color=INK)
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.suptitle("Plan view, true scale (nothing exaggerated). S1 is the anchor the other stations are tied to; tracker heights shown beside each station.",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(5, len(handles)), fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    fig.text(0.5, 0.045, _caption(reviews[0]), ha="center", va="top", fontsize=7, color=INK_SECONDARY)
    return fig


def risk_by_characteristic_figure(reviews: Sequence[PlanReview]):
    """For each characteristic, the expected wrong decisions (PFA + PFR) under each plan, one marker per plan (shape and colour by
    plan number, jittered sideways so plans do not overlap). A hollow marker is a characteristic the plan does not measure,
    priced at its process prior. Infeasible plans have no numbers and are listed in the legend as such. At most eight plans are drawn."""
    feasible = [(i, r) for i, r in enumerate(reviews) if r.feasible][:MAX_SERIES]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(reviews[0].characteristics or [1] * 6) + 4.0), 5.2))
    if not feasible:
        ax.text(0.5, 0.5, "no plan is feasible: nothing to draw", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig
    names = [c["name"] for c in feasible[0][1].characteristics]
    x = np.arange(len(names))
    width = 0.7
    offsets = np.linspace(-width / 2, width / 2, len(feasible)) if len(feasible) > 1 else np.array([0.0])
    for slot, (i, review) in enumerate(feasible):
        risks = np.array([c["risk"] for c in review.characteristics])
        measured = np.array([c["measured"] for c in review.characteristics])
        colour, marker = SERIES_COLOURS[i % MAX_SERIES], SERIES_MARKERS[i % MAX_SERIES]
        ax.scatter(x[measured] + offsets[slot], risks[measured], marker=marker, s=46, color=colour, edgecolors=INK, linewidths=0.6, zorder=3,
                   label=f"#{i + 1} {review.plan['name'][:34]}")
        if (~measured).any():
            ax.scatter(x[~measured] + offsets[slot], risks[~measured], marker=marker, s=46, facecolors="none", edgecolors=colour, linewidths=1.4,
                       zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n if len(n) <= 20 else n[:19] + "..." for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("characteristic")
    ax.set_ylabel("expected wrong decisions per part, PFA + PFR (tabulated risk curve)")
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    skipped = [f"#{i + 1} {r.plan['name'][:34]} (INFEASIBLE: no numbers)" for i, r in enumerate(reviews) if not r.feasible]
    ax.legend(handles + [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=INK_SECONDARY, label="hollow: not measured, priced at its process prior")]
              + [plt.Line2D([], [], ls="") for _ in skipped], labels + ["hollow: not measured, priced at its process prior"] + skipped,
              fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Risk of each characteristic under each plan", fontsize=10, loc="left", color=INK)
    fig.text(0.01, 0.01, _caption(reviews[0]).replace("\n", "  |  "), fontsize=6.5, color=INK_SECONDARY, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def figure_bytes(fig, fmt: str) -> bytes:
    """The figure as PNG (200 dpi) or SVG bytes, for a download button or a file."""
    if fmt not in ("png", "svg"):
        raise ValueError("fmt must be 'png' or 'svg'")
    buffer = BytesIO()
    fig.savefig(buffer, format=fmt, dpi=200, bbox_inches="tight")
    return buffer.getvalue()


def save_figure(fig, stem) -> List[str]:
    """Write `<stem>.png` and `<stem>.svg` (vector, for anything that goes in a document)."""
    stem = Path(stem)
    written = []
    for fmt in ("png", "svg"):
        path = stem.with_suffix("." + fmt)
        path.write_bytes(figure_bytes(fig, fmt))
        written.append(str(path))
    return written
