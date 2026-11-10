"""Plan view of the default station-track cell, with a connectivity diagnostic.

    python -m viz.make_station_track_figure

Panel (a): the part footprint, the track (rows and end loops, s ticks every
5 m, s = 0 and the direction of travel), the candidates at 1 m spacing, one
*illustrative* blocked zone, and the probe targets used in (b) and (c).
Panel (b): how many probe targets each candidate station can see, by group.
Panel (c): which pairs of candidates share at least 3 probe targets, the
minimum for one station to be tied to another in the network solve
(`planning.objectives.MIN_TARGETS_PER_NON_ANCHOR_STATION`).

**Every dimension is a placeholder** (part 15 x 2.5 m on a jig, underside 0.5 m,
top 2.0 m; standoff 3 m; heights 1.0 and 2.4 m): TODO(source), GKN cell
layout. The probe targets are a provisional stand-in for the scenario that is
built later, used only to see which stations can see what. Visibility uses the
repo's own limits (range 30 m from CLAUDE.md §1; incidence 80 deg, the
convention `planning.experiment.WING_MAX_INCIDENCE_ANGLE_RAD` already uses).
Writes results/station_track_cell.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from geometry.mesh import box_mesh
from geometry.station_track import SEGMENT_NAMES, StationTrack
from geometry.visibility import VisibilityLimits, visibility_matrix
from planning.candidates import track_candidate_poses

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LENGTH_M, WIDTH_M, STANDOFF_M = 15.0, 2.5, 3.0
BOTTOM_M, TOP_M = 0.5, 2.0
HEIGHTS_M = (1.0, 2.4)
SEGMENT_COLOURS = {"-y row": "#1b6ca8", "+x loop": "#e0a030", "+y row": "#c8443c", "-x loop": "#0e7c86"}
GROUP_STYLE = {"+y face": ("#c8443c", "o"), "-y face": ("#1b6ca8", "s"), "end faces": ("#e0a030", "^"),
               "top face (optional ties)": ("#444444", "D")}


def probe_targets():
    xs = np.linspace(2.5, 12.5, 5)
    pts, normals, groups = [], [], []
    for x in xs:
        pts.append([x, WIDTH_M / 2, 1.25]); normals.append([0, 1, 0]); groups.append("+y face")
        pts.append([x, -WIDTH_M / 2, 1.25]); normals.append([0, -1, 0]); groups.append("-y face")
        pts.append([x, 0.0, TOP_M]); normals.append([0, 0, 1]); groups.append("top face (optional ties)")
    for y in (-0.6, 0.0, 0.6):
        pts.append([0.0, y, 1.25]); normals.append([-1, 0, 0]); groups.append("end faces")
        pts.append([LENGTH_M, y, 1.25]); normals.append([1, 0, 0]); groups.append("end faces")
    return np.array(pts), np.array(normals, float), np.array(groups)


def main():
    track = StationTrack(0.0, LENGTH_M, WIDTH_M, STANDOFF_M, blocked_intervals_m=((30.0, 33.0),))
    poses = track_candidate_poses(track, 1.0, HEIGHTS_M)
    positions = np.array([p.position_m for p in poses])
    s_all = np.array([p.s_m for p in poses])
    h_all = np.array([p.height_m for p in poses])
    targets, normals, groups = probe_targets()
    part = box_mesh(np.array([0.0, -WIDTH_M / 2, BOTTOM_M]), np.array([LENGTH_M, WIDTH_M / 2, TOP_M]))
    limits = VisibilityLimits(max_range_m=30.0, max_incidence_angle_rad=np.deg2rad(80.0))
    visible = visibility_matrix(positions, targets, limits, normals, part, 1e-3)  # (candidates, probes)

    fig = plt.figure(figsize=(17, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.45, 1.0], hspace=0.38, wspace=0.18,
                            left=0.05, right=0.985, top=0.9, bottom=0.07)

    # (a) plan view
    ax = fig.add_subplot(grid[0, :])
    ax.add_patch(Rectangle((0, -WIDTH_M / 2), LENGTH_M, WIDTH_M, facecolor="0.8", edgecolor="k", linewidth=1.5, zorder=2))
    ax.text(LENGTH_M / 2, 0, f"part on jig\n{LENGTH_M:g} × {WIDTH_M:g} m, z {BOTTOM_M:g}–{TOP_M:g} m", ha="center", va="center", fontsize=10, zorder=3)
    s_dense = np.linspace(0, track.perimeter_m, 800)
    xy_dense = np.array([track.position_xy_m(v) for v in s_dense])
    for name in SEGMENT_NAMES:
        sel = np.array([track.segment(v) == name for v in s_dense])
        pts = np.where(sel[:, None], xy_dense, np.nan)
        ax.plot(pts[:, 0], pts[:, 1], color=SEGMENT_COLOURS[name], linewidth=2.5, zorder=1, label=f"{name}")
    blocked = np.array([track.is_blocked(v) for v in s_dense])
    ax.plot(np.where(blocked, xy_dense[:, 0], np.nan), np.where(blocked, xy_dense[:, 1], np.nan), color="k", linewidth=9,
            alpha=0.5, solid_capstyle="butt", zorder=1.5, label="blocked zone (illustrative: s 30–33 m)")
    xy_cand = np.array([track.position_xy_m(v) for v in np.unique(s_all)])
    ax.scatter(xy_cand[:, 0], xy_cand[:, 1], s=14, color="k", zorder=4, label="candidates, 1 m spacing (each at 2 heights)")
    for s_tick in np.arange(0, track.perimeter_m, 5.0):
        x, y = track.position_xy_m(s_tick)
        nearest = np.array([np.clip(x, 0.0, LENGTH_M), np.clip(y, -WIDTH_M / 2, WIDTH_M / 2)])
        away = np.array([x, y]) - nearest
        away = away / np.linalg.norm(away)
        ax.plot([x, x + 0.25 * away[0]], [y, y + 0.25 * away[1]], color="0.3", linewidth=1, zorder=5)
        ax.text(x + 0.62 * away[0], y + 0.62 * away[1], f"{s_tick:g}", ha="center", va="center", fontsize=8.5, color="0.2", zorder=5)
    x0, y0 = track.position_xy_m(0.0)
    ax.annotate("s = 0\n(-x end of the -y row)", (x0, y0), xytext=(-3.6, -5.3), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k"), ha="center")
    ax.annotate("", xy=track.position_xy_m(3.2), xytext=track.position_xy_m(0.6),
                arrowprops=dict(arrowstyle="-|>", color="k", linewidth=2))
    ax.text(7.5, -track.y_left_row_m - 1.2, "s increases anticlockwise  →", fontsize=9.5, ha="center")
    for grp, (colour, marker) in GROUP_STYLE.items():
        sel = groups == grp
        ax.scatter(targets[sel, 0], targets[sel, 1], marker=marker, s=46, color=colour, edgecolors="k", linewidths=0.7, zorder=6, label=f"probe targets: {grp}")
    ax.set_aspect("equal")
    ax.set_xlim(-6.6, LENGTH_M + 6.6)
    ax.set_ylim(-track.y_left_row_m - 2.4, track.y_left_row_m + 3.2)
    ax.set_xlabel("x, along the part (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper center", ncol=4)
    ax.set_title(f"(a) The cell: station track {track.perimeter_m:.1f} m round, standoff {STANDOFF_M:g} m — PLACEHOLDER dimensions (TODO(source))",
                 fontsize=11, loc="left", pad=8)

    # (b) visible probe targets by group along s
    ax = fig.add_subplot(grid[1, 0])
    for grp, (colour, _) in GROUP_STYLE.items():
        cols = np.where(groups == grp)[0]
        for h, style in zip(HEIGHTS_M, ("--", "-")):
            sel = h_all == h
            counts = visible[sel][:, cols].sum(axis=1)
            ax.plot(s_all[sel], counts, style, color=colour, linewidth=2 if h == 2.4 else 1.3,
                    label=f"{grp}, h = {h:g} m" if grp == "+y face" or h == 2.4 else None)
    edges = np.cumsum([0.0, track.row_length_m, np.pi * track.loop_radius_m, track.row_length_m, np.pi * track.loop_radius_m])
    for name, a, b in zip(SEGMENT_NAMES, edges[:-1], edges[1:]):
        ax.axvspan(a, b, color=SEGMENT_COLOURS[name], alpha=0.1)
        ax.text((a + b) / 2, 5.55, name, ha="center", fontsize=9)
    ax.set_xlim(0, track.perimeter_m)
    ax.set_ylim(0, 6.2)
    ax.set_xlabel("s along the track (m)")
    ax.set_ylabel("probe targets visible (of 5, 5, 6, 5)")
    ax.set_title("(b) What each station can see (thick: h = 2.4 m, thin dashed: h = 1.0 m)", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.5, loc="lower center", ncol=2)
    ax.grid(alpha=0.25)

    # (c) which pairs of candidates share >= 3 targets
    ax = fig.add_subplot(grid[1, 1])
    shared = visible.astype(int) @ visible.astype(int).T
    ax.imshow(shared >= 3, origin="lower", cmap="Blues", aspect="equal", extent=(0, len(poses), 0, len(poses)), vmin=0, vmax=1.4)
    tick_s = np.arange(0, 56, 10)
    tick_idx = [int(np.searchsorted(s_all, t)) for t in tick_s]
    ax.set_xticks(tick_idx); ax.set_xticklabels([f"{t}" for t in tick_s])
    ax.set_yticks(tick_idx); ax.set_yticklabels([f"{t}" for t in tick_s])
    for e in np.cumsum([track.row_length_m, np.pi * track.loop_radius_m, track.row_length_m]):
        k = int(np.searchsorted(s_all, e))
        ax.axvline(k, color="k", linewidth=0.5); ax.axhline(k, color="k", linewidth=0.5)
    ax.set_xlabel("station s (m), both heights")
    ax.set_ylabel("station s (m)")
    ax.set_title("(c) Pairs that share ≥ 3 probe targets\n(dark = can be tied directly)", fontsize=10.5, loc="left")

    fig.suptitle("Default station-track cell: layout, candidates, and which stations can see and tie to which", fontsize=13, x=0.05, ha="left")
    fig.savefig(RESULTS_DIR / "station_track_cell.png", dpi=150)
    plt.close(fig)

    # console diagnostic
    seg = np.array([p.segment for p in poses])
    plus_y = np.isin(seg, ["+y row"]); minus_y = np.isin(seg, ["-y row"])
    cross = shared[np.ix_(plus_y, minus_y)] >= 3
    print("candidates:", len(poses), "perimeter", round(track.perimeter_m, 2))
    print("+y-row station and -y-row station share >=3 probe targets directly (any pair):", bool(cross.any()), f"({cross.sum()} of {cross.size} pairs)")
    for grp in GROUP_STYLE:
        cols = groups == grp
        print(f"  {grp}: stations seeing >=3: " + ", ".join(f"{name} {int(((visible[:, cols].sum(axis=1) >= 3) & (seg == name)).sum())}/{int((seg == name).sum())}" for name in SEGMENT_NAMES))
    print("wrote station_track_cell.png")


if __name__ == "__main__":
    main()
