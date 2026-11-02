"""Explainer figure: what a "3x3 covariance per target" means.

    python -m viz.make_covariance_explainer

One target 2.5 m from one laser tracker, using the step-1 error model with
its fixed CLAUDE.md §4a inputs (`instruments.laser_tracker.LaserTracker`).
Three panels:

  (a) the 3x3 matrix itself, with its real numbers, diagonal and
      off-diagonal cells explained;
  (b) 2000 simulated measurements of that one point, seen from above (x-y),
      so the tilted cloud shows what a non-zero off-diagonal term means,
      against the cloud you would get if it were zero;
  (c) the same uncertainty as a 3D ellipsoid around the point, with the
      station and beam drawn, so the shape (thin along the beam, wide across
      it) is visible. The ellipsoid is exaggerated and the factor is printed
      on the figure (CLAUDE.md §8).

Writes results/explainer_covariance.png and .svg (vector, for slides and the
thesis). The scatter uses a fixed random seed, so the figure is identical on
every run. Units: micrometres for display, metres internally.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
UM = 1e6
DIAG_COLOUR = "#0e7c86"
OFF_COLOUR = "#e0a030"
BEAM_COLOUR = "#c8443c"

# A point 2.5 m from the tracker (the CLAUDE.md §4b reference range), off to
# one side and slightly above, so the beam is not lined up with any axis and
# the off-diagonal terms are visibly non-zero.
RANGE_M, AZIMUTH_DEG, ELEVATION_DEG = 2.5, 35.0, 20.0


def target_and_covariance():
    az, el = np.deg2rad(AZIMUTH_DEG), np.deg2rad(ELEVATION_DEG)
    point = RANGE_M * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    station = InstrumentPose(position_m=np.zeros(3))
    return point, LaserTracker().covariance_global(point, station)


def _ellipse_2d(cov2, n_std=1.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov2)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return Ellipse((0, 0), 2 * n_std * np.sqrt(vals[1]), 2 * n_std * np.sqrt(vals[0]), angle=angle, **kwargs)


def matrix_panel(ax, cov_um2):
    ax.set_xlim(0, 4.2)
    ax.set_ylim(-1.55, 4.3)
    ax.axis("off")
    names = ["x", "y", "z"]
    x0, y0, w, h = 0.75, 3.0, 1.12, 0.85
    for j, n in enumerate(names):
        ax.text(x0 + j * w + w / 2, y0 + h + 0.12, n, ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.text(x0 - 0.15, y0 - j * h + h / 2, n, ha="right", va="center", fontsize=14, fontweight="bold")
    for i in range(3):
        for j in range(3):
            diag = i == j
            ax.add_patch(Rectangle((x0 + j * w, y0 - i * h), w, h, facecolor=DIAG_COLOUR if diag else OFF_COLOUR,
                                   alpha=0.85 if diag else 0.55, edgecolor="white", linewidth=2,
                                   hatch=None if diag else "//"))
            value = cov_um2[i, j]
            label = f"{value:.1f}" if diag else f"{value:+.1f}"
            ax.text(x0 + j * w + w / 2, y0 - i * h + h * 0.6, label, ha="center", va="center", fontsize=13,
                    color="white" if diag else "black", fontweight="bold")
            if diag:
                ax.text(x0 + j * w + w / 2, y0 - i * h + h * 0.25, f"±{np.sqrt(value):.1f} µm", ha="center",
                        va="center", fontsize=10, color="white")
    ax.text(x0 + 1.5 * w, y0 - 2 * h - 0.3, "units: µm² (micrometres squared)", ha="center", fontsize=10.5, color="0.3")
    ax.add_patch(FancyBboxPatch((0.05, -0.55), 0.3, 0.3, boxstyle="round,pad=0.02", facecolor=DIAG_COLOUR, alpha=0.85))
    ax.text(0.45, -0.4, "Diagonal: how much the error spreads along x, y\nand z separately. Its square root is the\nfamiliar \"± µm\" for that axis.",
            va="center", fontsize=10.5)
    ax.add_patch(FancyBboxPatch((0.05, -1.35), 0.3, 0.3, boxstyle="round,pad=0.02", facecolor=OFF_COLOUR, alpha=0.55,
                                hatch="//"))
    ax.text(0.45, -1.2, "Off-diagonal: whether errors in two axes move\ntogether (+) or opposite (−). Zero means\nunrelated. The table is mirror-symmetric.",
            va="center", fontsize=10.5)
    ax.set_title("(a) The numbers: one target's 3×3 covariance", fontsize=13, loc="left", pad=6)


def scatter_panel(ax, cov_um2, point):
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(np.zeros(3), cov_um2, size=2000)
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.25, color="0.35", linewidths=0, label="2000 simulated measurements")
    cov_xy = cov_um2[:2, :2]
    ax.add_patch(_ellipse_2d(cov_xy, n_std=2.0, fill=False, edgecolor=DIAG_COLOUR, linewidth=2.5,
                             label=f"real spread (2σ): tilted, because x and y errors are linked\n(xy term = {cov_um2[0, 1]:+.1f}: when x reads high, y tends to read low)"))
    no_off = np.diag(np.diag(cov_xy))
    ax.add_patch(_ellipse_2d(no_off, n_std=2.0, fill=False, edgecolor=OFF_COLOUR, linewidth=2.2, linestyle="--",
                             label="same diagonal, off-diagonal set to zero: no tilt"))
    beam = point[:2] / np.linalg.norm(point[:2])
    lim = 3.2 * np.sqrt(cov_xy.max())
    ax.annotate("", xy=beam * lim * 0.95, xytext=-beam * lim * 0.95,
                arrowprops=dict(arrowstyle="->", color=BEAM_COLOUR, linewidth=1.8))
    ax.text(*(beam * lim * 0.62 + np.array([0.6, -1.4])), "laser beam\ndirection:\ncloud is narrow\nalong it", color=BEAM_COLOUR, fontsize=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_xlabel("x error (µm)")
    ax.set_ylabel("y error (µm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9.5, frameon=False)
    ax.set_title("(b) Measure the same point many times\n(seen from above)", fontsize=13, loc="left")


def ellipsoid_panel(ax, cov_m2, point):
    vals, vecs = np.linalg.eigh(cov_m2)
    radii = np.sqrt(vals)
    target_size_m = 0.45
    raw = target_size_m / radii.max()
    scale = float(10 ** np.floor(np.log10(raw)) * (5 if raw / 10 ** np.floor(np.log10(raw)) >= 5 else
                                                    2 if raw / 10 ** np.floor(np.log10(raw)) >= 2 else 1))
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1)
    surface = point + (sphere * radii * scale) @ vecs.T
    ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], color=DIAG_COLOUR, alpha=0.55, linewidth=0)
    ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color=BEAM_COLOUR, linewidth=1.8)
    ax.scatter([0], [0], [0], marker="^", s=120, color="k")
    ax.text(0, 0, -0.25, "tracker", fontsize=10, ha="center")
    ax.text(*(point * 0.45 + np.array([0, 0, 0.12])), "beam, 2.5 m", color=BEAM_COLOUR, fontsize=10)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 2.6)
    ax.set_zlim(-0.4, 1.4)
    ax.set_box_aspect((2.6, 2.6, 1.8))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=22, azim=-120)
    ax.set_title("(c) The same thing as a shape in 3D", fontsize=13, loc="left")
    lateral = np.sqrt(np.sort(vals)[1:]) * UM
    radial = np.sqrt(np.sort(vals)[0]) * UM
    ax.text2D(0.02, -0.02, f"ellipsoid exaggerated ×{scale:,.0f}\n"
                           f"thin along the beam (±{radial:.1f} µm: distance is measured precisely)\n"
                           f"wide across it (±{lateral[0]:.1f} and ±{lateral[1]:.1f} µm: angles are not)",
              transform=ax.transAxes, fontsize=10, va="top")


def main():
    point, cov_m2 = target_and_covariance()
    cov_um2 = cov_m2 * UM ** 2
    fig = plt.figure(figsize=(16, 7.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.28, left=0.02, right=0.98, top=0.84, bottom=0.25)
    matrix_panel(fig.add_subplot(grid[0]), cov_um2)
    scatter_panel(fig.add_subplot(grid[1]), cov_um2, point)
    ellipsoid_panel(fig.add_subplot(grid[2], projection="3d"), cov_m2, point)
    fig.suptitle("What \"a 3×3 covariance per target\" means: one point, 2.5 m from one laser tracker (step-1 model)",
                 fontsize=15, x=0.02, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"explainer_covariance.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote", RESULTS_DIR / "explainer_covariance.png", "and .svg")


if __name__ == "__main__":
    main()
