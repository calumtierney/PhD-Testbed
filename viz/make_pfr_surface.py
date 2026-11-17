"""3D surface: false-rejection probability against process capability (Cpk)
and measurement uncertainty, for a +-50 um tolerance.

    python -m viz.make_pfr_surface

PFR here is JCGM 106's joint probability that a part is conforming *and* is
rejected under simple acceptance (`risk.decision.decision_risk`): the share of
all parts that are good but wrongly failed. Process known exactly (Gaussian).

**Cpk alone does not fix PFR.** Cpk = (50 um - |mean|) / (3 std) is met by a
wide centred process or a narrow off-centre one, and they behave very
differently: the off-centre one has its parts bunched near a limit, where a
measurement error flips the decision. So two surfaces are drawn, each a
one-parameter family through one of the headline experiment's clusters:

  left   centred process: mean 0, std = 50 / (3 Cpk) um
         (Cpk 2.0 is `HIGH_CAPABILITY_CLUSTER`: std 8.3 um)
  right  off-centre process, spread fixed at 3 um: mean = 50 - 9 Cpk um
         (Cpk 1.33 is `MARGINAL_CLUSTER`: mean 38 um, std 3 um)

The black points are the table values used in the slides (4, 8, 20 um at the
two clusters), recomputed here. Writes results/pfr_surface_cpk_uncertainty.png/.svg.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from risk.decision import ProcessKnowledge, decision_risk

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HALF_M = 50e-6
OFFSET_STD_M = 3e-6
CPK = np.linspace(1.0, 2.0, 41)
U_UM = np.linspace(5.0, 25.0, 41)
# sequential blue ramp, light -> dark (dataviz reference palette)
RAMP = colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED = "#2b2b2b", "#6b6b6b"


def centred(cpk):
    return 0.0, HALF_M / (3.0 * cpk)


def off_centre(cpk):
    return HALF_M - 3.0 * OFFSET_STD_M * cpk, OFFSET_STD_M


def pfr_percent(process, cpk, u_um):
    mean, std = process(cpk)
    return 100.0 * decision_risk(ProcessKnowledge.known(mean, std), -HALF_M, HALF_M, u_um * 1e-6)[1]


def surface(process):
    return np.array([[pfr_percent(process, c, u) for c in CPK] for u in U_UM])


def main():
    families = [
        ("Centred process\nmean 0, std = 50 / (3 Cpk) µm", centred, [(2.0, u) for u in (8.0, 20.0)]),
        (f"Off-centre process, spread fixed\nstd {OFFSET_STD_M * 1e6:.0f} µm, mean = 50 − 9 Cpk µm", off_centre,
         [(4.0 / 3.0, u) for u in (8.0, 20.0)]),
    ]
    grids = [surface(p) for _, p, _ in families]
    z_max = max(g.max() for g in grids)
    norm = colors.Normalize(0.0, z_max)
    c_grid, u_grid = np.meshgrid(CPK, U_UM)

    fig = plt.figure(figsize=(13.5, 6.4))
    for k, ((title, process, marks), z) in enumerate(zip(families, grids)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d", computed_zorder=False)  # draw points over the surface
        ax.plot_surface(c_grid, u_grid, z, cmap=RAMP, norm=norm, rstride=1, cstride=1, linewidth=0.15,
                        edgecolor=(1, 1, 1, 0.35), antialiased=True, shade=False, zorder=1)
        for cpk, u in marks:
            value = pfr_percent(process, cpk, u)
            ax.scatter([cpk], [u], [value], color=INK, edgecolors="white", linewidths=1.5, s=48, depthshade=False, zorder=10)
            ax.text(cpk, u, value + 0.04 * z_max, f"  {value:.1f}%" if value >= 1.0 else f"  {value:.1g}%",
                    color=INK, fontsize=9, zorder=11)
        ax.set_xlim(2.0, 1.0)            # low capability at the front right, where the risk is
        ax.set_ylim(5.0, 25.0)
        ax.set_zlim(0.0, z_max * 1.05)
        ax.set_xlabel("Process capability Cpk", color=INK, labelpad=8)
        ax.set_ylabel("Measurement uncertainty u (µm)", color=INK, labelpad=8)
        ax.set_zlabel("Good parts wrongly failed, PFR (%)", color=INK, labelpad=6)
        ax.set_title(title, color=INK, fontsize=11)
        ax.view_init(elev=24, azim=-128)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor("#d9d9d9")
            axis._axinfo["grid"]["color"] = "#e6e6e6"
            axis._axinfo["grid"]["linewidth"] = 0.6
        ax.tick_params(colors=MUTED, labelsize=8)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), ax=fig.axes, shrink=0.55, pad=0.04)
    bar.set_label("PFR (% of all parts)", color=INK)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("False rejection against process capability and measurement uncertainty, ±50 µm tolerance, "
                 "simple acceptance", color=INK, fontsize=12, y=0.98)
    fig.text(0.5, 0.025, "Same z scale on both panels. Cpk does not determine PFR on its own: at equal Cpk an off-centre "
             "process loses far more good parts, because its parts sit close to a limit. Black points: values in the "
             "slide table.", ha="center", color=MUTED, fontsize=9)
    RESULTS_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(RESULTS_DIR / f"pfr_surface_cpk_uncertainty{suffix}", dpi=200, bbox_inches="tight")
    print("max PFR %: centred", round(grids[0].max(), 2), "off-centre", round(grids[1].max(), 2))
    for name, g in (("centred", grids[0]), ("off-centre", grids[1])):
        print(name, "PFR % at u=5,15,25 for Cpk 1, 1.5, 2:",
              [[round(g[np.argmin(abs(U_UM - u)), np.argmin(abs(CPK - c))], 3) for c in (1.0, 1.5, 2.0)] for u in (5, 15, 25)])


if __name__ == "__main__":
    main()
