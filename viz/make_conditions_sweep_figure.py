"""Figure for the conditions sweep (results/conditions_sweep.json, written by
`python -m viz.run_conditions_sweep`).

    python -m viz.make_conditions_sweep_figure

One panel per swept condition: how far above objective C's risk each other
objective's best plan lands (percent), at 2 added stations (solid) and 3 (dashed),
with C's own total risk underneath for scale. Writes
results/panel12_conditions_sweep.png and .svg.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STYLE = {"A0": ("#555555", "^"), "A1": ("#d55e00", "s"), "B": ("#009e73", "D")}
LABELS = {
    "tolerance_um": ("tolerance half-width (µm), priors scaled: Cpk fixed", True),
    "marginal_mean_um": ("marginal cluster mean (µm), σ 3 µm, ±50 µm", False),
    "separation_m": ("distance of each cluster from the centreline, d (m)", False),
    "systematic_um": ("isotropic floor no placement can reduce (µm)", False),
}
TITLES = {
    "tolerance_um": "(a) uncertainty relative to tolerance",
    "marginal_mean_um": "(b) how different the two processes are",
    "separation_m": "(c) how far apart the clusters are",
    "systematic_um": "(d) uncertainty placement cannot reduce",
}


def main():
    data = json.loads((RESULTS_DIR / "conditions_sweep.json").read_text())
    baseline = data["baseline"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), gridspec_kw=dict(height_ratios=[2.2, 1.0]), sharex="col")
    for col, (axis, (xlabel, logx)) in enumerate(LABELS.items()):
        top, bottom = axes[0, col], axes[1, col]
        for n, ls in zip(data["station_counts"], ("-", "--")):
            pts = sorted((p for p in data["points"] if p["axis"] == axis and p["n_stations"] == n), key=lambda p: p["value"])
            x = [p["value"] for p in pts]
            for name, (colour, marker) in STYLE.items():
                y = [p["plans"][name]["percent_above_C"] for p in pts]
                top.plot(x, y, ls, color=colour, marker=marker, markersize=6, linewidth=1.8,
                         label=f"{name}, {n} stations" if col == 0 else None)
            bottom.plot(x, [p["plans"]["C"]["risk"] for p in pts], ls, color="#cc2f6a", marker="o", markersize=4,
                        label=f"C, {n} stations" if col == 0 else None)
        for ax in (top, bottom):
            ax.axvline(baseline[axis], color="k", linestyle=":", linewidth=1)
            ax.grid(alpha=0.25)
            if logx:
                ax.set_xscale("log")
                ax.set_xticks([25, 50, 100, 250])
                ax.set_xticklabels(["25", "50", "100", "250"])
                ax.minorticks_off()
        top.set_ylim(-0.5, 18)
        bottom.set_yscale("log")
        bottom.set_xlabel(xlabel, fontsize=9)
        top.set_title(TITLES[axis], fontsize=10.5, loc="left")
    axes[0, 0].set_ylabel("risk above C's best plan (%)")
    axes[1, 0].set_ylabel("C's total risk\n(expected wrong decisions)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[1, 0].legend(fontsize=8, loc="lower left")
    axes[0, 3].text(0.98, 0.95, "dotted line: the headline baseline", transform=axes[0, 3].transAxes, ha="right", va="top", fontsize=8.5)
    fig.suptitle("When does the objective change the plan? Ring scene, exhaustive grid, one condition moved at a time "
                 "(B is within 0.8% of C everywhere; A1 within 7.8%;\nat 2 stations A0 and A1 choose the same plan except at d = 3 m, so the grey solid line is hidden under the orange one)", fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"panel12_conditions_sweep.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)
    print("wrote", RESULTS_DIR / "panel12_conditions_sweep.png")


if __name__ == "__main__":
    main()
