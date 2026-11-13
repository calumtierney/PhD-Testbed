"""Explainer figure: the two process clusters of the step-6 headline experiment,
and why objective B's risk weights are taken at a reference uncertainty of
8 um.

    python -m viz.make_cluster_weight_explainer

Left: each cluster's total decision risk (PFA + PFR, `risk.jcgm106`) against
measurement uncertainty, with the 8 um reference marked inside the 5-9 um
band that CLAUDE.md §4b predicts for one tracker at 2.5 m. Objective B's
weight for a characteristic is the *slope* of its curve there.
Right: the normalised weights `planning.objectives.risk_derived_weights`
gives for the two-cluster scene at every reference from 3 to 14 um -- the
choice of 8 um changes the numbers slightly, not which cluster dominates.

Clusters are `planning.experiment.MARGINAL_CLUSTER` and
`HIGH_CAPABILITY_CLUSTER` (tolerance +-50 um). Writes
results/explainer_cluster_weights.png and .svg.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planning.experiment import HIGH_CAPABILITY_CLUSTER, MARGINAL_CLUSTER, default_two_cluster_scenario
from planning.objectives import risk_derived_weights
from risk.jcgm106 import ClusterAssignment, DecisionRule, evaluate_conformity_risk

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
UM = 1e6
REFERENCE_UM = 8.0
COLOURS = {"marginal": "#c8443c", "high-capability": "#0e7c86"}
STYLES = {"marginal": "-", "high-capability": "--"}


def total_risk(cluster, u_m):
    r = evaluate_conformity_risk(ClusterAssignment("x", cluster), u_m, DecisionRule.from_zone_width(1e-4))
    return r.probability_false_acceptance + r.probability_false_rejection


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    ax = axes[0]
    u_um = np.geomspace(1, 30, 60)
    for cluster in (MARGINAL_CLUSTER, HIGH_CAPABILITY_CLUSTER):
        risk = np.array([total_risk(cluster, u / UM) for u in u_um])
        ax.loglog(u_um, risk, STYLES[cluster.name], color=COLOURS[cluster.name], linewidth=2.2,
                  label=f"{cluster.name}: μ {cluster.prior.mean_m * UM:.0f} µm, σ {cluster.prior.std_m * UM:.1f} µm, "
                        f"Cpk {cluster.prior.cpk(-50e-6, 50e-6):.2f}")
        h = 0.5
        slope = (total_risk(cluster, (REFERENCE_UM + h) / UM) - total_risk(cluster, (REFERENCE_UM - h) / UM)) / (2 * h)
        r0 = total_risk(cluster, REFERENCE_UM / UM)
        ax.scatter([REFERENCE_UM], [r0], color=COLOURS[cluster.name], s=60, zorder=5, edgecolors="white")
        ax.annotate(f"slope at 8 µm:\n{slope:.2g} per µm", (REFERENCE_UM, r0), textcoords="offset points",
                    xytext=(14, -26 if cluster.name == "marginal" else 10), fontsize=9.5, color=COLOURS[cluster.name])
    ax.axvspan(5, 9, color="0.88", zorder=0)
    ax.axvline(REFERENCE_UM, color="k", linewidth=1, linestyle=":")
    ax.text(5.1, 2e-9, "one tracker at 2.5 m:\nlateral 5-9 µm\n(Hughes et al. 2011)", fontsize=9, color="0.3")
    ax.set_xlabel("measurement uncertainty along the tolerance, u (µm)")
    ax.set_ylabel("decision risk per characteristic, PFA + PFR")
    ax.set_ylim(1e-10, 1)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("(a) Same uncertainty, very different risk: the weight is the slope at 8 µm", fontsize=11, loc="left")

    ax = axes[1]
    scenario = default_two_cluster_scenario()
    names = [scenario.assignments[c.name].cluster.name for c in scenario.characteristics]
    refs = np.arange(3.0, 14.5, 0.5)
    weights = np.array([risk_derived_weights(scenario, r / UM) for r in refs])
    for name in ("marginal", "high-capability"):
        col = names.index(name)
        ax.plot(refs, weights[:, col], STYLES[name], color=COLOURS[name], linewidth=2.2, label=f"{name} characteristics")
    w8 = risk_derived_weights(scenario, REFERENCE_UM / UM)
    ax.axvline(REFERENCE_UM, color="k", linewidth=1, linestyle=":")
    ax.axvspan(5, 9, color="0.88", zorder=0)
    ax.text(REFERENCE_UM + 0.2, 1.55,
            f"at 8 µm:\nmarginal {w8[names.index('marginal')]:.3f}\nhigh-cap. {w8[names.index('high-capability')]:.3f}",
            fontsize=9.5)
    ax.set_xlabel("reference uncertainty the weights are taken at (µm)")
    ax.set_ylabel("objective B weight (mean of all six = 1)")
    ax.set_ylim(-0.1, 2.2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="center right")
    ax.set_title("(b) The choice of 8 µm barely matters:\nthe marginal cluster carries ~all the weight anywhere from 3 to 14 µm",
                 fontsize=11, loc="left")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(RESULTS_DIR / f"explainer_cluster_weights.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote", RESULTS_DIR / "explainer_cluster_weights.png", "and .svg")


if __name__ == "__main__":
    main()
