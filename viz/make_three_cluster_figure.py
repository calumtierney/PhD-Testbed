"""The step-6 headline experiment on a three-cluster scene: one
high-capability cluster and TWO marginal clusters with the same Cpk (1.33)
made up differently -- one biased towards a limit with a tight spread, one
centred with a wide spread.

    python -m viz.make_three_cluster_figure        # ~7 minutes

Runs the identical four-objective grid search
(`planning.experiment.run_headline_experiment`) on
`planning.experiment.three_cluster_scenario`, with 2 searched stations (the
headline's setting, 276 plans) and 3 (2,024 plans), and writes
results/three_cluster_results.json and results/panel6d_three_cluster.png.
Deterministic: grid search, no random numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planning.experiment import run_headline_experiment, three_cluster_scenario

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CLUSTER_NAMES = {1: "1  high-capability\n(centred, Cpk 2.0)", 2: "2  marginal, off-centre\n(μ 38, σ 3 µm, Cpk 1.33)",
                 3: "3  marginal, centred\n(μ 0, σ 12.5 µm, Cpk 1.33)"}
CLUSTER_COLOURS = {1: "#0e7c86", 2: "#c8443c", 3: "#e0a030"}
OBJECTIVE_STYLE = {"a0": ("A0", "o", "#444444"), "a1": ("A1", "s", "#8e7dbe"), "b": ("B", "^", "#1b6ca8"), "c": ("C", "D", "#c8443c")}


def summarise(result):
    out = dict(weights=result.weights_risk.tolist())
    for name in OBJECTIVE_STYLE:
        plan = getattr(result, "plan_" + name)
        per = {}
        for i in (1, 2, 3):
            keys = [k for k in plan.risk_by_characteristic if k.startswith(f"cluster{i}")]
            per[i] = dict(
                u_sum_um=sum(plan.per_characteristic_uncertainty_m[k] for k in keys) * 1e6,
                risk=sum(plan.risk_by_characteristic[k].probability_false_acceptance
                         + plan.risk_by_characteristic[k].probability_false_rejection for k in keys))
        out[name] = dict(stations=[p.position_m.tolist() for p in plan.station_poses], clusters=per,
                         total=sum(c["risk"] for c in per.values()))
    return out


def draw(data, scenario):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.15, 0.9, 1.2]))

    ax = axes[0]
    for i, (lo, hi) in {1: (0, 3), 2: (3, 6), 3: (6, 9)}.items():
        pts = scenario.target_points_m[lo:hi]
        ax.scatter(pts[:, 0], pts[:, 1], s=90, color=CLUSTER_COLOURS[i], edgecolors="k", zorder=4,
                   label=CLUSTER_NAMES[i].replace("\n", " "))
    anchor = scenario.anchor_pose.position_m
    ax.scatter([anchor[0]], [anchor[1]], marker="*", s=260, color="k", zorder=5, label="anchor (fixed)")
    two = data["n2"]
    for name, (label, marker, colour) in OBJECTIVE_STYLE.items():
        for j, st in enumerate(two[name]["stations"]):
            ax.scatter([st[0]], [st[1]], marker=marker, s=120, facecolors="none", edgecolors=colour, linewidths=2.2,
                       zorder=6, label=f"{label} station" if j == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-4.4, 8.2)
    ax.legend(fontsize=7.5, loc="upper center", ncol=2, framealpha=0.95)
    ax.set_title("(a) Plan view, 2 searched stations:\nthe objectives place them differently", fontsize=10.5, loc="left")

    ax = axes[1]
    w = np.array(data["n2"]["weights"]).reshape(3, 3)[:, 0]
    bars = ax.bar([1, 2, 3], w, color=[CLUSTER_COLOURS[i] for i in (1, 2, 3)], edgecolor="k")
    for x, v in zip((1, 2, 3), w):
        ax.text(x, v + 0.05, f"{v:.3f}" if v < 0.5 else f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1  high-\ncapability", "2  marginal,\noff-centre", "3  marginal,\ncentred"], fontsize=9)
    ax.set_ylabel("objective B weight per characteristic\n(mean of all nine = 1)")
    ax.set_title("(b) Same Cpk, 67× different weight:\nB sees what Cpk hides", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    xs, labels = [], []
    pos = 0.0
    for key, title in (("n2", "2 searched stations"), ("n3", "3 searched stations")):
        for name, (label, _, _) in OBJECTIVE_STYLE.items():
            bottom = 0.0
            for i in (1, 2, 3):
                r = data[key][name]["clusters"][str(i)]["risk"] if str(i) in data[key][name]["clusters"] else data[key][name]["clusters"][i]["risk"]
                ax.bar(pos, r, bottom=bottom, color=CLUSTER_COLOURS[i], edgecolor="k", linewidth=0.6,
                       label=CLUSTER_NAMES[i].split("\n")[0] if (key == "n2" and name == "a0") else None)
                bottom += r
            ax.text(pos, bottom + 0.004, f"{bottom:.3f}", ha="center", fontsize=8.5)
            xs.append(pos)
            labels.append(label)
            pos += 1.0
        pos += 0.7
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.text(1.5, -0.15, "2 searched stations", ha="center", fontsize=10, transform=ax.get_xaxis_transform())
    ax.text(6.2, -0.15, "3 searched stations", ha="center", fontsize=10, transform=ax.get_xaxis_transform())
    ax.set_ylabel("global risk of the chosen plan, PFA + PFR\n(stacked by cluster)")
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("(c) Risk of each objective's plan:\nthe off-centre marginal cluster carries nearly all of it", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Step-6 headline experiment, three clusters: one high-capability and two marginal clusters sharing Cpk 1.33",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(RESULTS_DIR / "panel6d_three_cluster.png", dpi=170)
    plt.close(fig)


def main(redraw_only=False):
    scenario = three_cluster_scenario()
    if redraw_only:
        draw(json.loads((RESULTS_DIR / "three_cluster_results.json").read_text()), scenario)
        return
    data = {}
    for key, n in (("n2", 2), ("n3", 3)):
        data[key] = summarise(run_headline_experiment(scenario=scenario, n_stations=n))
        print(f"{n} searched stations done", flush=True)
    (RESULTS_DIR / "three_cluster_results.json").write_text(json.dumps(data, indent=1))
    draw(data, scenario)
    print("wrote panel6d_three_cluster.png and three_cluster_results.json")


if __name__ == "__main__":
    main()
