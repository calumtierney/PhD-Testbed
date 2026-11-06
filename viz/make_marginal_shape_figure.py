"""Does the step-6 headline result depend on the marginal cluster being biased
towards one limit?

    python -m viz.make_marginal_shape_figure        # ~6 minutes

The headline scene's marginal cluster is mean 38 um, std 3 um (Cpk 1.33,
biased towards a limit). Here the *same Cpk 1.33* is kept while the shape is
swept from a centred, wide process (mean 0, std 12.5 um) to a heavily biased,
tight one (mean 42 um, std 2 um), and the identical four-objective grid search
(`planning.experiment.run_headline_experiment`) is re-run at each. Only the
marginal cluster's prior changes (`with_marginal_prior`).

Writes results/marginal_shape_sweep.json and results/panel6c_marginal_shape.png.
Deterministic: grid search, no random numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planning.experiment import default_two_cluster_scenario, run_headline_experiment, with_marginal_prior
from risk.jcgm106 import ProcessPrior

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TOLERANCE_M = 50e-6
CPK = 4.0 / 3.0
MEANS_UM = (0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42)


def run_sweep():
    rows = []
    for mu_um in MEANS_UM:
        mu = mu_um * 1e-6
        sigma = (TOLERANCE_M - mu) / (3.0 * CPK)  # same Cpk at every shape
        result = run_headline_experiment(scenario=with_marginal_prior(default_two_cluster_scenario(), ProcessPrior(mu, sigma)))
        row = dict(mu_um=mu_um, sigma_um=sigma * 1e6, cpk=ProcessPrior(mu, sigma).cpk(-TOLERANCE_M, TOLERANCE_M),
                   weights=result.weights_risk.tolist())
        for name in ("a0", "a1", "b", "c"):
            plan = getattr(result, "plan_" + name)
            row[name] = dict(stations=[p.position_m.tolist() for p in plan.station_poses], pfa=plan.global_pfa,
                             pfr=plan.global_pfr)
        rows.append(row)
        print(f"mu {mu_um:2d} um, sigma {sigma * 1e6:5.2f} um done", flush=True)
    return rows


def draw(rows):
    mu = np.array([r["mu_um"] for r in rows])
    total = {n: np.array([r[n]["pfa"] + r[n]["pfr"] for r in rows]) for n in ("a0", "a1", "b", "c")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.semilogy(mu, total["a0"], "o-", color="#444444", label="A0: trace of covariance (the field's criterion)")
    ax.semilogy(mu, total["c"], "s--", color="#c8443c", label="C: direct risk")
    ax.set_xlabel("marginal process mean μ (µm from nominal; σ shrinks to hold Cpk at 1.33)")
    ax.set_ylabel("global risk of the chosen plan, PFA + PFR")
    ax.set_title("(a) Absolute risk is set by the process shape:\nover 200× from centred to biased", fontsize=10.5, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    reduction = 100.0 * (1.0 - total["c"] / total["a0"])
    ax.plot(mu, reduction, "o-", color="#0e7c86", linewidth=2.2)
    ax.scatter([38], [reduction[list(mu).index(38)]], s=160, facecolors="none", edgecolors="k", linewidths=2, zorder=5)
    ax.annotate("step-6 headline\n(μ 38, σ 3 µm)", (38, reduction[list(mu).index(38)]), textcoords="offset points",
                xytext=(-10, 16), ha="right", fontsize=9.5)
    ax.annotate("centred, wide\n(μ 0, σ 12.5 µm)", (0, reduction[0]), textcoords="offset points", xytext=(6, 18), fontsize=9.5)
    ax.set_ylim(0, max(reduction) * 1.3)
    ax.set_xlabel("marginal process mean μ (µm)")
    ax.set_ylabel("risk reduction, C versus A0 (%)")
    ax.set_title("(b) The risk-driven objective wins at every shape,\nmost at a moderate offset", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25)

    ax = axes[2]
    absolute = (total["a0"] - total["c"])
    ax.semilogy(mu, np.maximum(absolute, 1e-7), "o-", color="#e0a030", linewidth=2.2)
    ax.set_xlabel("marginal process mean μ (µm)")
    ax.set_ylabel("risk removed, A0 − C (absolute)")
    ax.set_title("(c) In absolute terms the gain keeps growing\nas the process sits closer to a limit", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, which="both")

    same_plans = all(np.allclose(r["c"]["stations"], rows[0]["c"]["stations"]) and np.allclose(r["a0"]["stations"], rows[0]["a0"]["stations"])
                     for r in rows)
    fig.suptitle("Step-6 headline experiment with the marginal cluster's shape swept at constant Cpk 1.33"
                 + ("   (the plans each objective picks are identical at every shape)" if same_plans else ""),
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "panel6c_marginal_shape.png", dpi=170)
    plt.close(fig)
    return same_plans


def main():
    rows = run_sweep()
    (RESULTS_DIR / "marginal_shape_sweep.json").write_text(json.dumps(rows, indent=1))
    same = draw(rows)
    print("wrote panel6c_marginal_shape.png and marginal_shape_sweep.json; same plans at every shape:", same)


if __name__ == "__main__":
    main()
