"""Explainer figures for global decision risk (JCGM 106): where false accepts
and false rejects come from, how they move with measurement uncertainty, and
how they depend on the process prior (mean and spread) rather than on Cpk.

    python -m viz.make_risk_explainer_figures

Every number is computed with `risk.jcgm106` (the same code the testbed's
step 5 and the wing-box experiment use), not drawn by hand. The two process
priors are the wing-box experiment's *placeholders* (`planning.inspection.
cluster_priors`, tolerance +-50 um, "offset" style) -- they illustrate the
mechanics, they are not measured process data (TODO(source)).

Writes results/explainer_risk_decision.png, explainer_risk_vs_uncertainty.png
and explainer_risk_vs_prior.png. Units are micrometres for display; the risk
code itself works in metres.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from planning.inspection import cluster_priors, nonconformance_rate
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, evaluate_conformity_risk

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
T_M = 50e-6  # tolerance half-width (the repo's reference scale, CLAUDE.md §4d)
UM = 1e6
COLOURS = {"marginal": "#c0392b", "high-capability": "#1b6ca8"}


def _rule() -> DecisionRule:
    return DecisionRule.from_zone_width(2 * T_M)


def _risk(prior: ProcessPrior, u_m: float):
    r = evaluate_conformity_risk(ClusterAssignment("x", FeatureCluster("x", prior)), u_m, _rule())
    return r.probability_false_acceptance, r.probability_false_rejection


def decision_figure():
    marginal, high = cluster_priors(T_M, "offset")
    rule = _rule()
    y_um = np.linspace(-95, 95, 1900)
    y_m = y_um / UM
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for u_um, style in ((2.0, "-"), (6.25, "--"), (20.0, ":")):
        u = u_um / UM
        p_accept = norm.cdf((rule.acceptance_upper_m - y_m) / u) - norm.cdf((rule.acceptance_lower_m - y_m) / u)
        ax.plot(y_um, p_accept, style, c="k", label=f"u = {u_um:g} µm")
    ax.axvspan(-50, 50, color="0.9", zorder=0)
    ax.set_xlabel("true deviation from nominal, y (µm)")
    ax.set_ylabel("P(measurement says 'accept' | true value y)")
    ax.set_title("(a) The measurement: a blurred cliff at each limit\nshaded = tolerance ±50 µm", fontsize=10)
    ax.legend(fontsize=9, loc="center")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for name, cluster in (("marginal", marginal), ("high-capability", high)):
        p = cluster.prior
        ax.plot(y_um, norm.pdf(y_m, p.mean_m, p.std_m) / UM, c=COLOURS[name],
                label=f"{name}: μ = {p.mean_m * UM:.0f}, σ = {p.std_m * UM:.1f} µm")
    ax.axvspan(-50, 50, color="0.9", zorder=0)
    ax.axvline(50, c="k", lw=0.8)
    ax.axvline(-50, c="k", lw=0.8)
    ax.set_xlabel("true deviation from nominal, y (µm)")
    ax.set_ylabel("probability density of the process, f(y) (per µm)")
    ax.set_title("(b) The process prior: where true parts fall\n"
                 f"out of tolerance: marginal {nonconformance_rate(marginal.prior, T_M):.1e}, "
                 f"high-capability {nonconformance_rate(high.prior, T_M):.1e}", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[2]
    u = 6.25 / UM
    p = marginal.prior
    p_accept = norm.cdf((rule.acceptance_upper_m - y_m) / u) - norm.cdf((rule.acceptance_lower_m - y_m) / u)
    f = norm.pdf(y_m, p.mean_m, p.std_m) / UM
    outside = np.abs(y_um) > 50
    ax.semilogy(y_um[outside], (f * p_accept)[outside], c="#c0392b")
    ax.fill_between(y_um[outside], 1e-12, (f * p_accept)[outside], color="#c0392b", alpha=0.3)
    ax.semilogy(y_um[~outside], (f * (1 - p_accept))[~outside], c="#1b6ca8")
    ax.fill_between(y_um[~outside], 1e-12, (f * (1 - p_accept))[~outside], color="#1b6ca8", alpha=0.3)
    pfa, pfr = _risk(p, u)
    ax.axvline(50, c="k", lw=0.8)
    ax.axvline(-50, c="k", lw=0.8)
    ax.set_ylim(1e-9, 1)
    ax.set_xlabel("true deviation from nominal, y (µm)")
    ax.set_ylabel("f(y) × P(wrong decision | y)   (per µm, log scale)")
    ax.set_xlim(20, 66)
    ax.set_title("(c) Marginal process, u = 6.25 µm: what is integrated (log scale)\n"
                 f"red (outside, accepted) PFA = {pfa:.1e}\nblue (inside, rejected) PFR = {pfr:.1e}", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    fig.suptitle("Global risk = the measurement's blur, averaged over where the process puts true parts "
                 "(placeholder priors, simple acceptance)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "explainer_risk_decision.png", dpi=150)
    plt.close(fig)


def uncertainty_figure():
    marginal, high = cluster_priors(T_M, "offset")
    u_um = np.geomspace(0.5, 50, 36)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, cluster in (("marginal", marginal), ("high-capability", high)):
        risks = np.array([_risk(cluster.prior, u / UM) for u in u_um])
        axes[0].loglog(u_um, risks[:, 0], c=COLOURS[name], label=f"{name}")
        axes[0].axhline(nonconformance_rate(cluster.prior, T_M), c=COLOURS[name], ls="--", lw=1)
        axes[1].loglog(u_um, np.maximum(risks[:, 1], 1e-14), c=COLOURS[name], label=f"{name}")
    for ax in axes:
        ax.axvline(6.25, c="0.4", ls=":", lw=1)
        ax.text(6.4, ax.get_ylim()[0] * 3, "TUR 4:1\n6.25 µm", fontsize=8, color="0.3")
        ax.set_xlabel("measurement uncertainty u (µm)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("PFA (per characteristic)")
    axes[0].set_title("False accept: capped by the process (dashed = its out-of-tolerance rate),\n"
                      "so it barely moves with u", fontsize=10)
    axes[1].set_ylabel("PFR (per characteristic)")
    axes[1].set_title("False reject: grows steeply with u,\nfaster the closer the process sits to a limit", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "explainer_risk_vs_uncertainty.png", dpi=150)
    plt.close(fig)


def prior_figure(n: int = 28):
    mu_rel = np.linspace(0.0, 0.85, n)    # process mean / T
    sd_rel = np.linspace(0.025, 0.30, n)  # process std / T
    u = 6.25 / UM
    log_pfa = np.zeros((n, n))
    log_pfr = np.zeros((n, n))
    for i, sd in enumerate(sd_rel):
        for j, mu in enumerate(mu_rel):
            pfa, pfr = _risk(ProcessPrior(mu * T_M, sd * T_M), u)
            log_pfa[i, j] = np.log10(max(pfa, 1e-16))
            log_pfr[i, j] = np.log10(max(pfr, 1e-16))
    marginal, high = cluster_priors(T_M, "offset")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    mm, ss = np.meshgrid(mu_rel, sd_rel)
    cpk = (1.0 - mm) / (3.0 * ss)  # for a process mean at or above zero
    for ax, data, title in ((axes[0], log_pfa, "log₁₀ PFA"), (axes[1], log_pfr, "log₁₀ PFR")):
        mesh = ax.pcolormesh(mu_rel * T_M * UM, sd_rel * T_M * UM, data, cmap="viridis", shading="auto")
        cs = ax.contour(mu_rel * T_M * UM, sd_rel * T_M * UM, cpk, levels=[1.0, 1.33, 1.67, 2.0], colors="white", linewidths=1.2)
        ax.clabel(cs, fmt=lambda v: f"Cpk {v:g}", fontsize=8)
        for cluster, name in ((marginal, "marginal"), (high, "high-capability")):
            ax.scatter([cluster.prior.mean_m * UM], [cluster.prior.std_m * UM], marker="*", s=220, c=COLOURS[name],
                       edgecolors="white", zorder=5)
            ax.annotate(name, (cluster.prior.mean_m * UM, cluster.prior.std_m * UM), textcoords="offset points",
                        xytext=(-62, 10) if name == "marginal" else (8, 8), color="white", fontsize=9)
        ax.set_xlabel("process mean μ (µm from nominal; tolerance ±50 µm)")
        ax.set_ylabel("process spread σ (µm)")
        ax.set_title(f"{title} at u = 6.25 µm", fontsize=10)
        fig.colorbar(mesh, ax=ax, label=title)
    fig.suptitle("Risk is a function of the prior's mean and spread; Cpk contours (white) only label the same plane",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "explainer_risk_vs_prior.png", dpi=150)
    plt.close(fig)


def main():
    decision_figure()
    uncertainty_figure()
    prior_figure()
    print("wrote explainer_risk_*.png to", RESULTS_DIR)


if __name__ == "__main__":
    main()
