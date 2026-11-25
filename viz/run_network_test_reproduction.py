"""Gate 5 of the NPL-model task: the Hughes et al. (2011) network test, reproduced.

    python -m viz.run_network_test_reproduction        # about 40 s

On the reconstructed layout (`network.hughes_network`; every coordinate is a
reconstruction, see its docstring) this

  * simulates both-face readings from the paper's uncompensated API T3 (Tables
    2-3) with Table 4 noise, and fits all 16 parameters together with the targets
    and station poses (`network.solve`, estimate mode, prior on mu only), over
    100 seeds: do the estimates cover the truth as their fitted uncertainties say?
  * compares the fitted standard uncertainties with Tables 2-3 as ratios, and the
    target, station and pointing uncertainties with the ranges the paper quotes;
  * reports the three correlations the paper names and the lambda-e_x correlation
    with and without the birdbath lines (and the effect of the lines on
    sigma(lambda)), and the a_E2-a_E0 correlation as the maximum elevation rises.

The layout was not tuned to improve any of these. Writes
results/npl_network_test.json, .png and .svg.
"""
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from instruments.npl_model import ARCSEC_TO_RAD, FRONT, INDEX, N_PARAMETERS, PARAMETER_NAMES, NPLErrorParameters
from network.hughes_network import (
    fit_network_test, hughes_2011_network_scenario, network_test_covariance, simulate_network_test,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TRUE = NPLErrorParameters.hughes_2011_uncompensated()
N_SEEDS = 100
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"
LABELS = ["λ", "μ", "e_x", "b_y,0", "b_z,0", "α", "γ", "a_A,1", "b_A,1", "a_A,2", "b_A,2", "a_E,0", "a_E,1", "b_E,1", "a_E,2", "b_E,2"]


def correlation_and_sigma(network):
    covariance = network_test_covariance(network, TRUE).parameter_estimates[0].covariance
    sigma = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigma, sigma), sigma


def main():
    network = hughes_2011_network_scenario()
    ranges, elevations = network.ranges_and_elevations()
    measured = network.visible
    out = {"layout": {
        "readings": network.n_readings, "range_m": [float(ranges[measured].min()), float(ranges[measured].max())],
        "elevation_deg": [float(elevations[measured].min()), float(elevations[measured].max())],
        "mean_cos_2_elevation": float(np.mean(np.cos(2 * np.deg2rad(elevations[measured])))),
        "paper": {"readings": 126, "range_m": [0.27, 6.4], "max_elevation_deg": 57, "mean_cos_2_elevation": 0.69}}}

    # 1. recovery over seeds
    z = []
    for seed in range(N_SEEDS):
        estimate = fit_network_test(network, simulate_network_test(network, TRUE, seed), seed).parameter_estimates[0]
        z.append((estimate.values - TRUE.values) / estimate.standard_uncertainties())
    z = np.array(z)
    keep = np.arange(N_PARAMETERS) != INDEX["mu"]  # mu's truth is its prior mean: trivially recovered
    a = np.abs(z[:, keep])
    out["recovery"] = {
        "seeds": N_SEEDS, "within_1_sigma": float((a < 1).mean()), "within_2_sigma": float((a < 2).mean()),
        "within_3_sigma": float((a < 3).mean()), "expected": [0.683, 0.954, 0.997], "max_abs_z": float(a.max()),
        "seeds_with_a_parameter_beyond_3_sigma": int((a > 3).any(axis=1).sum()),
        "rms_z_by_parameter": {n: float(np.sqrt((z[:, i] ** 2).mean())) for i, n in enumerate(PARAMETER_NAMES)}}

    # 2. uncertainties against the paper
    result = network_test_covariance(network, TRUE)
    ours = result.parameter_estimates[0].standard_uncertainties()
    paper = TRUE.standard_uncertainties()
    ratio = ours / paper
    n_targets = len(network.targets_m)
    target_um = [float(np.sqrt(np.trace(result.target_covariance_m2[3 * i:3 * i + 3, 3 * i:3 * i + 3])) * 1e6) for i in range(n_targets)]
    full = result.full_covariance_m2
    position_um, pointing_arcsec = [], []
    for k in range(4):  # free stations 1-4
        base = 3 * n_targets + 6 * k
        position_um.append(float(np.sqrt(np.trace(full[base:base + 3, base:base + 3])) * 1e6))
        pointing_arcsec.extend(float(np.sqrt(full[base + 3 + r, base + 3 + r]) / ARCSEC_TO_RAD) for r in range(3))
    out["uncertainties"] = {
        "ratio_to_paper": {n: float(r) for n, r in zip(PARAMETER_NAMES, ratio)},
        "target_rss_um": [min(target_um), max(target_um)], "paper_target_um": [2.1, 4.9],
        "station_position_rss_um": [min(position_um), max(position_um)], "paper_station_um": [1.1, 2.2],
        "station_pointing_arcsec": [min(pointing_arcsec), max(pointing_arcsec)], "paper_pointing_arcsec": [0.33, 0.45]}

    # 3. correlations
    c, sigma = correlation_and_sigma(network)
    named = {"gamma-b_y0": ("gamma_rad", "b_y0_m"), "a_E0-b_z0": ("a_E0_rad", "b_z0_m"), "a_E2-a_E0": ("a_E2_rad", "a_E0_rad")}
    out["correlations"] = {k: float(c[INDEX[a], INDEX[b]]) for k, (a, b) in named.items()}
    i, j = INDEX["lambda_m"], INDEX["e_x_m"]
    out["birdbath"] = {}
    for label, faces in (("two_faces", network.faces), ("front_face_only", ((FRONT,),) * 5)):
        with_lines = replace(network, faces=faces)
        without = replace(hughes_2011_network_scenario(include_birdbath=False), faces=faces)
        c1, s1 = correlation_and_sigma(with_lines)
        c0, s0 = correlation_and_sigma(without)
        rng = np.random.default_rng(0)
        pairs = [(s, t) for s in (3, 4) for t in range(n_targets) if network.visible[s, t] and (s, t) not in {(3, 5), (3, 7), (4, 8), (4, 9)}]
        controls = []
        for _ in range(12):
            visible = network.visible.copy()
            for k in rng.choice(len(pairs), 4, replace=False):
                visible[pairs[k]] = False
            controls.append(correlation_and_sigma(replace(network, faces=faces, visible=visible))[1])
        out["birdbath"][label] = {
            "corr_lambda_ex_with_lines": float(c1[i, j]), "corr_lambda_ex_without_lines": float(c0[i, j]),
            "sigma_lambda_um_with": float(s1[i] * 1e6), "sigma_lambda_um_without": float(s0[i] * 1e6),
            "sigma_lambda_um_control_removing_4_other_pairs": float(np.mean([s[i] for s in controls]) * 1e6),
            "sigma_ex_um_with": float(s1[j] * 1e6), "sigma_ex_um_without": float(s0[j] * 1e6)}
    out["elevation"] = {}
    for n_high in (0, 1, 2):
        net = hughes_2011_network_scenario(n_high_targets=n_high)
        el = net.ranges_and_elevations()[1][net.visible]
        out["elevation"][f"{n_high}_high_targets"] = {
            "max_elevation_deg": float(el.max()), "mean_cos_2_elevation": float(np.mean(np.cos(2 * np.deg2rad(el)))),
            "corr_a_E2_a_E0": float(correlation_and_sigma(net)[0][INDEX["a_E2_rad"], INDEX["a_E0_rad"]])}
    (RESULTS_DIR / "npl_network_test.json").write_text(json.dumps(out, indent=1))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.axhspan(0.5, 2.0, color="#eef3fb", zorder=0)
    ax.axhline(1.0, color=MUTED, lw=1.0, zorder=1)
    colours = ["#6250d6" if n == "mu" else "#2a78d6" for n in PARAMETER_NAMES]
    ax.bar(range(N_PARAMETERS), ratio, color=colours, width=0.7, zorder=3)
    ax.set_xticks(range(N_PARAMETERS), LABELS, fontsize=9)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("fitted standard uncertainty / the paper's (Tables 2–3)", color=INK)
    ax.set_title("(a) Same order of magnitude as the paper; layout not tuned", loc="left", fontsize=11, color=INK)
    ax.text(N_PARAMETERS - 0.5, 1.37, "shaded: 0.5 to 2 times the paper's value", ha="right", va="top", color=MUTED, fontsize=9)
    ax.text(1, ratio[1] + 0.03, "prior\n(μ)", ha="center", va="bottom", color="#6250d6", fontsize=8.5)
    ax = axes[1]
    x = np.linspace(-4, 4, 400)
    ax.hist(z[:, keep].ravel(), bins=np.linspace(-4, 4, 33), density=True, color="#2a78d6", edgecolor="white", zorder=3)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.6, zorder=4)
    ax.text(0.03, 0.95, f"{N_SEEDS} seeds × 15 parameters\nwithin 1σ: {100 * out['recovery']['within_1_sigma']:.1f}%  (68.3%)\n"
            f"within 2σ: {100 * out['recovery']['within_2_sigma']:.1f}%  (95.4%)\nwithin 3σ: {100 * out['recovery']['within_3_sigma']:.1f}%  (99.7%)",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK)
    ax.set_xlabel("(estimate − truth) / fitted standard uncertainty", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.set_title("(b) Recovery: error in units of the fitted uncertainty;  line: N(0, 1)", loc="left", fontsize=11, color=INK)
    for ax in axes:
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("The Hughes et al. 2011 network test on a reconstructed layout: 15 nests, 5 positions, both faces, 126 readings", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in (".png", ".svg"):
        fig.savefig(RESULTS_DIR / f"npl_network_test{suffix}", dpi=200, bbox_inches="tight")
    print(json.dumps({k: out[k] for k in ("recovery", "correlations", "birdbath", "elevation")}, indent=1)[:2600])
    print("uncertainties:", {k: out["uncertainties"][k] for k in ("target_rss_um", "station_position_rss_um", "station_pointing_arcsec")})


if __name__ == "__main__":
    main()
