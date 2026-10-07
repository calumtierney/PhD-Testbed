"""Interactive explorer (SETUP.md's "Run the interactive explorer"):

    streamlit run app.py

A thin Streamlit front-end over existing `visualization.plotting` calls
-- no new physics, just sliders over functions the six build steps
already provide. Deliberately small: `src/visualization/plotting.py`'s
own module docstring calls a persistent interactive dashboard "tier B",
a materially bigger addition than the static-figure "tier A" this
codebase otherwise builds, worth deciding on deliberately rather than
growing by default -- this file is that deliberate, scoped addition, not
an attempt to reproduce every `viz.make_figures` panel interactively.

Two tabs:

1. **Tracker error ellipsoid** (step 1) -- move range/azimuth/elevation/
   systematic-term sliders, watch the per-point covariance ellipsoid and
   its anisotropy ratio respond live. The same `LaserTracker.
   covariance_global` call `viz.make_figures.make_panel1_scene` uses,
   just re-run on every slider move instead of once.
2. **Conformity risk vs uncertainty** (step 5) -- move process mean/std
   (equivalently, Cpk) and tolerance-zone-width sliders, watch
   `risk.jcgm106.evaluate_conformity_risk`'s curve and a chosen
   characteristic's PFA/PFR update live. Same call
   `viz.make_figures.make_panel5_risk` makes once per cluster; here any
   process, not just the two fixed clusters.
"""
import numpy as np
import streamlit as st

from geometry.pose import InstrumentPose
from geometry.scene import Scene
from instruments.laser_tracker import LaserTracker
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, evaluate_conformity_risk
from visualization.plotting import plot_risk_vs_uncertainty, plot_scene

st.set_page_config(page_title="LVM Testbed Explorer", layout="wide")
st.title("LVM measurement planning testbed -- interactive explorer")
st.caption(
    "A thin Streamlit layer over existing plotting calls -- see app.py's own "
    "docstring for scope, and README.md for the static-figure command "
    "(`python -m viz.make_figures`) this complements."
)

tab_tracker, tab_risk = st.tabs(["Tracker error ellipsoid (step 1)", "Conformity risk (step 5)"])

with tab_tracker:
    st.markdown(
        "A laser tracker's measurement error ellipsoid is flattened across the "
        "beam (angular-encoder noise, growing with range) and thin along it "
        "(interferometer noise, roughly range-independent) -- CLAUDE.md §4b. "
        "Move the sliders and watch the ellipsoid, and its anisotropy ratio, "
        "respond."
    )
    col_controls, col_figure = st.columns([1, 2])
    with col_controls:
        range_m = st.slider("Range to target [m]", min_value=0.5, max_value=20.0, value=2.5, step=0.1)
        azimuth_deg = st.slider("Azimuth [deg]", min_value=-180.0, max_value=180.0, value=0.0, step=1.0)
        elevation_deg = st.slider("Elevation [deg]", min_value=-80.0, max_value=80.0, value=0.0, step=1.0)
        systematic_std_um = st.slider(
            "Systematic (placement-irreducible) term [µm]", min_value=0.0, max_value=20.0, value=0.0, step=0.5,
            help="LaserTracker.systematic_std_m -- calibration, thermal drift, "
            "refraction, nest repeatability; see that field's docstring.",
        )

    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    point_local_m = range_m * np.array(
        [np.cos(elevation_rad) * np.cos(azimuth_rad), np.cos(elevation_rad) * np.sin(azimuth_rad), np.sin(elevation_rad)]
    )
    tracker = LaserTracker(systematic_std_m=systematic_std_um * 1e-6)
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=point_local_m[None, :], instrument_pose=pose)

    covariance = tracker.covariance_global(point_local_m, pose)
    eigenvalues_m = np.sqrt(np.linalg.eigvalsh(covariance))[::-1]  # largest first
    anisotropy_ratio = eigenvalues_m[0] / eigenvalues_m[-1] if eigenvalues_m[-1] > 0 else float("inf")

    with col_figure:
        fig, ax = plot_scene(scene, tracker=tracker, title=f"Single point at {range_m:.1f} m")
        st.pyplot(fig)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Largest semi-axis (1σ)", f"{eigenvalues_m[0] * 1e6:.2f} µm")
    metric_cols[1].metric("Smallest semi-axis (1σ)", f"{eigenvalues_m[-1] * 1e6:.2f} µm")
    metric_cols[2].metric("Anisotropy ratio", f"{anisotropy_ratio:.1f} : 1")
    st.caption(
        "CLAUDE.md §4b's gate: at 2.5 m range, azimuth/elevation ~0, sensor noise "
        "alone, expect a ratio of roughly 5:1-7:1 with the long axes perpendicular "
        "to the beam. Try range_m=2.5, azimuth=0, elevation=0, systematic=0."
    )

with tab_risk:
    st.markdown(
        "JCGM 106 conformity risk (step 5): a process prior (how the true "
        "values are actually distributed) plus a measurement uncertainty gives "
        "a probability of false acceptance/rejection. Move the process's mean "
        "offset and spread (equivalently, its Cpk against the tolerance you "
        "set) and watch the risk-vs-uncertainty curve, and one chosen "
        "uncertainty's PFA/PFR, respond."
    )
    col_controls, col_figure = st.columns([1, 2])
    with col_controls:
        zone_width_um = st.slider("Tolerance zone width [µm]", min_value=20.0, max_value=200.0, value=100.0, step=5.0)
        process_mean_um = st.slider("Process mean offset from nominal [µm]", min_value=-40.0, max_value=40.0, value=0.0, step=1.0)
        process_std_um = st.slider("Process std (part-to-part spread) [µm]", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
        chosen_uncertainty_um = st.slider("Measurement uncertainty to evaluate [µm]", min_value=1.0, max_value=60.0, value=8.0, step=0.5)

    decision_rule = DecisionRule.from_zone_width(zone_width_um * 1e-6)
    cluster = FeatureCluster("interactive", ProcessPrior(mean_m=process_mean_um * 1e-6, std_m=process_std_um * 1e-6))
    assignment = ClusterAssignment("interactive characteristic", cluster)

    uncertainties_um = np.linspace(1, 60, 60)
    results = [evaluate_conformity_risk(assignment, u * 1e-6, decision_rule) for u in uncertainties_um]
    chosen_result = evaluate_conformity_risk(assignment, chosen_uncertainty_um * 1e-6, decision_rule)

    with col_figure:
        fig, ax = plot_risk_vs_uncertainty(
            uncertainties_um * 1e-6, {f"Cpk~{chosen_result.process_cpk:.2f}": results},
            title="Global risk (PFA + PFR) vs measurement uncertainty",
        )
        ax.axvline(chosen_uncertainty_um, color="gray", linestyle="--", linewidth=1)
        st.pyplot(fig)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Process Cpk", f"{chosen_result.process_cpk:.2f}")
    metric_cols[1].metric("PFA at chosen uncertainty", f"{chosen_result.probability_false_acceptance:.4g}")
    metric_cols[2].metric("PFR at chosen uncertainty", f"{chosen_result.probability_false_rejection:.4g}")
    st.caption(
        "CLAUDE.md §5 step 5's gate: push Cpk high (mean near 0, std small "
        "relative to the zone) and risk should go flat near zero regardless of "
        "uncertainty; push it down near AS13006's 1.33 floor (or below) and risk "
        "should respond sharply to uncertainty across the whole range."
    )
