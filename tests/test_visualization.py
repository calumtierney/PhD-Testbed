"""Smoke tests for visualization.plotting.

These check that every plotting function runs without error and returns
sane objects for a range of inputs -- not that the pixels look right
(matplotlib itself is trusted for that). Headless (Agg) backend so this
runs the same in CI as on a desktop.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from characteristics.characteristic import Characteristic, evaluate_characteristic
from characteristics.tolerances import FlatnessTolerance, PositionTolerance
from geometry.mesh import load_stl
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from geometry.visibility import VisibilityLimits, scene_visibility
from instruments.laser_tracker import LaserTracker, scene_point_covariances
from network.solve import (
    mean_positional_uncertainty_m,
    perturbed_initial_guess,
    simulate_observations,
    solve_network,
)
from planning.experiment import run_headline_experiment
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior, evaluate_conformity_risk
from visualization.plotting import (
    plot_characteristic_comparison,
    plot_error_ellipsoid,
    plot_network_result,
    plot_plan_comparison,
    plot_risk_vs_uncertainty,
    plot_scene,
    plot_uncertainty_vs_station_count,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_error_ellipsoid_returns_wireframe_artist():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    covariance = np.diag([1e-11, 4e-10, 4e-10])  # step-1-scale, metres^2
    artist = plot_error_ellipsoid(ax, np.array([1.0, 2.0, 3.0]), covariance, scale=1000.0)
    assert artist is not None


def test_plot_scene_bare():
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=np.array([[2.5, 0, 0], [0, 3.0, 0]]), instrument_pose=pose)
    fig, ax = plot_scene(scene)
    assert fig is not None and ax is not None


def test_plot_scene_with_tracker_ellipsoids():
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=np.array([[2.5, 0, 0], [0, 3.0, 0]]), instrument_pose=pose)
    tracker = LaserTracker()
    fig, ax = plot_scene(scene, tracker=tracker, title="with ellipsoids")
    assert ax.get_title() == "with ellipsoids"


def test_plot_scene_with_visibility_and_obstacles():
    wall = load_stl("scenes/wall.stl")
    pose = InstrumentPose(position_m=np.zeros(3))
    points = np.array([[3.0, 0.0, 0.0], [3.0, 5.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    scene = Scene(target_points_m=points, instrument_pose=pose, target_normals_m=normals, obstacles=wall)
    limits = VisibilityLimits(max_range_m=100.0, max_incidence_angle_rad=np.deg2rad(80.0))
    results = scene_visibility(scene, limits)

    fig, ax = plot_scene(scene, visibility_results=results)
    assert fig is not None


def test_plot_network_result_with_and_without_truth():
    rng = np.random.default_rng(0)
    tracker = LaserTracker()
    true_targets_m = np.array([[2.5, 0.0, 0.0], [0.0, 3.0, 0.5], [1.0, -1.0, 2.0]])
    true_poses = [
        InstrumentPose(position_m=np.array([0.0, 0.0, 0.0])),
        InstrumentPose(position_m=np.array([0.0, 4.0, 0.0])),
    ]
    observations = simulate_observations(true_targets_m, true_poses, tracker, rng)
    init_targets_m, init_poses = perturbed_initial_guess(true_targets_m, true_poses, 0.02, 0.005, rng)
    result = solve_network(observations, init_targets_m, init_poses, tracker)

    fig, ax = plot_network_result(result)
    assert fig is not None

    fig2, ax2 = plot_network_result(result, true_target_points_m=true_targets_m, title="check")
    assert ax2.get_title() == "check"


def test_plot_uncertainty_vs_station_count_with_reference():
    fig, ax = plot_uncertainty_vs_station_count(
        station_counts=[1, 2, 3, 4],
        mean_uncertainties_um=[17.6, 12.9, 11.3, 7.9],
        reference_um={1: 26.5, 2: 16.6, 3: 13.7, 4: 10.7},
    )
    assert ax.get_ylabel().startswith("mean positional")


def test_plot_functions_accept_existing_axes():
    """Passing an existing `ax` should reuse it, not create a new figure."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=np.array([[1.0, 0, 0]]), instrument_pose=pose)
    returned_fig, returned_ax = plot_scene(scene, ax=ax)
    assert returned_fig is fig
    assert returned_ax is ax


def test_plot_characteristic_comparison():
    tracker = LaserTracker()
    pose = InstrumentPose(position_m=np.zeros(3))
    scene = Scene(target_points_m=np.array([[2.5, 0.0, 0.0]]), instrument_pose=pose)
    covariances = scene_point_covariances(scene, tracker)

    position = Characteristic("position", [0], PositionTolerance(zone_diameter_m=1e-4))
    flatness = Characteristic(
        "flatness", [0], FlatnessTolerance(zone_width_m=5e-5, surface_normal=np.array([0.0, 1.0, 0.0]))
    )
    results = [
        evaluate_characteristic(position, covariances)[0],
        evaluate_characteristic(flatness, covariances)[0],
    ]

    fig, ax = plot_characteristic_comparison(["position", "flatness"], results)
    assert fig is not None
    assert ax.get_ylabel().startswith("characteristic")


def test_plot_risk_vs_uncertainty():
    tolerance = DecisionRule.simple_acceptance(-50e-6, 50e-6)
    uncertainties_m = [1e-6, 5e-6, 10e-6]
    high_capability = ClusterAssignment("c1", FeatureCluster("A", ProcessPrior(0.0, 5e-6)))
    marginal = ClusterAssignment("c2", FeatureCluster("B", ProcessPrior(40e-6, 6e-6)))

    results_by_case = {
        "high-capability": [evaluate_conformity_risk(high_capability, u, tolerance) for u in uncertainties_m],
        "marginal": [evaluate_conformity_risk(marginal, u, tolerance) for u in uncertainties_m],
    }
    fig, ax = plot_risk_vs_uncertainty(uncertainties_m, results_by_case)
    assert fig is not None
    assert ax.get_yscale() == "log"


def test_plot_plan_comparison():
    result = run_headline_experiment(azimuths_deg=[0, 120, 240], elevations_deg=[30], n_stations=2)
    names = [c.name for c in result.scenario.characteristics]
    u_a = [result.plan_a.per_characteristic_uncertainty_m[n] for n in names]
    u_b = [result.plan_b.per_characteristic_uncertainty_m[n] for n in names]

    fig, ax = plot_plan_comparison(names, u_a, u_b)
    assert fig is not None
    assert len(ax.patches) == 2 * len(names)
