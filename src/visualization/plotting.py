"""Plotting for metrology engineers: scenes, error ellipsoids, visibility
results, and network-solve outcomes.

**Not one of the six build steps.** CLAUDE.md §5's ladder is about the
physics chain (§2); this module sits alongside it, turning the objects
that chain already produces (a `Scene`, a per-point covariance, a list of
`VisibilityResult`, a `NetworkSolveResult`) into a figure. Per CLAUDE.md
§9 ("if a requested feature would take the project outside the six-step
ladder, say so before building it"): this is exactly such a feature, so
it lives in its own package, touches nothing in `geometry`, `instruments`
or `network` beyond reading their public objects, and has no gate of its
own -- there's nothing here CLAUDE.md §4 has a reference number for.

**Why matplotlib, and why static figures.** matplotlib is already an
approved dependency (CLAUDE.md §6); nothing new was added to build this.
Every function here returns a `(figure, axes)` pair rather than opening a
window or a server -- a caller can `.show()` it, save it to `results/`
(per §8, generated outputs, never edited by hand), or embed it in a
notebook, without this module needing to know which. This is deliberately
the smaller of two possible tiers of "interface":

  - **What's built here (tier A):** call a function, get a figure, in a
    script or a notebook. No new dependency, fits the existing library
    list exactly.
  - **What's not built (tier B):** a persistent, interactive GUI or web
    dashboard (Streamlit, Dash, PyVista's interactive viewer, ...) that a
    metrology engineer could click around in without writing any Python.
    That's a materially bigger addition -- new dependencies well outside
    CLAUDE.md §6's numpy/scipy/matplotlib/pytest list, its own UI code to
    maintain, and design decisions (what's a "session", what's editable,
    who else can see it) that are worth deciding deliberately rather than
    picking by default. If an interactive tool turns out to be wanted,
    that's a separate conversation, not an extension of this module.

**Scale.** Covariance ellipsoids in this codebase are a few micrometres
to a few tens of micrometres across; scenes are metres across. Drawn at
true size, an ellipsoid would be a few pixels at best -- invisible, or
misleadingly indistinguishable from a point. Every function that draws
ellipsoids therefore exaggerates them by an explicit `scale` factor and
prints that factor directly on the figure (`"ellipsoids exaggerated Nx"`)
so nobody mistakes the drawn size for the physical one. If a scale isn't
supplied, one is picked automatically so the largest ellipsoid spans
roughly 8% of the scene's extent -- a visible-but-not-scene-dominating
default, not a claim about the real uncertainty's relative size.
"""
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from characteristics.characteristic import CharacteristicPointUncertainty
from geometry.mesh import TriangleMesh
from geometry.pose import InstrumentPose
from geometry.scene import Scene
from geometry.visibility import VisibilityReason, VisibilityResult
from network.solve import NetworkSolveResult
from risk.jcgm106 import RiskResult

_VISIBILITY_COLORS = {
    VisibilityReason.VISIBLE: "tab:green",
    VisibilityReason.OUT_OF_RANGE: "tab:orange",
    VisibilityReason.INCIDENCE_TOO_STEEP: "tab:purple",
    VisibilityReason.OCCLUDED: "tab:red",
}


def _new_3d_axes(ax=None, figsize=(8, 7)):
    if ax is not None:
        return ax.figure, ax
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    return fig, ax


def _set_equal_3d_aspect(ax, points_m: np.ndarray):
    """matplotlib 3D axes don't default to an equal aspect ratio -- without
    this, a sphere plots as an ellipsoid and distances read wrong by eye.
    Centres the view on the point cloud and gives every axis the same
    half-range, the radius of the point furthest from the centroid."""
    centre_m = points_m.mean(axis=0)
    radius_m = max(float(np.max(np.linalg.norm(points_m - centre_m, axis=1))), 1e-6)
    ax.set_xlim(centre_m[0] - radius_m, centre_m[0] + radius_m)
    ax.set_ylim(centre_m[1] - radius_m, centre_m[1] + radius_m)
    ax.set_zlim(centre_m[2] - radius_m, centre_m[2] + radius_m)
    ax.set_box_aspect((1, 1, 1))


def _label_exaggeration(ax, scale: float):
    ax.text2D(
        0.02, 0.02, f"ellipsoids exaggerated {scale:.0f}x",
        transform=ax.transAxes, fontsize=8, color="gray",
    )


def plot_error_ellipsoid(ax, center_m, covariance_m2, scale=1.0, n_std=1.0, resolution=14, **kwargs):
    """Draw one covariance ellipsoid as a wireframe on 3D axes `ax`.

    Semi-axes are `n_std` standard deviations (`n_std * sqrt(eigenvalue)`)
    along each of the covariance's eigenvectors -- the principal-axis
    picture from `docs/step1_laser_tracker_physics.md`, drawn directly:
    for a laser tracker's per-point covariance this comes out as a
    flattened disc, thin along the beam. `scale` is a visual exaggeration
    factor (see module docstring); it does not change the ellipsoid's
    *shape*, only its drawn size.

    Remaining keyword arguments are forwarded to `Axes3D.plot_wireframe`
    (colour, alpha, linewidth, ...).
    """
    center_m = np.asarray(center_m, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_m2)
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # guard tiny negative numerical noise at 0
    radii_m = n_std * np.sqrt(eigenvalues)

    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    unit_sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ]
    ).reshape(3, -1)

    axes_scaled = eigenvectors @ np.diag(radii_m * scale)
    points_m = center_m[:, None] + axes_scaled @ unit_sphere
    x, y, z = points_m.reshape(3, resolution, resolution)
    return ax.plot_wireframe(x, y, z, **kwargs)


def _plot_mesh(ax, mesh: TriangleMesh, color="0.7", alpha=0.3):
    triangles_m = mesh.vertices_m[mesh.faces]  # (n_triangles, 3, 3)
    collection = Poly3DCollection(triangles_m, facecolor=color, edgecolor="0.4", alpha=alpha)
    ax.add_collection3d(collection)


def _plot_instrument_pose(ax, pose: InstrumentPose, length_m=0.3, label=None):
    """A station marker plus an arrow along its boresight -- the local +x
    axis, per geometry.spherical's convention (theta = phi = 0)."""
    boresight = pose.orientation[:, 0]
    ax.scatter(*pose.position_m, c="black", marker="^", s=60)
    ax.quiver(*pose.position_m, *(boresight * length_m), color="black", linewidth=1.2, arrow_length_ratio=0.3)
    if label:
        ax.text(*pose.position_m, label, fontsize=7)


def _auto_ellipsoid_scale(max_semi_axis_m: float, scene_radius_m: float, target_fraction: float = 0.08) -> float:
    if max_semi_axis_m <= 0.0:
        return 1.0
    return (target_fraction * scene_radius_m) / max_semi_axis_m


def plot_scene(
    scene: Scene,
    ax=None,
    visibility_results: Optional[List[VisibilityResult]] = None,
    tracker=None,
    ellipsoid_scale: Optional[float] = None,
    show_obstacles: bool = True,
    title: Optional[str] = None,
):
    """Plot a scene: the instrument station, its target points, and (if
    present) obstacle geometry -- the objects steps 1 and 2 work with.

    - Pass `visibility_results` (from `geometry.visibility.
      scene_visibility`, same order as `scene.target_points_m`) to colour
      each target by whether -- and why not -- it's measurable.
    - Pass `tracker` (a `LaserTracker`) to additionally draw every
      target's step 1 covariance ellipsoid, exaggerated per the module
      docstring's scale convention.
    """
    fig, ax = _new_3d_axes(ax)

    instrument_position_m = scene.instrument_pose.position_m
    ax.scatter(*instrument_position_m, c="black", marker="^", s=90, label="instrument")

    points_m = scene.target_points_m
    if visibility_results is not None:
        for reason in VisibilityReason:
            mask = np.array([r.reason is reason for r in visibility_results])
            if np.any(mask):
                ax.scatter(*points_m[mask].T, c=_VISIBILITY_COLORS[reason], label=reason.value, s=30)
    else:
        ax.scatter(*points_m.T, c="tab:blue", label="targets", s=30)

    if show_obstacles and scene.obstacles is not None:
        _plot_mesh(ax, scene.obstacles)

    all_points_m = np.vstack([points_m, instrument_position_m[None, :]])

    if tracker is not None:
        covariances = [tracker.covariance_global(points_m[i], scene.instrument_pose) for i in range(scene.n_points)]
        if ellipsoid_scale is None:
            centre_m = all_points_m.mean(axis=0)
            scene_radius_m = max(float(np.max(np.linalg.norm(all_points_m - centre_m, axis=1))), 1e-6)
            max_semi_axis_m = max((np.sqrt(np.linalg.eigvalsh(c).max()) for c in covariances), default=0.0)
            ellipsoid_scale = _auto_ellipsoid_scale(max_semi_axis_m, scene_radius_m)
        for point_m, covariance_m2 in zip(points_m, covariances):
            plot_error_ellipsoid(ax, point_m, covariance_m2, scale=ellipsoid_scale, color="tab:blue", linewidth=0.5, alpha=0.6)
        _label_exaggeration(ax, ellipsoid_scale)

    _set_equal_3d_aspect(ax, all_points_m)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return fig, ax


def plot_network_result(
    result: NetworkSolveResult,
    ax=None,
    ellipsoid_scale: Optional[float] = None,
    true_target_points_m: Optional[np.ndarray] = None,
    title: Optional[str] = None,
):
    """Plot a network solve's outcome (CLAUDE.md §5, step 3): every solved
    station pose (marker + boresight arrow), every solved target with its
    full 3x3 covariance ellipsoid (the actual `target_point_covariance`
    block -- not a diagonal approximation), and, if given, the true
    target positions for a visual accuracy check.
    """
    fig, ax = _new_3d_axes(ax)

    points_m = result.target_points_m
    ax.scatter(*points_m.T, c="tab:blue", s=25, label="solved targets")

    if true_target_points_m is not None:
        ax.scatter(*np.asarray(true_target_points_m).T, c="black", marker="x", s=25, label="true targets")

    for i, pose in enumerate(result.station_poses):
        _plot_instrument_pose(ax, pose, label=f"station {i}")

    station_positions_m = np.array([pose.position_m for pose in result.station_poses])
    all_points_m = np.vstack([points_m, station_positions_m])
    if true_target_points_m is not None:
        all_points_m = np.vstack([all_points_m, true_target_points_m])

    if ellipsoid_scale is None:
        centre_m = all_points_m.mean(axis=0)
        scene_radius_m = max(float(np.max(np.linalg.norm(all_points_m - centre_m, axis=1))), 1e-6)
        max_semi_axis_m = max(
            (np.sqrt(np.linalg.eigvalsh(result.target_point_covariance(i)).max()) for i in range(len(points_m))),
            default=0.0,
        )
        ellipsoid_scale = _auto_ellipsoid_scale(max_semi_axis_m, scene_radius_m)

    for i, point_m in enumerate(points_m):
        plot_error_ellipsoid(
            ax, point_m, result.target_point_covariance(i),
            scale=ellipsoid_scale, color="tab:blue", linewidth=0.5, alpha=0.5,
        )
    _label_exaggeration(ax, ellipsoid_scale)

    _set_equal_3d_aspect(ax, all_points_m)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return fig, ax


def plot_uncertainty_vs_station_count(
    station_counts: Iterable[int],
    mean_uncertainties_um: Iterable[float],
    ax=None,
    reference_um: Optional[Dict[int, float]] = None,
    title: str = "Mean positional uncertainty vs station count",
):
    """The step 3 gate, plotted: mean positional uncertainty (see
    `network.solve.mean_positional_uncertainty_m`) against station count
    for a fixed target set -- should fall and flatten (CLAUDE.md §5).

    `reference_um`, if given (e.g. `{1: 26.5, 2: 16.6, 3: 13.7, 4: 10.7}`,
    Wang, Forbes & Maropoulos 2014, CLAUDE.md §4d), overlays a literature
    series as a dashed line, explicitly labelled as a different
    scene/instrument -- CLAUDE.md's gate asks for the same *shape*, not
    the same numbers, and the legend says so rather than implying a
    direct comparison.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure

    station_counts = list(station_counts)
    mean_uncertainties_um = list(mean_uncertainties_um)
    ax.plot(station_counts, mean_uncertainties_um, "o-", color="tab:blue", label="this testbed")

    if reference_um is not None:
        ax.plot(
            list(reference_um.keys()), list(reference_um.values()), "s--", color="gray",
            label="Wang, Forbes & Maropoulos 2014 (different scene -- shape only)",
        )

    ax.set_xlabel("number of stations")
    ax.set_ylabel("mean positional uncertainty [µm]")
    ax.set_title(title)
    ax.set_xticks(station_counts)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_characteristic_comparison(
    labels: List[str],
    results: List[CharacteristicPointUncertainty],
    ax=None,
    title: str = "Characteristic uncertainty by tolerance type",
):
    """Bar chart of `uncertainty_m` (see `characteristics.characteristic.
    evaluate_characteristic`) across several characteristics -- the step 4
    gate (CLAUDE.md §5), plotted: different tolerance types on the same
    feature reading different uncertainties off the same covariance.

    `labels` and `results` must be the same length and in the same order
    (one label per `CharacteristicPointUncertainty`, e.g. from evaluating
    a position, a flatness and a parallelism characteristic on one point).
    This deliberately plots only the scalar `uncertainty_m` summary, not
    each result's full `projected_covariance_m2` -- a bar chart has one
    number per bar; see `docs/step4_characteristic_layer_physics.md` for
    the full projected covariances behind each bar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure

    uncertainties_um = [r.uncertainty_m * 1e6 for r in results]
    ax.bar(labels, uncertainties_um, color="tab:blue")
    ax.set_ylabel("characteristic uncertainty [µm]")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return fig, ax


def plot_risk_vs_uncertainty(
    uncertainties_m: Iterable[float],
    results_by_case: Dict[str, List[RiskResult]],
    ax=None,
    title: str = "Global conformity risk vs measurement uncertainty",
    log_scale: bool = True,
):
    """The step 5 gate, plotted (CLAUDE.md §5): total global risk
    (PFA + PFR, from `risk.jcgm106.evaluate_conformity_risk`) against
    measurement uncertainty, for one or more cases -- typically a
    high-capability process next to a marginal one, so the qualitative
    difference the gate describes ("near-zero... regardless of
    uncertainty" vs "strongly sensitive to uncertainty") is visible
    directly on a figure, not just asserted in a test.

    `results_by_case` maps a case label to a list of `RiskResult`, one per
    entry in `uncertainties_m`, in the same order (exactly what sweeping
    `evaluate_conformity_risk` across a range of uncertainty values for a
    fixed `ClusterAssignment`/`DecisionRule` produces).

    A high-capability and a marginal process's risk at the same
    uncertainty typically differ by many *orders of magnitude* (a
    difference in kind, not degree -- see
    `docs/step5_risk_layer_physics.md`), so the y-axis defaults to log
    scale: on a linear axis the high-capability case would flatten
    invisibly onto the x-axis next to the marginal one.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    uncertainties_um = [u * 1e6 for u in uncertainties_m]
    for label, results in results_by_case.items():
        total_risk = [r.probability_false_acceptance + r.probability_false_rejection for r in results]
        ax.plot(uncertainties_um, total_risk, "o-", label=label)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("measurement uncertainty [µm]")
    ax.set_ylabel("global risk (PFA + PFR)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return fig, ax
