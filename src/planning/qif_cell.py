"""From a QIF document to a factory cell the chain can plan, and a plan for it.

`interchange.qif.read_qif` turns a file into data. This module decides what of
that data the chain of CLAUDE.md §2 can use today, builds the scene round it,
and says plainly what it left out and what it assumed. Nothing is planned for
a requirement the model cannot represent: it is listed in `QifCell.skipped`
with the reason, not approximated silently.

**1. Which characteristics.** The chain handles a characteristic that is one
measured point judged along the direction(s) its tolerance constrains
(CLAUDE.md §5 step 4). In QIF terms:

  * `PointProfile` on a feature with a location and a normal -> the testbed's
    `ProfileTolerance`: the point's deviation along the surface normal against
    +- half the zone width. This is exact.
  * `Position` with a diametrical zone on a feature with a location and an axis
    or normal (a hole) -> `PositionTolerance` across that axis. The zone is
    round; the risk model is one-dimensional. The hole is therefore judged on
    its deviation along one direction against +- half the zone diameter, with
    the measurement uncertainty of the worst-measured direction across the
    axis. That understates how often a round zone is violated and is marked
    `approximate` on every such characteristic. A two-dimensional risk
    calculation is the proper fix and belongs in `risk`, not here.
  * `Position` with a spherical zone -> the same, in three directions.
  * Everything else is skipped with a reason: sizes (a diameter is not a
    location), form and orientation tolerances and surface profiles of whole
    faces (they need several points and their joint covariance, which the
    characteristic layer does not have yet), unequally disposed zones, features
    the file gives no location or direction for.

**2. Where the part is.** A QIF file is in the part's (or the vehicle's, or the
aircraft's) coordinates; the sample sheet-metal part sits 2.5 m from its
origin. `default_placement` puts the part on the cell floor plan: the file's
`up_axis` (z unless told otherwise; the file does not say) stays vertical, the
longer plan-view side lies along the cell's x, the part starts at x = 0, is
centred on y = 0 and its lowest point is at the jig height. Pass `placement`
(a 4x4 matrix, part to cell) when the real fixture position is known.

**3. How big the cell is.** The station track (`geometry.station_track`) is
drawn round the part's plan-view extent at the rules' standoff, and the
reference network is laid out round the track by `planning.cell.
reference_nests_for_track`. So a 0.1 m bracket and a 15 m spar get cells of
different size from the same rules. The extent is that of the *features the
file locates* (plus an obstacle mesh if one is given): a QIF file need not
contain the part's shape, so the true envelope may be larger. Every rule in
`CellRules` is a placeholder for a real cell, TODO(source).

**4. What blocks a sight line.** Only what is given: range, the incidence
angle at each target's normal, and an obstacle mesh if the caller supplies one
(e.g. the part's STL, in part coordinates). With no mesh the part cannot hide
its own far side except through the incidence limit, which is optimistic, and
the cell says so in its notes.

**5. What the part frame is.** A deviation is toleranced against datums. QIF
names datum *features* (a plane, a hole), not the points an inspector will
probe on them. `alignment="auto"` uses the located features of the datums the
planned characteristics refer to, if there are at least three and they are not
in a line; otherwise it falls back to a best fit through all the planned
points themselves, and says so. Either way the frame is a rigid best fit
(`planning.cell.PartFrameScorer`), not a 3-2-1 datum construction.

**6. What the process does.** If the file holds earlier inspections, each
characteristic's deviations are computed from the measured feature locations
(measured minus nominal, along the constrained direction) and pooled by
**cluster**: characteristics of the same type, tolerance and datum frame on the
same part (CLAUDE.md §3: priors belong to feature clusters, not to single
characteristics). The cluster's mean and standard deviation over n parts give a
Student-t predictive distribution (`risk.decision.ProcessKnowledge.
from_builds`). That clustering is a rule, not an engineer's judgement, and the
cell reports each cluster's members so it can be overruled. With no earlier
inspections the cluster gets a centred placeholder at `CellRules.default_cpk`,
marked as such: risk numbers from a placeholder are illustrative only.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from characteristics.characteristic import Characteristic
from characteristics.tolerances import PositionTolerance, ProfileTolerance
from geometry.mesh import TriangleMesh
from geometry.pose import InstrumentPose
from geometry.station_track import StationTrack
from geometry.visibility import VisibilityLimits
from instruments.laser_tracker import LaserTracker, default_planning_tracker
from interchange.qif import QifCharacteristic, QifDocument
from planning.cell import (
    CellRiskModel, MaxUncertaintyObjective, PartFrameScorer, RiskObjective, TraceObjective, WeightedUncertaintyObjective,
    build_risk_curve, layout_poses, measurement_time_minutes, reference_nests_for_track, search_cell_layout,
)
from planning.candidates import levelled_orientation
from planning.objectives import PlanningScenario
from risk.decision import ProcessKnowledge
from risk.jcgm106 import ClusterAssignment, DecisionRule, FeatureCluster, ProcessPrior

SIZE_KINDS = {"Diameter", "Radius", "Width", "Length", "Height", "Depth", "Thickness", "SphericalDiameter", "SphericalRadius",
              "Chord", "ConicalTaper", "FlatTaper", "Thread"}
MULTI_POINT_KINDS = {"SurfaceProfile", "LineProfile", "SurfaceProfileNonUniform", "Flatness", "Straightness", "Circularity",
                     "Cylindricity", "Perpendicularity", "Parallelism", "Angularity", "Symmetry", "Concentricity",
                     "Coaxiality", "CircularRunout", "TotalRunout", "Sphericity", "Conicity"}


@dataclass(frozen=True)
class CellRules:
    """How a cell is drawn round a part. Every number is a placeholder for a
    real cell's layout (TODO(source)); the defaults are the ones the long-part
    experiments use, so an imported part and that scene are built alike."""

    standoff_m: float = 3.0                       # track rows this far from the part's long sides
    part_bottom_height_m: float = 0.5             # the jig lifts the part's lowest located point to here
    station_height_bounds_m: Tuple[float, float] = (0.8, 3.0)
    nest_row_beyond_track_m: float = 2.25
    nest_pillar_spacing_m: float = 5.0
    nest_overhang_m: float = 2.5
    nest_end_beyond_loop_m: float = 3.25
    nest_end_half_spacing_m: float = 2.5
    nest_low_height_m: float = 1.0
    nest_above_part_m: float = 0.8
    max_range_m: float = 30.0                     # CLAUDE.md §1's upper range
    max_incidence_angle_rad: float = float(np.deg2rad(80.0))  # the repo's convention, not a literature value
    large_volume_from_m: float = 2.0              # CLAUDE.md §1: LVM is 2 m to 30 m
    default_cpk: float = 1.33                     # placeholder capability when the file holds no earlier inspections


@dataclass(frozen=True)
class PlannedCharacteristic:
    name: str
    qif_id: str
    kind: str
    feature_name: str
    target_index: int
    tolerance_half_width_m: float
    cluster: str
    approximate: bool           # a round or spherical zone judged in one dimension (see the module docstring)


@dataclass(frozen=True)
class ClusterSummary:
    name: str
    members: Tuple[str, ...]
    mean_m: float
    std_m: float
    n_builds: Optional[int]     # None: no earlier inspections, a placeholder
    source: str


@dataclass(frozen=True)
class QifCell:
    scenario: PlanningScenario
    track: StationTrack
    datum_indices: Tuple[int, ...]
    part_to_cell: np.ndarray
    planned: Tuple[PlannedCharacteristic, ...]
    clusters: Tuple[ClusterSummary, ...]
    knowledge: Dict[str, ProcessKnowledge]        # by cluster name
    skipped: Tuple[Tuple[str, str], ...]          # (characteristic name, why it is not planned)
    notes: Tuple[str, ...]
    envelope_min_m: np.ndarray
    envelope_max_m: np.ndarray
    alignment: str
    n_part_targets: int
    rules: CellRules = field(default_factory=CellRules)

    def scorer(self, require_all_characteristics: bool = True) -> PartFrameScorer:
        return PartFrameScorer(self.scenario, self.datum_indices, require_all_characteristics)

    def risk_model(self, rule: str = "simple") -> CellRiskModel:
        """Total decision risk against each characteristic's uncertainty, from
        each cluster's process knowledge. `rule`: "simple" (accept inside the
        tolerance limits) or "best" (accept when the part is more likely
        conforming than not). The two agree closely when the uncertainty is
        small against the tolerance; "best" takes minutes to tabulate for
        process knowledge from few builds (about four minutes for the sheet-
        metal sample's six clusters), "simple" about a second."""
        curves = {}
        for p in self.planned:
            key = (p.cluster, p.tolerance_half_width_m)
            if key not in curves:
                curves[key] = build_risk_curve(self.knowledge[p.cluster], -p.tolerance_half_width_m,
                                               p.tolerance_half_width_m, rule)
        return CellRiskModel(tuple(p.name for p in self.planned),
                             tuple(curves[(p.cluster, p.tolerance_half_width_m)] for p in self.planned), f"{rule} rule")

    def summary(self) -> dict:
        size = self.envelope_max_m - self.envelope_min_m
        return {
            "part_extent_m": size.tolist(),
            "track": {"perimeter_m": float(self.track.perimeter_m), "row_length_m": float(self.track.row_length_m),
                      "rows_y_m": [float(self.track.y_right_row_m), float(self.track.y_left_row_m)],
                      "standoff_m": self.rules.standoff_m},
            "n_part_targets": self.n_part_targets,
            "n_reference_nests": int(len(self.scenario.target_points_m) - self.n_part_targets),
            "alignment": self.alignment,
            "n_datum_targets": len(self.datum_indices),
            "part_to_cell": self.part_to_cell.tolist(),
            "planned": [{"name": p.name, "kind": p.kind, "feature": p.feature_name, "cluster": p.cluster,
                         "tolerance_half_width_um": p.tolerance_half_width_m * 1e6, "approximate": p.approximate}
                        for p in self.planned],
            "skipped": [{"name": n, "reason": r} for n, r in self.skipped],
            "clusters": [{"name": c.name, "members": list(c.members), "mean_um": c.mean_m * 1e6, "std_um": c.std_m * 1e6,
                          "n_builds": c.n_builds, "source": c.source} for c in self.clusters],
            "notes": list(self.notes),
        }


# -- placement ---------------------------------------------------------------------------
_UP_TO_Z = {
    "z": np.eye(3),
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "x": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
}


def default_placement(points_m: np.ndarray, rules: CellRules = CellRules(), up_axis: str = "z") -> np.ndarray:
    """The 4x4 matrix taking file coordinates to cell coordinates when nothing
    is known about the fixture: `up_axis` of the file becomes the cell's
    vertical, the longer plan-view side of the points' bounding box is laid
    along the cell's x, and the box is moved to start at x = 0, straddle y = 0
    and stand with its lowest point at `rules.part_bottom_height_m`."""
    if up_axis not in _UP_TO_Z:
        raise ValueError(f"up_axis must be 'x', 'y' or 'z', got {up_axis!r}")
    rotation = _UP_TO_Z[up_axis]
    points = np.asarray(points_m, dtype=float) @ rotation.T
    extent = points.max(axis=0) - points.min(axis=0)
    if extent[1] > extent[0]:  # lay the longer side along x: a quarter turn about the vertical
        quarter = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotation = quarter @ rotation
        points = np.asarray(points_m, dtype=float) @ rotation.T
    low, high = points.min(axis=0), points.max(axis=0)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [-low[0], -(low[1] + high[1]) / 2.0, rules.part_bottom_height_m - low[2]]
    return matrix


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


# -- building the cell -------------------------------------------------------------------
def _classify(c: QifCharacteristic, doc: QifDocument) -> Tuple[Optional[str], str]:
    """`(mapping, reason)`: mapping is "profile", "position_axis",
    "position_sphere" or None (not planned, for `reason`)."""
    if c.kind in SIZE_KINDS:
        return None, f"{c.kind}: a size, not a location; it needs points round the feature, not one point"
    if c.kind in MULTI_POINT_KINDS:
        return None, (f"{c.kind}: needs several points on the feature and their joint covariance; "
                      "the characteristic layer handles single points only so far")
    if c.kind not in ("PointProfile", "Position"):
        return None, f"{c.kind}: not a characteristic type the chain can represent yet"
    if c.tolerance_m is None or not c.tolerance_m > 0.0:
        return None, "no usable tolerance value in the file"
    if c.unequal_zone:
        return None, "unequally disposed or offset zone: its limits are not +- half the zone, and are not read yet"
    if len(c.feature_ids) != 1:
        return None, f"refers to {len(c.feature_ids)} features; exactly one is needed"
    feature = doc.features.get(c.feature_ids[0])
    if feature is None or feature.location_m is None:
        return None, "its feature has no location in the file (defined by CAD faces only?)"
    if feature.direction is None:
        return None, "its feature has no normal or axis, so neither the tolerance direction nor the side it can be seen from is known"
    if c.kind == "PointProfile":
        return "profile", ""
    if c.zone_shape == "Diametrical":
        return "position_axis", ""
    if c.zone_shape == "Spherical":
        return "position_sphere", ""
    return None, f"position with a {c.zone_shape or 'unstated'} zone: the zone's direction is not read yet"


def _collinear(points: np.ndarray) -> bool:
    if len(points) < 3:
        return True
    centred = points - points.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return singular[1] <= 1e-9 * max(singular[0], 1e-300)


def build_qif_cell(
    doc: QifDocument,
    scope_id: Optional[str] = None,
    rules: CellRules = CellRules(),
    placement: Optional[np.ndarray] = None,
    up_axis: str = "z",
    obstacle_part_frame: Optional[TriangleMesh] = None,
    alignment: str = "auto",
    tracker: Optional[LaserTracker] = None,
    with_reference_network: bool = True,
) -> QifCell:
    """Build the cell for one part or assembly of a QIF document (`scope_id`
    from `doc.scope_names()`; None = the document's root). See the module
    docstring for every decision made here. `alignment`: "auto", "datums" or
    "best_fit". `obstacle_part_frame`: a mesh in the file's coordinates, in
    metres, that blocks sight lines (the part itself, a jig)."""
    if alignment not in ("auto", "datums", "best_fit"):
        raise ValueError(f"alignment must be 'auto', 'datums' or 'best_fit', got {alignment!r}")
    notes: List[str] = []
    placements = doc.part_placements(scope_id)
    multiply_instanced = [doc.parts[p].name for p, t in placements.items() if len(t) > 1 and p in doc.parts]
    if multiply_instanced:
        notes.append(f"parts instanced more than once ({', '.join(multiply_instanced)}): only the first instance is planned")

    def to_scope(feature_id: str) -> np.ndarray:
        part_id = doc.part_of_feature(feature_id)
        return placements[part_id][0] if part_id in placements else np.eye(4)

    # 1. which characteristics, and the features they need
    planned_specs, skipped = [], []
    for c in doc.characteristics_in(scope_id):
        mapping, reason = _classify(c, doc)
        if mapping is None:
            skipped.append((c.name, reason))
        else:
            planned_specs.append((c, mapping))
    if not planned_specs:
        reasons = sorted({r.split(":")[0] for _, r in skipped})
        raise ValueError("the document has no characteristic the chain can plan"
                         + (f" (skipped: {', '.join(reasons)})" if reasons else " (it has no characteristics)"))
    feature_order: List[str] = list(dict.fromkeys(c.feature_ids[0] for c, _ in planned_specs))

    # 5. the part frame: datum features if they can fix it, else a best fit through the planned points
    datum_feature_ids: List[str] = []
    for c, _ in planned_specs:
        for label in c.datum_labels:
            for datum in doc.datums.values():
                if datum.label == label:
                    datum_feature_ids.extend(f for f in datum.feature_ids if f in doc.features
                                             and doc.features[f].location_m is not None and doc.features[f].direction is not None)
    datum_feature_ids = list(dict.fromkeys(datum_feature_ids))
    datum_points = np.array([_apply(to_scope(f), doc.features[f].location_m) for f in datum_feature_ids]).reshape(-1, 3)
    datums_usable = not _collinear(datum_points)
    if alignment == "datums" and not datums_usable:
        raise ValueError(f"alignment='datums' needs at least three located datum features not in a line; the file gives {len(datum_points)}")
    use_datums = datums_usable and alignment in ("auto", "datums")
    if use_datums:
        alignment_used = f"best fit to the {len(datum_feature_ids)} located datum features"
        feature_order += [f for f in datum_feature_ids if f not in feature_order]
    else:
        alignment_used = "best fit through all the planned points (the file gives no usable datum points)"
        if alignment == "auto":
            notes.append("the file locates fewer than three datum features (or they are in a line), so the part frame is a "
                         "best fit through the planned points themselves; give datum points to change that")

    # 2. where the part is
    scope_points = np.array([_apply(to_scope(f), doc.features[f].location_m) for f in feature_order])
    scope_normals = np.array([to_scope(f)[:3, :3] @ doc.features[f].direction for f in feature_order])
    obstacle_vertices = None if obstacle_part_frame is None else np.asarray(obstacle_part_frame.vertices_m, dtype=float)
    everything = scope_points if obstacle_vertices is None else np.vstack([scope_points, obstacle_vertices])
    if placement is None:
        part_to_cell = default_placement(everything, rules, up_axis)
        notes.append(f"placement assumed: the file's {up_axis} axis is vertical, the part's longer side lies along the cell, "
                     f"its lowest located point is {rules.part_bottom_height_m} m above the floor")
    else:
        part_to_cell = np.asarray(placement, dtype=float)
        if part_to_cell.shape != (4, 4):
            raise ValueError("placement must be a 4x4 matrix (file coordinates to cell coordinates, metres)")
    points = _apply(part_to_cell, scope_points)
    normals = scope_normals @ part_to_cell[:3, :3].T
    obstacle = None
    if obstacle_part_frame is not None:
        obstacle = TriangleMesh(_apply(part_to_cell, obstacle_vertices), obstacle_part_frame.faces)
    else:
        notes.append("no part geometry was given: a sight line is limited by range and by the incidence angle at each "
                     "target only, so the part cannot hide its own far side (optimistic)")
    envelope = points if obstacle is None else np.vstack([points, obstacle.vertices_m])
    low, high = envelope.min(axis=0), envelope.max(axis=0)
    if obstacle is None:
        notes.append("the part's extent is taken from the features the file locates; the real envelope may be larger")

    # 3. the cell round it
    span = float(max(high[0] - low[0], high[1] - low[1], high[2] - low[2]))
    if span < rules.large_volume_from_m:
        notes.append(f"the located features span {span:.2f} m: below the {rules.large_volume_from_m:g} m where large-volume "
                     "metrology starts, so a tracker network is unlikely to be the instrument of choice for this part")
    if high[2] > rules.station_height_bounds_m[1]:
        notes.append(f"the part reaches {high[2]:.1f} m, above the highest station height allowed ({rules.station_height_bounds_m[1]} m)")
    track = StationTrack(float(low[0]), float(max(high[0], low[0] + 1e-3)), float(max(high[1] - low[1], 1e-3)),
                         rules.standoff_m, part_centre_y_m=float((low[1] + high[1]) / 2.0))
    n_part_targets = len(points)
    if with_reference_network:
        nest_points, nest_normals = reference_nests_for_track(
            track, float(high[2]), rules.nest_row_beyond_track_m, rules.nest_pillar_spacing_m, rules.nest_overhang_m,
            rules.nest_end_beyond_loop_m, rules.nest_end_half_spacing_m, rules.nest_low_height_m, rules.nest_above_part_m)
        points, normals = np.vstack([points, nest_points]), np.vstack([normals, nest_normals])
    index_of = {f: i for i, f in enumerate(feature_order)}
    datum_indices = tuple(index_of[f] for f in datum_feature_ids) if use_datums else tuple(
        dict.fromkeys(index_of[c.feature_ids[0]] for c, _ in planned_specs))

    # 1 (continued) and 6: tolerances, clusters, process knowledge
    part_names = {p.id: p.name for p in doc.parts.values()}
    members: Dict[str, List[Tuple[QifCharacteristic, str]]] = {}
    for c, mapping in planned_specs:
        half_um = c.tolerance_m / 2.0 * 1e6
        frame = "|".join(c.datum_labels) if c.datum_labels else "no datums"
        part = part_names.get(doc.part_of_feature(c.feature_ids[0]) or "", "")
        key = f"{c.kind} +-{half_um:g} um [{frame}]" + (f" on {part}" if part and len(doc.parts) > 1 else "")
        members.setdefault(key, []).append((c, mapping))
    clusters, knowledge = [], {}
    for key, group in members.items():
        half = group[0][0].tolerance_m / 2.0
        deviations, n_builds = [], 0
        for c, mapping in group:
            feature = doc.features[c.feature_ids[0]]
            if not feature.measured_locations_m:
                continue
            offsets = np.array(feature.measured_locations_m) - feature.location_m  # file coordinates
            n_builds = max(n_builds, len(offsets))
            if mapping == "profile":
                deviations.extend(offsets @ feature.direction)
            else:  # a round zone judged along the direction its process is biased in (the module docstring, point 1)
                basis = (PositionTolerance(c.tolerance_m, feature.direction).constraint_directions()
                         if mapping == "position_axis" else np.eye(3))
                components = offsets @ basis.T
                mean = components.mean(axis=0)
                direction = mean / np.linalg.norm(mean) if np.linalg.norm(mean) > 1e-12 else np.eye(len(basis))[0]
                deviations.extend(components @ direction)
        deviations = np.array(deviations, dtype=float)
        if n_builds >= 2 and len(deviations) >= 2 and float(np.std(deviations, ddof=1)) > 0.0:
            mean_m, std_m = float(np.mean(deviations)), float(np.std(deviations, ddof=1))
            knowledge[key] = ProcessKnowledge.from_builds(mean_m, std_m, n_builds)
            source = f"{n_builds} measured parts in the file ({len(deviations)} deviations pooled over the cluster)"
            clusters.append(ClusterSummary(key, tuple(c.name for c, _ in group), mean_m, std_m, n_builds, source))
        else:
            std_m = half / (3.0 * rules.default_cpk)
            knowledge[key] = ProcessKnowledge.known(0.0, std_m)
            clusters.append(ClusterSummary(key, tuple(c.name for c, _ in group), 0.0, std_m, None,
                                           f"PLACEHOLDER: no earlier inspections in the file; centred, Cpk {rules.default_cpk} assumed"))
    if any(c.n_builds is None for c in clusters):
        notes.append("some clusters have no process data and use a placeholder capability: their risk figures are illustrative")

    rotation = part_to_cell[:3, :3]
    characteristics, assignments, decision_rules, planned = [], {}, {}, []
    used_names = set()
    cluster_of = {c.id: key for key, group in members.items() for c, _ in group}
    summary_of = {c.name: c for c in clusters}
    for c, mapping in planned_specs:
        name = c.name if c.name not in used_names else f"{c.name} ({c.id})"
        used_names.add(name)
        feature = doc.features[c.feature_ids[0]]
        direction = rotation @ (to_scope(feature.id)[:3, :3] @ feature.direction)
        if mapping == "profile":
            tolerance = ProfileTolerance(zone_width_m=c.tolerance_m, surface_normal=direction)
        else:
            tolerance = PositionTolerance(zone_diameter_m=c.tolerance_m, axis_direction=direction if mapping == "position_axis" else None)
        target = index_of[feature.id]
        characteristics.append(Characteristic(name=name, target_indices=[target], tolerance=tolerance))
        cluster = summary_of[cluster_of[c.id]]
        assignments[name] = ClusterAssignment(name, FeatureCluster(cluster.name, ProcessPrior(cluster.mean_m, cluster.std_m)), 1.0)
        decision_rules[name] = DecisionRule.from_zone_width(c.tolerance_m)
        planned.append(PlannedCharacteristic(name, c.id, c.kind, feature.name, target, c.tolerance_m / 2.0, cluster.name,
                                             approximate=mapping != "profile"))
        if c.material_condition in ("MAXIMUM", "LEAST"):
            notes.append(f"{name}: a material-condition modifier gives bonus tolerance, which is ignored (planned at the stated value)")
    if any(p.approximate for p in planned):
        notes.append("position characteristics have a round zone but are judged in one dimension (worst-measured direction "
                     "across the axis, +- half the zone diameter): their risk is indicative and understates a round zone's")
        notes.append("a hole's normal in the file may point into the part; if so the side it can be seen from is wrong here")

    anchor_xy = track.position_xy_m(track.tip_minus_x_s_m)
    anchor = np.array([anchor_xy[0], anchor_xy[1], float(np.mean(rules.station_height_bounds_m))])
    scenario = PlanningScenario(
        target_points_m=points, tracker=tracker if tracker is not None else default_planning_tracker(),
        characteristics=characteristics, assignments=assignments, decision_rules=decision_rules,
        anchor_pose=InstrumentPose(position_m=anchor, orientation=levelled_orientation(anchor, points.mean(axis=0))),
        target_normals_m=normals, obstacles=obstacle,
        visibility_limits=VisibilityLimits(max_range_m=rules.max_range_m, max_incidence_angle_rad=rules.max_incidence_angle_rad),
    )
    notes.extend(dict.fromkeys(doc.warnings))
    return QifCell(scenario, track, datum_indices, part_to_cell, tuple(planned), tuple(clusters), knowledge, tuple(skipped),
                   tuple(dict.fromkeys(notes)), low, high, alignment_used, n_part_targets, rules)


# -- planning ----------------------------------------------------------------------------
@dataclass(frozen=True)
class WorstRatioObjective:
    """The TUR plan's criterion when tolerances differ: the worst ratio of
    uncertainty to tolerance half-width over the characteristics."""

    half_widths_m: np.ndarray

    def __call__(self, score) -> float:
        return float(np.max(score.u_m / self.half_widths_m)) if np.all(np.isfinite(score.u_m)) else float("inf")


DEFAULT_DE = dict(maxiter=60, popsize=15, tol=1e-8, workers=1)


def describe_layout(cell: QifCell, stations: Sequence[Sequence], scorer: PartFrameScorer, model: CellRiskModel,
                    objective_fns: Optional[Dict[str, object]] = None) -> dict:
    """What a layout of the cell's track stations delivers, as `plan_qif_cell` reports it.

    `stations` are `(s_m, height_m, segment)` tuples (levelled stations on the
    track, facing the centroid of the scene's targets: `planning.cell.layout_poses`).
    Returns the stations (cell and file coordinates), each characteristic's
    uncertainty (um), test uncertainty ratio (tolerance half-width / 2u) and risk,
    the worst ratio, total risk (expected wrong decisions, from `model`'s risk
    curves), shot count, the time estimates of `measurement_time_minutes` and the
    value of each of `objective_fns` (name -> callable on the `LayoutScore`).
    The layout must be feasible for `scorer`: an infeasible one has no
    uncertainties to report (NaN everywhere) and is the caller's to filter out.
    """
    scenario, track = cell.scenario, cell.track
    half = np.array([p.tolerance_half_width_m for p in cell.planned])
    cell_to_part = np.linalg.inv(cell.part_to_cell)
    poses = layout_poses(stations, scenario, track)
    score = scorer.score(poses)
    per_risk = model.per_characteristic(score.u_m)
    tur = half / (2.0 * score.u_m)
    return {
        "stations": [{"s_m": float(s[0]), "height_m": float(s[1]), "segment": s[2],
                      "position_cell_m": p.position_m.tolist(),
                      "position_file_m": _apply(cell_to_part, p.position_m[None, :])[0].tolist()}
                     for s, p in zip(stations, poses)],
        "u_um": {p.name: float(u * 1e6) for p, u in zip(cell.planned, score.u_m)},
        "tur": {p.name: float(t) for p, t in zip(cell.planned, tur)},
        "worst_tur": float(np.min(tur)),
        "risk": {p.name: float(r) for p, r in zip(cell.planned, per_risk)},
        "total_risk": float(per_risk.sum()),
        "n_shots": score.n_shots,
        "time_min": measurement_time_minutes(len(stations), score.n_shots),
        "objective_values": {name: float(fn(score)) for name, fn in (objective_fns or {}).items()},
    }


def plan_qif_cell(cell: QifCell, station_counts: Sequence[int] = (2, 3, 4), objectives: Sequence[str] = ("A1", "risk"),
                  rule: str = "simple", seed: int = 0, de: Optional[dict] = None) -> dict:
    """Search the best station layout on the cell's track for each station
    count and each objective, and describe every layout found the same way.

    Objectives: "A0" (summed covariance trace: what the field minimises), "A1"
    (summed uncertainty along the tolerances), "TUR" (worst uncertainty-to-
    tolerance ratio) and "risk" (expected wrong decisions under `rule`: see
    `QifCell.risk_model` for the two rules and what the slower one costs). The
    optimiser is the one every other track search uses (differential evolution
    over arc length and height per station), with the same settings and seed
    for every objective. A layout found for one objective is also scored under
    the others, and each objective keeps the best layout seen at that count:
    the search finds a good plan, not provably the best.

    Returns a JSON-serialisable dict: the cell's `summary()`, the risk with no
    measurement at all, and per count and objective the stations (in cell and
    in file coordinates), each characteristic's uncertainty, test uncertainty
    ratio (tolerance half-width / 2u) and risk, the shot count and the time
    estimate of `planning.cell.measurement_time_minutes`.
    """
    settings = dict(DEFAULT_DE if de is None else de)
    track = cell.track
    half = np.array([p.tolerance_half_width_m for p in cell.planned])
    model = cell.risk_model(rule)
    available = {"A0": TraceObjective(), "A1": WeightedUncertaintyObjective(np.ones(len(half))),
                 "TUR": WorstRatioObjective(half), "risk": RiskObjective(model), "max_u": MaxUncertaintyObjective()}
    unknown = [o for o in objectives if o not in available]
    if unknown:
        raise ValueError(f"unknown objectives {unknown}; choose from {sorted(available)}")
    scorer = cell.scorer()
    out = {"cell": cell.summary(), "rule": rule, "de": {**settings, "seed": seed}, "objectives": list(objectives),
           "risk_if_nothing_is_measured": model.unmeasured_total(), "plans": {}}
    objective_fns = {name: available[name] for name in objectives}

    def describe(stations) -> dict:
        return describe_layout(cell, stations, scorer, model, objective_fns)

    for k in station_counts:
        found = {}
        for name in objectives:
            result = search_cell_layout(scorer, available[name], track, int(k), cell.rules.station_height_bounds_m,
                                        seed=seed, **settings)
            if result.value is not None:
                found[name] = [list(s) for s in result.stations]
        described = {name: describe(stations) for name, stations in found.items()}
        for name in objectives:
            if not described:
                out["plans"][f"{name}|{k}"] = {"feasible": False}
                continue
            best = min(described, key=lambda other: described[other]["objective_values"][name])
            out["plans"][f"{name}|{k}"] = {"feasible": True, "found_by": best, **described[best]}
    return out
