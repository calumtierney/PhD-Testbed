"""The scene a plan is reviewed against: the part, the cell round it, the targets, the
characteristics and what is known about the processes that made them.

Two sources, one wrapper (`ReviewScene`):

  * a QIF document (`from_qif_cell`, built by `planning.qif_cell.build_qif_cell`: what is
    planned, what is skipped and why, every assumption are in the cell's notes and are
    carried through);
  * the built-in synthetic factory cell (`from_factory_cell`: the long part on its track
    with a reference network, `planning.cell.factory_cell_scenario`). Its process
    statistics are placeholders.

**What is assumed about the cell round a QIF part.** The station track and the reference
network of nests are laid out *by rule* round the part (`planning.qif_cell.CellRules`),
and the part's placement in the cell is assumed unless one is supplied. A plan reviewed
against such a scene is therefore scored against an assumed reference network, not the
real cell's. `placement_assumed` and the notes say so, and the review prints it.

**Target names.** A plan may list the targets a station measures (`interchange.plan_file`).
The names this module recognises: a characteristic's name, the name of the feature it
refers to, `nest_<k>` for the k-th reference nest and `part_target_<i>` for a part target
that is neither. How a measurement package names its targets is not known here
(TODO(source): GKN's target naming); a name that is not recognised, or that two targets
share, is an error the review reports, never a guess.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from geometry.mesh import TriangleMesh, load_stl_bytes
from geometry.station_track import StationTrack
from interchange.qif import read_qif
from planning.cell import (
    CellRiskModel, factory_cell_scenario, part_datum_indices, process_knowledge_for,
)
from planning.experiment import long_part_track
from planning.objectives import PlanningScenario
from planning.qif_cell import QifCell, build_qif_cell
from review.provenance import digest
from risk.decision import ProcessKnowledge


class SceneError(ValueError):
    """A scene problem the review reports (an unknown or ambiguous target name)."""


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (n, 3) points."""
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class ReviewScene:
    """A scene, as the review sees it. Build with `from_qif_cell` or `from_factory_cell`."""

    label: str
    kind: str                                   # "qif" | "factory_cell"
    scenario: PlanningScenario                  # the tracker in it is replaced by the review's settings
    track: StationTrack
    datum_indices: Tuple[int, ...]
    characteristic_names: Tuple[str, ...]
    characteristic_kinds: Tuple[str, ...]       # QIF characteristic type, or the tolerance class for the built-in cell
    clusters: Tuple[str, ...]                   # per characteristic
    cluster_summaries: Tuple[Dict, ...]
    target_names: Tuple[str, ...]               # one per target (some may be None: unnamed)
    part_to_cell: Optional[np.ndarray]          # None: no file frame
    placement_assumed: bool
    notes: Tuple[str, ...]
    qif_cell: Optional[QifCell] = None
    n_part_targets: int = 0
    feature_names: Tuple[Tuple[int, str], ...] = ()   # (target index, QIF feature name): a second name a plan may use
    station_height_bounds_m: Tuple[float, float] = (0.8, 3.0)   # what the algorithm search may use; plans are not limited by it

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_qif_cell(cls, cell: QifCell, label: str) -> "ReviewScene":
        names = [None] * len(cell.scenario.target_points_m)
        for p in cell.planned:
            names[p.target_index] = p.name
        feature_names = {p.target_index: p.feature_name for p in cell.planned}
        for i in range(len(names)):
            if i >= cell.n_part_targets:
                names[i] = f"nest_{i - cell.n_part_targets}"
            elif names[i] is None:
                names[i] = f"part_target_{i}"
        summaries = tuple({"name": c.name, "members": list(c.members), "mean_um": c.mean_m * 1e6, "std_um": c.std_m * 1e6,
                           "n_builds": c.n_builds, "placeholder": c.n_builds is None, "source": c.source} for c in cell.clusters)
        return cls(
            label=label, kind="qif", scenario=cell.scenario, track=cell.track, datum_indices=tuple(cell.datum_indices),
            characteristic_names=tuple(p.name for p in cell.planned), characteristic_kinds=tuple(p.kind for p in cell.planned),
            clusters=tuple(p.cluster for p in cell.planned), cluster_summaries=summaries, target_names=tuple(names),
            part_to_cell=np.array(cell.part_to_cell), placement_assumed=any(n.startswith("placement assumed") for n in cell.notes),
            notes=tuple(cell.notes) + tuple(f"skipped: {n}: {r}" for n, r in cell.skipped), qif_cell=cell,
            n_part_targets=cell.n_part_targets, feature_names=tuple(sorted(feature_names.items())),
            station_height_bounds_m=tuple(cell.rules.station_height_bounds_m),
        )

    @classmethod
    def from_qif_bytes(cls, qif: bytes, label: str, stl: Optional[bytes] = None, stl_metres_per_unit: Optional[float] = None,
                       scope_id: Optional[str] = None, up_axis: str = "z", alignment: str = "auto", placement: Optional[np.ndarray] = None
                       ) -> "ReviewScene":
        """A scene from a QIF document held in memory (an upload) and, optionally, the part's STL (in the file's coordinates, scaled
        by `stl_metres_per_unit`, default the QIF file's own unit) to block sight lines. Nothing is written to disk.
        `placement` (4x4, file to cell) is *assumed* by `planning.qif_cell.default_placement` when not given, and the scene says so.
        Raises what `read_qif` and `build_qif_cell` raise (e.g. a document with nothing the chain can plan)."""
        doc = read_qif(qif)
        obstacle = None
        if stl is not None:
            mesh = load_stl_bytes(stl)
            scale = stl_metres_per_unit if stl_metres_per_unit is not None else doc.metres_per_unit
            obstacle = TriangleMesh(mesh.vertices_m * scale, mesh.faces)
        cell = build_qif_cell(doc, scope_id=scope_id, up_axis=up_axis, obstacle_part_frame=obstacle, alignment=alignment, placement=placement)
        return cls.from_qif_cell(cell, label)

    @classmethod
    def from_factory_cell(cls, track: Optional[StationTrack] = None, label: str = "built-in factory cell (synthetic)") -> "ReviewScene":
        track = long_part_track() if track is None else track
        scenario = factory_cell_scenario(track=track)
        n_part = 26  # planning.cell.DatumCellScorer's convention for this scene
        names = [None] * len(scenario.target_points_m)
        for c in scenario.characteristics:
            names[c.target_indices[0]] = c.name
        for i in range(len(names)):
            if i >= n_part:
                names[i] = f"nest_{i - n_part}"
            elif names[i] is None:
                names[i] = f"part_target_{i}"
        clusters, summaries = [], {}
        for c in scenario.characteristics:
            cluster = scenario.assignments[c.name].cluster
            clusters.append(cluster.name)
            summaries.setdefault(cluster.name, {
                "name": cluster.name, "members": [], "mean_um": cluster.prior.mean_m * 1e6, "std_um": cluster.prior.std_m * 1e6,
                "n_builds": None, "placeholder": True,
                "source": "PLACEHOLDER: the built-in factory cell's synthetic process statistics (TODO(source): real data)"})
            summaries[cluster.name]["members"].append(c.name)
        return cls(
            label=label, kind="factory_cell", scenario=scenario, track=track, datum_indices=tuple(part_datum_indices(scenario, n_part)),
            characteristic_names=tuple(c.name for c in scenario.characteristics),
            characteristic_kinds=tuple(type(c.tolerance).__name__ for c in scenario.characteristics),
            clusters=tuple(clusters), cluster_summaries=tuple(summaries.values()), target_names=tuple(names), part_to_cell=None,
            placement_assumed=False,
            notes=("synthetic scene: every dimension, tolerance, process statistic and the reference network are placeholders "
                   "(docs/cell_experiment.md, 'Placeholders needing real data')",),
            n_part_targets=n_part,
        )

    # -- what the scene says --------------------------------------------------------------
    @property
    def aim_point_m(self) -> np.ndarray:
        """Where every station faces: the centroid of the scene's targets (`planning.cell.layout_poses`'s convention)."""
        return self.scenario.target_points_m.mean(axis=0)

    @property
    def n_characteristics(self) -> int:
        return len(self.characteristic_names)

    def part_extent_xy_m(self) -> Tuple[np.ndarray, np.ndarray]:
        """Plan-view extent of the part, `(low, high)` as (x, y): the located features (and the obstacle mesh, if one was given)
        of a QIF scene, the footprint of the built-in cell's part."""
        if self.qif_cell is not None:
            return np.array(self.qif_cell.envelope_min_m[:2]), np.array(self.qif_cell.envelope_max_m[:2])
        t = self.track
        half = t.part_width_m / 2.0
        return np.array([t.part_x_min_m, t.part_centre_y_m - half]), np.array([t.part_x_max_m, t.part_centre_y_m + half])

    def tolerance_limits_m(self) -> np.ndarray:
        """(n_characteristics, 2): lower and upper tolerance limit of each, from its decision rule."""
        rules = [self.scenario.decision_rules[n] for n in self.characteristic_names]
        return np.array([[r.tolerance_lower_m, r.tolerance_upper_m] for r in rules])

    def process_knowledge(self, n_builds: Optional[int]) -> Tuple[ProcessKnowledge, ...]:
        """Process knowledge of each characteristic's cluster. A QIF scene's carries its own evidence (earlier parts
        in the file, or a placeholder); the built-in cell's statistics are read as coming from `n_builds` parts."""
        if self.kind == "qif":
            return tuple(self.qif_cell.knowledge[c] for c in self.clusters)
        return tuple(process_knowledge_for(self.scenario, n, n_builds) for n in self.characteristic_names)

    def risk_model(self, rule: str, n_builds: Optional[int]) -> CellRiskModel:
        """The one risk model every plan in a comparison is scored with (`CellRiskModel`: total decision risk against
        a characteristic's uncertainty, tabulated once)."""
        if self.kind == "qif":
            return self.qif_cell.risk_model(rule)
        return CellRiskModel.build(self.scenario, n_builds, rule)

    def target_index(self, name: str) -> int:
        """The index of the target called `name`. Raises `SceneError` for an unknown or ambiguous name."""
        matches = {i for i, n in enumerate(self.target_names) if n == name}
        matches |= {i for i, n in self.feature_names if n == name}
        if not matches:
            raise SceneError(f"no target is called {name!r}")
        if len(matches) > 1:
            raise SceneError(f"{name!r} names {len(matches)} different targets; it cannot be resolved")
        return matches.pop()

    def to_cell(self, positions_file_m: np.ndarray) -> np.ndarray:
        if self.part_to_cell is None:
            raise SceneError("this scene has no file frame (it is not built from a file): give the plan in 'cell' coordinates")
        return _apply(self.part_to_cell, np.atleast_2d(positions_file_m))

    def to_file(self, positions_cell_m: np.ndarray) -> Optional[np.ndarray]:
        if self.part_to_cell is None:
            return None
        return _apply(np.linalg.inv(self.part_to_cell), np.atleast_2d(positions_cell_m))

    # -- identity -------------------------------------------------------------------------
    def scene_hash(self) -> str:
        """Hash of what defines the scene -- geometry, characteristics and their tolerances, process statistics,
        datums, track, visibility limits, obstacle -- and not of the instrument model (a setting)."""
        s = self.scenario
        obstacle = None if s.obstacles is None else [np.asarray(s.obstacles.vertices_m), np.asarray(s.obstacles.faces, dtype=float)]
        chars = [[c.name, [int(i) for i in c.target_indices], type(c.tolerance).__name__,
                  np.asarray(c.tolerance.constraint_directions(), dtype=float)] for c in s.characteristics]
        rules = [[s.decision_rules[n].tolerance_lower_m, s.decision_rules[n].tolerance_upper_m] for n in self.characteristic_names]
        priors = [[c["name"], c["mean_um"], c["std_um"], c["n_builds"]] for c in self.cluster_summaries]
        track = self.track
        limits = s.visibility_limits
        return digest(
            self.kind, s.target_points_m, s.target_normals_m, [int(i) for i in self.datum_indices], chars, rules, list(self.clusters), priors,
            [track.part_x_min_m, track.part_x_max_m, track.part_width_m, track.standoff_m, track.part_centre_y_m],
            [limits.max_range_m, limits.max_incidence_angle_rad], obstacle,
            None if self.part_to_cell is None else self.part_to_cell,
        )

    def summary(self) -> Dict:
        return {
            "label": self.label, "kind": self.kind, "scene_hash": self.scene_hash(),
            "n_targets": int(len(self.scenario.target_points_m)), "n_characteristics": self.n_characteristics,
            "n_datum_targets": len(self.datum_indices), "placement_assumed": self.placement_assumed,
            "has_file_frame": self.part_to_cell is not None, "notes": list(self.notes),
            "clusters": [dict(c) for c in self.cluster_summaries],
        }


def qif_scopes(qif: bytes) -> Dict[str, str]:
    """The parts and assemblies a QIF document offers as scopes (`id -> label`), for choosing which to review against."""
    return dict(read_qif(qif).scope_names())
