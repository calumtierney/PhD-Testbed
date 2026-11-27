"""From a QIF document to a planned cell (`planning.qif_cell`).

The strongest check is the round trip: the long-part factory cell, written out
as a QIF document and read back, must give the same scene and the same
uncertainties as the hand-built one. The rest checks the decisions the module
makes on a real file and on small documents written for the purpose.
"""
from pathlib import Path

import numpy as np
import pytest

from interchange.qif import read_qif
from interchange.qif_write import write_point_inspection_qif
from planning.cell import DatumCellScorer, factory_cell_scenario, layout_poses
from planning.experiment import long_part_track
from planning.qif_cell import CellRules, build_qif_cell, default_placement, plan_qif_cell

SAMPLE = Path(__file__).resolve().parent.parent / "scenes" / "qif" / "SheetMetal_QIF_Results_6_samples.QIF"


@pytest.fixture(scope="module")
def sheet_metal_cell():
    return build_qif_cell(read_qif(SAMPLE))


def _write_long_part(path, measured=None):
    scenario = factory_cell_scenario(track=long_part_track())
    points, normals = scenario.target_points_m[:26], scenario.target_normals_m[:26]
    names = [c.name for c in scenario.characteristics] + [f"tie_{i}" for i in range(20)]
    specs = [{"name": c.name, "feature_index": c.target_indices[0], "zone_width_m": c.tolerance.zone_width_m}
             for c in scenario.characteristics]
    write_point_inspection_qif(path, "long part", points, normals, names, specs, datum_feature_indices=range(6, 26),
                               measured_points_m=measured)
    return scenario


# -- the round trip -------------------------------------------------------------------------
def test_long_part_survives_a_round_trip_through_qif(tmp_path):
    scenario = _write_long_part(tmp_path / "long_part.qif")
    doc = read_qif(tmp_path / "long_part.qif")
    cell = build_qif_cell(doc, placement=np.eye(4), obstacle_part_frame=scenario.obstacles)
    np.testing.assert_allclose(cell.scenario.target_points_m, scenario.target_points_m, atol=1e-9)
    np.testing.assert_allclose(cell.scenario.target_normals_m, scenario.target_normals_m, atol=1e-12)
    assert cell.datum_indices == tuple(range(6, 26)) and cell.n_part_targets == 26
    assert not cell.skipped and len(cell.planned) == 6
    track = long_part_track()
    assert cell.track.perimeter_m == pytest.approx(track.perimeter_m)
    assert (cell.track.y_right_row_m, cell.track.y_left_row_m) == pytest.approx((track.y_right_row_m, track.y_left_row_m))

    old, new = DatumCellScorer(scenario), cell.scorer()
    rng = np.random.default_rng(11)
    n_feasible = 0
    for _ in range(10):
        stations = [(rng.uniform(0, track.perimeter_m), rng.uniform(0.8, 3.0)) for _ in range(4)]
        a = old.score(layout_poses(stations, scenario, track))
        b = new.score(layout_poses(stations, cell.scenario, cell.track))
        assert a.feasible == b.feasible
        if a.feasible:
            n_feasible += 1
            np.testing.assert_allclose(b.u_m, a.u_m, rtol=1e-6)
    assert n_feasible >= 2


def test_process_statistics_come_from_the_measured_parts_in_the_file(tmp_path):
    scenario = factory_cell_scenario(track=long_part_track())
    points, normals = scenario.target_points_m[:26], scenario.target_normals_m[:26]
    z = np.array([-1.2, -0.4, 0.1, 0.5, 1.0])                      # five builds
    deviations = 20e-6 + 5e-6 * z                                  # the same on all six characteristics
    measured = np.repeat(points[None, :, :], len(z), axis=0)
    measured[:, :6, :] += deviations[:, None, None] * normals[None, :6, :]
    _write_long_part(tmp_path / "with_results.qif", measured)
    cell = build_qif_cell(read_qif(tmp_path / "with_results.qif"), placement=np.eye(4))
    assert len(cell.clusters) == 1                                  # one type, one tolerance, one datum frame
    cluster = cell.clusters[0]
    assert cluster.n_builds == 5 and len(cluster.members) == 6
    assert cluster.mean_m == pytest.approx(np.mean(np.tile(deviations, 6)), rel=1e-6)
    assert cluster.std_m == pytest.approx(np.std(np.tile(deviations, 6), ddof=1), rel=1e-6)
    component = cell.knowledge[cluster.name].components[0]
    assert component.dof == 4.0                                     # Student-t on n - 1: five parts, not thirty values


def test_without_earlier_inspections_the_prior_is_a_marked_placeholder(tmp_path):
    _write_long_part(tmp_path / "no_results.qif")
    cell = build_qif_cell(read_qif(tmp_path / "no_results.qif"), placement=np.eye(4), rules=CellRules(default_cpk=2.0))
    cluster = cell.clusters[0]
    assert cluster.n_builds is None and "PLACEHOLDER" in cluster.source
    assert cluster.std_m == pytest.approx(50e-6 / (3 * 2.0))
    assert any("placeholder" in n for n in cell.notes)


# -- a real file ----------------------------------------------------------------------------
def test_sheet_metal_sample_is_mapped_and_its_limits_are_stated(sheet_metal_cell):
    cell = sheet_metal_cell
    assert len(cell.planned) == 21 and not cell.skipped
    assert sum(p.approximate for p in cell.planned) == 4            # the four hole positions
    assert {p.kind for p in cell.planned if p.approximate} == {"Position"}
    by_name = {p.name: p for p in cell.planned}
    assert by_name["W1RFTMRA02V"].tolerance_half_width_m == pytest.approx(2e-3)
    assert by_name["W1RXXMRA19P"].tolerance_half_width_m == pytest.approx(0.625e-3)
    notes = " ".join(cell.notes)
    assert "large-volume" in notes and "no part geometry" in notes and "best fit through the planned points" in notes
    assert all(c.n_builds == 6 for c in cell.clusters)


def test_sheet_metal_cell_is_sized_round_the_part(sheet_metal_cell):
    cell = sheet_metal_cell
    extent = cell.envelope_max_m - cell.envelope_min_m
    assert extent[0] >= extent[1]                                   # laid with its longer side along the cell
    assert cell.envelope_min_m[0] == pytest.approx(0.0, abs=1e-9)
    assert cell.envelope_min_m[2] == pytest.approx(cell.rules.part_bottom_height_m)
    assert cell.track.row_length_m == pytest.approx(extent[0])
    assert cell.track.y_left_row_m == pytest.approx(extent[1] / 2 + cell.rules.standoff_m)
    part = cell.scenario.target_points_m[:cell.n_part_targets]
    nests = cell.scenario.target_points_m[cell.n_part_targets:]
    assert len(nests) == 20 and np.all(np.abs(nests[:, 1]).max() > np.abs(part[:, 1]).max() + cell.rules.standoff_m)
    # placement is rigid: distances between features are the file's
    doc = read_qif(SAMPLE)
    a, b = (doc.features[doc.characteristics[i].feature_ids[0]].location_m for i in ("15", "25"))
    assert np.linalg.norm(part[0] - part[1]) == pytest.approx(np.linalg.norm(a - b), rel=1e-12)


def test_hole_deviation_agrees_with_the_files_own_position_value():
    """QIF reports a position as the diameter of the zone the measured axis
    would need: twice its offset across the hole's axis. The offsets this
    module pools for the process statistics must reproduce the file's value."""
    doc = read_qif(SAMPLE)
    hole = doc.characteristics["173"]
    feature = doc.features[hole.feature_ids[0]]
    offset = feature.measured_locations_m[0] - feature.location_m
    across = offset - (offset @ feature.direction) * feature.direction
    assert 2.0 * np.linalg.norm(across) == pytest.approx(hole.measured_values_m[0], rel=1e-9)


def test_default_placement_is_a_rigid_motion_that_puts_the_part_on_the_cell():
    rng = np.random.default_rng(5)
    points = rng.uniform([2.0, -8.0, 3.0], [2.5, -1.0, 3.4], size=(30, 3))  # longer in y, far from the origin
    matrix = default_placement(points, CellRules(part_bottom_height_m=0.7))
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    placed = points @ rotation.T + matrix[:3, 3]
    low, high = placed.min(axis=0), placed.max(axis=0)
    assert low[0] == pytest.approx(0.0) and high[0] - low[0] > high[1] - low[1]
    assert low[1] == pytest.approx(-high[1]) and low[2] == pytest.approx(0.7)
    np.testing.assert_allclose(default_placement(points, up_axis="y")[:3, :3] @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-12)


# -- what is left out, and assemblies -------------------------------------------------------
ASSEMBLY = """<QIFDocument xmlns="http://qifstandards.org/xsd/qif3" versionQIF="3.0.0">
  <Transforms n="1"><Transform id="20"><Origin>2 0 0</Origin></Transform></Transforms>
  <Product>
    <PartSet n="2">
      <Part id="1" label="skin"><FeatureNominalIds n="2"><Id>30</Id><Id>31</Id></FeatureNominalIds>
        <CharacteristicNominalIds n="3"><Id>50</Id><Id>51</Id><Id>54</Id></CharacteristicNominalIds></Part>
      <Part id="2" label="rib"><FeatureNominalIds n="2"><Id>32</Id><Id>33</Id></FeatureNominalIds>
        <CharacteristicNominalIds n="2"><Id>52</Id><Id>53</Id></CharacteristicNominalIds></Part>
    </PartSet>
    <AssemblySet n="1"><Assembly id="3" label="box"><ComponentIds n="2"><Id>10</Id><Id>11</Id></ComponentIds></Assembly></AssemblySet>
    <ComponentSet n="2">
      <Component id="10"><Part><Id>1</Id></Part></Component>
      <Component id="11"><Transform><Id>20</Id></Transform><Part><Id>2</Id></Part></Component>
    </ComponentSet>
    <RootAssembly><Id>3</Id></RootAssembly>
  </Product>
  <Features><FeatureNominals n="4">
    <PointFeatureNominal id="30"><Location>0 0 0</Location><Normal>0 1 0</Normal></PointFeatureNominal>
    <PointFeatureNominal id="31"><Location>1 0 0.2</Location><Normal>0 1 0</Normal></PointFeatureNominal>
    <PointFeatureNominal id="32"><Location>0 0 0.1</Location><Normal>0 -1 0</Normal></PointFeatureNominal>
    <PointFeatureNominal id="33"><Location>1 0 0.3</Location><Normal>0 -1 0</Normal></PointFeatureNominal>
  </FeatureNominals></Features>
  <Characteristics>
    <CharacteristicDefinitions n="2">
      <PointProfileCharacteristicDefinition id="40"><ToleranceValue>0.0001</ToleranceValue></PointProfileCharacteristicDefinition>
      <FlatnessCharacteristicDefinition id="41"><ToleranceValue>0.0001</ToleranceValue></FlatnessCharacteristicDefinition>
    </CharacteristicDefinitions>
    <CharacteristicNominals n="5">
      <PointProfileCharacteristicNominal id="50"><CharacteristicDefinitionId>40</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>30</Id></FeatureNominalIds><Name>skin a</Name></PointProfileCharacteristicNominal>
      <PointProfileCharacteristicNominal id="51"><CharacteristicDefinitionId>40</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>31</Id></FeatureNominalIds><Name>skin b</Name></PointProfileCharacteristicNominal>
      <PointProfileCharacteristicNominal id="52"><CharacteristicDefinitionId>40</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>32</Id></FeatureNominalIds><Name>rib a</Name></PointProfileCharacteristicNominal>
      <PointProfileCharacteristicNominal id="53"><CharacteristicDefinitionId>40</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>33</Id></FeatureNominalIds><Name>rib b</Name></PointProfileCharacteristicNominal>
      <FlatnessCharacteristicNominal id="54"><CharacteristicDefinitionId>41</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>30</Id></FeatureNominalIds><Name>skin flatness</Name></FlatnessCharacteristicNominal>
    </CharacteristicNominals>
  </Characteristics>
</QIFDocument>"""


def test_a_part_and_its_assembly_give_different_cells():
    doc = read_qif(ASSEMBLY.encode())
    skin = build_qif_cell(doc, scope_id="1")
    assert [p.name for p in skin.planned] == ["skin a", "skin b"]
    assert skin.track.row_length_m == pytest.approx(1.0)
    box = build_qif_cell(doc, scope_id="3")
    assert [p.name for p in box.planned] == ["skin a", "skin b", "rib a", "rib b"]
    assert box.track.row_length_m == pytest.approx(3.0)             # the rib sits 2 m along the assembly
    assert len(box.clusters) == 2                                    # the same callout on two parts is two processes
    assert build_qif_cell(doc).track.row_length_m == pytest.approx(3.0)  # no scope: the root assembly


def test_what_cannot_be_planned_is_listed_with_its_reason_not_dropped():
    cell = build_qif_cell(read_qif(ASSEMBLY.encode()), scope_id="1")
    assert len(cell.skipped) == 1
    name, reason = cell.skipped[0]
    assert name == "skin flatness" and "several points" in reason
    with pytest.raises(ValueError, match="no characteristic the chain can plan"):
        build_qif_cell(read_qif(b'<QIFDocument versionQIF="3.0.0"/>'))


# -- planning -------------------------------------------------------------------------------
def test_plan_reports_every_layout_the_same_way(sheet_metal_cell):
    plan = plan_qif_cell(sheet_metal_cell, station_counts=(2,), objectives=("A1", "risk"), rule="simple",
                         de=dict(maxiter=6, popsize=6, tol=1e-8, workers=1))
    assert plan["risk_if_nothing_is_measured"] > 0.0
    a1, risk = plan["plans"]["A1|2"], plan["plans"]["risk|2"]
    assert a1["feasible"] and risk["feasible"]
    assert len(a1["stations"]) == 2 and set(a1["u_um"]) == {p.name for p in sheet_metal_cell.planned}
    # each objective keeps the best layout seen under its own measure
    assert risk["objective_values"]["risk"] <= a1["objective_values"]["risk"] * (1 + 1e-12)
    assert a1["objective_values"]["A1"] <= risk["objective_values"]["A1"] * (1 + 1e-12)
    assert risk["total_risk"] < plan["risk_if_nothing_is_measured"]
    # a station's file coordinates map back onto its cell coordinates
    station = a1["stations"][0]
    back = sheet_metal_cell.part_to_cell[:3, :3] @ station["position_file_m"] + sheet_metal_cell.part_to_cell[:3, 3]
    np.testing.assert_allclose(back, station["position_cell_m"], atol=1e-9)
    worst = min(a1["tur"].values())
    assert a1["worst_tur"] == pytest.approx(worst)


def test_describe_layout_is_what_plan_qif_cell_reports_for_the_same_stations(sheet_metal_cell):
    """`describe_layout` was the `describe()` closure inside `plan_qif_cell`; lifted to module level it must give the
    same record for the same stations, so anything built on it (the plan review) agrees with the planner."""
    from planning.qif_cell import describe_layout

    plan = plan_qif_cell(sheet_metal_cell, station_counts=(2,), objectives=("A1",), rule="simple",
                         de=dict(maxiter=4, popsize=5, tol=1e-8, workers=1))
    entry = plan["plans"]["A1|2"]
    stations = [(s["s_m"], s["height_m"], s["segment"]) for s in entry["stations"]]
    again = describe_layout(sheet_metal_cell, stations, sheet_metal_cell.scorer(), sheet_metal_cell.risk_model("simple"))
    for key in ("stations", "u_um", "tur", "worst_tur", "risk", "total_risk", "n_shots", "time_min"):
        assert again[key] == entry[key], key
    assert again["objective_values"] == {}  # none asked for
