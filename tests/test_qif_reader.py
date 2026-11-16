"""The QIF reader (`interchange.qif`): against a real sample file written by
other software, and against small hand-written documents for the cases no
public sample covers (assemblies with transforms, per-value units, hostile XML).
"""
from pathlib import Path

import numpy as np
import pytest

from interchange.qif import QifError, read_qif

SAMPLE = Path(__file__).resolve().parent.parent / "scenes" / "qif" / "SheetMetal_QIF_Results_6_samples.QIF"


def test_sample_units_counts_and_structure():
    doc = read_qif(SAMPLE)
    assert doc.version == "3.0.0"
    assert doc.linear_unit_name == "mm" and doc.metres_per_unit == pytest.approx(1e-3)
    kinds = [f.kind for f in doc.features.values()]
    assert (kinds.count("Point"), kinds.count("EdgePoint"), kinds.count("Circle")) == (13, 4, 4)
    char_kinds = [c.kind for c in doc.characteristics.values()]
    assert (char_kinds.count("PointProfile"), char_kinds.count("Position")) == (17, 4)
    assert doc.n_inspections == 6
    assert [p.name for p in doc.parts.values()] == ["Wing mirror reinforcement"]
    assert sorted(d.label for d in doc.datums.values()) == ["A", "B", "C"]


def test_sample_feature_is_in_metres_with_unit_normal():
    doc = read_qif(SAMPLE)
    feature = doc.features["20"]  # <Location>2466.729... 774.269... 945.002...</Location> in mm
    assert feature.name == "W1RFSMRA05"
    np.testing.assert_allclose(feature.location_m, [2.46672924804688, 0.774269897460938, 0.945002746582031], rtol=1e-12)
    assert np.linalg.norm(feature.direction) == pytest.approx(1.0)
    assert feature.direction_role == "normal"
    assert len(feature.measured_locations_m) == 6


def test_sample_characteristic_links_tolerance_and_datums():
    doc = read_qif(SAMPLE)
    profile = doc.characteristics["15"]
    assert profile.kind == "PointProfile" and profile.name == "W1RFTMRA02V"
    assert profile.tolerance_m == pytest.approx(4e-3)  # ToleranceValue 4 (mm): the zone's full width
    assert profile.feature_ids == ("9",)                # item 15 -> feature item 10 -> feature nominal 9
    position = doc.characteristics["173"]
    assert position.kind == "Position" and position.zone_shape == "Diametrical"
    assert position.tolerance_m == pytest.approx(1.25e-3)
    assert position.datum_labels == ("A", "B", "C")
    assert doc.features[position.feature_ids[0]].kind == "Circle"


def test_sample_measured_deviation_matches_the_files_own_value():
    """The file reports -0.014288 mm for characteristic 15 on the first part;
    the same number must come out of (measured - nominal) . normal, which is how
    the planner derives process statistics."""
    doc = read_qif(SAMPLE)
    feature = doc.features[doc.characteristics["15"].feature_ids[0]]
    deviation = (feature.measured_locations_m[0] - feature.location_m) @ feature.direction
    assert deviation == pytest.approx(-0.014288276431175e-3, rel=1e-6)
    assert doc.characteristics["15"].measured_values_m[0] == pytest.approx(-0.014288276431175e-3)


MBD = """<?xml version="1.0"?>
<QIFDocument xmlns="http://qifstandards.org/xsd/qif3" versionQIF="3.0.0">
  <FileUnits><PrimaryUnits>
    <LinearUnit><SIUnitName>meter</SIUnitName><UnitName>mm</UnitName><UnitConversion><Factor>0.001</Factor></UnitConversion></LinearUnit>
  </PrimaryUnits><OtherUnits n="1">
    <PMILinearUnit><SIUnitName>meter</SIUnitName><UnitName>inch</UnitName><UnitConversion><Factor>0.0254</Factor></UnitConversion></PMILinearUnit>
  </OtherUnits></FileUnits>
  <DatumDefinitions n="1"><DatumDefinition id="5"><DatumLabel>A</DatumLabel><FeatureNominalIds n="1"><Id>30</Id></FeatureNominalIds></DatumDefinition></DatumDefinitions>
  <DatumReferenceFrames n="1"><DatumReferenceFrame id="6"><Datums n="1"><Datum>
    <SimpleDatum><DatumDefinitionId>5</DatumDefinitionId></SimpleDatum><Precedence><PrecedenceEnum>PRIMARY</PrecedenceEnum></Precedence>
  </Datum></Datums></DatumReferenceFrame></DatumReferenceFrames>
  <Transforms n="1"><Transform id="20">
      <Rotation><XDirection>0 1 0</XDirection><YDirection>-1 0 0</YDirection><ZDirection>0 0 1</ZDirection></Rotation>
      <Origin>1000 0 0</Origin>
    </Transform></Transforms>
  <Product>
    <PartSet n="2">
      <Part id="1" label="skin"><FeatureNominalIds n="1"><Id>31</Id></FeatureNominalIds><CharacteristicNominalIds n="1"><Id>41</Id></CharacteristicNominalIds></Part>
      <Part id="2" label="rib"><FeatureNominalIds n="1"><Id>30</Id></FeatureNominalIds><CharacteristicNominalIds n="1"><Id>42</Id></CharacteristicNominalIds></Part>
    </PartSet>
    <AssemblySet n="1"><Assembly id="3" label="box"><ComponentIds n="2"><Id>10</Id><Id>11</Id></ComponentIds></Assembly></AssemblySet>
    <ComponentSet n="2">
      <Component id="10"><Part><Id>1</Id></Part></Component>
      <Component id="11"><Transform><Id>20</Id></Transform><Part><Id>2</Id></Part></Component>
    </ComponentSet>
    <RootAssembly><Id>3</Id></RootAssembly>
  </Product>
  <Features><FeatureNominals n="2">
    <PlaneFeatureNominal id="30"><Name>rib face</Name><Location>0 0 0</Location><Normal>0 0 2</Normal></PlaneFeatureNominal>
    <CylinderFeatureNominal id="31"><Axis><AxisPoint>100 200 0</AxisPoint><Direction>0 0 -1</Direction></Axis></CylinderFeatureNominal>
  </FeatureNominals></Features>
  <Characteristics>
    <CharacteristicDefinitions n="2">
      <PositionCharacteristicDefinition id="40"><ToleranceValue linearUnit="inch">0.01</ToleranceValue><DatumReferenceFrameId>6</DatumReferenceFrameId>
        <ZoneShape><DiametricalZone/></ZoneShape></PositionCharacteristicDefinition>
      <SurfaceProfileCharacteristicDefinition id="43"><ToleranceValue linearUnit="furlong">1</ToleranceValue></SurfaceProfileCharacteristicDefinition>
    </CharacteristicDefinitions>
    <CharacteristicNominals n="2">
      <PositionCharacteristicNominal id="41"><CharacteristicDefinitionId>40</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>31</Id></FeatureNominalIds><Name>hole 1</Name></PositionCharacteristicNominal>
      <SurfaceProfileCharacteristicNominal id="42"><CharacteristicDefinitionId>43</CharacteristicDefinitionId><FeatureNominalIds n="1"><Id>30</Id></FeatureNominalIds><Name>rib profile</Name></SurfaceProfileCharacteristicNominal>
    </CharacteristicNominals>
  </Characteristics>
</QIFDocument>"""


def test_model_based_definition_links_nominals_directly():
    doc = read_qif(MBD.encode())
    hole = doc.characteristics["41"]
    assert hole.name == "hole 1" and hole.feature_ids == ("31",) and hole.datum_labels == ("A",)
    cylinder = doc.features["31"]
    np.testing.assert_allclose(cylinder.location_m, [0.1, 0.2, 0.0])
    np.testing.assert_allclose(cylinder.direction, [0.0, 0.0, -1.0])
    assert cylinder.direction_role == "axis"
    np.testing.assert_allclose(doc.features["30"].direction, [0.0, 0.0, 1.0])  # normalised


def test_a_value_carries_its_own_unit_and_an_unknown_unit_is_never_guessed():
    doc = read_qif(MBD.encode())
    assert doc.characteristics["41"].tolerance_m == pytest.approx(0.01 * 0.0254)
    assert doc.characteristics["42"].tolerance_m is None
    assert any("furlong" in w for w in doc.warnings)


def test_assembly_places_each_part_and_scopes_its_characteristics():
    doc = read_qif(MBD.encode())
    assert doc.scope_names() == {"3": "assembly: box", "1": "part: skin", "2": "part: rib"}
    placements = doc.part_placements()  # the root assembly
    np.testing.assert_allclose(placements["1"][0], np.eye(4))
    # the rib's own x axis lies along the assembly's y, and its origin is 1 m along the assembly's x
    np.testing.assert_allclose(placements["2"][0] @ [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], atol=1e-12)
    assert [c.name for c in doc.characteristics_in("1")] == ["hole 1"]
    assert [c.name for c in doc.characteristics_in("2")] == ["rib profile"]
    assert len(doc.characteristics_in("3")) == 2
    np.testing.assert_allclose(doc.part_placements("2")["2"][0], np.eye(4))  # a part on its own is in place
    assert doc.part_of_feature("30") == "2"


def test_hostile_or_wrong_documents_are_refused():
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><QIFDocument>&a;</QIFDocument>'
    with pytest.raises(QifError, match="DOCTYPE"):
        read_qif(bomb)
    with pytest.raises(QifError, match="QIFDocument"):
        read_qif(b"<NotQif/>")
    with pytest.raises(QifError, match="well-formed"):
        read_qif(b"<QIFDocument>")
    with pytest.raises(QifError, match="limit"):
        read_qif(b"<QIFDocument/>", max_bytes=5)


def test_a_document_without_units_is_si_and_says_so():
    doc = read_qif(b'<QIFDocument versionQIF="2.1.0" xmlns="http://qifstandards.org/xsd/qif2"/>')
    assert doc.metres_per_unit == 1.0 and doc.version == "2.1.0"
    assert any("FileUnits" in w for w in doc.warnings)
