"""Writing a *minimal* QIF results document for single-point inspections.

This exists so that a scene the testbed already understands can be sent through
the same door as a file from someone else's software (`interchange.qif.read_qif`
-> `planning.qif_cell`), and so tests have a large-volume part to read: no
public QIF file of an aerospace-scale structure was found.

It writes only what the reader reads: units, one part, point features (location
and surface normal), point-profile characteristics on some of them, one datum
whose features are the part's reference points, and optionally the measured
locations from earlier builds. It uses QIF 3.0's element names and nesting as
found in the community's sample files. **It has not been validated against the
QIF schema** and leaves out elements a conforming writer must include; do not
treat its output as a QIF file for another system.
"""
from pathlib import Path
from typing import Optional, Sequence, Union
from xml.sax.saxutils import escape

import numpy as np


def _triple(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def write_point_inspection_qif(
    path: Union[str, Path],
    part_name: str,
    points_m: np.ndarray,
    normals: np.ndarray,
    feature_names: Sequence[str],
    profile_characteristics: Sequence[dict],
    datum_feature_indices: Sequence[int] = (),
    measured_points_m: Optional[np.ndarray] = None,
    unit_name: str = "mm",
    metres_per_unit: float = 1e-3,
) -> Path:
    """Write one part's point features and point-profile characteristics.

    `profile_characteristics`: one dict per characteristic with keys `name`,
    `feature_index` (into `points_m`) and `zone_width_m` (the profile zone's full
    width). `datum_feature_indices`: the features that realise datum A.
    `measured_points_m`: (n_builds, n_points, 3) measured locations from earlier
    builds, or None. Coordinates are written in `unit_name`."""
    points = np.asarray(points_m, dtype=float) / metres_per_unit
    normals = np.asarray(normals, dtype=float)
    n = len(points)
    next_id = [0]

    def new_id() -> int:
        next_id[0] += 1
        return next_id[0]

    part_id, component_id = new_id(), new_id()
    definition_id = new_id()
    nominal_ids = [new_id() for _ in range(n)]
    item_ids = [new_id() for _ in range(n)]
    datum_id, frame_id = new_id(), new_id()
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<QIFDocument xmlns="http://qifstandards.org/xsd/qif3" versionQIF="3.0.0">',
           "  <FileUnits><PrimaryUnits><LinearUnit>",
           f"    <SIUnitName>meter</SIUnitName><UnitName>{escape(unit_name)}</UnitName>",
           f"    <UnitConversion><Factor>{metres_per_unit!r}</Factor></UnitConversion>",
           "  </LinearUnit></PrimaryUnits></FileUnits>"]
    if len(datum_feature_indices):
        ids = "".join(f"<Id>{nominal_ids[i]}</Id>" for i in datum_feature_indices)
        out += [f'  <DatumDefinitions n="1"><DatumDefinition id="{datum_id}"><DatumLabel>A</DatumLabel>',
                f'    <FeatureNominalIds n="{len(datum_feature_indices)}">{ids}</FeatureNominalIds>',
                "  </DatumDefinition></DatumDefinitions>",
                f'  <DatumReferenceFrames n="1"><DatumReferenceFrame id="{frame_id}"><Datums n="1"><Datum>',
                f"    <SimpleDatum><DatumDefinitionId>{datum_id}</DatumDefinitionId></SimpleDatum>",
                "    <Precedence><PrecedenceEnum>PRIMARY</PrecedenceEnum></Precedence>",
                "  </Datum></Datums></DatumReferenceFrame></DatumReferenceFrames>"]
    out += [f'  <Product><PartSet n="1"><Part id="{part_id}"><Name>{escape(part_name)}</Name></Part></PartSet>',
            f'    <ComponentSet n="1"><Component id="{component_id}"><Part><Id>{part_id}</Id></Part></Component></ComponentSet>',
            "  </Product>",
            f'  <Features><FeatureDefinitions n="1"><PointFeatureDefinition id="{definition_id}"/></FeatureDefinitions>',
            f'    <FeatureNominals n="{n}">']
    for i in range(n):
        out.append(f'      <PointFeatureNominal id="{nominal_ids[i]}"><FeatureDefinitionId>{definition_id}</FeatureDefinitionId>'
                   f"<Location>{_triple(points[i])}</Location><Normal>{_triple(normals[i])}</Normal></PointFeatureNominal>")
    out.append(f'    </FeatureNominals><FeatureItems n="{n}">')
    for i in range(n):
        out.append(f'      <PointFeatureItem id="{item_ids[i]}"><FeatureNominalId>{nominal_ids[i]}</FeatureNominalId>'
                   f"<FeatureName>{escape(str(feature_names[i]))}</FeatureName></PointFeatureItem>")
    out.append("    </FeatureItems></Features>")

    definitions, nominals, items = [], [], []
    by_width = {}
    for spec in profile_characteristics:
        width = float(spec["zone_width_m"])
        if width not in by_width:
            by_width[width] = new_id()
            frame = f"<DatumReferenceFrameId>{frame_id}</DatumReferenceFrameId>" if len(datum_feature_indices) else ""
            definitions.append(f'      <PointProfileCharacteristicDefinition id="{by_width[width]}">'
                               f"<ToleranceValue>{width / metres_per_unit!r}</ToleranceValue>{frame}"
                               "</PointProfileCharacteristicDefinition>")
        nominal_id, item_id = new_id(), new_id()
        nominals.append(f'      <PointProfileCharacteristicNominal id="{nominal_id}">'
                        f"<CharacteristicDefinitionId>{by_width[width]}</CharacteristicDefinitionId></PointProfileCharacteristicNominal>")
        items.append(f'      <PointProfileCharacteristicItem id="{item_id}"><Name>{escape(str(spec["name"]))}</Name>'
                     f'<FeatureItemIds n="1"><Id>{item_ids[int(spec["feature_index"])]}</Id></FeatureItemIds>'
                     f"<CharacteristicNominalId>{nominal_id}</CharacteristicNominalId></PointProfileCharacteristicItem>")
    out += [f'  <Characteristics><CharacteristicDefinitions n="{len(definitions)}">', *definitions,
            f'    </CharacteristicDefinitions><CharacteristicNominals n="{len(nominals)}">', *nominals,
            f'    </CharacteristicNominals><CharacteristicItems n="{len(items)}">', *items,
            "    </CharacteristicItems></Characteristics>"]

    if measured_points_m is not None:
        measured = np.asarray(measured_points_m, dtype=float) / metres_per_unit
        out.append(f'  <Results><MeasurementResultsSet n="{len(measured)}">')
        for build in measured:
            out.append(f'    <MeasurementResults id="{new_id()}"><MeasuredFeatures n="{n}">')
            for i in range(n):
                out.append(f'      <PointFeatureMeasurement id="{new_id()}"><FeatureItemId>{item_ids[i]}</FeatureItemId>'
                           f"<Location>{_triple(build[i])}</Location></PointFeatureMeasurement>")
            out.append("    </MeasuredFeatures></MeasurementResults>")
        out.append("  </MeasurementResultsSet></Results>")
    out.append("</QIFDocument>")
    path = Path(path)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
