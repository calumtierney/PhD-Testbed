"""A reader for QIF documents (ISO 23952, Quality Information Framework).

**What a QIF file is, for this project.** QIF is the XML format a CAD/PMI
system uses to hand an inspection requirement to a measurement system: the
*features* to be measured (a point on a surface, a hole, a plane), the
*characteristics* toleranced on them (a profile, a position), the *datums*
those tolerances refer to, the *product* structure (parts, assemblies, where
each part sits in its assembly) and, in a results file, what earlier
inspections *measured*. That is the input the planning chain of CLAUDE.md §2
has so far been given by hand.

**What this module does.** It reads those five things into plain dataclasses,
in **SI units** (metres), and nothing else. It does not decide what can be
planned: `planning.qif_cell` does, and reports what it had to leave out.

**How the element names here were established.** From the QIF 3.0 sample files
and XSD schemas published by the QIF community
(github.com/QualityInformationFramework/qif-community; Boost licence), not from
memory: the sheet-metal results sample (point and circle features, point-profile
and position characteristics, six measured parts; `scenes/qif/`) and the NIST
MBD test cases (model-based definition: features and characteristics with no
inspection items). Two conventions for linking a characteristic to its feature
occur in those files and both are read:

  * results/plan files: `CharacteristicItem -> FeatureItemIds -> FeatureItem ->
    FeatureNominalId`;
  * MBD files: `CharacteristicNominal -> FeatureNominalIds` (no items at all).

Assembly structure (`AssemblySet`, `Component/Transform`, `TransformMatrix`) is
read as the schema text describes it. **No public sample contains an assembly
with transforms, so that part is untested against a vendor file**; the
convention assumed is stated on `QifComponent`.

**Units.** Coordinates and tolerance values are in the file's primary linear
unit (`FileUnits/PrimaryUnits/LinearUnit`, a name and a conversion factor to
metres) unless a value carries its own `linearUnit` attribute, which is looked
up among the file's declared units (a value in an undeclared unit is dropped
with a warning, never guessed). A file with no `FileUnits` is in SI units
by the standard's default; that is assumed and a warning recorded.

**Safety.** An uploaded file is untrusted. The reader uses the standard
library's XML parser, refuses any document with a DOCTYPE or entity
declaration (the route for entity-expansion and external-entity attacks; QIF
needs neither), refuses files above a size limit, and never follows the
`ExternalQIFReferences` a document may list (they are recorded as a warning).

**Not read.** Geometry (BRep, tessellation), plans, rules, measurement
resources, statistics blocks, and every feature or characteristic field not
listed on the dataclasses below.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

MAX_FILE_BYTES = 200 * 1024 * 1024


class QifError(ValueError):
    """The file is not a QIF document this reader can accept."""


@dataclass(frozen=True)
class QifFeature:
    """One nominal feature. `location_m` is the point a single-point
    measurement of it would be taken at (a point feature's location, a circle's
    centre, the axis point of a cylinder, the origin point of a plane), or None
    if the file gives none (e.g. a feature defined only by CAD faces).
    `direction` is a unit vector and `direction_role` says what it is:
    "normal" (the surface normal at a point, the normal of a circle's or
    plane's own plane) or "axis" (a cylinder's or cone's axis)."""

    id: str
    kind: str                     # "Point", "EdgePoint", "Circle", "Cylinder", "Plane", ...
    name: str
    location_m: Optional[np.ndarray]
    direction: Optional[np.ndarray]
    direction_role: Optional[str]
    measured_locations_m: Tuple[np.ndarray, ...] = ()   # one per earlier inspection that measured it


@dataclass(frozen=True)
class QifCharacteristic:
    """One toleranced characteristic. `tolerance_m` is the zone size as the
    file states it (`ToleranceValue`: the full width of a profile zone, the
    diameter of a position zone), None for characteristic types that carry
    limits some other way (sizes, distances, angles), which this reader records
    but does not interpret. `measured_values_m` are earlier inspections'
    reported values, in the file's own convention for that characteristic type
    (a point profile's signed normal deviation; a position's zone-equivalent
    value, not a signed deviation)."""

    id: str
    kind: str                     # "PointProfile", "Position", "SurfaceProfile", "Flatness", ...
    name: str
    tolerance_m: Optional[float]
    zone_shape: Optional[str]     # "Diametrical", "Spherical", "NonDiametrical" (position only)
    feature_ids: Tuple[str, ...]
    datum_reference_frame_id: Optional[str]
    datum_labels: Tuple[str, ...]
    nominal_id: Optional[str]
    unequal_zone: bool = False    # an unequally disposed or offset profile zone: limits are not +-T/2
    material_condition: Optional[str] = None
    criticality: Optional[str] = None
    measured_values_m: Tuple[float, ...] = ()


@dataclass(frozen=True)
class QifDatum:
    id: str
    label: str
    feature_ids: Tuple[str, ...]


@dataclass(frozen=True)
class QifPart:
    """A part definition. `feature_ids` / `characteristic_nominal_ids` are the
    nominals the file lists as belonging to it (empty if it does not say)."""

    id: str
    name: str
    feature_ids: Tuple[str, ...] = ()
    characteristic_nominal_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class QifAssembly:
    id: str
    name: str
    component_ids: Tuple[str, ...] = ()
    feature_ids: Tuple[str, ...] = ()
    characteristic_nominal_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class QifComponent:
    """One instance of a part or assembly inside its parent.

    `transform` is the 4x4 matrix taking the instance's own coordinates to its
    parent's, translation in metres, or None for "in place". **Assumed
    convention** (from the schema's `CoordinateSystemCoreType`: an `Origin` and
    a `Rotation` of `XDirection`, `YDirection`, `ZDirection`): a point
    (x, y, z) of the instance is at `Origin + x XDirection + y YDirection +
    z ZDirection` in the parent. Untested against a vendor file."""

    id: str
    part_id: Optional[str]
    assembly_id: Optional[str]
    transform: Optional[np.ndarray]


@dataclass
class QifDocument:
    version: str
    linear_unit_name: str
    metres_per_unit: float
    features: Dict[str, QifFeature] = field(default_factory=dict)
    characteristics: Dict[str, QifCharacteristic] = field(default_factory=dict)
    datums: Dict[str, QifDatum] = field(default_factory=dict)
    datum_reference_frames: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # id -> datum labels, in precedence order
    parts: Dict[str, QifPart] = field(default_factory=dict)
    assemblies: Dict[str, QifAssembly] = field(default_factory=dict)
    components: Dict[str, QifComponent] = field(default_factory=dict)
    root_id: Optional[str] = None      # the root part, assembly or component, if the file names one
    n_inspections: int = 0             # MeasurementResults blocks: earlier measured parts
    warnings: List[str] = field(default_factory=list)

    # -- product structure --------------------------------------------------------------
    def scope_names(self) -> Dict[str, str]:
        """`{id: "part: name" | "assembly: name"}` for everything a plan can be asked for."""
        out = {a.id: f"assembly: {a.name}" for a in self.assemblies.values()}
        out.update({p.id: f"part: {p.name}" for p in self.parts.values()})
        return out

    def part_placements(self, scope_id: Optional[str] = None) -> Dict[str, List[np.ndarray]]:
        """`{part id: [4x4 transform, ...]}`: where each part's coordinates sit
        in the coordinates of `scope_id` (a part or an assembly; None = the
        whole document), one transform per instance. A part asked for on its
        own is in place (identity). For an assembly the component transforms
        are composed down the tree."""
        identity = np.eye(4)
        if scope_id is None:
            scope_id = self._default_scope()
            if scope_id is None:
                return {p: [identity] for p in self.parts}
        if scope_id in self.parts:
            return {scope_id: [identity]}
        if scope_id not in self.assemblies:
            raise QifError(f"no part or assembly with id {scope_id!r}; available: {self.scope_names()}")
        placements: Dict[str, List[np.ndarray]] = {}

        def descend(assembly_id: str, to_scope: np.ndarray, depth: int):
            if depth > 50:
                raise QifError("assembly structure is nested more than 50 deep (a cycle?)")
            for component_id in self.assemblies[assembly_id].component_ids:
                component = self.components.get(component_id)
                if component is None:
                    self.warnings.append(f"assembly {assembly_id} lists component {component_id}, which the file does not define")
                    continue
                here = to_scope @ (component.transform if component.transform is not None else identity)
                if component.part_id is not None:
                    placements.setdefault(component.part_id, []).append(here)
                elif component.assembly_id in self.assemblies:
                    descend(component.assembly_id, here, depth + 1)

        descend(scope_id, identity, 0)
        return placements

    def _default_scope(self) -> Optional[str]:
        root = self.root_id
        if root in self.components:
            component = self.components[root]
            root = component.assembly_id or component.part_id
        if root in self.assemblies or root in self.parts:
            return root
        if len(self.assemblies) == 1:
            return next(iter(self.assemblies))
        if len(self.parts) == 1 and not self.assemblies:
            return next(iter(self.parts))
        return None

    def characteristics_in(self, scope_id: Optional[str] = None) -> List[QifCharacteristic]:
        """The characteristics belonging to a part or assembly. A file that
        does not say which part a characteristic belongs to (results files
        often do not) yields every characteristic, with a warning when that
        loses a distinction the caller asked for."""
        everything = list(self.characteristics.values())
        if scope_id is None:
            return everything
        part_ids = list(self.part_placements(scope_id))
        nominal_ids = set()
        for part_id in part_ids:
            nominal_ids.update(self.parts[part_id].characteristic_nominal_ids)
        if scope_id in self.assemblies:
            nominal_ids.update(self.assemblies[scope_id].characteristic_nominal_ids)
        if not nominal_ids:
            if len(self.parts) + len(self.assemblies) > 1:
                self.warnings.append(
                    f"the file does not say which characteristics belong to {scope_id}; all {len(everything)} are returned")
            return everything
        return [c for c in everything if (c.nominal_id or c.id) in nominal_ids]

    def part_of_feature(self, feature_id: str) -> Optional[str]:
        for part in self.parts.values():
            if feature_id in part.feature_ids:
                return part.id
        return None


# -- reading -----------------------------------------------------------------------------
def read_qif(source: Union[str, Path, bytes], max_bytes: int = MAX_FILE_BYTES) -> QifDocument:
    """Read a QIF document from a path or from the bytes of an uploaded file."""
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    if len(data) > max_bytes:
        raise QifError(f"file is {len(data) / 1e6:.0f} MB; the limit is {max_bytes / 1e6:.0f} MB")
    lowered = bytes(data).lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise QifError("the document declares a DOCTYPE or entities, which QIF does not need and this reader refuses")
    try:
        root = ET.fromstring(bytes(data))
    except ET.ParseError as error:
        raise QifError(f"not well-formed XML: {error}") from error
    for element in root.iter():  # QIF 2.x and 3.0 differ in namespace, not in the names read here
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.rsplit("}", 1)[1]
    if root.tag != "QIFDocument":
        raise QifError(f"root element is <{root.tag}>, not <QIFDocument>")
    return _Reader(root).read()


def _text(element: Optional[ET.Element], path: str) -> Optional[str]:
    if element is None:
        return None
    found = element.find(path)
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    return text or None


def _ids(element: Optional[ET.Element], path: str) -> Tuple[str, ...]:
    if element is None:
        return ()
    holder = element.find(path)
    if holder is None:
        return ()
    return tuple(i.text.strip() for i in holder.findall("Id") if i.text and i.text.strip())


def _vector(text: Optional[str]) -> Optional[np.ndarray]:
    if text is None:
        return None
    try:
        values = np.array([float(v) for v in text.split()])
    except ValueError:
        return None
    return values if values.shape == (3,) and np.all(np.isfinite(values)) else None


def _unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else None


# Where a feature's representative point and direction are, by element path,
# in the order tried. (Verified on Point, EdgePoint, Circle, Plane, Cylinder and
# OppositeParallelPlanes nominals in the sample files.)
_LOCATION_PATHS = ("Location", "Axis/AxisPoint", "CenterPlane/Point", "Center", "Point")
_DIRECTION_PATHS = (("Normal", "normal"), ("Axis/Direction", "axis"), ("CenterPlane/Normal", "normal"), ("Direction", "axis"))


class _Reader:
    def __init__(self, root: ET.Element):
        self.root = root
        self.by_id = {e.get("id"): e for e in root.iter() if e.get("id") is not None}
        self.warnings: List[str] = []
        self.units: Dict[str, float] = {}
        self.primary_unit_name = "m"
        self.metres_per_unit = 1.0

    def read(self) -> QifDocument:
        self._read_units()
        doc = QifDocument(self.root.get("versionQIF", "unknown"), self.primary_unit_name, self.metres_per_unit,
                          warnings=self.warnings)
        if self.root.find("ExternalQIFReferences") is not None:
            self.warnings.append("the document refers to other QIF documents (ExternalQIFReferences); they are not followed")
        item_to_nominal = self._read_features(doc)
        self._read_datums(doc)
        self._read_characteristics(doc, item_to_nominal)
        self._read_product(doc)
        return doc

    # -- units ---------------------------------------------------------------------------
    def _read_units(self):
        file_units = self.root.find("FileUnits")
        if file_units is None:
            self.warnings.append("no FileUnits: SI units (metres) assumed, the standard's default")
            return
        for unit in file_units.iter():  # LinearUnit, and PMILinearUnit (the unit PMI values may be stated in)
            if not unit.tag.endswith("LinearUnit"):
                continue
            name = _text(unit, "UnitName")
            factor = _text(unit, "UnitConversion/Factor")
            if name:
                self.units[name] = float(factor) if factor else 1.0
        primary = file_units.find("PrimaryUnits/LinearUnit")
        if primary is None:
            self.warnings.append("FileUnits has no primary linear unit: metres assumed")
            return
        self.primary_unit_name = _text(primary, "UnitName") or "m"
        factor = _text(primary, "UnitConversion/Factor")
        self.metres_per_unit = float(factor) if factor else 1.0
        if not (self.metres_per_unit > 0.0 and np.isfinite(self.metres_per_unit)):
            raise QifError(f"primary linear unit has conversion factor {factor!r}")

    def _length_m(self, element: Optional[ET.Element]) -> Optional[float]:
        """A length element's value in metres, honouring its own `linearUnit`."""
        if element is None or element.text is None:
            return None
        try:
            value = float(element.text)
        except ValueError:
            return None
        own = element.get("linearUnit")
        if own is None:
            return value * self.metres_per_unit
        if own not in self.units:  # never guess a unit: a tolerance read 25 times too tight is worse than none
            self.warnings.append(f"a value is given in unit {own!r}, which FileUnits does not define; it is not used")
            return None
        return value * self.units[own]

    # -- features ------------------------------------------------------------------------
    def _read_features(self, doc: QifDocument) -> Dict[str, str]:
        features = self.root.find("Features")
        item_to_nominal: Dict[str, str] = {}
        if features is None:
            self.warnings.append("the document has no Features section")
            return item_to_nominal
        names: Dict[str, str] = {}
        for item in features.findall("FeatureItems/*"):
            nominal_id = _text(item, "FeatureNominalId")
            if item.get("id") and nominal_id:
                item_to_nominal[item.get("id")] = nominal_id
                name = _text(item, "FeatureName")
                if name:
                    names.setdefault(nominal_id, name)
        measured: Dict[str, List[np.ndarray]] = {}
        for measurement in self.root.findall("Results/MeasurementResultsSet/MeasurementResults/MeasuredFeatures/*"):
            nominal_id = item_to_nominal.get(_text(measurement, "FeatureItemId") or "")
            location = _vector(_text(measurement, "Location"))
            if nominal_id and location is not None:
                measured.setdefault(nominal_id, []).append(location * self.metres_per_unit)
        for nominal in features.findall("FeatureNominals/*"):
            feature_id = nominal.get("id")
            if feature_id is None or not nominal.tag.endswith("FeatureNominal"):
                continue
            location = next((v for v in (_vector(_text(nominal, p)) for p in _LOCATION_PATHS) if v is not None), None)
            direction, role = None, None
            for path, path_role in _DIRECTION_PATHS:
                direction = _unit(_vector(_text(nominal, path)))
                if direction is not None:
                    role = path_role
                    break
            doc.features[feature_id] = QifFeature(
                id=feature_id, kind=nominal.tag[:-len("FeatureNominal")],
                name=names.get(feature_id) or _text(nominal, "Name") or f"feature {feature_id}",
                location_m=None if location is None else location * self.metres_per_unit,
                direction=direction, direction_role=role,
                measured_locations_m=tuple(measured.get(feature_id, ())),
            )
        return item_to_nominal

    # -- datums --------------------------------------------------------------------------
    def _read_datums(self, doc: QifDocument):
        for definition in self.root.findall("DatumDefinitions/DatumDefinition"):
            datum_id = definition.get("id")
            if datum_id:
                doc.datums[datum_id] = QifDatum(datum_id, _text(definition, "DatumLabel") or "?", _ids(definition, "FeatureNominalIds"))
        order = {"PRIMARY": 0, "SECONDARY": 1, "TERTIARY": 2, "QUATERNARY": 3, "QUINARY": 4, "SENARY": 5}
        for frame in self.root.findall("DatumReferenceFrames/DatumReferenceFrame"):
            entries = []
            for datum in frame.findall("Datums/Datum"):
                definition_id = _text(datum, "SimpleDatum/DatumDefinitionId")
                if definition_id is None:
                    self.warnings.append(f"datum reference frame {frame.get('id')} uses a compound datum, which is not read")
                    continue
                label = doc.datums[definition_id].label if definition_id in doc.datums else "?"
                entries.append((order.get(_text(datum, "Precedence/PrecedenceEnum") or "", 99), label))
            doc.datum_reference_frames[frame.get("id")] = tuple(label for _, label in sorted(entries, key=lambda e: e[0]))

    # -- characteristics -----------------------------------------------------------------
    def _read_characteristics(self, doc: QifDocument, item_to_nominal: Dict[str, str]):
        section = self.root.find("Characteristics")
        if section is None:
            self.warnings.append("the document has no Characteristics section")
            return
        values: Dict[str, List[float]] = {}
        results = self.root.findall("Results/MeasurementResultsSet/MeasurementResults")
        doc.n_inspections = len(results)
        for result in results:
            for measurement in result.findall("MeasuredCharacteristics/CharacteristicMeasurements/*"):
                item_id = _text(measurement, "CharacteristicItemId")
                value = self._length_m(measurement.find("Value"))
                if item_id and value is not None:
                    values.setdefault(item_id, []).append(value)

        def build(char_id, nominal, item) -> Optional[QifCharacteristic]:
            definition = self.by_id.get(_text(nominal, "CharacteristicDefinitionId") or "")
            kind = nominal.tag[:-len("CharacteristicNominal")]
            feature_ids = list(_ids(nominal, "FeatureNominalIds"))
            if item is not None:
                for item_id in _ids(item, "FeatureItemIds"):
                    if item_id in item_to_nominal:
                        feature_ids.append(item_to_nominal[item_id])
                    else:
                        self.warnings.append(f"characteristic {char_id} refers to feature item {item_id}, which is not defined")
            frame_id = _text(definition, "DatumReferenceFrameId")
            zone = definition.find("ZoneShape") if definition is not None else None
            zone_shape = None
            if zone is not None and len(zone):
                zone_shape = zone[0].tag[:-len("Zone")] if zone[0].tag.endswith("Zone") else zone[0].tag
            unequal = definition is not None and any(definition.find(t) is not None for t in ("UnequallyDisposedZone", "OffsetZone"))
            source = item if item is not None else nominal
            name = (_text(source, "Name") or _text(source, "CharacteristicDesignator/Designator")
                    or _text(nominal, "Name") or f"{kind} {char_id}")
            return QifCharacteristic(
                id=char_id, kind=kind, name=name,
                tolerance_m=self._length_m(definition.find("ToleranceValue")) if definition is not None else None,
                zone_shape=zone_shape, feature_ids=tuple(dict.fromkeys(feature_ids)),
                datum_reference_frame_id=frame_id, datum_labels=doc.datum_reference_frames.get(frame_id, ()),
                nominal_id=nominal.get("id"), unequal_zone=bool(unequal),
                material_condition=_text(definition, "MaterialCondition"),
                criticality=(_text(source, "CharacteristicDesignator/Criticality/CriticalityEnum")
                             or _text(nominal, "CharacteristicDesignator/Criticality/CriticalityEnum")),
                measured_values_m=tuple(values.get(char_id, ())),
            )

        nominals = {n.get("id"): n for n in section.findall("CharacteristicNominals/*") if n.get("id")}
        items = [i for i in section.findall("CharacteristicItems/*") if i.get("id")]
        if items:  # an inspection file: one characteristic per item (what is actually to be checked)
            for item in items:
                nominal = nominals.get(_text(item, "CharacteristicNominalId") or "")
                if nominal is None:
                    self.warnings.append(f"characteristic item {item.get('id')} has no nominal; skipped")
                    continue
                doc.characteristics[item.get("id")] = build(item.get("id"), nominal, item)
        else:      # a model-based definition: the nominals are the requirement
            for nominal_id, nominal in nominals.items():
                if nominal.tag.endswith("CharacteristicNominal"):
                    doc.characteristics[nominal_id] = build(nominal_id, nominal, None)

    # -- product -------------------------------------------------------------------------
    def _read_product(self, doc: QifDocument):
        product = self.root.find("Product")
        if product is None:
            return
        for part in product.findall("PartSet/Part"):
            name = _text(part, "Name") or part.get("label") or _text(part, "ModelNumber") or f"part {part.get('id')}"
            doc.parts[part.get("id")] = QifPart(part.get("id"), name, _ids(part, "FeatureNominalIds"),
                                                _ids(part, "CharacteristicNominalIds"))
        for assembly in product.findall("AssemblySet/Assembly"):
            name = _text(assembly, "Name") or assembly.get("label") or _text(assembly, "ModelNumber") or f"assembly {assembly.get('id')}"
            doc.assemblies[assembly.get("id")] = QifAssembly(
                assembly.get("id"), name, _ids(assembly, "ComponentIds"), _ids(assembly, "FeatureNominalIds"),
                _ids(assembly, "CharacteristicNominalIds"))
        for component in product.findall("ComponentSet/Component"):
            transform = None
            transform_id = _text(component, "Transform/Id")
            if transform_id is not None:
                transform = self._transform(self.by_id.get(transform_id), transform_id)
            doc.components[component.get("id")] = QifComponent(
                component.get("id"), _text(component, "Part/Id"), _text(component, "Assembly/Id"), transform)
        for tag in ("RootAssembly", "RootComponent", "RootPart"):
            root_id = _text(product, f"{tag}/Id")
            if root_id:
                doc.root_id = root_id
                break

    def _transform(self, element: Optional[ET.Element], transform_id: str) -> Optional[np.ndarray]:
        if element is None:
            self.warnings.append(f"component transform {transform_id} is not defined in the file; the component is taken as in place")
            return None
        matrix = np.eye(4)
        origin = _vector(_text(element, "Origin"))
        if origin is not None:
            matrix[:3, 3] = origin * self.metres_per_unit
        axes = [_vector(_text(element, f"Rotation/{axis}")) for axis in ("XDirection", "YDirection", "ZDirection")]
        if all(a is not None for a in axes):
            rotation = np.column_stack(axes)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
                self.warnings.append(f"transform {transform_id} has a rotation that is not orthonormal; it is used as given")
            matrix[:3, :3] = rotation
        elif any(a is not None for a in axes):
            self.warnings.append(f"transform {transform_id} gives only some rotation axes; no rotation is applied")
        return matrix
