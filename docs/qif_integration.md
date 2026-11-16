# Planning from a QIF file

**Status: first working slice.** A QIF document can be read, turned into a
factory cell sized round the part, and planned, from the command line:

    python -m viz.plan_from_qif scenes/qif/SheetMetal_QIF_Results_6_samples.QIF
    python -m viz.plan_from_qif file.qif --list              # what is in the file
    python -m viz.plan_from_qif file.qif --scope <id>        # one part or assembly

This sits outside the six-step ladder of CLAUDE.md §5: it is an input route to
the chain, not a link in it. It adds no physics. Code: `interchange.qif`
(reader), `interchange.qif_write` (a minimal writer for tests),
`planning.qif_cell` (file to cell to plan), `planning.cell.PartFrameScorer` and
`reference_nests_for_track` (the scene-independent pieces), `viz.plan_from_qif`.

## 1. What QIF gives the chain, and what it does not

QIF (ISO 23952) carries the inspection requirement: features, the
characteristics toleranced on them, datums, product structure and, in results
files, what earlier inspections measured. Element names and nesting were taken
from the QIF community's published QIF 3.0 samples and schemas, not from
memory. What was found there shapes everything below.

| The chain needs | QIF gives | Consequence |
|---|---|---|
| Points to measure, with a surface normal | Point, edge-point and circle features carry a location and a normal. Planes, cylinders and "other" features in model-based files carry a feature-level location or only references to CAD faces | Inspection-style files map directly. Model-based (MBD) files mostly do not: the points to probe are a measurement-strategy decision the file does not contain |
| The direction each tolerance constrains | Point profile: the normal. Hole position: the axis. Zone shape is stated | Point profile maps exactly. Position maps with an approximation (§3) |
| Datum frame | Datum *features* and their precedence. Often no geometry at all (the sheet-metal sample's datums name no features) | The part frame is a rigid best fit to located datum features, or to the planned points when there are none. It is not a 3-2-1 construction |
| Process capability per feature cluster | Nothing as such. Results files hold measured feature locations for n parts | Deviations are computed from those and pooled by a clustering *rule* (§4) |
| The part's shape, for occlusion | Optional BRep in MBD files; none in results files | No self-occlusion unless an STL is supplied alongside |
| Where the part is in the cell | Nothing. Coordinates are the part's or vehicle's own (the sample part is 2.5 m from its origin) | A placement is assumed and reported, or supplied |
| Parts within an assembly | `AssemblySet`, `Component`, `Transform` | Read as the schema describes. No public sample has one, so this is untested against a vendor file |

## 2. The scene changes size with the part

It does, and by rule rather than by hand. The station track is drawn round the
plan-view extent of the located features at the standoff of `CellRules`, and
the reference network is laid round the track by the same rules that produce
the 15 m cell (pillar rows behind each track row, end walls beyond each loop;
`reference_nests_for_track` reproduces `ers_nests` exactly for that cell). A
0.16 m bracket gets a 19 m track and 20 nests; the 15 m part gets its 57 m
track and 28.

Three things follow that are worth knowing before trusting a plan:

* **The extent is that of the features, not of the part.** A file that locates
  six points on a 10 m skin describes a much smaller footprint than the skin.
  Supplying the STL fixes this.
* **Every rule is a placeholder** for a real cell (standoff, pillar spacing,
  station heights, jig height): `TODO(source)`. A real cell has fixed
  monuments; the right input there is the cell's own nest coordinates, not a
  rule. That is a small change (pass the nests in) and should replace the rule
  once a real layout is available.
* **Small parts are not this instrument's job.** Below 2 m (CLAUDE.md §1) the
  cell says so. The plan is still computed, since it exercises the pipeline.

## 3. What is planned and what is skipped

Planned: point profile (exact) and position with a diametrical or spherical
zone (approximate). Skipped, each with its reason in the output: sizes, form
and orientation tolerances, profiles of whole surfaces, unequally disposed
zones, non-diametrical position zones, characteristics on features the file
does not locate. On the NIST MBD test cases almost everything is skipped, and
that is the correct answer today: those callouts need several points per
feature and their joint covariance, which the characteristic layer (step 4)
does not yet have.

**The position approximation.** A hole's zone is round; the risk model is
one-dimensional. A hole is judged along one direction against half the zone
diameter, using the uncertainty of the worst-measured direction across its
axis. This understates how often a round zone is violated. Options:

1. Keep it and flag it (done): cheap, and the ranking of layouts is probably
   right, but the risk number is not.
2. A two-dimensional risk calculation in `risk/` (bivariate prior and
   measurement, round acceptance zone): correct, perhaps two days' work with
   tests, and it thickens step 5 for every scene, not only QIF.
3. Leave positions out until (2) exists: honest but drops the holes.

Recommendation: (2) next, before any result involving holes is quoted.

## 4. Process knowledge from the file

CLAUDE.md §3 requires priors per feature cluster. QIF has no clusters, so one
is formed by rule: same characteristic type, tolerance, datum frame and part.
Deviations (measured minus nominal along the constrained direction) are pooled
over the cluster's members and the n measured parts; the degrees of freedom
are n - 1, the number of parts, not the number of pooled values, because
features on one part are not independent draws.

Two cautions. The rule can put unlike features together: in the sample, four
+-0.75 mm points pool to a standard deviation of 0.45 mm because they sit at
different offsets. An engineer would split them. Cluster membership is
reported for exactly that reason and should be overridable (not built yet).
And the characteristic values in the sample's results contain a duplicate
zero-valued entry per item, so they are not used; feature locations are.

With no earlier inspections a cluster gets a centred placeholder at Cpk 1.33,
labelled as a placeholder in every output.

## 5. Checks

* The long-part factory cell written out as QIF and read back gives the same
  54 targets, the same datum set and the same uncertainties as the hand-built
  scene (`tests/test_qif_cell.py`).
* On the real sample: units, counts, links, and two agreements with the file's
  own numbers: a point profile's deviation from (measured - nominal) . normal
  equals the reported value, and twice a hole's offset across its axis equals
  the reported position value.
* Hostile XML (DOCTYPE, entities) is refused; a value in an undeclared unit is
  dropped rather than guessed. The second mattered at once: one NIST file
  states tolerances in inches inside a millimetre document.

## 6. First result on a real file, and what it does not show

Sheet-metal sample, 21 characteristics, six measured parts, simple acceptance:

| Stations | Objective | Expected wrong decisions | Worst TUR | Largest u |
|---|---|---|---|---|
| 0 | (nothing measured) | 3.03 | | |
| 1 | any | no feasible layout: the points face several ways | | |
| 2 | A0 / A1 / risk | 0.110 / 0.118 / 0.107 | 22 to 27:1 | 9.6 um |
| 3 | A0 / A1 / risk | 0.063 / 0.062 / 0.062 | 40:1 | 9.4 um |
| 4 | A0 / A1 / risk | 0.053 / 0.053 / 0.052 | 40:1 | 9.4 um |

Read this as a pipeline check, not a finding. The part is 0.16 m across with
millimetre tolerances; a tracker is not how it would be measured, the cell
rules are placeholders, there is no part geometry, and the search settings are
light. The objectives barely differ here, which is what the earlier
experiments predict when every characteristic is far inside the instrument's
capability. What is worth noticing is that risk stays well above zero at a TUR
of 40:1, because several of this process's clusters sit near their limits.

## 7. Open issues, in the order they bite

1. **Multi-point characteristics** (flatness, surface profile, orientation):
   the bulk of real aerospace PMI. Needs step 4 to take several points and a
   fit. Until then MBD files are mostly skipped.
2. **Two-dimensional risk for positions** (§3).
3. **A real large-volume file.** No public QIF of an aerospace-scale part was
   found. One from GKN or AMIC would test the assembly path, the datum
   handling and the cell rules at once.
4. **Target accessibility.** A normal says which way a surface faces, not
   whether an SMR nest fits there or whether a hole's normal points out of the
   part. SMR offset from the surface is ignored.
5. **Datum construction.** Best fit, not 3-2-1; material-condition modifiers
   give bonus tolerance that is ignored (conservative).
6. **Overriding the automatic choices**: cluster membership, process
   statistics, placement and real nest positions from a side file.
7. **Speed of the conformance-probability rule**: about four minutes to
   tabulate for this file, against a second for simple acceptance, so the
   runner defaults to simple acceptance.
8. **Writing the plan back as QIF Plans**, and an upload tab in the Streamlit
   app (the builder and planner are already plain functions the app can call).
