# QIF sample files

`SheetMetal_QIF_Results_6_samples.QIF` is an unmodified copy of
`samples/Results/Sheet_Metal/SheetMetal_QIF_Results_6_samples.QIF` from the QIF
community repository (github.com/QualityInformationFramework/qif-community),
distributed under the Boost Software License 1.0 (`LICENSE_qif-community.md`).

It is a QIF 3.0 results document for a car-body sheet-metal part ("Wing mirror
reinforcement", about 0.13 m across, in vehicle coordinates): 17 point-profile
and 4 hole-position characteristics, measured on six parts. It is here because
it is a real file written by someone else's software, so `tests/test_qif_reader.py`
checks the reader against the format as it is used, not as this repo imagines it.

It is **not** a large-volume part. No public QIF file of an aerospace-scale
structure was found; `interchange.qif_write.write_point_inspection_qif` writes the testbed's
own long-part scene as a QIF document for that purpose (`tests/test_qif_cell.py`).
