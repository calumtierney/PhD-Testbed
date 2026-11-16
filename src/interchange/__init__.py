"""Reading inspection requirements from exchange files (QIF first).

Nothing here knows about trackers, networks or risk: this package turns a file
into plain data (`interchange.qif.QifDocument`). `planning.qif_cell` turns that
data into a scene the chain of CLAUDE.md §2 can plan.
"""
