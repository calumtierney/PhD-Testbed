"""Figure-generation entry point (SETUP.md's "daily use" section).

Not one of the six build-step packages under `src/` (CLAUDE.md §5) -- like
`visualization` (see that package's own module docstring), this sits
alongside the physics chain, turning what it already produces into the
PNGs committed under `results/`. Kept as a top-level package (not under
`src/`) so `python -m viz.make_figures` works straight from a fresh clone
with no install step of its own beyond `pip install -e .` (SETUP.md step
7), which is what makes this package's own imports of `geometry`,
`instruments`, etc. resolve.
"""
