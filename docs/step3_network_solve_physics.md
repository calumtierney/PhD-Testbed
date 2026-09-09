# Step 3 physics: why more stations help, and why the help runs out

Companion to `src/network/solve.py`. Read this before the code, or to
check the physics independently of it.

## What changes between step 1 and step 3

Step 1 gave one station's covariance for one point: a flattened ellipsoid,
thin along that station's beam, wide across it. Step 3 asks what happens
when several stations, at different positions, each get their own noisy
look at the *same* target, and everything — every target's coordinates,
every station's pose — is solved jointly rather than station-by-station.

## Why one station's weak direction is (usually) another's strong one

A single laser tracker station's ellipsoid for one point is anisotropic —
tight along its own beam, loose across it (step 1). Put a second station
somewhere *not* along the same line of sight, and its own ellipsoid is
tight along *its* beam and loose across *that* — a different direction
from the first station's tight direction, provided the two stations view
the target from sufficiently different angles.

Combining two independent estimates of the same point (by least squares)
doesn't just average their uncertainties — it multiplies their
likelihoods, and the combined ellipsoid is roughly the *intersection* of
the two individual ellipsoids' regions of "plausible position". Two thin,
differently-oriented discs intersect in a much smaller region than either
disc alone, in every direction — including the direction that was each
station's own weak axis, because the other station is (if the geometry is
good) tight exactly there. This is the real reason uncertainty drops with
station count, and it's why *where* stations are placed matters as much
as how many there are: two stations both looking down the same line
merely double-confirm the same weak direction and barely help; two
stations at a wide angle to each other cover each other's blind spot.

Beyond the *shared observed points*, every additional station also adds
redundant constraints on *station poses themselves* — with enough shared
targets, station poses become better determined too, which feeds back
into better target estimates. This is a real coupling this codebase
solves for directly (target coordinates and station poses are unknowns in
the same least-squares problem, not solved in separate passes).

## Why the improvement flattens

Once a target already has two or three stations viewing it from
meaningfully different angles, its position is already fairly
well-triangulated in every direction — there's no direction left where a
single station's uncertainty dominates unchallenged. A fourth or fifth
station, viewed from an angle similar to ones already present, mostly
overlaps information the network already has; it adds redundancy (which
does still help — more independent noisy looks always reduce variance
somewhat, roughly as `1/sqrt(n)` in the direction that's already
well-covered) but not the qualitative "someone finally covers this weak
direction" gain the second or third station provided. That's the
diminishing-returns shape CLAUDE.md §5's step 3 gate describes: "falls...
and flattens after four or five stations", matching Wang, Forbes &
Maropoulos (2014)'s reported 26.5 -> 16.6 -> 13.7 -> 10.7 µm for one to
four stations (§4d) — big early drops, smaller later ones. This codebase
doesn't try to reproduce those literal numbers (different scene,
different instrument parameters), only the shape; see
`tests/test_network_solve.py::test_mean_uncertainty_falls_and_flattens_with_station_count`.

## The frame-of-reference constraint, restated physically

A metrologist would put this as: the network of range-and-angle
observations tells you the *shape* of the target/station arrangement
precisely, but nothing in it tells you where that shape sits in the room,
or which way it's turned. Pick the whole rigid assembly up, translate and
rotate it together, and every single observation — every range, every
angle — reads back exactly the same. That's not a flaw in the
instrument; it's a basic fact about relative measurements. Something
external has to pin the assembly down. Real networks do this with a
known reference artefact, a levelled/oriented instrument, or by
convention; here, the simplest version is used — station 0's pose is
declared exact, and everything else is solved relative to it (see the
"frame-of-reference constraint" section of `network/solve.py`'s module
docstring for the mechanics). This is also why `perturbed_initial_guess`
must *not* perturb the anchor's pose when calling `solve_network` with a
matching `anchor_index`: doing so wouldn't give the solver "a rough guess
to refine," it would quietly redefine what "the global frame" means, and
every recovered target and station would come back offset by exactly
that error — a real, easy-to-make mistake, and one
`tests/test_network_solve.py::test_solved_targets_are_close_to_the_truth`
would have caught (it did, during development of this step).

## What the single-station case checks

With exactly one station, each target has exactly three unknowns (x, y,
z) and exactly three observations (d, theta, phi) from that one station —
no redundancy, so the least-squares fit passes exactly through the noisy
data (zero residual) and the covariance reduces to a closed-form Jacobian
propagation identical to step 1's. This isn't a coincidence to route
around; it's a consistency check that step 3 is really the same physics
as step 1, just solved a more general way — see
`tests/test_network_solve.py::test_single_station_network_solve_matches_step1_covariance`,
which checks the two independently-computed covariances agree to within
~2%.
