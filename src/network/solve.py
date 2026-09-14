"""Multi-station network solve (CLAUDE.md §5, step 3): weighted nonlinear
least squares over target coordinates *and* station poses, solved
together, from raw per-station spherical observations.

Step 1 asked "given one station and one target, what's the covariance?"
Step 3 asks the network version: several stations each observe some of a
shared set of targets; none of the station poses are known exactly (only
roughly); solve for everything -- every target's coordinates and every
station's pose -- simultaneously, in a way that's consistent with every
observation at once. This is standard surveying/photogrammetry practice,
usually called a network solve or bundle adjustment.

**The frame-of-reference constraint (removing 6 degrees of freedom).**
A network built purely from relative range/angle observations has no way
to know where it sits, or how it's oriented, in an absolute sense: pick
up every station and every target and rotate/translate the whole rigid
assembly together, and every observation (every range, every angle
between beams) comes out exactly the same. That's a 6-parameter family
(3 translation + 3 rotation) of equally good solutions -- a classic gauge
freedom. Left in, the normal equations are singular and the solve can't
converge to a unique answer.

The fix used here: hold one station's pose fixed exactly (the "anchor",
`anchor_index`, station 0 by default) and solve everything else --
every other station's pose, and every target's coordinates -- relative to
it. This is equivalent to declaring the anchor station's own local frame
to *be* the global frame. It removes exactly six unknowns (the anchor's
position and orientation) from the parameter vector, which is exactly the
six degrees of freedom the observations alone can't resolve. (Other
schemes exist -- e.g. a minimal set of external constraints on a few
target points, or solving the full gauge-free system and removing the
freedom afterwards with a Helmert transform -- fixing one station is the
simplest to implement and explain, and is what's used here.)

This makes the resulting covariance genuinely *datum-dependent*: which
station gets fixed is a real modelling choice, not an interchangeable
convention, and picking a different one legitimately changes the
covariance (a different datum, not a bug) -- exactly what the geodetic
network-design literature this laser-tracker-network setting descends
from predicts (Baarda's S-transformations; inner-constraint/free-network
solutions). See `planning.experiment` (CLAUDE.md §5 step 6) for where
this matters in practice: every candidate a search compares must share
the *same* fixed anchor, or the comparison is confounded by an
uncontrolled choice of datum, not by genuine differences in placement
quality.

**Weighting.** Every residual is a spherical-coordinate difference
(observed minus predicted range/azimuth/elevation), divided by that
sensor's own sigma_d / sigma_theta / sigma_phi (`instruments.
laser_tracker.LaserTracker`) -- literally "weight each observation by its
sensor standard deviation", per CLAUDE.md §5. All stations here share one
`LaserTracker` noise model; per-station instrument variation would be a
straightforward extension (give each `Observation` its own sigmas) but
isn't needed for this step's gate, so it isn't built.

`LaserTracker.systematic_std_m` -- an isotropic, placement-irreducible
variance term the tracker can carry alongside sigma_d/theta/phi (see that
field's docstring) -- deliberately never enters this spherical-residual
weighting: it isn't modelled as range/angle noise. `solve_network` adds
it separately, once per target, straight onto the diagonal of the fitted
target covariance, the same way `LaserTracker.covariance_local` adds it
after its own Jacobian propagation rather than before. Skipping this step
would silently discard the term the moment a scenario has more than one
station -- a real bug an earlier revision had (a `PlanningScenario` built
with a nonzero-systematic tracker changed nothing about a multi-station
plan's covariance).

**Exploiting the block-diagonal structure.** Each observation involves
exactly one target and one station -- never two targets, never two
stations. So in the Jacobian of the whole residual vector with respect to
the whole parameter vector, each observation's three residual rows are
zero everywhere except in that target's 3 columns and that station's 6
columns. This is passed to `scipy.optimize.least_squares` as an explicit
sparsity pattern (`jac_sparsity`), which both (a) makes scipy estimate
the Jacobian by finite differences on *groups* of parameters that don't
share any residual rows, perturbing many parameters per function
evaluation instead of one -- the standard graph-colouring/compressed
finite-difference trick, and (b) switches the trust-region solver to a
sparse linear algebra backend (`lsmr`) for the per-iteration linear
solve. Together, the cost of the *solve* -- finding the best-fit
parameters -- grows roughly linearly in the number of targets for a fixed
number of stations, rather than the roughly-cubic cost a dense
finite-difference Jacobian plus a dense linear solve would give at
`O((3 n_targets + 6 n_stations)^3)` per iteration if the block structure
were ignored (CLAUDE.md §3 asks this cost to be noted explicitly: with
`n_targets` targets and `n_obs` observations, ignoring the structure costs
`O(n_obs * (3 n_targets)^2)` just to *evaluate* a dense Jacobian, before
any linear algebra).

Extracting the *covariance* at the end is not given the same treatment:
`solve_network` inverts the full, dense (3 n_targets + 6 (n_stations-1))
normal matrix once, at convergence, to get the joint covariance over
every parameter (then keeps only the target-coordinate block). That's a
one-time `O(n_params^3)` cost, separate from the (near-linear-in-target-
count) iterative solve above. For the target and station counts this
testbed's step 3 gate needs (tens of targets, a handful of stations),
that's negligible; if scenes grow much larger, the next optimisation
would be a Schur-complement reduction -- marginalise the (few) station
parameters out algebraically, leaving only a block-diagonal-per-target
system to invert -- not built here, per CLAUDE.md §6's "optimise later,
once tests pass."

See docs/step3_network_solve_physics.md for the physical explanation of
*why* uncertainty falls as stations are added, and why it flattens out
rather than falling forever.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from geometry.pose import InstrumentPose
from geometry.spherical import cartesian_to_spherical
from instruments.laser_tracker import LaserTracker


@dataclass(frozen=True)
class Observation:
    """One spherical measurement: station `station_index` observing target
    `target_index`."""

    station_index: int
    target_index: int
    d_m: float
    theta_rad: float
    phi_rad: float


@dataclass
class NetworkSolveResult:
    """The outcome of a network solve.

    Attributes
    ----------
    target_points_m : (n_targets, 3) ndarray -- solved target coordinates.
    station_poses : list[InstrumentPose] -- solved station poses, one per
        station, in the same order as the input; `station_poses[anchor_index]`
        is exactly the anchor pose that was held fixed.
    target_covariance_m2 : (3*n_targets, 3*n_targets) ndarray -- the FULL
        joint covariance over every target coordinate, including
        cross-target correlations (two targets' recovered positions are
        correlated whenever their estimates share an uncertain station
        pose) -- not just a per-target block-diagonal approximation.
        CLAUDE.md §3: never reduce this to a scalar before the
        characteristic layer; use `target_point_covariance` for one
        target's 3x3 block, not a diagonal or a trace, when precision
        matters downstream. If the tracker carries a nonzero
        `systematic_std_m`, each target's own 3x3 diagonal block also
        includes that placement-irreducible isotropic term (added by
        `solve_network`, once per target -- see the module docstring's
        "Weighting" section); the cross-target blocks do not, since that
        term is modelled as independent per point, not shared.
    residual_rms : float -- RMS of the final whitened residuals (each
        already divided by its sensor sigma); should be of order 1 if the
        noise model matches the data, as a rough sanity check.
    n_function_evals, success : optimizer diagnostics from `scipy.optimize.
        least_squares`.
    """

    target_points_m: np.ndarray
    station_poses: List[InstrumentPose]
    target_covariance_m2: np.ndarray
    residual_rms: float
    n_function_evals: int
    success: bool

    def target_point_covariance(self, target_index: int) -> np.ndarray:
        """3x3 covariance block for one target, sliced out of the full
        joint covariance."""
        i = 3 * target_index
        return self.target_covariance_m2[i : i + 3, i : i + 3]


def simulate_observations(
    true_target_points_m: np.ndarray,
    true_station_poses: List[InstrumentPose],
    tracker: LaserTracker,
    rng: np.random.Generator,
) -> List[Observation]:
    """Simulate a full network: every station observes every target.

    Each observation is the exact geometric (d, theta, phi) from that
    station to that target, with independent Gaussian noise added on each
    component at the tracker's own sigma_d / sigma_theta / sigma_phi --
    the same spherical noise step 1 propagates to a Cartesian covariance,
    used here in its native, un-propagated form, since that's what an
    actual tracker shot reports.

    No occlusion is modelled here -- this function is for exercising the
    network solve itself. A caller wanting a visibility-filtered network
    (only simulating observations for station/target pairs `geometry.
    visibility.check_target_visibility` reports VISIBLE) can filter the
    returned list, or build one directly; that composition belongs to the
    caller, not this function, per CLAUDE.md §9 (step 3's job is the
    solve, not re-deciding visibility).
    """
    observations = []
    n_targets = true_target_points_m.shape[0]
    for station_index, pose in enumerate(true_station_poses):
        for target_index in range(n_targets):
            point_local_m = pose.to_local(true_target_points_m[target_index])
            d_m, theta_rad, phi_rad = cartesian_to_spherical(point_local_m)
            observations.append(
                Observation(
                    station_index=station_index,
                    target_index=target_index,
                    d_m=d_m + rng.normal(0.0, tracker.sigma_d_m),
                    theta_rad=theta_rad + rng.normal(0.0, tracker.sigma_theta_rad),
                    phi_rad=phi_rad + rng.normal(0.0, tracker.sigma_phi_rad),
                )
            )
    return observations


def exact_observations(
    target_points_m: np.ndarray, station_poses: List[InstrumentPose]
) -> List[Observation]:
    """Every station-target observation, computed exactly -- no noise --
    at the given geometry: every station observes every target.

    This is `simulate_observations`'s noiseless counterpart, for network
    *design* questions ("how precise would this station arrangement be")
    rather than fitting real or simulated data. Feeding these into
    `solve_network` with the same geometry as the initial guess converges
    immediately (every residual is already exactly zero) and returns the
    design covariance -- to numerical precision, the same `(J^T J)^-1` a
    real noisy solve at this geometry would converge to (see CLAUDE.md §5
    step 6 / docs/step6_headline_experiment.md for the direct empirical
    check), without paying for an actual noisy nonlinear fit. Used by
    `planning` to score many candidate station arrangements cheaply and
    deterministically -- no RNG needed during a search, so every
    candidate is compared on equal footing.
    """
    observations = []
    n_targets = target_points_m.shape[0]
    for station_index, pose in enumerate(station_poses):
        for target_index in range(n_targets):
            point_local_m = pose.to_local(target_points_m[target_index])
            d_m, theta_rad, phi_rad = cartesian_to_spherical(point_local_m)
            observations.append(Observation(station_index, target_index, d_m, theta_rad, phi_rad))
    return observations


def perturbed_initial_guess(
    true_target_points_m: np.ndarray,
    true_station_poses: List[InstrumentPose],
    position_noise_m: float,
    rotation_noise_rad: float,
    rng: np.random.Generator,
    anchor_index: int = 0,
):
    """A rough starting point for the solve, built by perturbing the truth.

    Stands in for what a real network solve would seed from -- nominal
    CAD/as-designed positions, or a previous survey -- which this testbed
    doesn't otherwise have, since it controls the ground truth directly.
    `solve_network` itself takes any initial guess; this helper exists so
    tests and demos don't all need to hand-roll one.

    `anchor_index`'s pose is returned exactly, unperturbed -- whatever is
    passed as the initial guess at that index is what `solve_network`
    holds fixed as the frame's origin (see the module docstring's note on
    the frame-of-reference constraint), so perturbing it wouldn't give a
    "rough guess at the anchor", it would silently redefine the global
    frame to a slightly wrong place and offset every recovered target and
    station by that same error. Must match the `anchor_index` later
    passed to `solve_network`.
    """
    target_points_m = true_target_points_m + rng.normal(
        0.0, position_noise_m, size=true_target_points_m.shape
    )
    station_poses = []
    for i, pose in enumerate(true_station_poses):
        if i == anchor_index:
            station_poses.append(pose)
            continue
        position_m = pose.position_m + rng.normal(0.0, position_noise_m, size=3)
        rotvec = Rotation.from_matrix(pose.orientation).as_rotvec()
        rotvec = rotvec + rng.normal(0.0, rotation_noise_rad, size=3)
        station_poses.append(
            InstrumentPose(position_m=position_m, orientation=Rotation.from_rotvec(rotvec).as_matrix())
        )
    return target_points_m, station_poses


def _wrap_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap an angle difference to (-pi, pi], so a residual near the
    +-pi branch cut of atan2 doesn't read as a huge spurious error."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _pack_station_params(pose: InstrumentPose) -> np.ndarray:
    rotvec = Rotation.from_matrix(pose.orientation).as_rotvec()
    return np.concatenate([pose.position_m, rotvec])


def _unpack_station_params(params6: np.ndarray) -> InstrumentPose:
    position_m = params6[:3]
    orientation = Rotation.from_rotvec(params6[3:]).as_matrix()
    return InstrumentPose(position_m=position_m, orientation=orientation)


def _station_param_offsets(n_stations: int, anchor_index: int, n_target_params: int) -> dict:
    """Column offset, in the parameter vector, of each non-anchor station's
    6-parameter block. The anchor station has no entry -- it isn't a free
    parameter."""
    offsets = {}
    offset = n_target_params
    for i in range(n_stations):
        if i == anchor_index:
            continue
        offsets[i] = offset
        offset += 6
    return offsets


def _build_parameter_vector(target_points_m, station_poses, anchor_index) -> np.ndarray:
    blocks = [target_points_m.reshape(-1)]
    for i, pose in enumerate(station_poses):
        if i == anchor_index:
            continue
        blocks.append(_pack_station_params(pose))
    return np.concatenate(blocks)


def _unpack_parameter_vector(params, n_targets, n_stations, anchor_index, anchor_pose):
    target_points_m = params[: 3 * n_targets].reshape(n_targets, 3)
    station_poses = []
    offset = 3 * n_targets
    for i in range(n_stations):
        if i == anchor_index:
            station_poses.append(anchor_pose)
        else:
            station_poses.append(_unpack_station_params(params[offset : offset + 6]))
            offset += 6
    return target_points_m, station_poses


def _residuals(params, observations, n_targets, n_stations, anchor_index, anchor_pose, tracker):
    target_points_m, station_poses = _unpack_parameter_vector(
        params, n_targets, n_stations, anchor_index, anchor_pose
    )
    residuals = np.empty(3 * len(observations))
    for k, obs in enumerate(observations):
        pose = station_poses[obs.station_index]
        point_local_m = pose.to_local(target_points_m[obs.target_index])
        d_pred_m, theta_pred_rad, phi_pred_rad = cartesian_to_spherical(point_local_m)
        residuals[3 * k + 0] = (obs.d_m - d_pred_m) / tracker.sigma_d_m
        residuals[3 * k + 1] = (
            _wrap_angle_rad(obs.theta_rad - theta_pred_rad) / tracker.sigma_theta_rad
        )
        residuals[3 * k + 2] = (obs.phi_rad - phi_pred_rad) / tracker.sigma_phi_rad
    return residuals


def _build_jac_sparsity(observations, n_targets, n_stations, anchor_index):
    n_target_params = 3 * n_targets
    n_station_params = 6 * (n_stations - 1)
    n_params = n_target_params + n_station_params
    n_residuals = 3 * len(observations)
    station_param_offset = _station_param_offsets(n_stations, anchor_index, n_target_params)

    rows, cols = [], []
    for k, obs in enumerate(observations):
        row_base = 3 * k
        target_col_base = 3 * obs.target_index
        for r in range(3):
            for c in range(3):
                rows.append(row_base + r)
                cols.append(target_col_base + c)
        station_col_base = station_param_offset.get(obs.station_index)
        if station_col_base is not None:
            for r in range(3):
                for c in range(6):
                    rows.append(row_base + r)
                    cols.append(station_col_base + c)

    data = np.ones(len(rows), dtype=bool)
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_residuals, n_params)).tocsr()


def solve_network(
    observations: List[Observation],
    initial_target_points_m: np.ndarray,
    initial_station_poses: List[InstrumentPose],
    tracker: LaserTracker,
    anchor_index: int = 0,
) -> NetworkSolveResult:
    """Solve for every target's coordinates and every non-anchor station's
    pose jointly, by weighted nonlinear least squares over `observations`.

    See the module docstring for the frame-of-reference constraint
    (`anchor_index`'s pose is held fixed, not solved for), the weighting
    (each residual divided by the tracker's own sigma_d/theta/phi), and
    how the block-diagonal observation structure is exploited.
    """
    n_targets = initial_target_points_m.shape[0]
    n_stations = len(initial_station_poses)
    anchor_pose = initial_station_poses[anchor_index]

    x0 = _build_parameter_vector(initial_target_points_m, initial_station_poses, anchor_index)
    jac_sparsity = _build_jac_sparsity(observations, n_targets, n_stations, anchor_index)

    fit = least_squares(
        _residuals,
        x0,
        jac_sparsity=jac_sparsity,
        method="trf",
        args=(observations, n_targets, n_stations, anchor_index, anchor_pose, tracker),
    )

    target_points_m, station_poses = _unpack_parameter_vector(
        fit.x, n_targets, n_stations, anchor_index, anchor_pose
    )

    # Covariance of the fitted parameters. Residuals are already whitened
    # (each divided by its sensor sigma in _residuals), so the weighted
    # normal matrix is simply J^T J -- no separate weight matrix to fold
    # in. We take Cov(params) = (J^T J)^+ directly, trusting the input
    # sigma_d/theta/phi as the real measurement uncertainty, rather than
    # rescaling by an empirical chi-square-per-degree-of-freedom factor:
    # this codebase treats sensor noise as measured, not fitted, at every
    # other layer (CLAUDE.md §4e), and doing otherwise here would be
    # quietly recalibrating the noise model against its own residuals.
    # residual_rms below is exactly the diagnostic that would reveal a
    # sigma mismatch, without silently absorbing it into the covariance.
    jac = fit.jac
    if scipy.sparse.issparse(jac):
        normal_matrix = (jac.T @ jac).toarray()
    else:
        normal_matrix = jac.T @ jac
    full_covariance = np.linalg.pinv(normal_matrix)
    target_covariance_m2 = full_covariance[: 3 * n_targets, : 3 * n_targets].copy()

    # tracker.systematic_std_m (instruments.laser_tracker.LaserTracker) is
    # not a spherical-observation noise term -- it never entered
    # `_residuals`' weighting above, on purpose, for the same reason
    # `LaserTracker.covariance_local` adds it *after* the spherical->
    # Cartesian Jacobian rather than folding it into sigma_d/theta/phi: it
    # isn't modelled as arising from range/angle noise in the first place
    # (calibration-parameter uncertainty, thermal drift, refraction, SMR/
    # nest repeatability -- see that field's docstring). It has to be
    # added here too, once per target, in exactly the same isotropic
    # Cartesian form, or it silently vanishes the moment a scenario has
    # more than one station: a caller building a `PlanningScenario` with a
    # nonzero-systematic_std_m tracker (CLAUDE.md referee note M6) would
    # see no change at all in a multi-station plan's covariance, even
    # though `LaserTracker.covariance_local` alone would show one -- the
    # network solve would be silently throwing away a term the instrument
    # model reports having. Adding it once per target, independent of how
    # many stations observed it, is exactly "irreducible by placement": no
    # station arrangement can make this term smaller, so it should not get
    # smaller just because a network solve, rather than a single shot, was
    # used to reach this target's coordinate. This also keeps
    # `test_single_station_network_solve_matches_step1_covariance` true
    # for a nonzero systematic term: with one station, this is exactly
    # what `LaserTracker.covariance_global` already does, so the network
    # solve continues to reduce to step 1's answer in that limit.
    if tracker.systematic_std_m > 0.0:
        systematic_variance_m2 = tracker.systematic_std_m**2
        for i in range(n_targets):
            block = slice(3 * i, 3 * i + 3)
            target_covariance_m2[block, block] += systematic_variance_m2 * np.eye(3)

    return NetworkSolveResult(
        target_points_m=target_points_m,
        station_poses=station_poses,
        target_covariance_m2=target_covariance_m2,
        residual_rms=float(np.sqrt(np.mean(fit.fun**2))),
        n_function_evals=fit.nfev,
        success=bool(fit.success),
    )


def mean_positional_uncertainty_m(result: NetworkSolveResult) -> float:
    """A single scalar summarising overall network quality: the RMS, over
    all targets, of each target's RSS positional uncertainty
    (sqrt(trace(3x3 covariance block))).

    This is a *reporting* convenience for comparing solves at a glance
    (e.g. plotting uncertainty against station count, mirroring the
    single numbers Wang, Forbes & Maropoulos 2014 report) -- not a
    replacement for the full per-target 3x3 covariance, still returned in
    full by `NetworkSolveResult.target_covariance_m2`. CLAUDE.md §3's
    "never reduce to a scalar" is about the physics chain itself (an
    instrument model, a network solve, a characteristic uncertainty must
    never *only* carry a scalar) -- it doesn't forbid a labelled summary
    statistic for a plot or a headline number, provided the full
    covariance remains what's actually passed downstream.
    """
    n_targets = result.target_points_m.shape[0]
    per_target_rss_m = np.array(
        [np.sqrt(np.trace(result.target_point_covariance(i))) for i in range(n_targets)]
    )
    return float(np.sqrt(np.mean(per_target_rss_m**2)))
