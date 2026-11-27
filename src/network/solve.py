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
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from geometry.pose import InstrumentPose
from geometry.spherical import cartesian_to_spherical
from instruments.laser_tracker import LaserTracker
from instruments.npl_model import (
    BACK, FRONT, INDEX, N_PARAMETERS, PARAMETER_NAMES, NPLErrorParameters, reading_sensitivities, true_to_observed_batch,
)


@dataclass(frozen=True)
class Observation:
    """One spherical measurement: station `station_index` observing target
    `target_index`."""

    station_index: int
    target_index: int
    d_m: float
    theta_rad: float
    phi_rad: float
    face: str = FRONT  # "front" or "back": which face of the tracker took the shot (instruments.npl_model)


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
    parameter_estimates : estimate mode only -- one `NPLErrorParameters` per
        instrument id (sorted): the fitted values and their posterior
        covariance (zero rows and columns for parameters held fixed).
    full_covariance_m2 : estimate mode only -- the covariance of every unknown,
        in the order targets (3 each), non-anchor station poses (position,
        rotation vector: 6 each), then for each instrument its estimated
        parameters, in `PARAMETER_NAMES` order.
    n_parameters_estimated : parameters estimated per instrument.
    """

    target_points_m: np.ndarray
    station_poses: List[InstrumentPose]
    target_covariance_m2: np.ndarray
    residual_rms: float
    n_function_evals: int
    success: bool
    parameter_estimates: Optional[List[NPLErrorParameters]] = None
    full_covariance_m2: Optional[np.ndarray] = None
    n_parameters_estimated: int = 0

    def target_point_covariance(self, target_index: int) -> np.ndarray:
        """3x3 covariance block for one target, sliced out of the full
        joint covariance."""
        i = 3 * target_index
        return self.target_covariance_m2[i : i + 3, i : i + 3]

    def joint_point_covariance(self, indices: Sequence[int]) -> np.ndarray:
        """(3m, 3m) joint covariance of the m targets `indices`, in that order,
        cross-covariances included. What a quantity defined *relative to other
        points* needs (`characteristics.characteristic.
        evaluate_relative_characteristic`): the per-target blocks of
        `target_point_covariance` cannot express that errors shared by several
        targets cancel in their difference. Note the tracker's per-point
        `systematic_std_m` term, if any, is on each target's own diagonal block
        here, as in `target_covariance_m2`."""
        rows = np.concatenate([[3 * int(i) + c for c in range(3)] for i in indices])
        return self.target_covariance_m2[np.ix_(rows, rows)]


def simulate_observations(
    true_target_points_m: np.ndarray,
    true_station_poses: List[InstrumentPose],
    tracker: LaserTracker,
    rng: np.random.Generator,
    instrument_params: Optional[Sequence[NPLErrorParameters]] = None,
    instrument_ids: Optional[Sequence[int]] = None,
    faces: Optional[Sequence[Sequence[str]]] = None,
    visible: Optional[np.ndarray] = None,
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

    **With `instrument_params`** (a sequence of `NPLErrorParameters`, one per
    physical tracker, ordered by sorted instrument id) the readings are what
    those *real* instruments would report -- `instruments.npl_model.
    true_to_observed`, front face or back per `faces` (default front only) --
    plus the same random noise. `visible` (S, N) restricts which shots are
    taken. The noise-only branch below is unchanged.
    """
    if instrument_params is not None or faces is not None or visible is not None:
        return _simulate_with_instruments(true_target_points_m, true_station_poses, tracker, rng,
                                          instrument_params, instrument_ids, faces, visible)
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
    instrument_ids: Optional[Sequence[int]] = None,
    estimate_parameters: Union[None, bool, "ParameterPrior"] = None,
) -> NetworkSolveResult:
    """Solve for every target's coordinates and every non-anchor station's
    pose jointly, by weighted nonlinear least squares over `observations`.

    See the module docstring for the frame-of-reference constraint
    (`anchor_index`'s pose is held fixed, not solved for), the weighting
    (each residual divided by the tracker's own sigma_d/theta/phi), and
    how the block-diagonal observation structure is exploited.

    **Geometric parameters** (`instruments.npl_model`, `docs/step1b_npl_geometric_model.md`).
    With `tracker.geometric_params` unset (or zero), every observation on the
    front face and `estimate_parameters` unset, this is the noise-only solve,
    unchanged. Otherwise `_solve_extended` runs, in one of two modes:

      * **consider** (a tracker with `geometric_params`, `estimate_parameters`
        unset): readings are corrected with the parameters' values, which are not
        re-estimated; their covariance enters as Sigma = N^-1 + S V S^T with
        S = -N^-1 J^T W H (H the readings' sensitivity to the parameters,
        W the whitening) -- the same quantity `network.design` computes
        analytically (tests).
      * **estimate** (`estimate_parameters` a `ParameterPrior`, or True to use
        `tracker.geometric_params` as the prior): the parameters of each
        physical tracker are solved together with the network, under the prior
        (the paper's equation 18). The covariance comes from the full normal
        matrix, after a Jacobi-scaled rank test: a network that cannot
        determine a parameter or a pose raises `NetworkRankError` naming the
        weakest directions, instead of `pinv`'s zero variance.

    `instrument_ids` (one per station, default all the same tracker) says which
    stations share a physical instrument, and so one parameter vector.
    """
    if _extended_solve_needed(observations, tracker, estimate_parameters):
        return _solve_extended(observations, initial_target_points_m, initial_station_poses, tracker,
                               anchor_index, instrument_ids, estimate_parameters)
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


# =============================================================================================
# Geometric parameters: faces, instrument identity, consider and estimate modes
# =============================================================================================
class NetworkRankError(ValueError):
    """The network does not determine every unknown it was asked to solve for:
    a gauge freedom (a station not tied to the anchor) or a parameter the data
    cannot separate (e.g. the range scale mu with no length reference and no
    prior). `weak_directions` names the unknowns that carry the weakest
    direction of the Jacobi-scaled normal matrix."""

    def __init__(self, message: str, weak_directions: List[str]):
        super().__init__(message)
        self.weak_directions = weak_directions


@dataclass(frozen=True, eq=False)
class ParameterPrior:
    """What is known about one tracker's 16 geometric parameters before the
    network test (the paper's prior term (Bc - c0)^T (Bc - c0), equation 18).

    `mean` (16,); `covariance` (16, 16) for the parameters that have a prior;
    `flat` (16,) bool: no prior at all (estimated freely); `fixed` (16,) bool:
    held at `mean`, not estimated. A parameter is in exactly one of the three
    states. Cross-covariances between a prior-carrying parameter and a flat or
    fixed one are ignored.
    """

    mean: np.ndarray
    covariance: np.ndarray
    flat: np.ndarray
    fixed: np.ndarray

    @classmethod
    def flat_except(cls, prior_std: Optional[Dict[str, float]] = None, fixed: Sequence[str] = (),
                    mean: Optional[np.ndarray] = None) -> "ParameterPrior":
        """Every parameter flat, except those named in `prior_std` (a
        standard-deviation prior, SI units) and those in `fixed` (held at the
        mean, default zero). The paper's network test is
        `flat_except({"mu": 5e-7})`: the scale is traceable to the laser
        wavelength, nothing else is known in advance."""
        flat = np.ones(N_PARAMETERS, bool)
        fixed_mask = np.zeros(N_PARAMETERS, bool)
        covariance = np.zeros((N_PARAMETERS, N_PARAMETERS))
        for name, std in (prior_std or {}).items():
            flat[INDEX[name]] = False
            covariance[INDEX[name], INDEX[name]] = std**2
        for name in fixed:
            flat[INDEX[name]], fixed_mask[INDEX[name]] = False, True
        return cls(np.zeros(N_PARAMETERS) if mean is None else np.asarray(mean, float), covariance, flat, fixed_mask)

    @classmethod
    def from_parameters(cls, params: NPLErrorParameters) -> "ParameterPrior":
        """The prior a tracker's own parameter set states: its values and
        covariance, parameters of zero variance held fixed."""
        fixed = np.diag(params.covariance) == 0.0
        return cls(params.values.copy(), params.covariance.copy(), np.zeros(N_PARAMETERS, bool), fixed)


def _extended_solve_needed(observations, tracker: LaserTracker, estimate_parameters) -> bool:
    params = tracker.geometric_params
    if estimate_parameters is not None and estimate_parameters is not False:
        return True
    if params is not None and (params.values.any() or params.covariance.any()):
        return True
    return any(o.face != FRONT for o in observations)


def _simulate_with_instruments(targets, poses, tracker, rng, instrument_params, instrument_ids, faces, visible):
    n_stations, n_targets = len(poses), targets.shape[0]
    ids = np.zeros(n_stations, int) if instrument_ids is None else np.asarray(instrument_ids, dtype=int)
    unique = sorted(set(ids.tolist()))
    slot = {i: n for n, i in enumerate(unique)}
    if instrument_params is None:
        instrument_params = [NPLErrorParameters.zero()] * len(unique)
    if len(instrument_params) != len(unique):
        raise ValueError(f"instrument_params needs one entry per instrument id ({len(unique)}), got {len(instrument_params)}")
    faces = [(FRONT,)] * n_stations if faces is None else [tuple(f) for f in faces]
    vis = np.ones((n_stations, n_targets), bool) if visible is None else np.asarray(visible, dtype=bool)
    observations = []
    for s, pose in enumerate(poses):
        t_idx = np.nonzero(vis[s])[0]
        local = np.array([pose.to_local(targets[t]) for t in t_idx])
        values = instrument_params[slot[int(ids[s])]].values
        for face in faces[s]:
            readings = true_to_observed_batch(local, values, face)
            for t, (d, theta, phi) in zip(t_idx, readings):
                observations.append(Observation(
                    s, int(t), d + rng.normal(0.0, tracker.sigma_d_m), theta + rng.normal(0.0, tracker.sigma_theta_rad),
                    phi + rng.normal(0.0, tracker.sigma_phi_rad), face))
    return observations


@dataclass
class _Layout:
    """Where each unknown lives in the parameter vector: targets, non-anchor
    station poses, then each instrument's estimated parameters."""

    n_targets: int
    n_stations: int
    anchor_index: int
    station_offset: Dict[int, int]
    inst_slot_of_station: np.ndarray   # (S,) instrument slot of each station
    n_instruments: int
    estimated: np.ndarray              # parameter indices estimated per instrument (may be empty)
    param_offset: int                  # first parameter column
    n_unknowns: int

    def instrument_columns(self, slot: int) -> np.ndarray:
        n = len(self.estimated)
        return self.param_offset + slot * n + np.arange(n)


def _make_layout(n_targets, n_stations, anchor_index, instrument_ids, estimated: np.ndarray):
    ids = np.zeros(n_stations, int) if instrument_ids is None else np.asarray(instrument_ids, dtype=int)
    if ids.shape != (n_stations,):
        raise ValueError(f"instrument_ids needs one entry per station ({n_stations}), got shape {ids.shape}")
    unique = sorted(set(ids.tolist()))
    slot = {i: n for n, i in enumerate(unique)}
    inst_slot = np.array([slot[int(i)] for i in ids])
    n_target_params = 3 * n_targets
    station_offset = _station_param_offsets(n_stations, anchor_index, n_target_params)
    param_offset = n_target_params + 6 * (n_stations - 1)
    n_unknowns = param_offset + len(unique) * len(estimated)
    return _Layout(n_targets, n_stations, anchor_index, station_offset, inst_slot, len(unique), estimated, param_offset, n_unknowns), unique


def _unpack_extended(x, layout: _Layout, anchor_pose: InstrumentPose, base_values: np.ndarray):
    targets = x[: 3 * layout.n_targets].reshape(layout.n_targets, 3)
    positions = np.empty((layout.n_stations, 3))
    orientations = np.empty((layout.n_stations, 3, 3))
    free = [s for s in range(layout.n_stations) if s != layout.anchor_index]
    positions[layout.anchor_index], orientations[layout.anchor_index] = anchor_pose.position_m, anchor_pose.orientation
    if free:
        blocks = np.stack([x[layout.station_offset[s]: layout.station_offset[s] + 6] for s in free])
        positions[free] = blocks[:, :3]
        orientations[free] = Rotation.from_rotvec(blocks[:, 3:]).as_matrix()
    values = np.repeat(base_values[None, :], layout.n_instruments, axis=0) if base_values.ndim == 1 else base_values.copy()
    for slot in range(layout.n_instruments):
        values[slot, layout.estimated] = x[layout.instrument_columns(slot)]
    return targets, positions, orientations, values


def _extended_residuals(x, layout, anchor_pose, base_values, obs_s, obs_t, obs_back, readings, sigma, prior_rows):
    targets, positions, orientations, values = _unpack_extended(x, layout, anchor_pose, base_values)
    local = np.einsum("oji,oj->oi", orientations[obs_s], targets[obs_t] - positions[obs_s])
    predicted = true_to_observed_batch(local, values[layout.inst_slot_of_station[obs_s]], np.where(obs_back, BACK, FRONT))
    diff = readings - predicted
    diff[:, 1] = (diff[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    residual = (diff / sigma).reshape(-1)
    if prior_rows is not None:
        parts = [residual]
        for slot in range(layout.n_instruments):
            c = values[slot]
            parts.append(prior_rows["l_inv"] @ (c[prior_rows["index"]] - prior_rows["mean"]))
        residual = np.concatenate(parts)
    return residual


def _extended_sparsity(layout, obs_s, obs_t, n_prior_rows_per_instrument, n_prior_params):
    n_obs = len(obs_s)
    n_rows = 3 * n_obs + layout.n_instruments * n_prior_rows_per_instrument
    rows, cols = [], []
    for k in range(n_obs):
        s, t = int(obs_s[k]), int(obs_t[k])
        station = layout.station_offset.get(s)
        param_cols = layout.instrument_columns(int(layout.inst_slot_of_station[s]))
        for r in range(3):
            row = 3 * k + r
            rows.extend([row] * 3)
            cols.extend(range(3 * t, 3 * t + 3))
            if station is not None:
                rows.extend([row] * 6)
                cols.extend(range(station, station + 6))
            rows.extend([row] * len(param_cols))
            cols.extend(param_cols.tolist())
    for slot in range(layout.n_instruments):
        cols_slot = layout.instrument_columns(slot)
        position = [int(np.nonzero(layout.estimated == pi)[0][0]) for pi in n_prior_params]  # prior parameters are estimated ones
        for r in range(n_prior_rows_per_instrument):
            row = 3 * n_obs + slot * n_prior_rows_per_instrument + r
            rows.extend([row] * len(position))
            cols.extend(cols_slot[position].tolist())
    data = np.ones(len(rows), dtype=bool)
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, layout.n_unknowns)).tocsr()


def _unknown_label(j: int, layout: _Layout, instrument_ids_sorted) -> str:
    if j < 3 * layout.n_targets:
        return f"target {j // 3} {'xyz'[j % 3]}"
    if j < layout.param_offset:
        free = [s for s in range(layout.n_stations) if s != layout.anchor_index]
        k, r = divmod(j - 3 * layout.n_targets, 6)
        return f"station {free[k]} {['x', 'y', 'z', 'rotation x', 'rotation y', 'rotation z'][r]}"
    n = len(layout.estimated)
    slot, r = divmod(j - layout.param_offset, n)
    return f"instrument {instrument_ids_sorted[slot]} {PARAMETER_NAMES[int(layout.estimated[r])]}"


def _checked_inverse(normal: np.ndarray, layout: _Layout, instrument_ids_sorted) -> np.ndarray:
    """Inverse of the normal matrix after a Jacobi-scaled rank test (the one
    `network.design` uses): a nearly singular matrix means the data cannot
    determine every unknown, which `pinv` would hide as zero variance."""
    from network.design import CONNECTIVITY_EIGENVALUE_FLOOR

    diag = np.diag(normal)
    scale = 1.0 / np.sqrt(np.where(diag > 0.0, diag, 1.0))
    scaled = normal * scale[:, None] * scale[None, :]
    values, vectors = np.linalg.eigh(scaled)
    if np.any(diag <= 0.0) or values[0] < CONNECTIVITY_EIGENVALUE_FLOOR:
        weak = vectors[:, 0]
        labels = [_unknown_label(int(j), layout, instrument_ids_sorted) for j in np.argsort(-np.abs(weak))[:6] if abs(weak[j]) > 0.15]
        if np.any(diag <= 0.0):
            labels = [_unknown_label(int(j), layout, instrument_ids_sorted) for j in np.nonzero(diag <= 0.0)[0][:6]]
        raise NetworkRankError(
            "the network does not determine every unknown (smallest Jacobi-scaled eigenvalue "
            f"{values[0]:.2e}, floor {CONNECTIVITY_EIGENVALUE_FLOOR:.0e}); weakest directions involve: " + ", ".join(labels),
            labels)
    return (vectors * (1.0 / values)) @ vectors.T * scale[:, None] * scale[None, :]


def _solve_extended(observations, initial_targets, initial_poses, tracker, anchor_index, instrument_ids, estimate_parameters):
    """`solve_network` with faces, instrument identity, and either the consider
    covariance or estimated parameters (see `solve_network`)."""
    n_targets, n_stations = initial_targets.shape[0], len(initial_poses)
    anchor_pose = initial_poses[anchor_index]
    params = tracker.geometric_params
    estimate = estimate_parameters is not None and estimate_parameters is not False
    if estimate:
        if estimate_parameters is True:
            if params is None:
                raise ValueError("estimate_parameters=True needs tracker.geometric_params as the prior")
            prior = ParameterPrior.from_parameters(params)
        else:
            prior = estimate_parameters
        estimated = np.nonzero(~prior.fixed)[0]
        base_values = prior.mean.copy()
    else:
        prior = None
        estimated = np.zeros(0, dtype=int)
        base_values = np.zeros(N_PARAMETERS) if params is None else params.values.copy()
    layout, ids_sorted = _make_layout(n_targets, n_stations, anchor_index, instrument_ids, estimated)

    obs_s = np.array([o.station_index for o in observations])
    obs_t = np.array([o.target_index for o in observations])
    obs_back = np.array([o.face == BACK for o in observations])
    bad = [o.face for o in observations if o.face not in (FRONT, BACK)]
    if bad:
        raise ValueError(f"face must be '{FRONT}' or '{BACK}', got {bad[0]!r}")
    readings = np.array([[o.d_m, o.theta_rad, o.phi_rad] for o in observations])
    sigma = np.array([tracker.sigma_d_m, tracker.sigma_theta_rad, tracker.sigma_phi_rad])

    prior_rows, n_prior_rows, prior_params = None, 0, np.zeros(0, int)
    if estimate:
        prior_params = np.nonzero(~prior.flat & ~prior.fixed)[0]
        if len(prior_params):
            sub = prior.covariance[np.ix_(prior_params, prior_params)]
            prior_rows = {"index": prior_params, "mean": prior.mean[prior_params], "l_inv": np.linalg.inv(np.linalg.cholesky(sub))}
            n_prior_rows = len(prior_params)
    x0 = np.concatenate([
        initial_targets.reshape(-1),
        *[_pack_station_params(pose) for i, pose in enumerate(initial_poses) if i != anchor_index],
        *([base_values[estimated]] * layout.n_instruments if len(estimated) else []),
    ])
    args = (layout, anchor_pose, base_values, obs_s, obs_t, obs_back, readings, sigma, prior_rows if estimate else None)
    sparsity = _extended_sparsity(layout, obs_s, obs_t, n_prior_rows, prior_params) if len(estimated) else None
    if sparsity is None:  # no parameter unknowns: the same sparsity as the noise-only solve
        sparsity = _build_jac_sparsity(observations, n_targets, n_stations, anchor_index)
    fit = least_squares(_extended_residuals, x0, jac_sparsity=sparsity, method="trf", x_scale="jac", args=args)

    jac = fit.jac.toarray() if scipy.sparse.issparse(fit.jac) else fit.jac
    normal = jac.T @ jac
    targets, positions, orientations, values = _unpack_extended(fit.x, layout, anchor_pose, base_values)
    poses = [InstrumentPose(position_m=positions[i].copy(), orientation=orientations[i].copy()) for i in range(n_stations)]
    parameter_estimates, full_covariance = None, None
    if estimate:
        full_covariance = _checked_inverse(normal, layout, ids_sorted)
        target_covariance = full_covariance[: 3 * n_targets, : 3 * n_targets].copy()
        parameter_estimates = []
        for slot in range(layout.n_instruments):
            cols = layout.instrument_columns(slot)
            covariance = np.zeros((N_PARAMETERS, N_PARAMETERS))
            covariance[np.ix_(estimated, estimated)] = full_covariance[np.ix_(cols, cols)]
            parameter_estimates.append(NPLErrorParameters(values[slot], covariance, f"fitted, instrument {ids_sorted[slot]}"))
    else:
        full_covariance_u = np.linalg.pinv(normal)
        target_covariance = full_covariance_u[: 3 * n_targets, : 3 * n_targets].copy()
        if params is not None and params.covariance.any():
            target_covariance += _consider_term(jac, full_covariance_u, layout, positions, orientations, targets, obs_s, obs_t,
                                                obs_back, sigma, params)
    if tracker.systematic_std_m > 0.0:
        for i in range(n_targets):
            block = slice(3 * i, 3 * i + 3)
            target_covariance[block, block] += tracker.systematic_std_m**2 * np.eye(3)
    return NetworkSolveResult(
        target_points_m=targets, station_poses=poses, target_covariance_m2=target_covariance,
        residual_rms=float(np.sqrt(np.mean(fit.fun[: 3 * len(observations)] ** 2))), n_function_evals=fit.nfev,
        success=bool(fit.success), parameter_estimates=parameter_estimates, full_covariance_m2=full_covariance,
        n_parameters_estimated=len(estimated),
    )


def _consider_term(jac, covariance_u, layout, positions, orientations, targets, obs_s, obs_t, obs_back, sigma, params):
    """S V S^T for the targets: S = -N^-1 J^T W H, H the readings' sensitivity
    to each tracker's parameters at the solved geometry (see `solve_network`)."""
    local = np.einsum("oji,oj->oi", orientations[obs_s], targets[obs_t] - positions[obs_s])
    dr_dc = reading_sensitivities(local, params.values, np.where(obs_back, BACK, FRONT))[1]  # (O, 3, 16)
    n_obs = len(obs_s)
    h = np.zeros((3 * n_obs, layout.n_instruments * N_PARAMETERS))
    whitened = dr_dc / sigma[None, :, None]
    for k in range(n_obs):
        c0 = N_PARAMETERS * int(layout.inst_slot_of_station[obs_s[k]])
        h[3 * k:3 * k + 3, c0:c0 + N_PARAMETERS] = whitened[k]
    sensitivity = -covariance_u @ jac.T @ h  # (unknowns, 16 K)
    s_targets = sensitivity[: 3 * layout.n_targets]
    v_full = np.kron(np.eye(layout.n_instruments), params.covariance)
    return s_targets @ v_full @ s_targets.T
