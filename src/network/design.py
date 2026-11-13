"""Fast analytic *design* covariance for a multi-station network (CLAUDE.md
§3: "exploit the block-diagonal structure before reaching for surrogates").

`network.solve.solve_network` answers the same question by running a
nonlinear least-squares fit and inverting a dense normal matrix. That is
right for fitting real or simulated data, but wasteful for *planning*,
where observations are the exact, noiseless ones at the true geometry
(`network.solve.exact_observations`), the fit converges in one evaluation,
and all that is wanted is the covariance `(J^T J)^-1` at that geometry. With
~500 targets the dense route inverts a ~1500 x 1500 matrix per candidate
plan. This module gets the same numbers directly, from the block structure
alone, and is the fast baseline any later surrogate-model speedup claim
must be measured against (a slow baseline would inflate that claim).

**The derivation, in metrology terms.** One shot of target t from station s
carries a Cartesian information matrix

    Lambda_ts = inverse of the step-1 single-shot covariance (global frame)

(`instruments.laser_tracker.LaserTracker.covariance_global`): strong along
the beam, weak across it. Treat each shot as a pseudo-measurement of the
target's position with that information; the only way station s's *own*
pose error (3 position + 3 rotation parameters) enters is by shifting and
rotating that pseudo-measurement:

    x_hat(shot) = c_s + R_s p_measured
    d x_hat / d c_s      = I
    d x_hat / d(rot_s)   = -[x_t - c_s]_x      (small global rotation)

Stack every shot and the normal matrix has the block-arrow form

    N = [[A, B],       A  : block-diagonal, one 3x3 per target
         [B^T, D]]     D  : block-diagonal, one 6x6 per free station
                       B  : couples a target only to stations that saw it

(A, D and B are accumulated shot by shot.) The target covariance is the
target block of N^-1. Eliminating the targets first leaves a small
(6 x number-of-free-stations)-square matrix

    D' = D - B^T A^-1 B           (information about the station poses)

and, by block inversion, each target's own 3x3 covariance is

    Cov_t = A_t^-1 + A_t^-1 B_t D'^-1 B_t^T A_t^-1

which needs only that target's own row of B. Cost: linear in targets,
cubic only in the (tiny) number of stations.

**The anchor / datum.** Exactly as `network.solve` (see its module
docstring), one station's pose is held fixed -- its frame *is* the global
frame -- and the covariance is datum-dependent: a different anchor is a
different datum, not a bug.

**Rank deficiency, caught instead of hidden.** A network whose free
stations are not tied (through shared targets) to the anchor has a gauge
freedom: move that sub-network rigidly and every observation is unchanged.
`solve_network` silently handles this with `np.linalg.pinv`, which assigns
the unconstrained direction *zero* variance -- the opposite of "unknown".
Here that shows up as a singular `D'`; it is detected (after Jacobi
scaling, so the test is scale-free) and reported as `connected=False`
rather than returned as a number that looks fine.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from geometry.pose import InstrumentPose
from instruments.laser_tracker import LaserTracker

# Smallest allowed eigenvalue of the Jacobi-scaled station-pose information
# D'. An exact gauge freedom gives ~1e-16; a genuinely weak but valid
# geometry (e.g. three nearly collinear shared targets) gives something
# many orders larger, and is *supposed* to come out as a large uncertainty,
# not as a rejected plan.
CONNECTIVITY_EIGENVALUE_FLOOR = 1e-9


@dataclass(frozen=True)
class DesignCovarianceResult:
    """Outcome of `design_point_covariances`.

    Attributes
    ----------
    point_covariance_m2 : (N, 3, 3) -- each target's marginal covariance in
        the global frame; NaN for targets no station saw, and for every
        target when `connected` is False.
    observed : (N,) bool -- target seen by at least one station.
    connected : False if some free station is not rigidly tied to the
        anchor through shared targets (gauge freedom -- see module
        docstring).
    n_observations : number of station-target shots used.
    """

    point_covariance_m2: np.ndarray
    observed: np.ndarray
    connected: bool
    n_observations: int


def _skew(v: np.ndarray) -> np.ndarray:
    """(O, 3) vectors -> (O, 3, 3) cross-product matrices [v]_x."""
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def shot_information_global(
    delta_global_m: np.ndarray, orientations: np.ndarray, tracker: LaserTracker
) -> np.ndarray:
    """Global-frame 3x3 information matrix of each shot.

    `delta_global_m` (O, 3): vector from station to target, global frame.
    `orientations` (O, 3, 3): the observing station's rotation matrix per
    shot. Equals the inverse of `LaserTracker.covariance_global` for that
    shot -- computed in closed form from the local spherical frame
    (radial / azimuthal / elevation unit vectors, which are orthogonal --
    `geometry.spherical.spherical_jacobian`), so no matrix inversion is
    needed to get it.
    """
    p_local = np.einsum("oji,oj->oi", orientations, delta_global_m)  # R^T delta
    d = np.linalg.norm(p_local, axis=1)
    theta = np.arctan2(p_local[:, 1], p_local[:, 0])
    phi = np.arcsin(np.clip(p_local[:, 2] / d, -1.0, 1.0))
    cos_t, sin_t, cos_p, sin_p = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)

    frame = np.zeros((len(d), 3, 3))  # columns: radial, azimuthal, elevation
    frame[:, :, 0] = np.stack([cos_p * cos_t, cos_p * sin_t, sin_p], axis=1)
    frame[:, :, 1] = np.stack([-sin_t, cos_t, np.zeros_like(d)], axis=1)
    frame[:, :, 2] = np.stack([-sin_p * cos_t, -sin_p * sin_t, cos_p], axis=1)

    info_diag = np.stack(
        [
            np.full_like(d, 1.0 / tracker.sigma_d_m**2),
            1.0 / (d * np.maximum(cos_p, 1e-12) * tracker.sigma_theta_rad) ** 2,
            1.0 / (d * tracker.sigma_phi_rad) ** 2,
        ],
        axis=1,
    )
    info_local = np.einsum("oik,ok,ojk->oij", frame, info_diag, frame)
    return np.einsum("oia,oab,ojb->oij", orientations, info_local, orientations)


def design_point_covariances(
    target_points_m: np.ndarray,
    station_positions_m: np.ndarray,
    station_orientations: np.ndarray,
    tracker: LaserTracker,
    visible: np.ndarray,
    anchor_index: int = 0,
) -> DesignCovarianceResult:
    """Per-target design covariance for the shots in `visible` (S, N).

    Matches `network.solve.solve_network` fed with `exact_observations`
    restricted to the visible pairs (tests/test_design_covariance.py pins
    this), without the nonlinear fit or the dense inversion. Does not add
    `tracker.systematic_std_m`: that term is per-point and isotropic, so
    `projected_uncertainty_m` adds it after projection.
    """
    targets = np.asarray(target_points_m, dtype=float)
    positions = np.asarray(station_positions_m, dtype=float)
    orientations = np.asarray(station_orientations, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    n_stations, n_targets = visible.shape

    cov = np.full((n_targets, 3, 3), np.nan)
    observed = visible.any(axis=0)
    s_idx, t_idx = np.nonzero(visible)
    n_obs = len(s_idx)
    if n_obs == 0:
        return DesignCovarianceResult(cov, observed, True, 0)

    delta = targets[t_idx] - positions[s_idx]
    rot = orientations[s_idx]
    lam = shot_information_global(delta, rot, tracker)  # (O, 3, 3)

    a_info = np.zeros((n_targets, 3, 3))
    np.add.at(a_info, t_idx, lam)

    free = [s for s in range(n_stations) if s != anchor_index]
    n_free = len(free)
    a_inv = np.zeros((n_targets, 3, 3))
    a_inv[observed] = np.linalg.inv(a_info[observed])

    if n_free == 0:
        cov[observed] = a_inv[observed]
        return DesignCovarianceResult(cov, observed, True, n_obs)

    g_station = np.concatenate([np.broadcast_to(np.eye(3), (n_obs, 3, 3)), -_skew(delta)], axis=2)
    lam_g = np.einsum("oij,ojk->oik", lam, g_station)  # (O, 3, 6)

    n_cols = 6 * n_free
    b_full = np.zeros((n_targets, 3, n_cols))
    d_info = np.zeros((n_cols, n_cols))
    for k, s in enumerate(free):
        sel = s_idx == s
        cols = slice(6 * k, 6 * k + 6)
        b_full[t_idx[sel], :, cols] = -lam_g[sel]
        d_info[cols, cols] = np.einsum("oji,ojk->ik", g_station[sel], lam_g[sel])

    b_obs = b_full[observed]
    ainv_obs = a_inv[observed]
    # m_t = A_t^-1 B_t  (3 x cols); B^T A^-1 B = sum_t B_t^T m_t, one matrix
    # product over the stacked (3 N) x cols arrays
    m = np.matmul(ainv_obs, b_obs)
    d_prime = d_info - b_obs.reshape(-1, n_cols).T @ m.reshape(-1, n_cols)

    diag = np.diag(d_prime)
    if np.any(diag <= 0.0):
        return DesignCovarianceResult(cov, observed, False, n_obs)
    scale = 1.0 / np.sqrt(diag)
    scaled = d_prime * scale[:, None] * scale[None, :]
    if np.linalg.eigvalsh(scaled).min() < CONNECTIVITY_EIGENVALUE_FLOOR:
        return DesignCovarianceResult(cov, observed, False, n_obs)

    d_prime_inv = np.linalg.inv(d_prime)
    cov[observed] = ainv_obs + np.matmul(m @ d_prime_inv, m.transpose(0, 2, 1))
    return DesignCovarianceResult(cov, observed, True, n_obs)


def count_untied_stations(visible: np.ndarray, anchor_index: int = 0, min_shared: int = 3) -> int:
    """How many stations cannot be attached to the anchor's frame by at
    least `min_shared` shared targets, growing the anchor's group station
    by station (a target seen by any member of the group is a known point
    in the group's frame).

    A necessary condition for the network to be rigid (three shared,
    non-collinear targets pin a station's six degrees of freedom), and a
    *graded* one -- counting stations rather than answering yes/no -- so a
    search has something to descend when most candidate layouts fail. It
    ignores collinearity; `design_point_covariances`' rank test is the
    final arbiter."""
    seen = [set(np.nonzero(row)[0]) for row in np.asarray(visible, dtype=bool)]
    group_targets = set(seen[anchor_index])
    remaining = [s for s in range(len(seen)) if s != anchor_index]
    changed = True
    while changed and remaining:
        changed = False
        for s in list(remaining):
            if len(seen[s] & group_targets) >= min_shared:
                group_targets |= seen[s]
                remaining.remove(s)
                changed = True
    return len(remaining)


def design_covariances_for_poses(
    target_points_m: np.ndarray,
    station_poses: Sequence[InstrumentPose],
    tracker: LaserTracker,
    visible: np.ndarray,
    anchor_index: int = 0,
) -> DesignCovarianceResult:
    """Convenience wrapper over `design_point_covariances` taking
    `InstrumentPose` objects (the form `network.solve` uses)."""
    positions = np.array([p.position_m for p in station_poses])
    orientations = np.array([p.orientation for p in station_poses])
    return design_point_covariances(
        target_points_m, positions, orientations, tracker, visible, anchor_index
    )


def projected_uncertainty_m(
    point_covariance_m2: np.ndarray, directions: np.ndarray, systematic_std_m: float = 0.0
) -> np.ndarray:
    """Standard uncertainty of each point along its own constrained
    direction (step 4's projection, for a one-direction tolerance such as
    a profile callout): `sqrt(n^T Sigma n + sigma_sys^2)`.

    `directions` (N, 3) must be unit vectors. NaN covariances (unobserved
    points) give NaN."""
    variance = np.einsum("ti,tij,tj->t", directions, point_covariance_m2, directions)
    return np.sqrt(variance + systematic_std_m**2)


def design_joint_covariance(
    target_points_m: np.ndarray,
    station_poses: Sequence[InstrumentPose],
    tracker: LaserTracker,
    visible: np.ndarray,
    subset: Sequence[int],
    anchor_index: int = 0,
):
    """Joint design covariance of a *subset* of targets, cross-covariances
    included: `(covariance (3n x 3n), observed (n,), connected)` for the n
    targets in `subset`, in that order.

    `design_point_covariances` returns each target's own 3x3 block only, which
    is all a quantity defined at one point in the anchor's frame needs. A
    quantity defined *relative to other measured points* -- a deviation from a
    datum frame fitted to datum targets (`datum_referenced_functional`) --
    needs the covariances between targets too: two targets seen from the same
    uncertain station move together, and much of that common motion cancels
    in a relative quantity. By the same block elimination as the module
    docstring,

        Cov(t, s) = [t == s] A_t^-1  +  (A_t^-1 B_t) D'^-1 (A_s^-1 B_s)^T

    Rows and columns of unobserved targets are NaN. Matches the corresponding
    blocks of `network.solve.solve_network`'s full covariance (tests).
    """
    targets = np.asarray(target_points_m, dtype=float)
    positions = np.array([p.position_m for p in station_poses])
    orientations = np.array([p.orientation for p in station_poses])
    visible = np.asarray(visible, dtype=bool)
    n_stations, n_targets = visible.shape
    subset = np.asarray(list(subset), dtype=int)
    n_sub = len(subset)
    out = np.full((3 * n_sub, 3 * n_sub), np.nan)
    observed = visible.any(axis=0)
    obs_sub = observed[subset]
    s_idx, t_idx = np.nonzero(visible)
    if len(s_idx) == 0:
        return out, obs_sub, True

    delta = targets[t_idx] - positions[s_idx]
    lam = shot_information_global(delta, orientations[s_idx], tracker)
    a_info = np.zeros((n_targets, 3, 3))
    np.add.at(a_info, t_idx, lam)
    a_inv = np.zeros((n_targets, 3, 3))
    a_inv[observed] = np.linalg.inv(a_info[observed])

    free = [s for s in range(n_stations) if s != anchor_index]
    n_cols = 6 * len(free)
    keep = np.repeat(obs_sub, 3)
    block = np.zeros((3 * n_sub, 3 * n_sub))
    for i, t in enumerate(subset):
        block[3 * i:3 * i + 3, 3 * i:3 * i + 3] = a_inv[t]
    if n_cols == 0:
        out[np.ix_(keep, keep)] = block[np.ix_(keep, keep)]
        return out, obs_sub, True

    g_station = np.concatenate([np.broadcast_to(np.eye(3), (len(s_idx), 3, 3)), -_skew(delta)], axis=2)
    lam_g = np.einsum("oij,ojk->oik", lam, g_station)
    b_full = np.zeros((n_targets, 3, n_cols))
    d_info = np.zeros((n_cols, n_cols))
    for k, s in enumerate(free):
        sel = s_idx == s
        cols = slice(6 * k, 6 * k + 6)
        b_full[t_idx[sel], :, cols] = -lam_g[sel]
        d_info[cols, cols] = np.einsum("oji,ojk->ik", g_station[sel], lam_g[sel])
    m = np.matmul(a_inv[observed], b_full[observed])
    d_prime = d_info - b_full[observed].reshape(-1, n_cols).T @ m.reshape(-1, n_cols)
    diag = np.diag(d_prime)
    if np.any(diag <= 0.0):
        return out, obs_sub, False
    scale = 1.0 / np.sqrt(diag)
    if np.linalg.eigvalsh(d_prime * scale[:, None] * scale[None, :]).min() < CONNECTIVITY_EIGENVALUE_FLOOR:
        return out, obs_sub, False
    m_sub = np.matmul(a_inv[subset], b_full[subset]).reshape(3 * n_sub, n_cols)  # zero rows for unobserved targets
    full = block + m_sub @ np.linalg.inv(d_prime) @ m_sub.T
    out[np.ix_(keep, keep)] = full[np.ix_(keep, keep)]
    return out, obs_sub, True


def datum_referenced_functional(
    points_m: np.ndarray, point_index: int, datum_indices: Sequence[int], direction: Optional[np.ndarray] = None
):
    """Coefficients `a` (3 x 3n, or 1 x 3n with a `direction`) such that
    `a @ displacements` is the displacement of point `point_index` *relative to
    the frame fitted to the datum points*, to first order.

    A deviation of a toleranced point is defined in the part's datum frame, not
    in a tracker's. That frame is realised by measuring datum targets and
    best-fitting their measured positions to their nominal ones with a rigid
    motion (translation t, small rotation w; unweighted least squares, the usual
    best-fit). An error dx_i in each measured datum target moves the fitted
    frame by (t, w) = (G^T G)^-1 G^T dx_D, with G_i = [I, -[x_i - c]_x] and c
    the datum centroid, and so moves the nominal position of the toleranced
    point by G_p (t, w). What the inspection sees is the difference:

        deviation error = dx_p - G_p (G^T G)^-1 G^T dx_D

    This is the geodetic S-transformation to the datum defined by the datum
    points: the result does not depend on which station's frame the network
    covariance happens to be expressed in (tests check that invariance).

    `points_m` (n, 3) are the nominal positions of the subset the covariance
    is over; `datum_indices` index into it. Returns `None` if the datum points
    cannot fix six degrees of freedom (fewer than three, or collinear)."""
    points = np.asarray(points_m, dtype=float)
    datum = list(datum_indices)
    if len(datum) < 3:
        return None
    centre = points[datum].mean(axis=0)

    def g_rows(x):
        return np.concatenate([np.eye(3), -_skew((x - centre)[None, :])[0]], axis=1)  # 3 x 6

    g = np.vstack([g_rows(points[i]) for i in datum])  # 3m x 6
    normal = g.T @ g
    if np.any(np.diag(normal) <= 1e-12):  # a rotation no datum point resists (points on one line through the centroid)
        return None
    scale = 1.0 / np.sqrt(np.diag(normal))
    if np.linalg.eigvalsh(normal * scale[:, None] * scale[None, :]).min() < 1e-9:
        return None
    fit = np.linalg.solve(normal, g.T)  # 6 x 3m: datum displacements -> (t, w)
    transfer = g_rows(points[point_index]) @ fit  # 3 x 3m
    a = np.zeros((3, 3 * len(points)))
    a[:, 3 * point_index:3 * point_index + 3] += np.eye(3)
    for k, i in enumerate(datum):
        a[:, 3 * i:3 * i + 3] -= transfer[:, 3 * k:3 * k + 3]
    if direction is None:
        return a
    return np.asarray(direction, dtype=float)[None, :] @ a
