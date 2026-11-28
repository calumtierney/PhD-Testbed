"""Independent checks of a review's numbers: recompute them a different way and say whether they agree.

A review's figures come from a chain (stations -> visibility -> instrument error -> network covariance -> datum frame -> tolerance
direction -> decision risk). Each link has its own unit tests, but a reviewer who doubts a number should be able to ask "is it right?"
and get an answer that does not rely on the code that produced it. This module gives three, each sharing as little as possible with
the production path:

  1. **The decision step, closed form.** For a normal process N(mean, std), a measurement error N(0, u) and acceptance limits, PFA and
     PFR are one-dimensional integrals over the true deviation, evaluated here with adaptive quadrature (`scipy.integrate.quad`), not
     the panel scheme of `risk.decision`. They agree to 1e-6 relative or better.
  2. **The decision step, Monte Carlo.** Draw parts from the process prior (normal, Student-t, or a mixture: whatever the review used),
     add measurement error, apply the acceptance limits, count wrong decisions. Agreement is judged against the sampling error.
  3. **The whole measurement chain, Monte Carlo** (`monte_carlo_uncertainty`; slow, so opt-in): simulate the tracker's *readings*
     (range, azimuth, elevation, with the sensor noise and, for the NPL model, one draw of the parameter errors per physical
     tracker), solve the network by nonlinear least squares, fit the part frame to the measured datum targets with an explicit
     best-fit rigid transform (Kabsch / SVD), and take the deviation of each characteristic's point from nominal along its tolerance
     direction. Its standard deviation over many trials is compared with the review's projected uncertainty. This shares with the
     production path only the visibility matrix, the sensor and parameter definitions and the nonlinear solver; it does not use the
     design covariance (`network.design`), the S-transformation (`datum_referenced_functional`) or the scorer.

Agreement is statistical: a Monte Carlo standard deviation from n trials has a relative standard error of 1 / sqrt(2 (n - 1)) (1.3% at
3000 trials, 3.5% at 400). `Check.passed` is true when the difference is within `n_sigma` of that error (default 4).
"""
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate
from scipy.stats import norm

from geometry.pose import InstrumentPose
from instruments.npl_model import NPLErrorParameters
from network.solve import perturbed_initial_guess, simulate_observations, solve_network
from planning.candidates import levelled_orientation
from risk.decision import ProcessKnowledge


@dataclass(frozen=True)
class Check:
    """One comparison of a reviewed number with an independent recomputation."""

    name: str
    passed: bool
    reviewed: float
    independent: float
    tolerance: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "reviewed": float(self.reviewed), "independent": float(self.independent),
                "tolerance": self.tolerance, "detail": self.detail}


# -- 1. the decision step, closed form ------------------------------------------------------------------
def risk_closed_form(mean_m: float, std_m: float, lower_m: float, upper_m: float, u_m: float,
                     accept_lower_m: Optional[float] = None, accept_upper_m: Optional[float] = None) -> Tuple[float, float]:
    """(PFA, PFR) for a *normal* process N(mean, std), measurement error N(0, u) and acceptance limits (default: the tolerance limits).

    With x the true deviation and m = x + e the measured one, P(accept | x) = Phi((a_hi - x) / u) - Phi((a_lo - x) / u), so
        PFR = integral over x in tolerance of  f(x) (1 - P(accept | x)),
        PFA = integral over x outside tolerance of  f(x) P(accept | x).
    Evaluated by adaptive quadrature, with the integrand's breakpoints given to the integrator."""
    a_lo = lower_m if accept_lower_m is None else accept_lower_m
    a_hi = upper_m if accept_upper_m is None else accept_upper_m

    def accept(x):
        return norm.cdf((a_hi - x) / u_m) - norm.cdf((a_lo - x) / u_m)

    def f(x):
        return norm.pdf(x, mean_m, std_m)

    def quad(func, a, b, points):
        pts = sorted(p for p in points if a < p < b)
        return integrate.quad(func, a, b, points=pts or None, epsabs=0.0, epsrel=1e-10, limit=400)[0]

    reach = 14.0 * std_m + 14.0 * u_m + abs(mean_m) + (upper_m - lower_m)
    breaks = (lower_m, upper_m, a_lo, a_hi, mean_m)
    pfr = quad(lambda x: f(x) * (1.0 - accept(x)), lower_m, upper_m, breaks)
    pfa = (quad(lambda x: f(x) * accept(x), upper_m, upper_m + reach, breaks) +
           quad(lambda x: f(x) * accept(x), lower_m - reach, lower_m, breaks))
    return float(pfa), float(pfr)


# -- 2. the decision step, Monte Carlo ---------------------------------------------------------------------
def sample_process(knowledge: ProcessKnowledge, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` true deviations drawn from a `ProcessKnowledge` (a mixture of normal and Student-t components)."""
    weights = np.array([c.weight for c in knowledge.components])
    which = rng.choice(len(weights), size=n, p=weights / weights.sum())
    out = np.empty(n)
    for i, comp in enumerate(knowledge.components):
        sel = which == i
        k = int(sel.sum())
        if comp.dof is None:
            out[sel] = rng.normal(comp.mean_m, comp.scale_m, k)
        else:
            out[sel] = comp.mean_m + comp.scale_m * rng.standard_t(comp.dof, k)
    return out


def risk_monte_carlo(knowledge: ProcessKnowledge, lower_m: float, upper_m: float, u_m: float, n_parts: int, rng: np.random.Generator,
                     accept_lower_m: Optional[float] = None, accept_upper_m: Optional[float] = None, batch: int = 2_000_000) -> Dict[str, float]:
    """PFA and PFR by simulating `n_parts` parts: true deviation from the process, plus measurement error N(0, u), accepted if the
    measured value is inside the acceptance limits. Returns the estimates and their binomial standard errors."""
    a_lo = lower_m if accept_lower_m is None else accept_lower_m
    a_hi = upper_m if accept_upper_m is None else accept_upper_m
    false_accept = false_reject = done = 0
    while done < n_parts:
        n = min(batch, n_parts - done)
        x = sample_process(knowledge, n, rng)
        m = x + rng.normal(0.0, u_m, n)
        conforming = (x >= lower_m) & (x <= upper_m)
        accepted = (m >= a_lo) & (m <= a_hi)
        false_accept += int(np.sum(~conforming & accepted))
        false_reject += int(np.sum(conforming & ~accepted))
        done += n
    pfa, pfr = false_accept / n_parts, false_reject / n_parts
    return {"pfa": pfa, "pfr": pfr, "pfa_se": float(np.sqrt(max(pfa, 1.0 / n_parts) / n_parts)),
            "pfr_se": float(np.sqrt(max(pfr, 1.0 / n_parts) * (1 - pfr) / n_parts)), "n_parts": n_parts,
            "n_false_accepts": false_accept, "n_false_rejects": false_reject}


# -- 3. the whole chain, Monte Carlo --------------------------------------------------------------------------
def best_fit_rigid(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rotation R and translation t minimising sum |R s + t - d|^2 over the points (Kabsch's algorithm, by SVD)."""
    cs, cd = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - cd))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, cd - rotation @ cs


def monte_carlo_uncertainty(context, plan, n_trials: int, seed: int = 0,
                            progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """The whole-chain Monte Carlo of the module docstring, for a review `context` and the `plan` that was reviewed.

    Returns per characteristic the simulated projected uncertainty (the standard deviation of the deviation along the tolerance
    direction, or for several directions the root of the largest eigenvalue of its covariance, plus the systematic floor in
    quadrature, as the scorer reports it), its relative standard error, and the mean deviation (which should be zero)."""
    from review.core import measurement_pattern  # a local import: core imports nothing from here

    scene, scenario = context.scene, context.scenario
    tracker = scenario.tracker
    nominal = replace(tracker, geometric_params=None, systematic_std_m=0.0, floor_excludes_geometric_parameters=False)  # what the solver assumes
    truth_all = scenario.target_points_m
    pattern = measurement_pattern(context, plan)
    poses, visible = pattern.poses, pattern.visible
    instrument_ids = pattern.instrument_ids if pattern.instrument_ids is not None else [0] * len(poses)
    faces = pattern.faces if pattern.faces is not None else [("front",)] * len(poses)
    observed = np.nonzero(visible.any(axis=0))[0]
    position_in_subset = {int(t): i for i, t in enumerate(observed)}
    truth = truth_all[observed]
    visible_sub = visible[:, observed]
    datum = [position_in_subset[i] for i in scene.datum_indices if i in position_in_subset]
    characteristics = []
    for c in scenario.characteristics:
        t = c.target_indices[0]
        characteristics.append((position_in_subset.get(int(t)), np.atleast_2d(np.asarray(c.tolerance.constraint_directions(), dtype=float))))

    unique_ids = sorted(set(instrument_ids))
    params = tracker.geometric_params
    carries_parameters = params is not None and bool(params.covariance.any())
    rng = np.random.default_rng(seed)
    deviations = [np.zeros((n_trials, d.shape[0])) if i is not None else None for i, d in characteristics]
    for trial in range(n_trials):
        if carries_parameters:  # one draw of the parameter errors per physical tracker, shared by all its stations
            draws = [NPLErrorParameters(rng.multivariate_normal(np.zeros(len(params.values)), params.covariance, method="svd"),
                                        np.zeros_like(params.covariance)) for _ in unique_ids]
        else:
            draws = None
        observations = simulate_observations(truth, poses, nominal, rng, instrument_params=draws, instrument_ids=instrument_ids,
                                             faces=faces, visible=visible_sub)
        initial_targets, initial_poses = perturbed_initial_guess(truth, poses, 0.002, 0.0002, rng)
        solved = solve_network(observations, initial_targets, initial_poses, nominal, anchor_index=0).target_points_m
        rotation, translation = best_fit_rigid(solved[datum], truth[datum])  # the part frame: measured datum targets best-fitted to nominal
        for k, (index, directions) in enumerate(characteristics):
            if index is None:
                continue
            deviations[k][trial] = directions @ (rotation @ solved[index] + translation - truth[index])
        if progress is not None:
            progress(trial + 1, n_trials)
    floor_m = tracker.systematic_std_m
    rows = []
    for (index, directions), dev in zip(characteristics, deviations):
        if index is None:
            rows.append(None)
            continue
        covariance = np.atleast_2d(np.cov(dev, rowvar=False, ddof=1))
        u = float(np.sqrt(max(np.linalg.eigvalsh(covariance).max(), 0.0) + floor_m ** 2))
        rows.append({"u_m": u, "relative_se": 1.0 / np.sqrt(2.0 * (n_trials - 1)), "mean_deviation_m": dev.mean(axis=0).tolist()})
    return {"n_trials": n_trials, "seed": seed, "characteristics": rows, "tracker_model": ("npl parameters drawn per trial" if carries_parameters
                                                                                          else "sensor noise only")}


# -- the checks ---------------------------------------------------------------------------------------------------------------
def check_review(context, plan, review, n_parts: int = 1_000_000, n_trials: int = 0, seed: int = 0, n_sigma: float = 4.0,
                 progress: Optional[Callable[[int, int], None]] = None) -> List[Check]:
    """Recompute a feasible review's numbers independently and compare. `n_parts` parts are simulated per characteristic for the
    decision step; `n_trials` > 0 also runs the whole-chain Monte Carlo (about 0.2 s per trial on the built-in cell). `plan` is the
    `PlanFile` that was reviewed."""
    if not review.feasible:
        return []
    scene = context.scene
    limits = scene.tolerance_limits_m()
    rng = np.random.default_rng(seed)
    checks: List[Check] = []
    for k, c in enumerate(review.characteristics):
        if not c["measured"]:
            continue
        knowledge = context.knowledge[k]
        lower, upper = float(limits[k, 0]), float(limits[k, 1])
        u = c["u_um"] * 1e-6
        accept = (lower, upper) if c["acceptance_um"] is None else tuple(None if a is None else a * 1e-6 for a in c["acceptance_um"])
        a_lo, a_hi = (-np.inf if accept[0] is None else accept[0]), (np.inf if accept[1] is None else accept[1])
        if len(knowledge.components) == 1 and knowledge.components[0].dof is None and np.isfinite(a_lo) and np.isfinite(a_hi):
            comp = knowledge.components[0]
            pfa, pfr = risk_closed_form(comp.mean_m, comp.scale_m, lower, upper, u, a_lo, a_hi)
            for label, reviewed, independent in (("PFR", c["pfr"], pfr), ("PFA", c["pfa"], pfa)):
                scale = max(abs(independent), 1e-300)
                checks.append(Check(f"{c['name']}: {label}, closed form", abs(reviewed - independent) <= 1e-6 * scale + 1e-15,
                                    reviewed, independent, "relative 1e-6", "quadrature of the one-dimensional integral (not risk.decision's panels)"))
        mc = risk_monte_carlo(knowledge, lower, upper, u, n_parts, rng, a_lo, a_hi)
        for label, reviewed, key in (("PFR", c["pfr"], "pfr"), ("PFA", c["pfa"], "pfa")):
            se = mc[key + "_se"]
            checks.append(Check(f"{c['name']}: {label}, Monte Carlo of {n_parts:,} parts", abs(reviewed - mc[key]) <= n_sigma * se + 1e-12,
                                reviewed, mc[key], f"{n_sigma:g} standard errors ({se:.2e})",
                                f"{mc['n_false_rejects'] if key == 'pfr' else mc['n_false_accepts']} events"))
    if n_trials > 0:
        run = monte_carlo_uncertainty(context, plan, n_trials, seed, progress)
        for c, row in zip(review.characteristics, run["characteristics"]):
            if row is None or not c["measured"]:
                continue
            reviewed, independent = c["u_um"], row["u_m"] * 1e6
            checks.append(Check(f"{c['name']}: projected uncertainty u, whole-chain Monte Carlo ({n_trials} trials)",
                                abs(independent - reviewed) <= n_sigma * row["relative_se"] * reviewed, reviewed, independent,
                                f"{n_sigma:g} standard errors ({100 * row['relative_se']:.1f}% each)",
                                "simulated readings -> nonlinear network solve -> explicit best-fit datum frame -> deviation along the tolerance direction"))
    return checks
