"""Process knowledge and decision rules beyond the step-5 baseline.

`risk.jcgm106` computes global PFA and PFR for a *known normal* process and
*simple acceptance* (acceptance limits = tolerance limits). Both assumptions
are the conventional starting point and both matter a great deal at the
uncertainties large-volume metrology delivers:

**1. The process is never known exactly.** Large-volume work is small-batch
(CLAUDE.md §3): a cluster's mean and spread come from a handful of builds, not
from the ~50 samples classical capability statistics assume. With `n` builds,
sample mean `m` and sample standard deviation `s`, and no other information,
the probability distribution for the *next* part's deviation is not
N(m, s^2) but the Student-t predictive distribution

    x  ~  m + s * sqrt(1 + 1/n) * t_(n-1)

(the standard Bayesian result for a normal process with unknown mean and
variance under the reference prior). It has the same centre and almost the
same width, but much heavier tails: a process that looks like Cpk 1.33 from
five builds has a ~1% chance of producing a nonconforming next part, not the
0.003% the normal model with the same m and s claims. As n grows the
predictive distribution tends to the normal one. `ProcessKnowledge.from_builds`
builds it; `n_builds=None` means "known exactly" and reproduces `jcgm106`.
The number of builds is a statement about the evidence, not a tuned constant,
and results are reported across it.

Cluster assignment is itself uncertain (`ClusterAssignment.confidence`).
`ProcessKnowledge.mixed_with` propagates that as a two-component mixture: with
probability `confidence` the characteristic belongs to its assigned cluster,
otherwise to the alternative given.

**2. Simple acceptance is not the rule JCGM 106 recommends when uncertainty is
large.** JCGM 106 defines the *conformance probability* of an item given its
measured value, p_c(m) = P(true value in tolerance | m), and decision rules
that accept when p_c(m) is at least a required level (guarded acceptance).
`acceptance_interval` returns the acceptance limits that rule implies for a
given process knowledge and measurement uncertainty:

    min_conformance_probability = 0.5   the rule that minimises PFA + PFR (the
                                        Bayes rule for equal costs of the two
                                        errors): accept when the part is more
                                        likely conforming than not;
    higher values (e.g. 0.95)           guarded acceptance: fewer false
                                        accepts, more false rejects.

`decision_risk` evaluates PFA and PFR for *any* acceptance limits and any
`ProcessKnowledge`, so simple acceptance, guard-banded and conformance-
probability rules are all compared with the same integral.

**What this adds to planning.** Under the rule that minimises PFA + PFR, a
measurement can never make the decision worse than not measuring at all
(accepting everything is one of the rules available), so total risk as a
function of uncertainty rises monotonically from near zero to the unmeasured
level, the process's own nonconformance rate. The drop from that level is the
*value of the measurement*; `planning.cell_experiment` uses it as the planning
objective.

Numerics: one-dimensional integrals by composite Gauss-Legendre quadrature on
panels placed around the tolerance limits, the acceptance limits and each
prior component (vectorised, ~1 ms), checked against `risk.jcgm106`'s
adaptive quadrature in tests.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special, stats

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(10)
_SCALE_STEPS = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0])
_U_STEPS = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])


@dataclass(frozen=True)
class PriorComponent:
    """One component of a process-knowledge mixture: normal if `dof` is None,
    otherwise a location-scale Student-t with `dof` degrees of freedom."""

    weight: float
    mean_m: float
    scale_m: float
    dof: Optional[float] = None

    def _dist(self):
        if self.dof is None:
            return stats.norm(self.mean_m, self.scale_m)
        return stats.t(self.dof, self.mean_m, self.scale_m)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist().pdf(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist().logpdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist().cdf(x)

    def sf(self, x: np.ndarray) -> np.ndarray:
        return self._dist().sf(x)


@dataclass(frozen=True)
class ProcessKnowledge:
    """What is known about the true deviation of a characteristic before it is
    measured: a mixture of normal and Student-t components (weights sum to 1).
    See the module docstring for where each form comes from."""

    components: Tuple[PriorComponent, ...]

    def __post_init__(self):
        total = sum(c.weight for c in self.components)
        if not np.isclose(total, 1.0):
            raise ValueError(f"component weights must sum to 1, got {total}")
        if any(c.scale_m <= 0 for c in self.components):
            raise ValueError("component scales must be positive")

    @classmethod
    def known(cls, mean_m: float, std_m: float) -> "ProcessKnowledge":
        """A process known exactly: N(mean, std^2). `risk.jcgm106`'s model."""
        return cls((PriorComponent(1.0, mean_m, std_m, None),))

    @classmethod
    def from_builds(cls, mean_m: float, std_m: float, n_builds: Optional[int]) -> "ProcessKnowledge":
        """The predictive distribution for the next part when `mean_m` and
        `std_m` are the sample mean and standard deviation of `n_builds` earlier
        parts: Student-t, n-1 degrees of freedom, scale std * sqrt(1 + 1/n).
        `n_builds=None` is the known-process limit."""
        if n_builds is None:
            return cls.known(mean_m, std_m)
        if n_builds < 2:
            raise ValueError("at least two builds are needed for a sample standard deviation")
        return cls((PriorComponent(1.0, mean_m, std_m * np.sqrt(1.0 + 1.0 / n_builds), float(n_builds - 1)),))

    def mixed_with(self, other: "ProcessKnowledge", confidence: float) -> "ProcessKnowledge":
        """`confidence` x this + (1 - confidence) x `other`: cluster-assignment doubt."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        mine = tuple(PriorComponent(c.weight * confidence, c.mean_m, c.scale_m, c.dof) for c in self.components)
        theirs = tuple(PriorComponent(c.weight * (1.0 - confidence), c.mean_m, c.scale_m, c.dof) for c in other.components)
        return ProcessKnowledge(tuple(c for c in mine + theirs if c.weight > 0.0))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return sum(c.weight * c.pdf(x) for c in self.components)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        terms = np.array([np.log(c.weight) + c.logpdf(x) for c in self.components])
        return special.logsumexp(terms, axis=0)

    def nonconformance(self, tolerance_lower_m: float, tolerance_upper_m: float) -> float:
        """P(true value outside the tolerance): the risk of accepting unseen."""
        return float(sum(c.weight * (c.cdf(tolerance_lower_m) + c.sf(tolerance_upper_m)) for c in self.components))


def _panels(low: float, high: float, breakpoints: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [low, high], split at `breakpoints`."""
    if not high > low:
        return np.empty(0), np.empty(0)
    edges = np.unique(np.concatenate([[low, high], [b for b in breakpoints if low < b < high]]))
    a, b = edges[:-1], edges[1:]
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    x = (mid[:, None] + half[:, None] * _GL_NODES[None, :]).ravel()
    w = (half[:, None] * _GL_WEIGHTS[None, :]).ravel()
    return x, w


def _breakpoints(prior: ProcessKnowledge, u_m: float, limits: Sequence[float]) -> np.ndarray:
    points: List[float] = []
    for c in prior.components:
        points.extend(c.mean_m + _SCALE_STEPS * c.scale_m)
        points.extend(c.mean_m - _SCALE_STEPS * c.scale_m)
        points.append(c.mean_m)
    for limit in limits:
        if np.isfinite(limit):
            points.append(limit)
            points.extend(limit + _U_STEPS * u_m)
            points.extend(limit - _U_STEPS * u_m)
    return np.array(points)


def _acceptance_probability(x: np.ndarray, u_m: float, acceptance_lower_m: float, acceptance_upper_m: float) -> np.ndarray:
    """P(measured value in the acceptance interval | true value x), normal error."""
    return stats.norm.cdf((acceptance_upper_m - x) / u_m) - stats.norm.cdf((acceptance_lower_m - x) / u_m)


def decision_risk(
    prior: ProcessKnowledge, tolerance_lower_m: float, tolerance_upper_m: float, u_m: float,
    acceptance_lower_m: Optional[float] = None, acceptance_upper_m: Optional[float] = None,
) -> Tuple[float, float]:
    """Global `(PFA, PFR)` for process knowledge `prior`, measurement
    uncertainty `u_m` and the given acceptance limits (default: simple
    acceptance; `-inf`/`+inf` accept everything on that side)."""
    a_lo = tolerance_lower_m if acceptance_lower_m is None else acceptance_lower_m
    a_hi = tolerance_upper_m if acceptance_upper_m is None else acceptance_upper_m
    if a_lo == -np.inf and a_hi == np.inf:  # accept everything: no measurement needed
        return prior.nonconformance(tolerance_lower_m, tolerance_upper_m), 0.0
    if not a_hi > a_lo:  # reject everything
        return 0.0, 1.0 - prior.nonconformance(tolerance_lower_m, tolerance_upper_m)
    points = _breakpoints(prior, u_m, (tolerance_lower_m, tolerance_upper_m, a_lo, a_hi))
    reach = 12.0 * u_m
    # false accept: true value outside the tolerance, measured value inside the acceptance interval;
    # beyond the acceptance limit + 12 u the acceptance probability is < 1e-32
    span = max(c.scale_m for c in prior.components) * 2000.0
    upper_end = a_hi + reach if np.isfinite(a_hi) else tolerance_upper_m + span
    lower_end = a_lo - reach if np.isfinite(a_lo) else tolerance_lower_m - span
    pfa = 0.0
    for low, high in ((tolerance_upper_m, upper_end), (lower_end, tolerance_lower_m)):
        x, w = _panels(low, high, points)
        if x.size:
            pfa += float(np.sum(w * prior.pdf(x) * _acceptance_probability(x, u_m, a_lo, a_hi)))
    if not np.isfinite(a_hi):  # far tail accepted outright
        pfa += float(sum(c.weight * c.sf(upper_end) for c in prior.components))
    if not np.isfinite(a_lo):
        pfa += float(sum(c.weight * c.cdf(lower_end) for c in prior.components))
    # false reject: true value inside the tolerance, measured value outside the acceptance interval
    x, w = _panels(tolerance_lower_m, tolerance_upper_m, points)
    pfr = float(np.sum(w * prior.pdf(x) * (1.0 - _acceptance_probability(x, u_m, a_lo, a_hi))))
    return pfa, pfr


def conformance_probability(
    prior: ProcessKnowledge, tolerance_lower_m: float, tolerance_upper_m: float, u_m: float, measured_m: np.ndarray,
) -> np.ndarray:
    """JCGM 106's conformance probability p_c(m) = P(true value in tolerance |
    measured value m), for each entry of `measured_m`."""
    measured_m = np.atleast_1d(np.asarray(measured_m, dtype=float))
    points = _breakpoints(prior, u_m, (tolerance_lower_m, tolerance_upper_m))
    core_low = min(c.mean_m - 60.0 * c.scale_m for c in prior.components)
    core_high = max(c.mean_m + 60.0 * c.scale_m for c in prior.components)
    width = min(u_m, min(c.scale_m for c in prior.components))
    out = np.empty(len(measured_m))

    def log_posterior(x, m):  # unnormalised: log prior + log likelihood of the measured value
        return prior.logpdf(x) - 0.5 * ((m - x) / u_m) ** 2

    for k, m in enumerate(measured_m):
        # The posterior's mass can sit anywhere between the prior's core and the measured value (a
        # tight prior and a coarse measurement put it near the prior; heavy tails put it near m).
        # Pass 1 finds the peak on panels around both; pass 2 adds panels around the peak. Work in
        # logs, shifted by the peak, so far-apart prior and measurement do not underflow to 0/0.
        low = min(m - 12.0 * u_m, core_low, tolerance_lower_m)
        high = max(m + 12.0 * u_m, core_high, tolerance_upper_m)
        local = np.concatenate([points, m + _U_STEPS * u_m, m - _U_STEPS * u_m, [m]])
        x1, _ = _panels(low, high, local)
        log1 = log_posterior(x1, m)
        peak = x1[int(np.argmax(log1))]
        local = np.concatenate([local, peak + _SCALE_STEPS[:14] * width, peak - _SCALE_STEPS[:14] * width, [peak]])
        x_all, w_all = _panels(low, high, local)
        log_all = log_posterior(x_all, m)
        shift = float(log_all.max())
        total = np.sum(w_all * np.exp(log_all - shift))
        x_in, w_in = _panels(tolerance_lower_m, tolerance_upper_m, local)
        inside = np.sum(w_in * np.exp(log_posterior(x_in, m) - shift))
        out[k] = min(1.0, inside / total) if total > 0.0 else 0.0
    return out


def acceptance_interval(
    prior: ProcessKnowledge, tolerance_lower_m: float, tolerance_upper_m: float, u_m: float,
    min_conformance_probability: float = 0.5, n_grid: int = 241, reach_tolerances: float = 30.0,
) -> Tuple[float, float]:
    """Acceptance limits of the rule "accept when the conformance probability
    is at least `min_conformance_probability`" (see the module docstring).

    Returns `(lower, upper)`. A side on which every measured value out to
    `reach_tolerances` tolerance half-widths is still accepted is returned as
    infinite (accept everything on that side); `(nan, nan)` means no measured
    value is acceptable (reject everything). The accepted set is taken as one
    interval (its hull), which is exact for the unimodal priors used here."""
    half = 0.5 * (tolerance_upper_m - tolerance_lower_m)
    centre = 0.5 * (tolerance_upper_m + tolerance_lower_m)
    reach = reach_tolerances * half + 10.0 * u_m
    grid = centre + reach * np.sinh(np.linspace(-4.0, 4.0, n_grid)) / np.sinh(4.0)  # dense near the tolerance
    p = conformance_probability(prior, tolerance_lower_m, tolerance_upper_m, u_m, grid)
    ok = p >= min_conformance_probability
    if not ok.any():
        return float("nan"), float("nan")
    first, last = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))

    def crossing(i_out: int, i_in: int) -> float:
        # the measured value at which the conformance probability equals the required level,
        # between a rejected and an accepted grid point
        f = lambda m: conformance_probability(prior, tolerance_lower_m, tolerance_upper_m, u_m, [m])[0] - min_conformance_probability
        a, b = sorted((float(grid[i_out]), float(grid[i_in])))
        if f(a) * f(b) > 0.0:  # no sign change at this resolution: fall back to the grid point
            return float(grid[i_in])
        return float(optimize.brentq(f, a, b, xtol=1e-4 * min(u_m, half), rtol=1e-10))

    lower = -np.inf if first == 0 else crossing(first - 1, first)
    upper = np.inf if last == len(grid) - 1 else crossing(last + 1, last)
    return lower, upper


def best_rule_risk(
    prior: ProcessKnowledge, tolerance_lower_m: float, tolerance_upper_m: float, u_m: float,
    min_conformance_probability: float = 0.5,
) -> Tuple[float, float, Tuple[float, float]]:
    """`(PFA, PFR, acceptance limits)` under the conformance-probability rule."""
    lower, upper = acceptance_interval(prior, tolerance_lower_m, tolerance_upper_m, u_m, min_conformance_probability)
    if np.isnan(lower):
        return 0.0, 1.0 - prior.nonconformance(tolerance_lower_m, tolerance_upper_m), (lower, upper)
    pfa, pfr = decision_risk(prior, tolerance_lower_m, tolerance_upper_m, u_m, lower, upper)
    return pfa, pfr, (lower, upper)


def unmeasured_risk(prior: ProcessKnowledge, tolerance_lower_m: float, tolerance_upper_m: float) -> float:
    """Total risk of the better of the two decisions available without a
    measurement: accept everything (risk = nonconformance rate) or reject
    everything (risk = conformance rate)."""
    nc = prior.nonconformance(tolerance_lower_m, tolerance_upper_m)
    return float(min(nc, 1.0 - nc))
