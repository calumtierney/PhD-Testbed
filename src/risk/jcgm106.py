"""JCGM 106 conformity decision risk (CLAUDE.md §5, step 5).

Every step before this one answers "how uncertain is this measurement?"
This step asks the question an inspection actually exists to answer:
given that uncertainty, how often does the accept/reject decision come
out *wrong*? JCGM 106 ("Evaluation of measurement data -- The role of
measurement uncertainty in conformity assessment") is the standard that
defines this properly, rather than pretending a measurement's own
uncertainty doesn't matter to the decision it's used for.

**The setup.** A characteristic has a true value `Y` (unknown, but drawn
from some population -- the process that made the part) and a measured
value `X = Y + E`, where `E` is the measurement error (mean zero, std
`uncertainty_m`, the scalar this codebase already computes in step 4). A
decision rule accepts the part if `X` falls inside an acceptance interval
(equal to the tolerance limits themselves, for "simple acceptance" --
CLAUDE.md §5's starting point; narrower, for a guard-banded rule, see
`DecisionRule`). Two things can go wrong:

  - **False acceptance (PFA)**: `Y` is actually outside tolerance (a bad
    part), but noise on this particular measurement happened to land `X`
    inside the acceptance interval anyway -- the part is shipped when it
    shouldn't be.
  - **False rejection (PFR)**: `Y` is actually inside tolerance (a good
    part), but noise landed `X` outside -- the part is scrapped or
    reworked when it didn't need to be.

**Why this needs a population, not just one measurement.** A single
measurement's own error doesn't, by itself, say how likely a wrong
decision is -- that depends on where the *true* value was likely to be in
the first place. A part solidly in the middle of a tight, well-centred
process essentially cannot be pushed outside tolerance by realistic
measurement noise; a part sitting right at a tolerance limit can be
tipped either way by a small amount of noise. This is exactly what
CLAUDE.md §3 means by "risk depends on... a prior distribution over the
true value representing process capability": the *global* risk (the
quantity this module returns) is the measurement error's effect,
integrated over that population:

```
PFA = integral over y outside tolerance of:
        f_prior(y) * P(measured value lands in acceptance interval | true value = y) dy

PFR = integral over y inside tolerance of:
        f_prior(y) * P(measured value lands outside acceptance interval | true value = y) dy
```

with `P(accept | Y=y)` itself a Gaussian tail probability, since the
measurement error `E` is modelled as `N(0, uncertainty_m^2)` (consistent
with every other layer in this codebase treating sensor/measurement
noise as Gaussian). `f_prior` is the process prior (`ProcessPrior`,
Gaussian here too -- see below). Both integrals are one-dimensional and
computed numerically (`scipy.integrate.quad`); there's no need for a
closed form.

**The prior lives on a feature-cluster, not a characteristic.** CLAUDE.md
§3: LVM is small-batch, and classical process capability statistics need
~50 samples per characteristic that aerospace manufacturing simply
doesn't supply. Instead, a prior is elicited once for a *cluster* of
characteristics that share a production process (Schmitt et al.) --
`FeatureCluster` holds that shared prior, and `ClusterAssignment` records,
explicitly (per CLAUDE.md §5's own instruction), which cluster a given
characteristic draws its prior from. Because that assignment is itself a
judgement call, not a measurement, `ClusterAssignment.confidence` carries
how sure it is, and every `RiskResult` reports it back -- CLAUDE.md §3:
"cluster assignment carries its own uncertainty and must be reported."
Step 5 does *not* yet feed that confidence into the risk numbers
themselves (e.g. by mixing in a second cluster's prior weighted by
`1 - confidence`) -- it's reported, not (yet) propagated. A principled way
to do that is a plausible next elaboration, not required by this step's
gate, so it isn't built (CLAUDE.md §9).

**Scope: one-dimensional tolerances only, for now.** The risk integral
above is a classical 1D bilateral-tolerance calculation: a scalar `Y`
against a scalar `[lower, upper]` interval. That maps directly onto a
step 4 `FlatnessTolerance` or `ParallelismTolerance` characteristic (a
single projected direction, `uncertainty_m` is exactly that direction's
standard deviation). It does *not* map exactly onto a step 4
`PositionTolerance` with more than one constrained direction: a spherical
position tolerance's true acceptance region is a ball, not an interval,
and the magnitude of a multi-dimensional Gaussian deviation follows a
noncentral chi-type distribution, not a Gaussian one. Passing a position
characteristic's RSS `uncertainty_m` into this module still gives a
directionally sensible number (bigger uncertainty relative to the zone
size still means more risk), but it is an approximation, not the exact
multivariate calculation -- documented here rather than silently treated
as exact. Building the exact multivariate version is future work.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


@dataclass(frozen=True)
class ProcessPrior:
    """A Gaussian prior over a feature-cluster's true values -- what the
    production process itself does, independent of how well any one part
    is measured.

    Values are in the same "deviation from nominal target" convention
    `DecisionRule` uses: `mean_m = 0` means the process is perfectly
    centred on the nominal target; a nonzero mean represents a systematic
    process offset/bias.

    Attributes
    ----------
    mean_m : the process's mean true deviation from nominal.
    std_m : the process's own natural spread (NOT measurement
        uncertainty -- this is what varies part-to-part even measured
        perfectly).
    """

    mean_m: float
    std_m: float

    def __post_init__(self):
        if self.std_m <= 0:
            raise ValueError(f"std_m must be positive, got {self.std_m}")

    def cpk(self, tolerance_lower_m: float, tolerance_upper_m: float) -> float:
        """One-sided process capability index (CLAUDE.md glossary: 'how
        much room the process has inside the tolerance'), the smaller --
        worse -- of the two sides. Informational: doesn't feed into the
        risk integral directly (the integral uses the prior distribution
        itself), but is the standard single-number summary a metrologist
        would ask for, and is what this module calls a "high-capability
        process" when it's large (commonly Cpk >~ 1.33)."""
        upper_capability = (tolerance_upper_m - self.mean_m) / (3.0 * self.std_m)
        lower_capability = (self.mean_m - tolerance_lower_m) / (3.0 * self.std_m)
        return min(upper_capability, lower_capability)


@dataclass(frozen=True)
class FeatureCluster:
    """A cluster of characteristics sharing one production process, and
    therefore one process prior (CLAUDE.md §3)."""

    name: str
    prior: ProcessPrior


@dataclass(frozen=True)
class ClusterAssignment:
    """Explicit record of which feature-cluster a characteristic's prior
    is drawn from, and how confident that assignment is.

    confidence : float in [0, 1]. 1.0 = certain. Reported in every
        `RiskResult` (CLAUDE.md §3); not yet used to adjust the risk
        calculation itself (see module docstring).
    """

    characteristic_name: str
    cluster: FeatureCluster
    confidence: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class DecisionRule:
    """A conformity decision rule: accept if the measured deviation from
    nominal falls within [acceptance_lower_m, acceptance_upper_m].

    "Simple acceptance" (CLAUDE.md §5 step 5's starting point, and what
    `simple_acceptance` / `from_zone_width` below construct) means the
    acceptance limits equal the tolerance limits exactly -- JCGM 106's
    baseline rule, no guard band. Passing narrower acceptance limits than
    the tolerance limits gives a guard-banded rule instead (CLAUDE.md
    glossary: "acceptance limit pulled inside the tolerance limit to
    control risk") -- supported by this same dataclass for later use, but
    not exercised by step 5's own gate, which only asks about simple
    acceptance.
    """

    tolerance_lower_m: float
    tolerance_upper_m: float
    acceptance_lower_m: Optional[float] = None
    acceptance_upper_m: Optional[float] = None

    def __post_init__(self):
        if self.tolerance_lower_m >= self.tolerance_upper_m:
            raise ValueError("tolerance_lower_m must be less than tolerance_upper_m")
        if self.acceptance_lower_m is None:
            object.__setattr__(self, "acceptance_lower_m", self.tolerance_lower_m)
        if self.acceptance_upper_m is None:
            object.__setattr__(self, "acceptance_upper_m", self.tolerance_upper_m)

    @classmethod
    def simple_acceptance(cls, tolerance_lower_m: float, tolerance_upper_m: float) -> "DecisionRule":
        return cls(tolerance_lower_m, tolerance_upper_m)

    @classmethod
    def from_zone_width(cls, zone_width_m: float) -> "DecisionRule":
        """A symmetric bilateral tolerance of total width `zone_width_m`
        about the nominal target -- the natural reading of a step 4
        `FlatnessTolerance`/`ParallelismTolerance`'s `zone_width_m`."""
        half_width_m = zone_width_m / 2.0
        return cls.simple_acceptance(-half_width_m, half_width_m)


@dataclass(frozen=True)
class RiskResult:
    """Global conformity risk for one characteristic.

    Attributes
    ----------
    characteristic_name, cluster_name, cluster_assignment_confidence :
        carried through from the `ClusterAssignment` for full traceability
        (CLAUDE.md §3: assignment confidence must be reported).
    probability_false_acceptance, probability_false_rejection : the two
        global risk numbers CLAUDE.md §5 step 5 asks for.
    process_cpk : the process prior's one-sided Cpk against this
        characteristic's tolerance limits -- informational, the standard
        single-number capability summary.
    """

    characteristic_name: str
    cluster_name: str
    cluster_assignment_confidence: float
    probability_false_acceptance: float
    probability_false_rejection: float
    process_cpk: float


def _acceptance_probability_given_true_value(
    true_value_m: float, uncertainty_m: float, decision_rule: DecisionRule
) -> float:
    """P(measured value lands in the acceptance interval | true value),
    for Gaussian measurement error X | Y=y ~ N(y, uncertainty_m^2)."""
    if uncertainty_m <= 0.0:
        in_interval = decision_rule.acceptance_lower_m <= true_value_m <= decision_rule.acceptance_upper_m
        return 1.0 if in_interval else 0.0
    z_upper = (decision_rule.acceptance_upper_m - true_value_m) / uncertainty_m
    z_lower = (decision_rule.acceptance_lower_m - true_value_m) / uncertainty_m
    return float(norm.cdf(z_upper) - norm.cdf(z_lower))


def _prior_support_bounds(prior: ProcessPrior, n_sigma: float = 12.0):
    """A finite window that captures essentially all of the prior's
    probability mass.

    `scipy.integrate.quad` is asked here to integrate a prior density
    that's often sharply peaked (std of a few micrometres) and can be
    centred well away from the tolerance limits (a biased process) --
    against SI-unit (metre) coordinates, that peak sits at a tiny
    absolute x-value. Integrating literally to +-infinity in that regime
    is not just unnecessary, it's unreliable: quad's infinite-interval
    transform can fail to locate a narrow, offset peak and silently
    report a near-zero result with a deceptively small estimated error
    (confirmed directly -- with a peak at 6e-5 m, `quad(f, 5e-5, np.inf)`
    returned exactly 0.0, while the same integrand over the finite range
    `[5e-5, mean + 10*std]` correctly returned ~0.13). Integrating over a
    wide but *finite* window the prior's own mean and std define sidesteps
    this failure mode entirely, at a truncation error smaller than
    `norm.sf(12) ~ 1e-33` -- utterly negligible next to floating point
    precision.
    """
    return prior.mean_m - n_sigma * prior.std_m, prior.mean_m + n_sigma * prior.std_m


def _global_false_acceptance(
    prior: ProcessPrior, uncertainty_m: float, decision_rule: DecisionRule
) -> float:
    def integrand(y):
        return norm.pdf(y, prior.mean_m, prior.std_m) * _acceptance_probability_given_true_value(
            y, uncertainty_m, decision_rule
        )

    prior_low, prior_high = _prior_support_bounds(prior)
    below_tolerance = 0.0
    if prior_low < decision_rule.tolerance_lower_m:
        below_tolerance, _ = quad(integrand, prior_low, decision_rule.tolerance_lower_m)
    above_tolerance = 0.0
    if prior_high > decision_rule.tolerance_upper_m:
        above_tolerance, _ = quad(integrand, decision_rule.tolerance_upper_m, prior_high)
    return below_tolerance + above_tolerance


def _global_false_rejection(
    prior: ProcessPrior, uncertainty_m: float, decision_rule: DecisionRule
) -> float:
    def integrand(y):
        return norm.pdf(y, prior.mean_m, prior.std_m) * (
            1.0 - _acceptance_probability_given_true_value(y, uncertainty_m, decision_rule)
        )

    value, _ = quad(integrand, decision_rule.tolerance_lower_m, decision_rule.tolerance_upper_m)
    return value


def evaluate_conformity_risk(
    assignment: ClusterAssignment, uncertainty_m: float, decision_rule: DecisionRule
) -> RiskResult:
    """Global PFA and PFR for one characteristic (CLAUDE.md §5, step 5).

    `uncertainty_m` is the characteristic's measurement uncertainty --
    directly what step 4's `CharacteristicPointUncertainty.uncertainty_m`
    provides (see also `evaluate_characteristic_risk` below, which takes
    that object directly). `assignment` supplies the process prior (via
    its cluster, explicitly) and is reported back in full on the result.
    """
    prior = assignment.cluster.prior
    return RiskResult(
        characteristic_name=assignment.characteristic_name,
        cluster_name=assignment.cluster.name,
        cluster_assignment_confidence=assignment.confidence,
        probability_false_acceptance=_global_false_acceptance(prior, uncertainty_m, decision_rule),
        probability_false_rejection=_global_false_rejection(prior, uncertainty_m, decision_rule),
        process_cpk=prior.cpk(decision_rule.tolerance_lower_m, decision_rule.tolerance_upper_m),
    )


def evaluate_characteristic_risk(
    characteristic_uncertainty, assignment: ClusterAssignment, decision_rule: DecisionRule
) -> RiskResult:
    """Convenience wrapper: takes a step 4
    `characteristics.characteristic.CharacteristicPointUncertainty`
    directly rather than a bare `uncertainty_m` float, so the risk layer
    can be chained straight onto the characteristic layer's output.
    Delegates entirely to `evaluate_conformity_risk`.
    """
    return evaluate_conformity_risk(assignment, characteristic_uncertainty.uncertainty_m, decision_rule)
