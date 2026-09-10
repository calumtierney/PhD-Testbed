# Step 5 physics: why a high-capability process shrugs off uncertainty

Companion to `src/risk/jcgm106.py`. Read this before the code, or to
check the physics independently of it.

## The question this step actually answers

Every step before this one produces an uncertainty number. None of them
ask whether that uncertainty *matters*. It usually doesn't matter equally
everywhere: a measurement uncertain by 10 µm is nearly irrelevant to a
part that's nowhere near its tolerance limit, and can be decisive for a
part that's sitting right on one. Step 5 is where "how uncertain is this
measurement" turns into "how often does this uncertainty actually cause a
wrong ship/scrap decision" — CLAUDE.md's whole thesis (§1): decision risk,
not raw uncertainty, is the thing worth planning around.

## Why a single measurement's error isn't enough

Take one measured value, `X`, with its own known uncertainty. On its own,
that tells you nothing about how likely a wrong decision is — it depends
entirely on where the *true* value was likely to have been, which the one
measurement alone doesn't reveal (that's the whole reason it needed
measuring). What's needed is a model of the *population* the part came
from: the process. A process centred well inside tolerance, with a tight
spread, essentially never produces a true value anywhere near a limit —
almost no realistic amount of measurement noise can flip the decision. A
process sitting close to a limit, or spread wide relative to the
tolerance band, regularly produces true values right where a small amount
of noise *can* flip the decision. This population is the "prior
distribution over the true value representing process capability"
CLAUDE.md §5 asks for, and `ProcessPrior` is exactly that: not a
statement about any one part, but about what the process as a whole does.

## The two ways a decision can be wrong

- **False acceptance (PFA)**: the true value is outside tolerance (a bad
  part), but the measurement's own noise happened to land inside the
  acceptance interval — a bad part ships.
- **False rejection (PFR)**: the true value is inside tolerance (a good
  part), but noise landed outside — a good part gets scrapped or reworked
  for nothing.

Both are computed the same way: weight every possible true value `y` by
how likely the process is to actually produce it (`f_prior(y)`), then by
how likely a measurement of that particular `y` is to land on the *wrong*
side of the acceptance boundary, and add it all up:

```
PFA = integral over y outside tolerance:  f_prior(y) * P(measured value lands inside acceptance | true value = y) dy
PFR = integral over y inside tolerance:   f_prior(y) * P(measured value lands outside acceptance | true value = y) dy
```

`P(... | true value = y)` is a plain Gaussian tail probability, since
measurement error is modelled as `N(0, uncertainty_m^2)` — the same
Gaussian-noise assumption every other layer in this codebase makes.

## Why a high-capability process is insensitive to uncertainty

Picture the process prior as a bell curve sitting well inside the
tolerance band, several standard deviations from either limit (a high
Cpk, by definition). For a true value drawn from the *middle* of that
curve, even a fairly generous measurement uncertainty would have to swing
the reading by several of the process's own standard deviations — which
is also several tolerance-widths' worth of the *remaining room* — before
it could plausibly cross a limit. The probability of a Gaussian doing
that is small, and shrinks *fast* (the tail of a Gaussian is famously
thin) as the required swing grows relative to the noise. Meanwhile,
essentially none of the process's probability mass sits near enough to a
limit for a modest amount of noise to matter at all. Both false
acceptance and false rejection stay tiny — not because the measurement is
perfect, but because the part being measured was never close to the edge
in the first place. That's the whole content of "a high-capability
process shows near-zero risk even when uncertainty is moderately large"
(CLAUDE.md §5's step 5 gate): it's a statement about the *process*, and
the measurement uncertainty has to grow enormously (relative to how much
room the process has) before it starts to matter.

## Why a marginal process is the opposite

Now picture the prior's mean sitting close to a limit — a process barely
capable, or an off-centre one. A large fraction of its probability mass
is already near the boundary. For *those* true values, only a small
amount of measurement noise is needed to push the reading across the
line — and the required noise is directly `uncertainty_m` itself, so as
`uncertainty_m` grows, more and more of that near-the-edge mass becomes
newly reachable across the boundary. Risk grows quickly and substantially
with uncertainty, rather than staying flat — the opposite behaviour from
the high-capability case, from the exact same measurement error model.

## Worked numbers

`tests/test_jcgm106.py` uses a ±50 µm tolerance (CLAUDE.md §4d's own
reference scale) and two processes, swept across the uncertainty range
this codebase's earlier steps actually produce (1-14 µm):

| Uncertainty | High-capability (mean 0, std 5 µm, Cpk ≈ 3.3) total risk | Marginal (mean 40 µm, std 6 µm, Cpk ≈ 0.56) total risk |
|---|---|---|
| 1 µm | ~1e-22 | 0.8% |
| 5 µm | ~2e-12 | 6.8% |
| 8 µm | ~1e-7 | 12.9% |
| 14 µm | ~8e-4 | 22.9% |

The high-capability column crosses roughly *nineteen orders of
magnitude* and never leaves "utterly negligible". The marginal column
moves from "occasionally rejecting a good part" to "rejecting almost a
quarter of good parts" over the same, realistic uncertainty range. Same
measurement physics, same integral, wildly different sensitivity —
purely a consequence of how much room each process actually has.

## A numerical trap this module had to avoid

`scipy.integrate.quad` is the obvious tool for these integrals, and the
"outside tolerance" parts of PFA naturally look like they want infinite
bounds (`-infinity` to the lower limit, the upper limit to `+infinity`).
That turned out to be actively wrong to do directly: with a process prior
that's both sharply peaked (a few micrometres wide) and offset from the
tolerance limits (a biased process), `quad`'s infinite-interval transform
can fail to locate that narrow peak at all, and silently report a result
of exactly `0.0` with a deceptively tiny estimated error — not a warning,
not a NaN, just a plausible-looking wrong answer. Caught directly during
this step's own testing: a process centred 60 µm outside a ±50 µm
tolerance evaluated to `probability_false_acceptance = 0.0`, which is
obviously wrong for a process that's mostly outside tolerance being
measured with any realistic noise. `_prior_support_bounds` sidesteps this
by integrating over a wide but *finite* window sized from the prior's own
mean and standard deviation (±12σ, truncation error ~1e-33) instead of
literal infinity — see its docstring for the reproduction. Worth keeping
in mind for any *other* infinite-domain integral added later in this
codebase: infinite bounds are not automatically the "more correct,
harmless" choice next to a finite approximation.
