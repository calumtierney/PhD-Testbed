# Voiceover: Global decision risk (`docs/global_risk_explained.pptx`)

A spoken script for the 13-slide deck, one section per slide. It explains each
calculation from first principles: what is being counted, why, and how it is
computed. At a normal speaking pace (about 150 words a minute) it runs to
roughly 18 minutes. Text in *[square brackets]* is a stage direction, not
something to read aloud.

Every number quoted comes from the testbed's own risk code, using the
wing-box experiment's **placeholder** process priors. They illustrate the
mechanics; they are not measured process data.

---

## Slide 1 — Title

This presentation answers two questions. First: when we measure a part and
decide whether it passes, how often is that decision wrong, and how do we
calculate it? Second: that calculation needs to know how the manufacturing
process behaves, so how do we get that knowledge in aerospace, where we only
ever build a handful of parts?

I'll build everything up from the basics. By the end you should be able to
see where each number comes from, and which numbers we can trust today and
which are still placeholders.

---

## Slide 2 — A measurement is not the truth

*[Point to the number line on the left.]*

Every feature on a part has a true deviation from its nominal CAD position.
Call it Y. We never see Y directly. What we see is the measured value, X, and
X is Y plus some measurement error, E. No instrument is perfect, so E is never
exactly zero.

The tolerance is the band from minus T to plus T. Our rule is simple: if the
measured value X lands inside the band, we accept the part; if it lands
outside, we reject it.

Here is the problem. The blue dot is a true value just inside the limit, so
this is a good part. But the measurement error has pushed the measured value,
the red cross, just outside. We reject a good part.

*[Move to the table on the right.]*

If you list every combination, the true part is either good or bad, and the
measurement either accepts or rejects. That gives four boxes. Two of them are
correct. The other two are the mistakes.

A **false reject** is a good part that we scrap or rework. That costs money
and yield. Its probability is written PFR.

A **false accept** is a bad part that we ship. On a safety-critical part, that
is the one that matters. Its probability is written PFA.

Notice that both mistakes only happen near a limit. A part sitting in the
middle of the tolerance can't be pushed out by a few micrometres of noise. So
how often we make mistakes depends on two things: how big the noise is, and
how many true parts sit close to a limit. Those are the two things we have to
model.

---

## Slide 3 — Three ingredients

To calculate the risk we need three ingredients.

**First, the measurement error.** We model E as a normal, or Gaussian,
distribution with a mean of zero and a standard deviation u. The mean is zero
because we assume the instrument isn't biased; u says how spread out the
errors are. About two-thirds of measurements land within plus or minus u of
the truth, and about 95 percent within plus or minus 2u.

Where does u come from? That is the rest of the testbed. We place the laser
tracker stations, work out which points each station can see, propagate the
tracker's angle and distance noise into a three-by-three covariance for every
point, and project that onto the surface normal, because a profile tolerance
only cares about error along the normal. The result is one number, u, per
point. This is the part of the problem the measurement planner controls.

**Second, the tolerance and decision rule.** The tolerance comes from the
drawing. Our rule is the simplest one in the JCGM 106 standard: accept if the
measured value is within the tolerance. A guard band would pull the
acceptance limit inside the tolerance. That cuts false accepts but raises
false rejects.

**Third, the process prior.** This is the part people usually forget. Before
we measure anything, the manufacturing process has already put the true
value somewhere. If we built thousands of parts, their true deviations would
form a distribution. We model it as another normal distribution, with a mean
mu, which is the process's systematic offset, and a standard deviation sigma,
its natural part-to-part spread. "Prior" just means what we believe before
the measurement arrives.

*[Point to the equation bar.]*

So the measured value is the true value, drawn from the process, plus an
error, drawn from the instrument.

One more piece of vocabulary. Because we average over the whole population of
parts the process makes, this is called **global** risk. It answers "across
all the parts we'll ever measure, how often is the decision wrong?" It is not
the risk for one particular reading on one particular day.

---

## Slide 4 — The equations

Let's build the equations one step at a time.

*[First line.]*

**Step one.** Suppose we knew the true value was exactly y. What's the
chance the measurement says "accept"?

The measured value is y plus an error with standard deviation u. We accept if
it lands between minus T and plus T. For a normal distribution, the chance of
landing below some value is given by a standard function, written capital
phi, the normal cumulative distribution function. You feed it "how many
standard deviations away is this value", and it gives you the probability of
being below it.

The upper limit T is "T minus y, divided by u" standard deviations above the
true value. The lower limit is "minus T minus y, divided by u". The chance of
landing between them is phi of the first minus phi of the second.

*[Point to panel (a) of the figure.]*

That's this curve. Deep inside the tolerance it's essentially one: you always
accept. Far outside it's essentially zero. Exactly at a limit it's one half,
because the error is equally likely to push you either side. The larger the
error, the more blurred the cliff. With u of 2 micrometres it's a sharp step;
at 20 micrometres it's a gentle slope.

*[Panel (b).]*

**Step two.** We don't know y. The process prior tells us how likely each
value of y is. Here are the two placeholder processes. The red one is
marginal: it's biased, sitting at 38 micrometres, close to the 50 micrometre
limit, with a narrow spread of 3 micrometres. The blue one is high
capability: centred at zero with plenty of room.

**Step three.** Combine them. Think of it as counting parts. Imagine the
process makes a very large number of parts. Slice the possible true values
into thin strips. For each strip, the prior tells you what fraction of parts
fall in it, and the curve from step one tells you what fraction of those get
accepted. Multiply, and you get the fraction of all parts that have that true
value *and* get accepted.

Now add up the strips. If you add up only the strips **outside** the
tolerance, you've counted parts that are truly bad but got accepted. That
total is **PFA**.

If you add up only the strips **inside** the tolerance, using one minus the
accept probability, you've counted parts that are truly good but got
rejected. That's **PFR**.

Adding up infinitely thin strips is what an integral is. That's all these two
equations say. In the code, `scipy.integrate.quad` does the adding numerically.

*[Panel (c).]*

This panel shows the two things being added up, for the marginal process at
u of 6.25 micrometres. The blue region, inside the tolerance, is good parts
being rejected. The red region, just past the limit, is bad parts being
accepted. The vertical axis is logarithmic, so trust the printed numbers, not
the areas. PFR is about four in a hundred, PFA about one and a half in a
hundred thousand.

---

## Slide 5 — What the equations say

Three properties fall straight out of those equations.

**One. PFA can never exceed the process's own out-of-tolerance rate.** The
accept probability is at most one. So the false-accept total can't be more
than the total number of bad parts the process makes in the first place. For
the marginal process that's about 3 in 100,000; for the high-capability one
about 2 in a billion. Those are the dashed lines on the left. Measuring can
only catch some of those bad parts. And a part we don't measure at all is
accepted by default, so it escapes at exactly that rate.

Look at how flat the solid lines are. Once the uncertainty is more than a few
micrometres, making it bigger or smaller barely moves the false-accept rate.
Why? For a marginal process, the bad parts sit just past the limit, within a
few micrometres of it. Any realistic tracker error is comparable to that
distance, so the measurement catches roughly half of them whatever you do.

**Two. PFR grows very steeply with uncertainty.** On the right, the marginal
process goes from about 5 in 100,000 at 1 micrometre, to 4 in 100 at 6.25
micrometres, to over a quarter at 20 micrometres. That's because the
marginal process has lots of good parts just inside the limit, and the bigger
the noise, the more of them get pushed out.

**Three. If u goes to zero, both go to zero.** A perfect measurement makes no
mistakes. And u is the lever the planner holds: which instrument, how far
away, at what angle, how many stations.

So the practical message: beyond a few micrometres, measuring harder barely
changes the safety risk, but it changes the false-reject cost a great deal.
And where you spend your measurement effort should depend on which process
is the marginal one.

---

## Slide 6 — Cpk is a label, not an input

People in manufacturing usually describe a process with a single number,
Cpk. It's defined as the distance from the process mean to the nearer
tolerance limit, divided by three standard deviations. A Cpk of 1.33 means
there's four standard deviations of room on the tight side.

But notice what the risk equations actually used. They used the prior's mean
and its standard deviation directly. Cpk never appears.

*[Point to the heat maps.]*

Each square here is a separate process, with a different mean along the
bottom and a different spread up the side. The colour is the risk. The white
lines join all the processes with the same Cpk. Cpk is just a way of labelling
lines across this map.

And the risk is not constant along those lines. Take two processes that both
have Cpk 1.33. One is biased towards a limit, mean 38, spread 3. The other is
centred, mean 0, spread 12.5. At the same 6.25 micrometre uncertainty, the
false-reject probability is about 4 in 100 for the first and 3 in 10,000 for
the second. That's roughly 140 times different, for the same Cpk.

The reason is that Cpk squashes two numbers into one, and the risk depends on
both. So "we can't compute Cpk" isn't really the obstacle. The real question
is: where do the mean and the spread come from?

---

## Slide 7 — The aerospace problem: too few builds

Classically you estimate a process's mean and spread from data, and a
standard rule of thumb is about fifty samples of the same characteristic.

A characteristic is one toleranced thing you check. In the wing-box
experiment, each of the 500 inspection points is one characteristic. One
point, on one wing, gives you one sample per build. Aerospace builds a
handful of wings, so you'll never have fifty samples of any single point. A
per-point Cpk would just be noise.

*[Point to the diagram.]*

The way out is to notice that characteristics aren't independent. The 60
points on the leading-edge profile, the 120 on the free-form surface: they're
all made by the same forming and machining process. A process that runs
slightly biased will tend to be biased on all of them.

So we group characteristics that share a production process into a
**cluster**, and give each cluster one prior. You now pool all the knowledge
you have about that process, instead of trying to learn 500 separate things.

Here the leading-edge tasks, T1 to T3, form a marginal cluster, and the flat
top skin and flaps, T4 to T6, a high-capability cluster. I want to be clear
that this assignment is my placeholder. Nothing I had says which wing-box
features are actually harder to hold. That has to come from people who know
the process.

---

## Slide 8 — Four ways to get mu and sigma without 50 builds

So how do we fill in each cluster's mean and spread? Here are four routes.
None of them is built into the testbed yet; they're the options.

**One: elicitation.** Ask process engineers the questions they can answer.
Not "what's the Cpk?", but "what's a typical deviation on this feature?" and
"what's the worst you'd see on one part in twenty?" For a normal
distribution, the typical value gives you the mean, and the one-in-twenty
value sits about 1.6 standard deviations beyond it. So two answers pin down
both numbers. This is the route CLAUDE.md specifies.

**Two: pooling.** A cluster can borrow data from anywhere the same process
has been used: non-conformance reports, first-article inspections, trial
coupons, machine capability runs, other programmes. No single feature has
fifty samples, but the cluster might.

**Three: use the measurements themselves.** One wing already gives us 60 to
120 measured points in a cluster. Their measured values are spread out for
two reasons: the process varies, and the measurement adds noise. When two
independent sources of scatter add, their variances add. So the variance of
what we measured equals sigma squared plus u squared. The digital twin knows
u for every point, so we can subtract it and recover sigma. The catches:
neighbouring points on a surface are correlated, so they're not truly
independent samples, and an instrument bias would look like a process
offset. And this only works after the first measurement, so it can't plan
build one.

**Four: update as you go.** Start with the elicited prior, and each time a
build is measured, combine the prior with the new data using Bayes' rule:
your updated belief is proportional to the prior times how well each possible
process explains what you just saw. Early on the prior dominates; as builds
accumulate, the data takes over. The plan for build five is better informed
than the plan for build one.

To be completely honest: today, every prior in the testbed is a placeholder.
Slide 12 shows which conclusions survive that.

---

## Slide 9 — Which cluster is a point in? That is uncertain too

There's a second layer of uncertainty. Even if we knew each cluster's prior
perfectly, assigning a point to a cluster is a judgement, and judgements can
be wrong.

So we give each assignment a confidence, c. With probability c the point
belongs to the cluster we labelled it; otherwise it belongs to the other one.
To get the point's risk, we take the risk under each possibility and weight
it by how likely that possibility is. That's just the law of total
probability: c times the risk if the label is right, plus one minus c times
the risk if it's wrong.

*[Point to the numbers.]*

Here's why this matters. Suppose we decide not to measure a point we've
labelled high capability. If we're certain of that label, skipping it costs 2
in a billion of escape risk, which is nothing.

But at 90 percent confidence, there's a 10 percent chance it's actually a
marginal point, which escapes 3 in 100,000 times. Ten percent of 3 in 100,000
is 3 in a million. So skipping that point now costs about 1,600 times more,
purely because of a 10 percent doubt about which box it's in.

In the wing-box experiment, that showed up directly. The risk-driven plan
needed 2 stations when the labels were certain, 3 at 90 percent, and 5 at 50
percent. How much you can safely leave unmeasured depends on how sure you are
of the labels.

---

## Slide 10 — From one characteristic to a whole part

So far everything has been for one point. A part has 500. How do we combine
them?

**Expected counts.** The expected number of false accepts on a part is just
the sum of the individual false-accept probabilities. The same goes for
false rejects. This works because expected values always add, whether or not
the points are related. If 500 points each have a 1 in 1,000 chance of being
falsely rejected, you expect half a false reject per part.

**The chance of at least one escape.** For safety, the question is usually
"what's the chance that anything bad gets through?" The easiest way to get
that is through the opposite. The chance a given point does *not* escape is
one minus its PFA. If the points are independent, the chance that *none* of
them escapes is all those numbers multiplied together. And the chance that at
least one does is one minus that product.

That independence assumption is an approximation. Points measured from the
same station share that station's position error, so their errors are
correlated. I treat it as a convenient summary, not an exact result.

**The conventional rule.** The common rule of thumb, from the ANSI/NCSL
Z540.3 standard, is a test uncertainty ratio of at least 4 to 1: the
tolerance half-width divided by the expanded uncertainty, which is 2u. Four
to one therefore means u must be no more than one eighth of the tolerance,
so 6.25 micrometres on a plus or minus 50 tolerance.

*[Point to the three numbers.]*

Now a striking result. With the placeholder priors, the chance of at least
one escape on a 500-point part is about 7.6 in 1,000 if we measure nothing at
all, 3.5 in 1,000 if every point is measured at the 4-to-1 limit, and 2.2 in
1,000 if every point is measured at 1 micrometre, which is about as good as
the tracker gets.

So measuring everything almost perfectly only cuts the escape risk by about
three and a half times compared with measuring nothing. That's slide 5's
first property again, at part level: the process sets most of the consumer's
risk. What the plan really decides is which points get measured at all, and
how many false rejects it causes.

---

## Slide 11 — What counts as acceptable risk?

All of this gives us numbers. Whether those numbers are acceptable is a
different question, and the honest answer is that I couldn't find a published
limit for safety-critical aerospace inspection.

The sourced anchors are these. The ANSI/NCSL Z540.3 standard caps the
probability of false accept at 2 percent per measurement result, and uses the
4-to-1 ratio only as a fallback when you can't compute risk. JCGM 106, ILAC-G8
and the NASA measurement-assurance handbook explain how to compute and
control risk, and how to set guard bands, but they don't give a universal
number.

None of these is aerospace-specific. And Z540.3's 2 percent is about 600
times looser than the largest per-point false-accept risk we compute, so it
wouldn't constrain anything.

So the testbed doesn't invent a limit. It compares plans on relative terms:
a risk-driven plan is held to the risk the conventional plan actually
delivers, or to what the 4-to-1 rule guarantees, and the limit is swept to
show what changes. If a programme has a real limit, it drops straight in.

That real limit, for false accepts and false rejects, and whether any
inspection task may be left partly unmeasured, has to come from the
programme. It's a quality and safety decision, not a modelling one.

---

## Slide 12 — What this means for planning

So what can we say today, with placeholder priors? The wing-box experiment
tested exactly that by changing one assumption at a time.

**Robust.** The risk-driven plan needed 3 stations where the conventional,
uncertainty-driven plan needed 9. That held for tolerances from plus or minus
25 to plus or minus 250 micrometres, and for both prior shapes, because the
risk-driven plan is always held to whatever the conventional plan actually
delivers under those same assumptions.

**Sensitive.** It depends on how sure we are of the cluster labels: 2
stations if we're certain, 5 at 50 percent. And you can't buy a stricter limit
than the conventional plan delivers, because the process sets a floor on the
escape risk.

**Vanishes.** If the points that are hard to see, the underside of the flaps,
turn out to be the marginal ones, or if policy says at least 75 percent of
every task must be measured, the saving disappears.

That tells us exactly what we need from the programme to make this real:
the mean and spread for each cluster, the cluster assignments and how
confident we are in them, the real tolerances, and the acceptable risk
including any minimum sampling rule.

---

## Slide 13 — Takeaways

Four things to take away.

**One.** PFA and PFR are integrals: you take the process prior, multiply by
the chance the measurement accepts, and add up over the bad region or the good
region. False accepts are capped by the process; false rejects are driven by
the measurement.

**Two.** The inputs are the process mean and spread, not Cpk. Cpk is just a
label for a line across that space, and two processes with the same Cpk can
differ by more than a hundred times in false rejects.

**Three.** With few builds, set priors per cluster of features that share a
process, using elicitation, pooled data, the wing's own measurements and
Bayesian updating, and always report how confident the cluster assignment is.

**Four.** Every number here rests on those priors. So plan across plausible
priors rather than one, and get the real values, and the acceptable-risk
limit, from the programme.

Thank you.
