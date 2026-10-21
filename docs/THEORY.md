# Theory

Three laws. Each one is a closed form in `nawah/theory.py`, and each one
is checked against Monte Carlo in `tests/test_mathematics.py`. If a closed
form and a simulation disagree, the closed form is wrong — several of them
were, and the tests are how it was found.

---

## Law 1 — the cost of the Nth skill

```
cost(N) = C1 · N^(-beta)
```

`beta` decides which of three regimes you are in:

| beta | Regime | What it means |
|---|---|---|
| `< 0` | interference | each skill makes the next harder |
| `= 0` | warehouse | storing, not learning |
| `> 0` | **learning to learn** | each skill makes the next cheaper |

This is the criterion, not a benchmark score. Chollet's formulation is
that intelligence is *skill-acquisition efficiency*, distinct from skill
itself — "skill is merely the output of the process of intelligence".
A system that improves at acquiring skills is doing something categorically
different from one that accumulates them.

**Measured:** cost per task falls **×1.29** from the first quarter of a
run to the last (8/8 seeds). The exponent on cost per acquired skill is
**+0.327 ± 0.228** (7/8 seeds positive).

**Why the ratio is the claim and the exponent is only indicative:** cost
per task is *censored* — it cannot fall below 1 or exceed `k_max`. Fitting
a power law to a censored, S-shaped variable produces a confident number
that does not survive testing. The form-free first-quarter/last-quarter
ratio has no such problem. The power-law specification test passes on 6/8
seeds, which is enough to say "roughly a power law" and not enough to quote
an exponent without that caveat.

---

## Law 2 — how much a library can hold

The probability that retrieval picks the right skill out of `m` competing
ones, where the quality gap between best and runner-up is `r` in units of
representation noise:

```
P(correct) = ∫ φ(x) · Φ(x + r)^(m-1) dx
```

This integral used to be approximated. The approximation was wrong by up
to **0.20** in absolute probability, which is enormous when the quantity
being predicted is itself a probability. It is now evaluated exactly; the
residual against Monte Carlo is **0.0048**.

Inverting it gives the **critical library size** `T*`: the largest library
that still retrieves correctly at a target precision.

```
T* = A · H^a        (H = model width)
```

and the theory links that exponent to how fast the quality gap closes as
the library grows:

```
theta = 1 / (2a)
```

- `theta = 1` — a bounded quality distribution, best skills crowd
  together — predicts exactly `a = 0.5`.
- `theta < 1` — the gap closes more slowly — gives a steeper exponent and
  a library whose ceiling climbs fast with model width.

**Measured:** `T* = 0.302 · H^1.320` (r² = 0.94), so **theta = 0.38**.

Two caveats, stated because they matter:

1. The two narrowest widths both return 18, which is a **floor**: a
   library smaller than the number of variants cannot cover every kind of
   task. Those points are censored and the exponent is overstated.
2. From `H = 256` the router stops failing inside the range searched. A
   separate run at `H = 2048` held precision **1.00 with a library of
   9,216 skills**.

**No ceiling is hard-coded anywhere in the retrieval dynamics.** The
library is grown until precision falls through a threshold and whatever
size that happens at is reported. (In an earlier version the ceiling was
typed in as a constant and then printed in the results table — measuring
it would have returned the input. That is not a measurement, it is an
echo.)

---

## Law 3 — what the context is worth

Total experience held constant while the number of distinct contexts the
library must serve grows. A context-blind router must average over all of
them; a contextual router can specialise.

| Contexts | Skills | Contextual | Context-blind |
|---|---|---|---|
| 4 | 16 | 1.000 | 1.000 |
| 18 | 72 | 1.000 | 0.611 |
| 36 | 144 | 0.444 | 0.111 |
| 72 | 288 | 0.208 | 0.028 |

The gain is zero when there is nothing to specialise over and grows as the
library is asked to serve more distinct situations. Experience does not
become more valuable by being concentrated; it becomes more valuable by
being *sorted*.

---

## What is assumed and what is measured

| Quantity | Status |
|---|---|
| `P(correct) = ∫ φ Φ^(m-1)` | derived, verified against Monte Carlo |
| `theta = 1/(2a)` | derived |
| `sigma = sigma0/√H` | **assumed** — the one place width enters |
| `T* = A·H^a` | measured, not assumed |
| `beta > 0` | measured (form-free ratio) |
| optimum branching factor | derived, `theory.optimal_branching` |
| skill difficulty profile | a design choice, not a law |

The assumption that model width enters only through `sigma` is the weakest
link in the chain. It is worth saying plainly: if a wider model changed
something other than the sharpness of its eye, Law 2's exponent would move.
