# The router

The router is the part of NAWAH that is easiest to get wrong and hardest to
notice getting wrong. This document is the record of how it works and what
was found broken in it.

---

## The scoring function

Every candidate at every level is scored the same way:

```
score(choice) = 0.30 · quality(choice)
              + 0.45 · posterior(choice, context)
              + 0.25 · similarity(task, choice)
              + noise,  noise ~ N(0, sigma0/√H)
```

- **quality** — `strength · (0.6 + 0.4 · win_rate)`. What the skill is
  worth in general.
- **posterior** — how well *this* choice has done in *this* context. The
  heart of it, described below.
- **similarity** — token Jaccard between what the task asks and what the
  skill claims. The semantic term.
- **noise** — representation noise. The only place model width enters.

An earlier version scored retrieval as `strength · (0.6 + 0.4 · win_rate)
+ noise`. There is no `task` in that formula. Retrieval was "pick the
biggest number, plus noise", and every claim about *semantic*
confusability was a story told about a mechanism that could not see the
question. Worse, no experiment that existed could have caught it: because
the corruption model rewarded only family-level match, retrieval quality
was invisible to every measurement available.

---

## The posterior, and the shrinkage

Each choice carries a Beta posterior over "how good is this choice, in
this context":

```
wins, losses          = record for (choice, context)
global wins/losses    = record for (choice, *)           [all contexts]
p_global              = (gw + a) / (gw + gl + a + b)

alpha = wins   + m · p_global       + 0.5
beta  = losses + m · (1 - p_global) + 0.5
```

The shrinkage toward the global record is not decoration. A few hundred
tasks spread over a few dozen contexts leaves **well under one observation
per cell**. Without shrinkage, every choice you have barely tried looks
like a coin flip, and routing degenerates.

`m` is the weight of that prior, expressed in pseudo-observations. The
default is a constant `2.0`. An adaptive form, `m / (1 + n/m)`, releases
the prior's grip as evidence accumulates. The adaptive form is the more
principled of the two — a fixed `m` is too strong for a context seen a
hundred times and too weak for one seen twice — **and it did not win.**
Both are implemented, switchable via `SkillStore(adaptive=...)`, and the
disagreement is left visible on purpose.

---

## Explore vs commit

This is the bug that took the longest to find, and the reason this document
exists.

**Routing precision was 0.55.** The router picked the right skill barely
more often than a coin, despite 3,398 recorded routing observations sitting
in the bandit.

The cause was Thompson sampling. Thompson draws *from* the posterior, so a
half-known option routinely draws higher than a well-known better one.
Concretely, measured: a skill with a record of **(3 wins, 1 loss) drew
0.97** and beat the correct skill's **(7, 0) drawing 0.81**.

That behaviour is correct **while learning** — it is how a router
discovers options it has barely tried. It is wrong **at test time**, where
it throws away knowledge that was already paid for.

### The fix

```python
route(task, explore=True)    # Thompson, during training
route(task, explore=False)   # UCB, at evaluation and composition
```

UCB scores `mean + c · sqrt(variance)` with `c = 0.5`. The bonus is large
when evidence is thin and vanishes when it is thick, so the router
explores early and commits late **with no schedule to tune and no switch
to flip**.

| | Routing precision |
|---|---|
| Thompson at test time | 0.55 |
| UCB at test time | **0.70** |

The constant matters and was swept, not guessed:

| `ucb_c` | Precision |
|---|---|
| 0.0 (pure mean) | 0.661 |
| 0.3 | 0.699 |
| **0.5** | **0.704** |
| 1.0 | 0.619 |
| 1.5 | 0.563 |
| 2.5 | 0.486 |

A large bonus is as bad as Thompson: it reintroduces exploration exactly
where it is no longer wanted.

### Why it cannot come back

`tests/test_router.py::test_ucb_beats_thompson_at_test_time` re-runs the
ablation on three seeds and fails if UCB stops beating Thompson or if
precision drops below 0.60.

---

## The measurements behind it

Per-variant routing precision with a 62-skill library, before the fix:

| Variant | Precision | Skills covering it |
|---|---|---|
| case:upper | 0.93 | 10 |
| reorder:reverse | 0.93 | 9 |
| strip:nodigits | 0.67 | 10 |
| strip:nospaces | 0.43 | 6 |
| case:swapcase | 0.20 | 7 |
| case:title | 0.17 | 5 |
| strip:nodots | 0.10 | 5 |

Skills covering `strip:nodots` existed — five of them. The router simply
kept reaching past them. That ruled out "the library lacks coverage" and
pointed straight at the decision rule.

---

## The ceiling this produces

Because the router is the thing that limits library size, a better router
moves the ceiling. With UCB, `T* = 0.302 · H^1.320` (r² = 0.94), and at
`H = 2048` precision held at **1.00 with a library of 9,216 skills**.
Before the fix the same measurement returned `T* = 2 · H^0.623`.

A better router does not just answer more questions correctly. It raises
how much the system is allowed to know.
