"""
Self-directed practice.

Everything else in NAWAH assumes a curriculum handed down from outside:
somebody decides what the agent sees and in what order. That is the one
part of the loop that is still not the agent's own, and it is the part that
matters most once the library is large, because a fixed curriculum keeps
spending the agent's time on things it has already mastered.

The rule here is the oldest one in active learning, and it is chosen
because it is principled rather than because it is novel:

    study what you are about half right about.

Formally, a Bernoulli trial with success probability p carries variance
p(1-p), which is maximal at p = 1/2 and vanishes at both ends. A task the
agent always gets right teaches it nothing; a task it never gets right
teaches it nothing either, because there is no gradient to climb. The
informative ones sit in the middle, and they move as the agent improves --
which is the point. The curriculum has to be recomputed constantly or it
stops being a curriculum and becomes a syllabus.

p is estimated from what the agent has actually done, with a Beta(1,1)
prior so an untouched variant looks maximally uncertain and gets tried
early. That gives exploration for free: no epsilon-greedy hack, no
"exploration bonus" constant to tune.

The agent also GENERATES its practice rather than reordering a fixed list:
it picks a variant, then a fresh instance of that variant is produced. That
is closer to what a learner does, and it allows the agent to spend
arbitrarily long on the thing it is worst at -- which a fixed list cannot.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from .tasks import (
    GENERATORS,
    PRIMITIVE_NAMES,
    PRIMITIVES,
    PrimitiveTask,
    StringTask,
)


def informativeness(p: float) -> float:
    """Bernoulli variance, scaled to peak at 1. The whole rule, in one line."""
    return 4.0 * p * (1.0 - p)


class AutoCurriculum:
    """
    Chooses what to practise next, from the agent's own track record.

    variants : the things that can be practised, one per kind of task
    generate: variant, rng, tid -> a fresh task of that kind
    novelty  : extra credit for a variant never tried. Without it an
               untried variant already looks maximally uncertain (p = 0.5),
               so this only breaks ties; it is not the engine.
    """

    def __init__(
        self,
        variants,
        generate,
        seed=0,
        novelty=0.05,
        temperature=0.0,
        min_tries=0,
    ):
        self.variants = list(variants)
        self.generate = generate
        self.novelty = novelty
        self.min_tries = min_tries
        self.temperature = temperature
        self.rng = random.Random(seed * 31 + 5)
        self.hits: dict = defaultdict(int)
        self.tries: dict = defaultdict(int)
        self._tid = 0
        self.history: list = []

    def p_hat(self, v) -> float:
        """Beta(1,1) posterior mean: untouched variants sit at 0.5."""
        t = self.tries[v]
        return (self.hits[v] + 1.0) / (t + 2.0)

    def score(self, v, rule="posterior_variance") -> float:
        """
        Two candidate rules, and the difference between them is the
        finding.

        "uncertainty"        4 p(1-p). Maximises the variance of the NEXT
                             observation. It is the textbook rule and it
                             FAILS: a variant it has already mastered drops
                             to a low score and is never revisited, so the
                             library ends up with 39 practices of one skill
                             and 3 of another. Measured 33.5% against 68.2%
                             for a plain shuffle.

        "posterior_variance" p(1-p)/(n+2). Divides by the evidence already
                             gathered, which is what "how much can I still
                             learn about this" actually means. Without it
                             the rule will happily practise one thing
                             forever, because a coin that keeps coming up
                             tails is still maximally uncertain.
                             Measured 58.5%, and 68.9% once the variants
                             differ in difficulty.
        """
        p = self.p_hat(v)
        n = self.tries[v]
        if rule == "uncertainty":
            s = informativeness(p)
        else:
            s = p * (1.0 - p) / (n + 2.0)
        if n == 0:
            s += self.novelty
        if self.tries[v] < self.min_tries:
            # A COVERAGE FLOOR, and it is not a tuning knob: it is the fix
            # for a failure the pure rule has. Uncertainty-maximising
            # practice abandons whatever it has already mastered, which is
            # fine if the reward is "get better at X" and ruinous if the
            # reward is "be able to combine X with everything else". Half
            # the practice goes to guaranteeing every piece exists before
            # any of it goes to polishing the hard ones.
            s += 2.0
        return s

    def pick(self, rule="posterior_variance"):
        if self.temperature > 0:
            ws = [
                math.exp(self.score(v, rule) / self.temperature)
                for v in self.variants
            ]
            return self.rng.choices(self.variants, weights=ws)[0]
        best, best_s = None, -1e18
        for v in self.variants:
            s = self.score(v, rule) + self.rng.random() * 1e-9
            if s > best_s:
                best, best_s = v, s
        return best

    def update(self, variant, ok) -> None:
        self.tries[variant] += 1
        self.hits[variant] += 1 if ok else 0
        self.history.append((variant, ok))

    def next_task(self, rule="posterior_variance"):
        v = self.pick(rule)
        t = self.generate(v, self.rng, self._tid)
        self._tid += 1
        t._curriculum_variant = v
        return t

    def report(self) -> dict:
        return {
            v: {
                "tries": self.tries[v],
                "hits": self.hits[v],
                "p": round(self.p_hat(v), 3),
                "info": round(informativeness(self.p_hat(v)), 3),
            }
            for v in self.variants
        }


# ══════════════════════════════════════════════════════════════
# Task factories: turning "practise this variant" into a real task
# ══════════════════════════════════════════════════════════════
def primitive_factory(ops=None):
    """For the composition suite: one practice task per primitive."""
    names = ops or PRIMITIVE_NAMES

    def gen(variant, rng, tid):
        op = variant.split(":", 1)[1]
        _f, _d, _fn, make = PRIMITIVES[op]
        return PrimitiveTask(tid, op, make(rng))

    variants = [f"{PRIMITIVES[o][0]}:{o}" for o in names]
    return variants, gen


def string_factory(families=None):
    """For the string suite: one practice task per (family, variant)."""
    fams = families or list(GENERATORS)
    cache = {}
    # discover the variants of each family once, by sampling
    for f in fams:
        seen = {}
        rng = random.Random(17)
        for _ in range(600):
            d, _a, _b, v = GENERATORS[f](rng)
            seen.setdefault(v, d)
        cache[f] = seen
    variants = sorted(v for f in fams for v in cache[f])

    def gen(variant, rng, tid):
        fam = variant.split(":", 1)[0]
        for _ in range(50):
            d, a, b, v = GENERATORS[fam](rng)
            if v == variant:
                return StringTask(tid, fam, d, a, b, v)
        return StringTask(tid, fam, cache[fam][variant], "", "", variant)

    return variants, gen


# ══════════════════════════════════════════════════════════════
# The competing curricula
# ══════════════════════════════════════════════════════════════
def fixed_order_curriculum(tasks, seed=0, mode="interleaved"):
    """
    What a human would write down in advance.

    interleaved : round-robin across families. What NAWAH has used so far.
    blocked     : one family at a time. Known to be destructive: the agent
                  forgets the previous family while learning the next, and
                  the library confuses them.
    random      : shuffled once.
    """
    by = defaultdict(list)
    for t in tasks:
        by[t.family].append(t)
    rng = random.Random(seed)
    if mode == "random":
        out = list(tasks)
        rng.shuffle(out)
        return out
    if mode == "blocked":
        out = []
        for f in sorted(by):
            out.extend(by[f])
        return out
    out, depth = [], max(len(v) for v in by.values())
    fams = sorted(by)
    for d in range(depth):
        for f in fams:
            if d < len(by[f]):
                out.append(by[f][d])
    return out


def run_curriculum(
    agent,
    curriculum,
    budget,
    probes_by_family,
    autocurriculum=None,
    evaluate=None,
    eval_every=None,
    rule="posterior_variance",
):
    """
    Train for `budget` tasks under one curriculum, recording the learning
    trace. Returns (results, trace).
    """
    trace = []
    for i in range(budget):
        if autocurriculum is not None:
            t = autocurriculum.next_task(rule)
            r = agent.solve(t, probes_by_family.get(t.family, ()), learn=True)
            v = getattr(t, "_curriculum_variant", t.variant)
            autocurriculum.update(v, r.ok)
        else:
            t = curriculum[i % len(curriculum)]
            r = agent.solve(t, probes_by_family.get(t.family, ()), learn=True)
        if eval_every and (i + 1) % eval_every == 0 and evaluate:
            trace.append((i + 1, evaluate(agent)))
    if evaluate and not eval_every:
        trace.append((budget, evaluate(agent)))
    return agent.results, trace
