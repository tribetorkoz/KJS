"""
NAWAH — the agent loop.

    route -> attempt -> verify -> (fail) forced decorrelation
                              -> (pass) distill
                                     |
                                A/B gate -> store or reject
                                     |
                                consolidate when a family outgrows its
                                clusters
                                     |
                                CREDIT GOES BACK TO THE ROUTER

That last arrow is the one that was missing. In v2 retrieval never learned:
it ranked skills by a number the task had no influence on, so the router
could not improve no matter how much the agent saw. Here every outcome is
fed back to the family, the cluster and the skill that produced it, so
routing is learned by experience rather than fixed by hand.

Two further things live here:

  * soundness is decided by measurement, not assumption. In the simulator we
    know whether a skill is corrupted (we corrupted it). With a real model
    we do not -- the gate alone decides, with a paired A/B test.
  * solve_composed: multi-step tasks, one skill per step. It is how a
    library becomes a plan, and it is the experiment that asks whether
    skills learned separately can be chained together.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from . import consolidate as C
from .gate import validate as gate_validate
from .skill import Skill, distill
from .store import SkillStore


@dataclass
class Result:
    tid: int
    family: str
    ok: bool
    cost: int = 0  # number of attempts
    tokens: int = 0
    seconds: float = 0.0
    skill_used: str = ""
    learned: str = ""
    # which route was actually taken, for anyone who wants to look
    route: str = ""
    # how many skills the router had to choose among on this task
    crowd: int = 0
    stages: int = 1


DISTILL_TMPL = """An agent succeeded at this task:

{desc}

Input:
{inp}

Correct output:
{out}

Write a reusable skill that helps solve this family of tasks, in exactly
this format and nothing outside it:

# Title of the skill

## When to use
When to use it, and when not to.

## Procedure
The general rule — do not just restate this one example.

## Test cases
- input -> output
(three cases, at least one of them different from the example above)"""


class NawahAgent:
    def __init__(
        self,
        backend,
        width=64,
        k_max=4,
        decorrelate=True,
        gate=True,
        consolidate=True,
        hierarchical=True,
        spurious_rate=0.35,
        gate_min_gain=0.10,
        gate_alpha=0.50,
        sigma0=0.55,
        branching=6,
        shrink=2.0,
        ucb_c=0.5,
        adaptive=False,
        seed=0,
        name="nawah",
    ):
        self.backend = backend
        self.store = SkillStore(
            width=width,
            seed=seed,
            hierarchical=hierarchical,
            sigma0=sigma0,
            branching=branching,
            shrink=shrink,
            ucb_c=ucb_c,
            adaptive=adaptive,
        )
        self.k_max = k_max
        self.decorrelate = decorrelate  # forced decorrelation
        self.gate = gate  # A/B gate
        self.consolidate = consolidate  # merge near the ceiling
        # Fraction of self-distilled skills that are corrupted —
        # simulator only. With a real model we don't know; the gate decides.
        self.spurious_rate = spurious_rate
        # Gate parameters are tunable, not buried in constants.
        # alpha=0.50 came from a sweep (0.05/0.20/0.50/0.90): being too
        # strict rejects sound skills and costs more than it protects.
        # ! Calibrated on my simulator — re-run the sweep on a real model.
        self.gate_min_gain = gate_min_gain
        self.gate_alpha = gate_alpha
        self.seed = seed
        self.name = name
        self.results: list[Result] = []
        self._rng = random.Random(seed * 977 + 13)

    # ── distillation ──────────────────────────────────────────
    def _distill(self, task, final_out):
        variant = getattr(task, "variant", "")
        if self.backend.name == "sim":
            # We corrupt part of them to check that the gate catches it.
            # The dangerous corruption is not a skill from the wrong family
            # -- that is easy to spot. It is a skill from the RIGHT family
            # that names the WRONG variant: it passes every superficial
            # check and then quietly does the wrong thing.
            if self._rng.random() < self.spurious_rate:
                from .tasks import family_variants, variant_descriptions

                alts = [
                    v
                    for v in family_variants().get(task.family, [])
                    if v != variant
                ]
                if not alts:
                    return distill(
                        task.family,
                        task.desc,
                        "[procedure]",
                        [(task.inp, final_out)],
                        variant,
                    )
                wrong = alts[self._rng.randrange(len(alts))]
                desc = variant_descriptions().get(wrong, task.desc)
                s = distill(
                    task.family,
                    desc,
                    "[corrupted procedure]",
                    [(task.inp, final_out)],
                    wrong,
                )
                s.truth = False  # we corrupt it to test the gate
                s.is_sound = False  # and the gate does not know yet
                return s
            return distill(
                task.family,
                task.desc,
                "[procedure]",
                [(task.inp, final_out)],
                variant,
            )

        # real model: it writes the skill, the gate judges it
        prompt = DISTILL_TMPL.format(
            desc=task.desc, inp=task.inp, out=final_out
        )
        body = self.backend.complete(prompt)
        return Skill(
            family=task.family,
            body=body,
            provenance="self",
            is_sound=False,
            variant=variant,
        )

    # ── one task ──────────────────────────────────────────────
    def solve(self, task, probes=(), learn=True):
        self.backend.begin_task()
        # learning: explore, so the router finds out about options
        # it has barely tried. Evaluation commits instead.
        route = self.store.route(task, explore=learn)
        skill = route.skill

        t0 = time.time()
        used, ok, out = 0, False, ""
        for i in range(self.k_max):
            used += 1
            out = self.backend.attempt(
                task, skill, decorrelate=self.decorrelate and i > 0
            )
            if task.verify(out):
                ok = True
                break
        tok = self.backend.cost.tokens_in + self.backend.cost.tokens_out
        r = Result(
            task.tid,
            task.family,
            ok,
            used,
            tok,
            time.time() - t0,
            skill.id if skill else "",
            route=">".join(route.levels) or "(empty)",
            crowd=self.store.last_crowd,
        )
        # the router gets told what happened, at every level it decided on
        self.store.record(route, ok)

        if ok and learn:
            cand = self._distill(task, out)
            accepted = True
            if self.gate:
                v = gate_validate(
                    self.store,
                    cand,
                    probes,
                    self.backend,
                    min_gain=self.gate_min_gain,
                    alpha=self.gate_alpha,
                )
                accepted = v.accepted
                if not accepted:
                    self.store.rejected += 1
                    self.store.reject_log.append(f"{cand.id}: {v.reason}")
            if accepted:
                self.store.add(cand)
                r.learned = cand.id
                if self.consolidate and self.store.needs_consolidation():
                    C.consolidate(
                        self.store,
                        self.backend,
                        {task.family: list(probes)},
                        max_merges=1,
                        min_gain=self.gate_min_gain,
                        alpha=self.gate_alpha,
                    )

        self.results.append(r)
        return r

    # ── multi-step tasks: the library becomes a plan ──────────
    def solve_composed(self, task, probes_by_family=None, learn=True):
        """
        Retrieve a skill for step 1, apply it, then retrieve a DIFFERENT
        skill for step 2 on the result. Nothing here was trained on
        compositions -- the library only ever saw single operations, so
        success means the pieces were found and chained, nothing else.
        """
        probes_by_family = probes_by_family or {}
        t0 = time.time()
        cur, ok_all, used, trace = task.inp, True, 0, []
        for stage in range(len(task.ops)):
            sub = task.sub(stage, cur)
            self.backend.begin_task()
            route = self.store.route(sub, explore=False)
            skill = route.skill
            got, out = False, ""
            for i in range(self.k_max):
                used += 1
                out = self.backend.attempt(
                    sub, skill, decorrelate=self.decorrelate and i > 0
                )
                if sub.verify(out):
                    got = True
                    break
            self.store.record(route, got)
            trace.append(f"{task.ops[stage]}={'ok' if got else 'X'}")
            if not got:
                ok_all = False
                break
            cur = out
        ok = ok_all and task.verify(cur)
        tok = self.backend.cost.tokens_in + self.backend.cost.tokens_out
        r = Result(
            task.tid,
            "compose",
            ok,
            used,
            tok,
            time.time() - t0,
            "+".join(task.ops),
            route=">".join(trace),
            crowd=self.store.last_crowd,
            stages=len(task.ops),
        )
        r.trace = trace
        self.results.append(r)
        return r

    def run_composed(self, tasks, probes_by_family=None, learn=True):
        for t in tasks:
            self.solve_composed(t, probes_by_family, learn)
        return self.results

    def run(self, tasks, probes_by_family, learn=True):
        for t in tasks:
            self.solve(t, probes_by_family.get(t.family, ()), learn)
        return self.results

    # ── transfer ──────────────────────────────────────────────
    def donate(self, receiver, probes_by_family, with_routing=True) -> int:
        """
        The receiver validates every skill on its own family first.

        with_routing: also hand over the routing table. Without it the
        receiver inherits the skills but not the knowledge of which one to
        use where, which is a fair test of how much of the value sits in
        the skills and how much in the router.
        """
        n = 0
        for sk in self.store.skills:
            clone = sk.clone()
            probes = probes_by_family.get(clone.family, ())
            if not probes:
                continue
            v = gate_validate(
                receiver.store,
                clone,
                probes,
                receiver.backend,
                min_gain=receiver.gate_min_gain,
                alpha=receiver.gate_alpha,
            )
            if v.accepted:
                clone.is_sound = True
                receiver.store.add(clone)
                n += 1
        if with_routing:
            receiver.store.import_bandit(self.store.export_bandit())
        return n

    def stats(self):
        ok = [r.ok for r in self.results]
        return dict(
            n=len(ok),
            acc=100 * sum(ok) / max(1, len(ok)),
            cost=sum(r.cost for r in self.results) / max(1, len(ok)),
            skills=len(self.store),
            **self.store.stats(),
        )


class BaselineAgent:
    """The model bare: naive retries, no skills, no decorrelation."""

    def __init__(self, backend, k_max=4):
        self.backend, self.k_max = backend, k_max
        self.store = SkillStore(width=1)
        self.results = []

    def solve(self, task, probes=(), learn=False):
        self.backend.begin_task()
        used, ok = 0, False
        t0 = time.time()
        for _ in range(self.k_max):
            used += 1
            if task.verify(
                self.backend.attempt(task, None, decorrelate=False)
            ):
                ok = True
                break
        self.results.append(
            Result(task.tid, task.family, ok, used, 0, time.time() - t0)
        )
        return self.results[-1]

    def solve_composed(self, task, probes_by_family=None, learn=False):
        """Two blind attempts in a row. It has nothing to chain."""
        t0 = time.time()
        cur, ok_all, used = task.inp, True, 0
        for stage in range(len(task.ops)):
            sub = task.sub(stage, cur)
            self.backend.begin_task()
            got, out = False, ""
            for _ in range(self.k_max):
                used += 1
                out = self.backend.attempt(sub, None, decorrelate=False)
                if sub.verify(out):
                    got, ok_all = True, True
                    break
                ok_all = False
            if not got:
                break
            cur = out
        ok = ok_all and task.verify(cur)
        self.results.append(
            Result(
                task.tid,
                "compose",
                ok,
                used,
                0,
                time.time() - t0,
                stages=len(task.ops),
            )
        )
        return self.results[-1]

    def run_composed(self, tasks, probes_by_family=None, learn=False):
        for t in tasks:
            self.solve_composed(t)
        return self.results

    def run(self, tasks, probes_by_family=None, learn=False):
        for t in tasks:
            self.solve(t)
        return self.results

    def stats(self):
        ok = [r.ok for r in self.results]
        return {
            "n": len(ok),
            "acc": 100 * sum(ok) / max(1, len(ok)),
            "cost": sum(r.cost for r in self.results) / max(1, len(ok)),
            "skills": 0,
        }
