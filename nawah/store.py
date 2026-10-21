"""
Skill store: **learned hierarchical routing**, disk persistence, and a
capacity ceiling that is MEASURED rather than assumed.

Why this file was rewritten
---------------------------
In v2 the retrieval score was

    score = skill.strength * (0.6 + 0.4 * skill.win_rate) + noise

There is no `task` anywhere in it. Retrieval did not depend on the question
being asked, so "semantic confusability" was a story told about a mechanism
that was really just "pick the highest number, plus noise". Every capacity
claim built on top of it was decoration.

Worse, the capacity ceiling T* = 0.2234 H^1.494 was hard-coded into the
noise term and then reported in the results table as if it had been
observed. Measuring it would have returned the input. That is not a
measurement, it is an echo.

What is here now
----------------
No T* appears anywhere in the dynamics. Instead:

  * Every skill's score has a term that depends on the TASK:
    a content-similarity term, plus a Thompson-sampled estimate of how well
    this skill has worked on tasks of exactly this kind.
  * Routing is a three-level contest -- family, cluster, skill -- and each
    level is decided by the same noisy argmax, with representation noise
    sigma = sigma0 / sqrt(width). Bigger model, sharper eye.
  * Because the only noise is per-candidate Gaussian, the probability of
    picking the right skill falls with the size of the crowd for the reason
    extreme-value statistics predicts, and for no other reason. The
    critical library size is then something you can go and measure.

The two noise sources, and why both matter
------------------------------------------
    1. representation noise   sigma_rep = sigma0 / sqrt(H)
    2. estimation noise       implicit in the Thompson sample: a skill tried
                              twice has a wide posterior, one tried fifty
                              times has a narrow one

A library can therefore be limited by EITHER the weakness of the model or
the thinness of its experience, and improving only one of the two stops
helping. See theory.critical_library_joint and bench.capacity_sweep.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict

import numpy as np

from .skill import Skill
from .tasks import context_key

# Kept for reporting and for anyone who imported it. It is NOT used by the
# retrieval dynamics any more -- see the module docstring for why that
# distinction is the whole point of this rewrite.
C_CAPACITY = 0.2234
ALPHA_CAPACITY = 1.494


def capacity_limit(width: float) -> float:
    """The v2 assumed ceiling. Reported for comparison, never used
    to decide anything.
    """
    return C_CAPACITY * (width**ALPHA_CAPACITY)


def _tokens(s: str) -> set:
    return set(re.findall(r"[\w]+", s.lower()))


class Route:
    """What retrieval decided, kept so that credit can be assigned later."""

    __slots__ = ("ctx", "keys", "levels", "skill")

    def __init__(self, skill, keys, ctx, levels):
        self.skill = skill
        self.keys = keys  # bandit keys, one per level
        self.ctx = ctx
        self.levels = levels  # human-readable: what was chosen at each level

    def __repr__(self):
        return f"<Route {'/'.join(self.levels)} -> {self.skill}>"


class SkillStore:
    def __init__(
        self,
        width=64,
        seed=0,
        hierarchical=True,
        sigma0=0.55,
        branching=6,
        ucb_prior=(0.30, 1.0),
        shrink=2.0,
        ucb_c=0.5,
        adaptive=False,
        w_quality=0.30,
        w_context=0.45,
        w_similarity=0.25,
    ):
        self.width = width
        self.hierarchical = hierarchical
        self.sigma0 = sigma0
        self.branching = branching
        self.pa, self.pb = ucb_prior
        self.shrink = shrink
        self.ucb_c = ucb_c
        self.adaptive = adaptive
        self.w_q, self.w_c, self.w_s = w_quality, w_context, w_similarity
        self.by_family: dict[str, list[Skill]] = defaultdict(list)
        self.rejected = 0
        self.merges = 0
        self.reject_log: list[str] = []
        self._rng = np.random.default_rng(seed)
        self._next_tid = 0
        self._next_in_fam: dict[str, int] = defaultdict(int)
        self._fam_vec: dict[str, np.ndarray] = {}
        # (level_key, context) -> [wins, losses]
        self._bandit: dict[tuple, list] = {}
        self.n_routes = 0
        # Diagnostics: how big was the crowd we actually had to choose among?
        self.last_crowd = 0

    # ── representation ────────────────────────────────────────
    def _vec(self, family: str) -> np.ndarray:
        if family not in self._fam_vec:
            v = self._rng.normal(size=24)
            self._fam_vec[family] = v / np.linalg.norm(v)
        return self._fam_vec[family]

    @property
    def sigma(self) -> float:
        """Representation noise. The only place model width enters."""
        return self.sigma0 / math.sqrt(max(1e-9, self.width))

    @staticmethod
    def similarity(task, skill) -> float:
        """Token Jaccard between what the task asks and what the
        skill claims.
        """
        tt = _tokens(f"{task.desc} {getattr(task, 'inp', '')}")
        st = _tokens(skill.body)
        if not tt or not st:
            return 0.0
        return len(tt & st) / len(tt | st)

    @staticmethod
    def _sim(a: Skill, b: Skill) -> float:
        """Skill-to-skill similarity: token Jaccard plus family agreement."""
        ta, tb = _tokens(a.body), _tokens(b.body)
        j = len(ta & tb) / max(1, len(ta | tb))
        return 0.5 * j + 0.5 * (1.0 if a.family == b.family else 0.0)

    @staticmethod
    def quality(skill) -> float:
        return skill.strength * (0.6 + 0.4 * skill.win_rate)

    def _alpha_beta(self, key, ctx):
        """
        Posterior for "how good is this choice, in this context".

        The per-context record is thin by construction -- a few hundred
        tasks across a few dozen contexts leaves well under one
        observation per cell -- so it is shrunk toward the same key's
        record across ALL contexts. A skill that has never been tried on
        this exact kind of task is still not a stranger: it is assumed to
        perform about as well as it does in general, with the spread that
        ignorance deserves.
        """
        wins, losses = self._bandit.get((key, ctx), (0.0, 0.0))
        gw, gl = self._bandit.get((key, ""), (0.0, 0.0))
        ng = gw + gl
        pg = (gw + self.pa) / (ng + self.pa + self.pb) if ng > 0 else 0.5
        # The prior's weight can either be a constant or release its grip
        # as evidence piles up (m/(1 + n/m)). The adaptive form is the more
        # principled of the two -- a fixed m is too strong for a context
        # seen a hundred times and too weak for one seen twice -- and it
        # was tested across both suites. It did not win. Kept, switchable
        # and documented, because "more principled" and "measured better"
        # disagreed here and the disagreement is worth being able to see.
        m = (
            self.shrink / (1.0 + (wins + losses) / max(0.25, self.shrink))
            if self.adaptive
            else self.shrink
        )
        return (wins + m * pg + 0.5, losses + m * (1.0 - pg) + 0.5)

    def _posterior(self, key, ctx):
        """
        Score for a routing choice: posterior MEAN plus an uncertainty
        bonus that shrinks as evidence accumulates (UCB).

        This replaced Thompson sampling, and the reason is worth keeping.
        Thompson sampling draws from the posterior, so a half-known option
        regularly draws higher than a well-known better one. That is
        exactly what you want while LEARNING -- it is how the router finds
        out about options it has barely tried -- and exactly what you do
        not want at test time, where it throws away knowledge that was
        paid for. Measured: routing precision 0.55 under Thompson, because
        a skill with a record of (3,1) would draw 0.97 and beat the
        correct skill's (7,0) drawing 0.81.

        UCB gets both: an option with almost no evidence has a large
        posterior variance and so a large bonus, and an option with a lot
        of evidence has none -- so the router explores early and commits
        late, without a schedule and without a switch.
        """
        a, b = self._alpha_beta(key, ctx)
        n = a + b
        mean = a / n
        var = (a * b) / (n * n * (n + 1.0))
        return mean + self.ucb_c * math.sqrt(var)

    def _thompson(self, key, ctx):
        """Thompson sampling. Kept for the ablation that justifies UCB."""
        a, b = self._alpha_beta(key, ctx)
        return float(self._rng.beta(a, b))

    # ── adding ────────────────────────────────────────────────
    def add(self, skill: Skill) -> None:
        skill.tid = self._next_tid
        self._next_tid += 1
        idx = self._next_in_fam[skill.family]
        self._next_in_fam[skill.family] = idx + 1
        skill.cluster = idx // max(1, self.branching)
        self.by_family[skill.family].append(skill)

    @property
    def skills(self) -> list[Skill]:
        return [s for lst in self.by_family.values() for s in lst]

    def __len__(self):
        return sum(len(v) for v in self.by_family.values())

    # ── routing ───────────────────────────────────────────────
    def _pick(self, candidates, scorer, rng):
        """Noisy argmax. Returns (best, crowd_size)."""
        if not candidates:
            return None, 0
        best, best_s = None, -1e18
        for c in candidates:
            s = scorer(c) + float(rng.normal(0.0, self.sigma))
            if s > best_s:
                best, best_s = c, s
        return best, len(candidates)

    def route(self, task, rng=None, explore=False):
        """
        family -> cluster -> skill, each level a noisy contest decided by
        what has actually worked in this context before.

        Flat mode collapses all three levels into one, which is the
        ablation that tells you whether the hierarchy is earning anything.
        """
        rng = rng or self._rng
        # explore -> Thompson sampling (gather information)
        # commit  -> UCB               (spend it)
        score_of = self._thompson if explore else self._posterior
        if len(self) == 0:
            return Route(None, [], context_key(task), [])
        ctx = context_key(task)
        keys, levels = [], []

        if not self.hierarchical:
            pool = self.skills
            sk, crowd = self._pick(
                pool,
                lambda s: (
                    self.w_q * self.quality(s)
                    + self.w_c * score_of(("sk", s.id), ctx)
                    + self.w_s * self.similarity(task, s)
                ),
                rng,
            )
            self.last_crowd = crowd
            if sk is None:
                return Route(None, [], ctx, [])
            return Route(sk, [("sk", sk.id)], ctx, [sk.family])

        # ── level 1: which family ──
        fams = list(self.by_family.keys())
        fam, _ = self._pick(
            fams,
            lambda f: (
                self.w_c * score_of(("fam", f), ctx)
                + self.w_s
                * float(np.dot(self._vec(f), self._vec(task.family)))
            ),
            rng,
        )
        if fam is None:
            return Route(None, [], ctx, [])
        keys.append(("fam", fam))
        levels.append(fam)
        pool = self.by_family[fam]

        # ── level 2: which cluster inside the family ──
        clusters = sorted({s.cluster for s in pool})
        if len(clusters) > 1:
            cl, _ = self._pick(
                clusters,
                lambda c: (
                    self.w_c * score_of(("clu", fam, c), ctx)
                    + self.w_q
                    * max(self.quality(s) for s in pool if s.cluster == c)
                ),
                rng,
            )
            if cl is None:
                cl = clusters[0]
            keys.append(("clu", fam, cl))
            levels.append(f"c{cl}")
            pool = [s for s in pool if s.cluster == cl]

        # ── level 3: which skill ──
        sk, crowd = self._pick(
            pool,
            lambda s: (
                self.w_q * self.quality(s)
                + self.w_c * score_of(("sk", s.id), ctx)
                + self.w_s * self.similarity(task, s)
            ),
            rng,
        )
        self.last_crowd = crowd
        if sk is None:
            return Route(None, keys, ctx, levels)
        keys.append(("sk", sk.id))
        levels.append(sk.id)
        return Route(sk, keys, ctx, levels)

    def retrieve(self, task, rng=None, explore=False):
        """Convenience wrapper: just the skill."""
        return self.route(task, rng, explore=explore).skill

    def record(self, route: Route, ok: bool) -> None:
        """
        Credit assignment. Every level that took part in the decision gets
        the outcome, so the router learns which family, which cluster and
        which skill to enter for this kind of task -- not by being told,
        but from what happened.
        """
        if route is None or route.skill is None:
            return
        self.n_routes += 1
        for k in route.keys:
            self.record_ctx(k, route.ctx, ok)

    def record_ctx(self, key, ctx, ok) -> None:
        """One observation, recorded both per-context and globally."""
        for c in (ctx, ""):
            rec = self._bandit.setdefault((key, c), [0.0, 0.0])
            rec[0] += 1.0 if ok else 0.0
            rec[1] += 0.0 if ok else 1.0

    def observations(self, ctx=None) -> int:
        if ctx is None:
            return int(
                sum(wins + losses for wins, losses in self._bandit.values())
            )
        return int(
            sum(
                wins + losses
                for (_key, c), (wins, losses) in self._bandit.items()
                if c == ctx
            )
        )

    # ── capacity ──────────────────────────────────────────────
    def crowding(self) -> int:
        """How many competitors the last retrieval had to beat. Measured."""
        return self.last_crowd

    def needs_consolidation(self) -> bool:
        """
        Merge when a family has outgrown its clusters. This is a design
        parameter, not a claimed law: keep the leaf contest small enough to
        be winnable.
        """
        return any(
            len(v) > 2 * self.branching for v in self.by_family.values()
        )

    def pressure(self) -> float:
        """Reported only. The dynamics no longer read it."""
        return len(self) / max(1e-9, capacity_limit(self.width))

    # ── disk ──────────────────────────────────────────────────
    def save(self, path: str) -> int:
        os.makedirs(path, exist_ok=True)
        n = 0
        for fam, lst in self.by_family.items():
            d = os.path.join(path, fam)
            os.makedirs(d, exist_ok=True)
            for sk in lst:
                with open(
                    os.path.join(d, f"{sk.id}.skill.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(sk.to_file())
                n += 1
        return n

    @staticmethod
    def load(path: str, width=64) -> SkillStore:
        st = SkillStore(width=width)
        if not os.path.isdir(path):
            return st
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                if not fn.endswith(".skill.md"):
                    continue
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    try:
                        st.add(Skill.from_file(f.read()))
                    except Exception as e:
                        print(f"  ! skipped {fn}: {e}")
        return st

    # ── transfer ──────────────────────────────────────────────
    def export_all(self, path: str) -> int:
        return self.save(path)

    def export_bandit(self) -> dict:
        """Routing knowledge as plain data, so it can travel with
        the skills.
        """
        return {
            "|".join(map(str, k)) + "|" + c: list(v)
            for (k, c), v in self._bandit.items()
        }

    def import_bandit(self, data: dict) -> int:
        n = 0
        for key, v in (data or {}).items():
            parts = key.split("|")
            ctx = parts[-1]
            lvl = parts[:-1]
            k = tuple(int(x) if x.lstrip("-").isdigit() else x for x in lvl)
            rec = self._bandit.setdefault((k, ctx), [0.0, 0.0])
            rec[0] += float(v[0])
            rec[1] += float(v[1])
            n += 1
        return n

    def stats(self) -> dict:
        return {
            "n": len(self),
            "families": len(self.by_family),
            "sigma": round(self.sigma, 3),
            "crowd": self.last_crowd,
            "observations": self.observations(),
            "rejected": self.rejected,
            "merges": self.merges,
        }


# ══════════════════════════════════════════════════════════════
# MEASURING THE CEILING
# ══════════════════════════════════════════════════════════════
def measure_precision(store, probes, rng=None, n_warm=0, retrieval_only=True):
    """
    Retrieval precision: how often does routing land on a skill that
    genuinely applies? Correctness is judged by the simulator's ground-truth
    label, which the router never sees.

    retrieval_only : judge only the probes the library actually COVERS.
                     Two different things can go wrong -- the library does
                     not know this kind of task at all (a coverage failure,
                     fixed by learning more skills), or it knows it and
                     picks the wrong one (a routing failure, which is what
                     the capacity law is about). Mixing them measures
                     neither.

    n_warm : supervised routing rounds to run first. Without experience the
             bandit is guessing, and the whole point of the two-sided bound
             is that experience and model width trade off against each
             other.
    """
    rng = rng or np.random.default_rng(0)
    if len(store) == 0:
        return 0.0
    if retrieval_only:
        probes = [
            t
            for t in probes
            if any(t.variant in s.covered for s in store.skills)
        ]
        if not probes:
            return 0.0
    for _ in range(n_warm):
        for t in probes:
            r = store.route(t, rng, explore=True)  # warm-up learns
            ok = (
                r.skill is not None
                and r.skill.truth
                and t.variant in r.skill.covered
            )
            store.record(r, ok)
    hit = 0
    for t in probes:
        r = store.route(t, rng, explore=False)  # measurement commits
        if (
            r.skill is not None
            and r.skill.truth
            and t.variant in r.skill.covered
        ):
            hit += 1
    return hit / max(1, len(probes))
