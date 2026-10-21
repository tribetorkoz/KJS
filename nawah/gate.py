"""
Validation gate — a real A/B test, not an arbitrary threshold.

Why: "skills an agent writes for itself *degrade* performance (-1.3 pp)."
So the question is not "does this skill clear a bar?" but:

    Does this skill make the model better than it would be without it?

Answered by a paired A/B run over held-out probe tasks, then McNemar's exact
test: we count flips (fail->pass = b) and (pass->fail = c), and reject the
skill when regressions rival improvements.

That makes the gate defensible in front of a reviewer — not just a cutoff
someone picked.
"""

from __future__ import annotations

import math

from .tasks import context_key


def _binom_tail(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided binomial tail (exact McNemar)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tot = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tot / (2.0**n))


class Verdict:
    __slots__ = (
        "accepted",
        "b",
        "c",
        "p_value",
        "rate_with",
        "rate_without",
        "reason",
    )

    def __init__(self, accepted, b, c, rw, ro, p, reason=""):
        self.accepted, self.b, self.c = accepted, b, c
        self.rate_with, self.rate_without = rw, ro
        self.p_value, self.reason = p, reason

    def __repr__(self):
        tag = "accepted" if self.accepted else "rejected"
        return (
            f"<{tag} b={self.b} c={self.c} "
            f"{self.rate_without:.0%}->{self.rate_with:.0%} "
            f"p={self.p_value:.3f}>"
        )


def validate(
    store, skill, probes, backend, k_tries=3, min_gain=0.10, alpha=0.20
) -> Verdict:
    """
    probes : held-out tasks, never used for training.
    k_tries: attempts per task (no decorrelation — we want the skill's
             effect alone, not the retry policy's).
    """
    probes = list(probes)
    n = len(probes)
    if n == 0:
        return Verdict(False, 0, 0, 0.0, 0.0, 1.0, "no probe tasks")

    def run(with_skill):
        ok = []
        for t in probes:
            backend.begin_task()
            hit = False
            for _ in range(k_tries):
                out = backend.attempt(
                    t, skill if with_skill else None, decorrelate=False
                )
                if t.verify(out):
                    hit = True
                    break
            ok.append(hit)
            # The probe runs are real attempts on real tasks. Throwing that
            # evidence away wastes most of what the gate costs.
            if (
                with_skill
                and store is not None
                and hasattr(store, "record_ctx")
            ):
                store.record_ctx(("sk", skill.id), context_key(t), hit)
        return ok

    without = run(False)
    with_ = run(True)

    b = sum(1 for x, y in zip(without, with_) if not x and y)  # improved
    c = sum(1 for x, y in zip(without, with_) if x and not y)  # regressed

    rw = sum(with_) / n
    ro = sum(without) / n
    p = _binom_tail(b, b + c)

    skill.wins += sum(with_)
    skill.losses += n - sum(with_)
    skill.verified_on = n

    gain = rw - ro
    if b + c == 0:
        return Verdict(False, b, c, rw, ro, 1.0, "no measurable effect")
    if gain < min_gain:
        return Verdict(
            False, b, c, rw, ro, p, f"effect too small ({gain:+.0%})"
        )
    if c > b:
        return Verdict(False, b, c, rw, ro, p, "hurts more than it helps")
    if p > alpha:
        return Verdict(False, b, c, rw, ro, p, f"not significant (p={p:.2f})")

    skill.is_sound = True
    return Verdict(True, b, c, rw, ro, p, "")
