"""The validation gate: skills are admitted on evidence, not optimism.

A candidate skill is run twice against held-out probes, with and
without itself, and admitted only if it demonstrably helps. These
tests drive the gate with a backend whose outcomes are controlled
directly, so accept and reject are both exercised on demand rather
than left to luck.
"""

from nawah.gate import _binom_tail, validate
from nawah.skill import Skill
from nawah.store import SkillStore


class FakeBackend:
    """A backend whose outcomes we control directly."""

    name = "fake"

    def __init__(self, pattern):
        self.pattern = pattern
        self.seen = {}

    def begin_task(self):
        pass

    def attempt(self, task, skill, decorrelate=False):
        key = (task.tid, bool(skill))
        i = self.seen.get(key, 0)
        self.seen[key] = i + 1
        seq = self.pattern.get(key, [False])
        ok = seq[min(i, len(seq) - 1)]
        return task.expected if ok else "wrong"


def _probes(n=6, fam="f"):
    from nawah.tasks import StringTask

    return [
        StringTask(100 + i, fam, "d", f"in{i}", f"out{i}") for i in range(n)
    ]


def test_gate_accepts_helpful_skill():
    """The gate accepts a skill that genuinely helps"""
    pr = _probes()
    pat = {(t.tid, False): [False] for t in pr}
    pat.update({(t.tid, True): [True] for t in pr})
    v = validate(SkillStore(), Skill("f", "x"), pr, FakeBackend(pat))
    assert v.accepted, f"rejected wrongly: {v}"
    assert v.b == len(pr) and v.c == 0


def test_gate_rejects_harmful_skill():
    """The gate rejects a skill that hurts"""
    pr = _probes()
    pat = {(t.tid, False): [True] for t in pr}
    pat.update({(t.tid, True): [False] for t in pr})
    v = validate(SkillStore(), Skill("f", "x"), pr, FakeBackend(pat))
    assert not v.accepted, f"accepted wrongly: {v}"
    assert v.c > v.b


def test_gate_rejects_no_effect():
    """The gate rejects a skill with no effect — no accepting by luck"""
    pr = _probes()
    pat = {(t.tid, False): [True] for t in pr}
    pat.update({(t.tid, True): [True] for t in pr})
    v = validate(SkillStore(), Skill("f", "x"), pr, FakeBackend(pat))
    assert not v.accepted, f"accepted with no effect: {v}"


def test_gate_empty_probes():
    """A gate with no probe tasks rejects and does not crash"""
    v = validate(SkillStore(), Skill("f", "x"), [], FakeBackend({}))
    assert not v.accepted and "no probe tasks" in v.reason


def test_mcnemar_known_values():
    """McNemar: hand-computed known values"""
    assert abs(_binom_tail(0, 0) - 1.0) < 1e-09
    assert abs(_binom_tail(0, 6) - 2 / 64) < 1e-09
    assert abs(_binom_tail(3, 6) - 1.0) < 1e-09
    assert _binom_tail(1, 10) < 0.05
    assert _binom_tail(4, 10) > 0.05
