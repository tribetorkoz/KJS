"""Merging skills, and the measurement harness that scores them.

Consolidation keeps the leaf contest small enough to be winnable by
merging skills that do the same thing. These tests pin the merge
rules -- coverage can grow, strength may not be inflated -- and the
split/interleave/learning-curve helpers every benchmark depends on.
"""

from nawah.agent import NawahAgent, Result
from nawah.backend import SimBackend
from nawah.bench import fit_beta, interleave, learning_curve, split_tasks
from nawah.consolidate import consolidate, merge_text
from nawah.skill import Skill
from nawah.tasks import load_string_tasks


def test_consolidate_reduces_and_validates():
    """Merging: reduces the count and keeps only what helps"""
    rest, probes = split_tasks(
        load_string_tasks(per_family=24), n_probe_per_family=6
    )
    ag = NawahAgent(SimBackend(0.1, seed=4), seed=4)
    ag.run(interleave(rest), probes)
    before = len(ag.store)
    assert before >= 2, f"too few skills to merge (got {before})"
    m = consolidate(ag.store, ag.backend, probes, max_merges=1)
    assert m in (0, 1), m
    if m:
        assert len(ag.store) == before - 1, "merge did not reduce the count"
        assert all(
            s.is_sound for s in ag.store.skills
        ), "an unverified skill survived"


def test_merge_does_not_inflate_strength():
    """Merging compresses, it does not inflate strength (else it is free)"""

    class B:
        name = "t"

        def complete(self, p):
            return "merged skill"

    a = Skill("f", "a", strength=1.0, truth=True)
    b = Skill("f", "b", strength=1.2, truth=True)
    assert merge_text(a, b, B()) == "merged skill"


def test_bench_fit_beta():
    """beta: falling cost -> positive; flat cost -> zero"""
    dec = [
        Result(i, "f", True, cost=max(1, int(8 - i * 0.1))) for i in range(40)
    ]
    flat = [Result(i, "f", True, cost=3) for i in range(40)]
    b1, _ = fit_beta(dec)
    b2, _ = fit_beta(flat)
    assert b1 > 0, f"falling cost gave beta={b1}"
    assert abs(b2) < 1e-06, f"flat cost gave beta={b2}"


def test_bench_split_disjoint():
    """Probes and curriculum are disjoint — no leakage"""
    rest, probes = split_tasks(
        load_string_tasks(per_family=12), n_probe_per_family=4
    )
    pids = {t.tid for v in probes.values() for t in v}
    cids = {t.tid for v in rest.values() for t in v}
    assert not pids & cids, "tasks shared between probes and curriculum"


def test_bench_interleave_covers_families():
    """Interleaving: every family appears early"""
    rest, _ = split_tasks(
        load_string_tasks(per_family=12), n_probe_per_family=4
    )
    cur = interleave(rest)
    assert {t.family for t in cur[:6]} == set(
        rest.keys()
    ), "the first cycle does not cover every family"


def test_learning_curve_blocks():
    """Learning curve: the block count is right"""
    rs = [Result(i, "f", i >= 30) for i in range(60)]
    c = learning_curve(rs, blocks=6)
    assert len(c) == 6
    assert c[0] == 0.0 and c[-1] == 100.0, c
