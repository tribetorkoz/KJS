"""Routing precision, the capacity law, and the explore/commit split.

How much a library can hold before retrieval starts confusing it,
measured rather than assumed; how much the routing table itself is
worth when it travels with the skills; and the separation between
exploring while learning and committing at test time.
"""

import collections

import numpy as np

from nawah import bench
from nawah.agent import NawahAgent
from nawah.backend import SimBackend
from nawah.bench import interleave, measure_critical_library, split_tasks
from nawah.store import SkillStore
from nawah.tasks import load_primitive_tasks, load_string_tasks
from nawah.theory import fit_capacity_exponent, gap_exponent_from_capacity


def test_critical_library_is_measured_not_assumed():
    """
    Grow the library until routing fails. The number must come out
    finite and must grow when the model gets wider -- and no ceiling may
    be hard-coded anywhere in the retrieval dynamics.
    """
    lo = measure_critical_library(
        64, n_warm=20, thresh=0.85, seed=0, n_max=512, seeds=3
    )
    hi = measure_critical_library(
        1024, n_warm=20, thresh=0.85, seed=0, n_max=512, seeds=3
    )
    assert lo["T_star"] >= 1 and hi["T_star"] >= 1
    assert (
        hi["T_star"] >= lo["T_star"]
    ), f"a wider model bought no capacity: {lo['T_star']} -> {hi['T_star']}"


def test_measured_capacity_exponent_is_positive():
    """T* grows with H, and the exponent comes out in a sane range"""
    rows = []
    for H in (64, 256, 1024):
        m = measure_critical_library(
            H, n_warm=20, thresh=0.85, seed=0, n_max=512, seeds=3
        )
        rows.append((H, max(1.0, m["T_star"])))
    a, _A, _r2 = fit_capacity_exponent(
        [r[0] for r in rows], [r[1] for r in rows]
    )
    assert a > 0.0, f"capacity does not grow with the model: a={a:.3f}"
    assert gap_exponent_from_capacity(a) > 0


def test_bandit_knowledge_travels_with_the_skills():
    """Routing knowledge exports and imports without loss"""
    st = SkillStore(width=64, seed=0)
    st.record_ctx(("sk", "a.1"), "ctx1", True)
    st.record_ctx(("sk", "a.1"), "ctx1", False)
    data = st.export_bandit()
    other = SkillStore(width=64, seed=1)
    other.import_bandit(data)
    assert other._bandit.get((("sk", "a.1"), "ctx1")) == [
        1.0,
        1.0,
    ], f"routing knowledge lost in transit: {other._bandit}"


def test_the_whole_report_renders():
    """End to end on the simulator, including the new sections"""
    tasks = load_string_tasks(per_family=8)
    rest, probes = bench.split_tasks(tasks, n_probe_per_family=3)
    curr = bench.interleave(rest)

    def mk(seed=0, **kw):
        return SimBackend(0.1, seed=seed)

    head = bench.headline(mk, curr, probes, seeds=range(2))
    abl = bench.ablation(mk, curr, probes, seeds=(0,))
    tr = bench.transfer(mk, curr, probes, seeds=(0,))
    cap = bench.capacity_sweep(mk, curr, probes, seeds=(0,))
    md = bench.report_md(head, abl, tr, cap)
    assert "NAWAH report" in md and "%" in md
    assert (
        head["delta_mean"] > 0
    ), f"layer hurt on the sim: {head['delta_mean']}"


def test_ucb_beats_thompson_at_test_time():
    """
    [repeat] The router must commit, not keep exploring.

    Thompson sampling draws from the posterior, so a half-known option
    regularly draws higher than a well-known better one -- correct while
    learning, wrong at test time, where it throws away knowledge that was
    paid for. Measured before the fix: routing precision 0.55, because a
    skill with a record of (3,1) drew 0.97 and beat the correct skill's
    (7,0) drawing 0.81. Switched to UCB: 0.70.
    """
    prim = load_primitive_tasks(per_op=32, seed=7)
    rest, probes = split_tasks(prim, n_probe_per_family=4)
    curr = interleave(rest)

    def precision(mode):
        out = []
        for sd in (0, 1, 2):
            ag = NawahAgent(SimBackend(0.1, seed=sd), seed=sd)
            if mode == "thompson":
                ag.store._posterior = ag.store._thompson
            for x in curr:
                ag.solve(x, probes.get(x.family, ()), True)
            covers = collections.Counter()
            for sk in ag.store.skills:
                for v in sk.covered:
                    covers[v] += 1
            rng = np.random.default_rng(0)
            hits = []
            for v in sorted(covers):
                ts = [t for t in curr if t.variant == v][:25]
                hits.append(
                    sum(
                        1
                        for t in ts
                        if (
                            lambda r: r.skill and t.variant in r.skill.covered
                        )(ag.store.route(t, rng))
                    )
                    / max(1, len(ts))
                )
            out.append(np.mean(hits))
        return np.mean(out)

    ucb = precision("ucb")
    th = precision("thompson")
    assert ucb > th, f"UCB {ucb:.3f} did not beat Thompson {th:.3f}"
    assert ucb > 0.6, f"routing precision is poor: {ucb:.3f}"
