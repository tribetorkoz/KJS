"""The agent: determinism, learning, transfer, and the baseline.

The agent is the loop: route, attempt, record, distill, gate.
These tests pin the properties that make the measurements mean
anything at all -- same seed gives same run, order does not change
the outcome, learning actually happens, and the baseline, which
never learns, stays flat.
"""

import numpy as np

from nawah.agent import BaselineAgent, NawahAgent
from nawah.backend import SimBackend
from nawah.bench import interleave, split_tasks
from nawah.tasks import load_string_tasks


def _run_agent(seed, n=60, **kw):
    from nawah.agent import NawahAgent
    from nawah.backend import SimBackend
    from nawah.bench import interleave, split_tasks
    from nawah.tasks import load_string_tasks

    rest, probes = split_tasks(
        load_string_tasks(per_family=12), n_probe_per_family=4
    )
    curr = interleave(rest)[:n]
    b = SimBackend(0.1, seed=seed)
    ag = NawahAgent(b, seed=seed, **kw)
    ag.run(curr, probes)
    return ([r.ok for r in ag.results], len(ag.store))


def test_agent_reproducible_same_seed():
    """[repeat] Agent: same seed -> exactly the same result"""
    a, na = _run_agent(7)
    b, nb = _run_agent(7)
    assert a == b, "different results for the same seed"
    assert na == nb, "different skill counts"


def test_agent_order_independent():
    """
    [order] Running two agents in sequence does not change the second.
    (This fixed the global id-counter bug.)
    """
    solo, _ = _run_agent(11)
    _run_agent(999)
    second, _ = _run_agent(11)
    assert solo == second, "results depend on run order"


def test_agent_learns():
    """
    The agent improves: its second half beats its first.

    Averaged over seeds, because a single learning curve is noisy and a
    test that passes only on one seed is a test that will fail on Tuesday.
    """
    rest, probes = split_tasks(
        load_string_tasks(per_family=36), n_probe_per_family=6
    )
    curr = interleave(rest)
    firsts, lasts, nbase, nnaw = ([], [], [], [])
    for s in range(5):
        ag = NawahAgent(SimBackend(0.1, seed=s), seed=s)
        ok = [r.ok for r in ag.run(curr, probes)]
        h = len(ok) // 2
        firsts.append(np.mean(ok[:h]))
        lasts.append(np.mean(ok[h:]))
        b = BaselineAgent(SimBackend(0.1, seed=s), k_max=4)
        nbase.append(np.mean([r.ok for r in b.run(curr, probes)]))
        nnaw.append(np.mean(ok))
    assert np.mean(lasts) > np.mean(
        firsts
    ), f"no improvement: {np.mean(firsts):.2f} -> {np.mean(lasts):.2f}"
    assert np.mean(nnaw) > np.mean(
        nbase
    ), f"layer did not beat the bare model: {np.mean(nnaw):.2f} vs {np.mean(nbase):.2f}"


def test_gate_blocks_junk():
    """The gate keeps corrupted skills out of the library"""
    rest, probes = split_tasks(
        load_string_tasks(per_family=12), n_probe_per_family=4
    )
    curr = interleave(rest)
    counts = {}
    for gate in (True, False):
        ag = NawahAgent(SimBackend(0.1, seed=5), seed=5, gate=gate)
        ag.run(curr, probes)
        counts[gate] = sum(1 for s in ag.store.skills if not s.truth)
    assert (
        counts[True] == 0
    ), f"{counts[True]} corrupted skills got past the gate"
    assert counts[False] > 0, "without the gate none entered — test is vacuous"
    assert counts[False] > counts[True]


def test_donate_transfers_and_validates():
    """Transfer: skills move, and the receiver validates"""
    rest, probes = split_tasks(
        load_string_tasks(per_family=12), n_probe_per_family=4
    )
    curr = interleave(rest)
    teacher = NawahAgent(SimBackend(0.1, seed=1), seed=1)
    teacher.run(curr, probes)
    assert len(teacher.store) > 0, "the teacher learned nothing"
    student = NawahAgent(SimBackend(0.1, seed=2), seed=2, decorrelate=False)
    acc = teacher.donate(student, probes)
    assert acc > 0, "no skill was accepted (transfer is broken)"
    assert acc <= len(teacher.store), "accepted more than were offered"
    assert all(
        s.is_sound for s in student.store.skills
    ), "an unverified skill got in"


def test_baseline_never_learns():
    """The baseline never learns: its library is always empty"""
    rest, probes = split_tasks(
        load_string_tasks(per_family=12), n_probe_per_family=4
    )
    ag = BaselineAgent(SimBackend(0.1, seed=1))
    ag.run(interleave(rest), probes)
    assert len(ag.store) == 0
    assert all(r.skill_used == "" for r in ag.results)
