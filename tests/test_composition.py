"""Chaining skills that were never trained together.

Systematicity: mastering A and B separately must pay off on AB.
The library only ever sees single operations, so these tests are the
only evidence that composition is learned rather than memorised.
"""

import numpy as np

from nawah.agent import BaselineAgent, NawahAgent
from nawah.backend import SimBackend
from nawah.bench import interleave, split_tasks
from nawah.curriculum import (
    AutoCurriculum,
    fixed_order_curriculum,
    informativeness,
    primitive_factory,
)
from nawah.tasks import (
    PRIMITIVES,
    ComposedTask,
    all_pairs,
    load_composed_tasks,
    load_primitive_tasks,
)


def test_composed_task_chains_correctly():
    """A composed task really is the composition of its parts"""
    t = ComposedTask(0, ["upper", "bracket"], "ab cd")
    assert t.expected == PRIMITIVES["bracket"][2](
        PRIMITIVES["upper"][2]("ab cd")
    )
    s0 = t.sub(0, t.inp)
    assert s0.verify(PRIMITIVES["upper"][2]("ab cd"))
    s1 = t.sub(1, PRIMITIVES["upper"][2]("ab cd"))
    assert s1.verify(t.expected)
    assert not t.verify("something else")


def test_all_pairs_are_type_compatible():
    """No generated composition blows up when it is evaluated"""
    pairs = all_pairs(2)
    assert len(pairs) > 50, f"only {len(pairs)} pairs survived"
    tasks = load_composed_tasks(pairs[:30], per_pair=2, seed=3)
    assert tasks and all(t.expected for t in tasks)


def test_composition_generalises_to_unseen_pairs():
    """
    Trained on single operations only, the agent solves pairs it has
    never seen. This is the systematicity claim, and it is the one that
    a lookup table cannot fake.
    """
    prim = load_primitive_tasks(per_op=32, seed=7)
    rest, probes = split_tasks(prim, n_probe_per_family=4)
    curr = interleave(rest)
    test = load_composed_tasks(all_pairs(2)[:30], per_pair=3, seed=11)
    wins = 0
    for s in range(3):
        b = BaselineAgent(SimBackend(0.1, seed=s), k_max=4)
        base = 100 * np.mean([r.ok for r in b.run_composed(test)])
        g = NawahAgent(SimBackend(0.1, seed=s), seed=s)
        g.run(curr, probes)
        g.results = []
        naw = 100 * np.mean([r.ok for r in g.run_composed(test)])
        wins += naw > base + 10
    assert wins >= 3, f"composition beat the baseline on only {wins}/3 seeds"


def test_composition_grows_with_the_library():
    """Systematicity accumulates: more skills, more combinations solved"""
    test = load_composed_tasks(all_pairs(2)[:40], per_pair=2, seed=11)
    got = []
    for per_op in (3, 12, 32):
        prim = load_primitive_tasks(per_op=per_op, seed=7)
        rest, probes = split_tasks(prim, n_probe_per_family=4)
        curr = interleave(rest)
        accs = []
        for s in range(3):
            g = NawahAgent(SimBackend(0.1, seed=s), seed=s)
            g.run(curr, probes)
            g.results = []
            accs.append(100 * np.mean([r.ok for r in g.run_composed(test)]))
        got.append(np.mean(accs))
    assert got[-1] > got[0] + 15, f"no growth in systematicity: {got}"


def test_informativeness_peaks_at_half():
    """The rule peaks where the agent is a coin flip"""
    assert abs(informativeness(0.5) - 1.0) < 1e-12
    assert informativeness(0.0) == 0.0 and informativeness(1.0) == 0.0
    assert informativeness(0.4) < informativeness(0.5) > informativeness(0.6)


def test_blocked_curriculum_is_destructive():
    """
    One family at a time cripples the library. This is the control that
    says the curriculum experiment is measuring something real.
    """
    tasks = load_primitive_tasks(per_op=6, seed=7)
    blocked = fixed_order_curriculum(tasks, seed=0, mode="blocked")
    mixed = fixed_order_curriculum(tasks, seed=0, mode="interleaved")
    fams_blocked = len({t.family for t in blocked[:12]})
    fams_mixed = len({t.family for t in mixed[:12]})
    assert fams_mixed > fams_blocked, "interleaving did not mix anything"


def test_posterior_variance_rule_does_not_starve_coverage():
    """
    The naive rule abandons what it has mastered and ends up practising
    one thing 39 times and another 3 times. Dividing by the evidence
    already gathered is the fix, and this is the test for it.
    """
    variants, gen = primitive_factory()
    ac = AutoCurriculum(variants, gen, seed=0)
    for i in range(200):
        t = ac.next_task("posterior_variance")
        ac.update(t._curriculum_variant, i % 3 != 0)
    tries = [v["tries"] for v in ac.report().values()]
    assert min(tries) >= 8, f"starved a variant: {sorted(tries)}"
    assert max(tries) < 4 * min(tries), f"still lopsided: {sorted(tries)}"
    naive = AutoCurriculum(variants, gen, seed=0)
    for i in range(200):
        t = naive.next_task("uncertainty")
        naive.update(t._curriculum_variant, i % 3 != 0)
    nt = [v["tries"] for v in naive.report().values()]
    assert max(nt) > 2 * min(nt), "the naive rule should be the lopsided one"
