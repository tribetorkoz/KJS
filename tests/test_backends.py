"""Model backends: the simulator, the accounting, the budget guard.

The simulator is the default because it lets every claim in this
repository be reproduced for the price of electricity. These tests
pin its determinism, the token ledger, and the guard that stops a
run before it spends more than you said it could.
"""

import os

from nawah.backend import (
    AnthropicBackend,
    BudgetExceeded,
    CostLedger,
    OpenAIBackend,
    SimBackend,
    make_backend,
)
from nawah.tasks import StringTask


def test_backend_sim_determinism():
    """[repeat] Same seed -> same result"""
    t = StringTask(0, "f", "d", "a", "b")

    def run():
        b = SimBackend(0.1, seed=42)
        out = []
        for _ in range(3):
            b.begin_task()
            out.append(
                [b.attempt(t, None, decorrelate=False) for _ in range(4)]
            )
        return out

    assert run() == run(), "simulator is not deterministic for a fixed seed"


def test_backend_seeds_differ():
    """Different seeds -> different results (otherwise seeds do nothing)"""
    t = StringTask(0, "f", "d", "a", "b")

    def run(seed):
        b = SimBackend(0.1, seed=seed)
        r = []
        for _ in range(20):
            b.begin_task()
            r.append(b.attempt(t, None) == t.expected)
        return r

    assert run(1) != run(2), "seeds have no effect"


def test_make_backend_filters_kwargs():
    """make_backend does not pass arguments the backend rejects (model->sim)"""
    b = make_backend("sim", model="should not be passed", seed=1)
    assert isinstance(b, SimBackend)


def test_real_backends_need_keys():
    """Real backends fail with a clear message when the key is missing"""
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    for cls in (AnthropicBackend, OpenAIBackend):
        try:
            cls().complete("hello")
            raise AssertionError(
                f"{cls.__name__} did not refuse without a key"
            )
        except RuntimeError as e:
            assert "API_KEY" in str(e), str(e)


def test_cost_ledger():
    """Cost ledger accumulates correctly"""
    c = CostLedger()
    c.record(10, 5, 0.1, 0.001)
    c.record(20, 10, 0.2, 0.002)
    assert c.calls == 2 and c.tokens_in == 30 and (c.tokens_out == 15)
    assert abs(c.seconds - 0.3) < 1e-09 and abs(c.usd - 0.003) < 1e-09


def test_budget_guard_stops():
    """The budget guard actually stops a run"""
    b = make_backend("sim", seed=0).set_budget(calls=3)
    t = StringTask(0, "f", "d", "a", "b")
    stopped = False
    try:
        for _ in range(20):
            b.begin_task()
            b.attempt(t, None)
    except BudgetExceeded:
        stopped = True
    assert stopped, "the guard never fired"
    assert b.cost.calls <= 4
