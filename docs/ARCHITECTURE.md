# Architecture

One page. How the pieces fit, and why each one exists.

---

## The loop

Every task goes through the same seven steps. Nothing else happens.

```
                 ┌──────────────────────────────────────────┐
   task ────────▶│ 1. context key   (from the task's words) │
                 └──────────────────┬───────────────────────┘
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ 2. route   family → cluster → skill      │
                 │    three contests, each scored by:       │
                 │    0.30 quality + 0.45 record + 0.25 sim │
                 └──────────────────┬───────────────────────┘
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ 3. attempt  up to k_max times            │
                 │    retries are decorrelated on purpose   │
                 └──────────────────┬───────────────────────┘
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ 4. record  every level that decided      │
                 │    gets the outcome, in this context     │
                 └──────────────────┬───────────────────────┘
                                    ▼
                        success? ─── no ──▶ done
                                    │
                                   yes
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ 5. distill   write a candidate skill     │
                 └──────────────────┬───────────────────────┘
                                    ▼
                 ┌──────────────────────────────────────────┐
                 │ 6. gate     run it against held-out      │
                 │             probes, with and without     │
                 └──────────────────┬───────────────────────┘
                          reject ◀──┴──▶ accept
                                          │
                                          ▼
                 ┌──────────────────────────────────────────┐
                 │ 7. add, then merge if crowded            │
                 └──────────────────────────────────────────┘
```

---

## The modules

| Module | Responsibility | Knows nothing about |
|---|---|---|
| `skill.py` | What a skill **is**: the file format | routing, agents, models |
| `store.py` | The library: routing, persistence, bandit memory | how skills are made |
| `gate.py` | Whether a candidate is **admitted** | where the candidate came from |
| `agent.py` | The loop above | how routing is scored internally |
| `consolidate.py` | Merging skills that duplicate | where skills came from |
| `curriculum.py` | What to practise next | the agent's internals |
| `tasks.py` | Task suites, mechanical verification | agents, skills |
| `backend.py` | Talking to a model, counting tokens | skills, routing |
| `theory.py` | Closed-form mathematics | everything else (pure) |
| `bench.py` | Every experiment | nothing (it drives the rest) |
| `cli.py` | Command-line surface | the algorithms |

The dependency direction is one-way: `bench` → `agent` → `store`/`gate`
→ `skill`. `theory.py` is pure and imports nothing from the package, so
it can be checked in isolation — which it is, by brute force, in
`tests/test_mathematics.py`.

---

## Two decisions worth defending

### Why three contests instead of one search

Retrieval quality degrades with the number of competitors. Splitting the
decision into family → cluster → skill keeps the final contest small
enough to be winnable.

Measured: removing the hierarchy costs **16.2 points** of accuracy
(51.8% → 35.6%). At 576 observations spread over 72 contexts, the
contextual hierarchical router scores 0.208 against 0.028 for a
context-blind one.

### Why exploration and commitment are different code paths

The router scores options with a posterior over "how good is this choice,
in this context". There are two ways to act on a posterior:

- **Thompson sampling** draws *from* it. A poorly-evidenced option
  routinely draws higher than a well-evidenced better one.
- **UCB** takes the mean plus a bonus proportional to the standard
  deviation. The bonus shrinks on its own as evidence accumulates.

Thompson is right **while learning** — it is how the router discovers
options. It is wrong **at test time**, where it throws away knowledge
that was already paid for. Measured: routing precision **0.55** under
Thompson (a skill with a record of (3,1) drew 0.97 and beat the correct
skill's (7,0) drawing 0.81) against **0.70** under UCB.

So `route(task, explore=True)` uses Thompson and is called during
training; `route(task, explore=False)` uses UCB and is called during
evaluation, composition and measurement. This is not a tunable
hyperparameter — it is the difference between gathering information and
spending it. `tests/test_router.py` pins it.

---

## Where the model enters

Exactly one place: `backend.attempt(task, skill, decorrelate)`.

Everything else in the system is model-agnostic, which is why the default
backend is a simulator. `SimBackend` has a competence parameter and a
seeded RNG, so an entire benchmark is reproducible with no network and no
cost. Swapping in a real backend changes nothing except the price.

Model width enters the mathematics in exactly one place too: the
representation noise, `sigma = sigma0 / sqrt(H)`. A wider model has a
sharper eye, sees less noise, and can therefore hold a larger library
before retrieval starts confusing it. That is Law 2 in
[`THEORY.md`](THEORY.md).

---

## State and persistence

- Skills live in memory as `Skill` objects and on disk as Markdown files
  (`docs/SKILL_FORMAT.md`).
- The routing table is a dictionary of (wins, losses) keyed by
  `(choice, context)`. It is exported as JSON and travels with the skills
  (`store.export_bandit` / `import_bandit`).
- A receiving agent re-validates every inherited skill on its own probes
  before trusting it. Measured: inheriting the skills **and** the routing
  table recovers 51.2% against 15.1% for a virgin agent with the same
  model.

---

## Failure modes the design guards against

| Risk | Guard |
|---|---|
| A bad skill poisons the library | `gate.py`: held-out probes, with/without |
| Retrieval gets lost in a big library | hierarchy (3 contests) + merging |
| Retrying repeats the same mistake | forced decorrelation from attempt 2 |
| A real backend bankrupts you | `BudgetExceeded`, caps on USD, calls, tokens |
| A formula is quietly wrong | `tests/test_mathematics.py`, brute force |
| A published number drifts from the code | `python3 -m nawah verify` |
