# NAWAH

**A skill layer over any model.**

It does not make the model smarter. It makes the model's experience
survive: every success is distilled into a **verified, reusable skill
file**, and a router learns which file to reach for in which situation.

```
Model alone                         12.3%
With NAWAH                          49.1%      (+36.9 pp, 8/8 seeds)
Two-step compositions, never trained  2.0%  ->  82.7%
Three-step, never trained             1.6%  ->  80.9%
```

The composition rows are the point. The library is trained **only on
single operations** and tested on pairs and triples it has never seen.
Memorisation cannot fake that number.

---

## 60 seconds

```bash
pip install numpy            # the only runtime dependency

python3 -m nawah verify      # re-check every claim the project makes
python3 -m nawah theory      # the three laws, with their measured numbers
python3 examples/trace_one_task.py   # watch one task go through the system
pytest                       # 70 tests
```

No API key is needed for any of that. The default backend is a
deterministic simulator that lets every number in this repository be
reproduced for the price of electricity.

---

## The idea in one paragraph

A model forgets everything between calls. NAWAH sits between you and the
model and refuses to let that happen. When a task succeeds, the solution
is distilled into a **plain-text skill file** — Markdown with YAML front
matter, readable and diffable by a human. A **gate** then runs that skill
against held-out probes and admits it only if it demonstrably helps. A
**router** learns, from outcomes alone, which skill to reach for in which
context. Skills **merge** when they start duplicating each other, and the
whole library can be **exported and handed to another agent**, which
re-validates every file before trusting it.

---

## Layout

```
nawah/                  the package
  skill.py              what a skill IS: the file format
  store.py              the library: routing, persistence, bandit memory
  gate.py               the admission test: evidence, not optimism
  agent.py              the loop: route -> attempt -> record -> distill
  consolidate.py        merging skills that duplicate each other
  curriculum.py         deciding what to practise next
  tasks.py              task suites and mechanical verification
  backend.py            model backends, token accounting, budget guard
  theory.py             the closed-form mathematics
  bench.py              every experiment in the project
  cli.py                the command-line entry point

tests/                  70 tests, one module per subsystem
docs/                   architecture, theory, results, running guide
examples/               one runnable end-to-end trace
scripts/                regenerates docs/RESULTS.md
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) next. It is one page
and it explains how the pieces fit.

---

## What it is not

- **Not a fine-tune.** No weights are touched. The model is a black box.
- **Not a prompt library.** Skills are gated on held-out evidence and
  merged when they duplicate. A prompt collection does neither.
- **Not a claim about AGI.** It is a claim about *efficiency*: the cost of
  acquiring skill N falls as N grows (`docs/THEORY.md`, Law 1). That is
  Chollet's criterion for intelligence-as-skill-acquisition, measured.
- **Not proven on a frontier model yet.** Every number here comes from the
  simulator unless it says otherwise. The simulator is honest about what
  it can and cannot see; see `docs/RESULTS.md`.

---

## Running it on a real model

See [`docs/RUNNING.md`](docs/RUNNING.md). In short:

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # in your shell, never in a file
export NAWAH_MAX_USD=1.00               # the guard, so a bug cannot bill you
python3 -m nawah bench --backend anthropic --suite code --seeds 2 --k-max 3
```

That first run costs about **two cents**. Ollama runs it for nothing.

---

## License

MIT. See `LICENSE`.
