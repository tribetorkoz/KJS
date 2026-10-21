# Changelog

All notable changes. The project is versioned by the claims it can support,
not by the features it has grown.

## 3.0.0

### Fixed — things that were actively wrong

- **The router kept exploring after it had already learned.** Thompson
  sampling was used at test time, so a half-known option routinely beat a
  well-known better one: a skill with a record of (3,1) drew 0.97 and beat
  the correct skill's (7,0) drawing 0.81. Routing precision was 0.55.
  Split into `explore=True` (Thompson, training) and `explore=False` (UCB,
  evaluation). Precision 0.55 → 0.70. See `docs/ROUTER.md`.
- **A duplicated test was being silently dropped.** The test registry was
  keyed by function name, so two tests sharing a name meant one never ran.
  The suite is now plain pytest, which reports both.
- **`bench.curriculum_experiment(suite="string")` raised `NameError`.**
  The task loaders were used but never imported. Unreachable from the CLI
  default, reachable by anyone passing the argument.
- **`np.nanmean` was called on possibly all-NaN arrays**, producing
  runtime warnings and a silent NaN. The guard checked length, not
  finiteness.
- **The retrieval integral was approximated** and wrong by up to 0.20 in
  absolute probability. Now evaluated exactly; residual against Monte
  Carlo 0.0048.
- **`m* ∝ H^(1/(2(1+θ)))` was the wrong closed form.** Corrected to
  `θ = 1/(2a)`. Found by measurement, not by reading.
- **`fit_beta_half` re-indexed its halves**, comparing the wrong ranges.
- **A hard-coded capacity ceiling** was printed in the results table as if
  it had been measured. Removed; the ceiling is now grown to.
- **The curriculum rule `4p(1−p)` starved coverage** — it abandoned
  whatever had already been mastered. Replaced by `p(1−p)/(n+2)`.

### Changed

- Routing defaults: `ucb_c=0.5`, `shrink=2.0`.
- Adaptive shrinkage (`m/(1 + n/m)`) is implemented and switchable. It is
  more principled than a fixed `m` and measured slightly worse. Both are
  kept; the disagreement is documented rather than hidden.

### Added

- `docs/` — architecture, theory, router, skill format, running guide.
- `tests/` as a proper package: 70 tests, one module per subsystem.
- `examples/trace_one_task.py` — one task, followed end to end.
- `pyproject.toml`, `Makefile`, `LICENSE`, `.gitignore`, `conftest.py`.
- `python -m nawah` as an alias for `python -m nawah.cli`.

### Known limitations

- Every number comes from the simulator unless explicitly stated.
- The curriculum result did not survive the routing fix: a plain
  interleaved order now beats the agent's own curriculum (70.0% vs 61.1%).
  Reported as a negative result in `docs/RESULTS.md`.
- Model width enters the theory through exactly one assumption,
  `sigma = sigma0/√H`. It is the weakest link in the chain.
