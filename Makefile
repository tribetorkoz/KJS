# NAWAH — common tasks. Run `make help` for the list.
PYTHON ?= python3

.PHONY: help install test lint format verify demo report clean

help:                          ## Show this list
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:                       ## Install in editable mode with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

test:                          ## Run the test suite
	$(PYTHON) -m pytest tests -q

lint:                          ## Static checks (must be clean to merge)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:                        ## Auto-format and auto-fix
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m black .

verify:                        ## Re-check every claim the project makes
	$(PYTHON) -m nawah verify --seeds 8 --per-family 36

demo:                          ## Trace one task through the whole system
	$(PYTHON) examples/trace_one_task.py

report:                        ## Regenerate docs/RESULTS.md
	$(PYTHON) scripts/make_report.py > docs/RESULTS.md

clean:                         ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
