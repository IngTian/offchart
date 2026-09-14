# The whole command surface. Two targets cover normal use:
#
#     make pull        fetch the latest data, then rebuild the tables Grafana reads
#     make grafana     start Grafana on http://localhost:3000
#
# Run `make` with no target for the list.
#
# WHY A MAKEFILE AND NOT A CLICK CLI
#
# There is nothing here that argparse does not already do -- every target is one
# `python -m scripts.*` invocation. What the Makefile adds is that `make pull` is ONE
# command where the honest version is two (ingest, then rebuild the derived tables),
# and forgetting the second one leaves Grafana showing last week's numbers with no
# error anywhere. A target that cannot be half-run is the entire point.

# WHICH INTERPRETER, AND WHY IN THIS ORDER
#
# An ACTIVATED environment wins over a directory that happens to be sitting in the tree.
# Activation is an explicit statement of intent; a leftover .venv/ is not, and the
# earlier version preferred the directory -- so someone working in an activated env got
# silently run against a stale .venv instead, with no line of output saying so.
#
# `make venv` puts .venv exactly where uv looks for it, so the ordinary case needs no
# activation at all. Nothing in this repo cares which tool built the environment: two
# pure-Python dependencies, no compiled extension, no C library.
# Override explicitly any time with `make test PYTHON=/path/to/python`.
PYTHON ?= $(shell \
  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then \
    echo "$$VIRTUAL_ENV/bin/python"; \
  elif [ -x .venv/bin/python ]; then echo .venv/bin/python; \
  else echo python3; fi)
COMPOSE ?= docker compose -f grafana/docker-compose.yml

PY_VERSION ?= 3.12

.DEFAULT_GOAL := help
.PHONY: help venv which-python pull backfill build ingest verify test status \
        grafana grafana-stop grafana-logs fixtures compact clean reset-db

help:
	@echo 'offchart -- CFTC and friends into SQLite, read by Grafana.'
	@echo
	@echo '  make venv          once: create .venv with uv and install requirements'
	@echo '  make which-python  show which interpreter make will use, and why'
	@echo '  make pull          fetch latest -> database -> rebuild board_* tables'
	@echo '  make backfill      same, but full history (first run: ~5 min, ~900 MB)'
	@echo '  make build         rebuild board_* only, no network'
	@echo '  make status        what is in the database and how stale it is'
	@echo
	@echo '  make grafana       start Grafana at http://localhost:3000'
	@echo '  make grafana-stop  stop it'
	@echo '  make grafana-logs  follow its logs'
	@echo
	@echo '  make test          the suite (no network, ~2s)'
	@echo '  make verify        prove the CFTC field maps against the live API'
	@echo '  make fixtures      regenerate tests/fixtures/fixture.sqlite from the store'
	@echo '  make compact       VACUUM the database'
	@echo '  make reset-db      delete the database (CONFIRM=1 required)'
	@echo
	@echo "PYTHON=$(PYTHON)"

# uv builds the environment. It needs no system Python at all -- `--python 3.12` fetches
# a standalone interpreter if none is installed -- which is the whole reason this replaced
# a conda target: the dependencies were always two pure wheels, and the environment
# manager was the only part with opinions.
#
# .venv is the default location uv reads and writes, so it is also what $(PYTHON) finds
# above with nothing activated. requirements.txt stays the single source of truth; there
# is no lock file because the dependency set is pandas and pytest.
# `--clear` because uv refuses an existing .venv outright, so without it this target
# works exactly once and then fails with "a virtual environment already exists" -- and
# rerunning it after a requirements or PY_VERSION change is the normal reason to type it.
# Rebuilding is a second's work for two pure wheels, and it self-heals a broken env.
venv:
	uv venv --clear --python $(PY_VERSION)
	uv pip install -r requirements.txt
	@echo
	@echo 'done -- now: make backfill'
	@echo '(no activation needed: make finds .venv on its own)'

# Worth having as a target rather than a comment: "which python is this actually using"
# is the first question when a suite passes in one shell and fails in another.
which-python:
	@echo "PYTHON      = $(PYTHON)"
	@echo "  version   = $$($(PYTHON) --version 2>&1)"
	@# 2>/dev/null and a fallback, not 2>&1: a traceback pasted into a diagnostic makes
	@# the one line you came here to read scroll off the screen.
	@echo "  pandas    = $$($(PYTHON) -c 'import pandas; print(pandas.__version__)' 2>/dev/null || echo 'MISSING -- run make venv')"
	@echo "  pytest    = $$($(PYTHON) -c 'import pytest; print(pytest.__version__)' 2>/dev/null || echo 'MISSING -- run make venv')"
	@# Mirrors the $(PYTHON) conditions above exactly, including the -x test. An earlier
	@# version checked only that $$VIRTUAL_ENV was non-empty, so a stale variable pointing
	@# at a deleted environment made this line contradict the interpreter it just printed.
	@echo "  chosen by = $$( \
	  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then echo "VIRTUAL_ENV ($$VIRTUAL_ENV)"; \
	  elif [ -x .venv/bin/python ]; then echo 'the .venv/ directory (no env activated)'; \
	  else echo 'PATH fallback (no env activated, no .venv)'; fi)"

# The one command. Incremental fetch, then rebuild the derived tables.
#
# The rebuild is unconditional and total rather than incremental, and that is a
# correctness requirement, not laziness: CFTC revises already-published weeks, and a
# revision at week t changes every causal percentile from t onward. Rebuilding
# everything makes a stale percentile -- a plausible wrong number -- impossible.
pull: ingest build
	@echo
	@echo 'pulled and rebuilt. Grafana will show it on the next panel refresh.'

# Full history. Needed on a first run, after changing a field map, or after widening
# the market universe. A source with nothing stored backfills itself anyway.
backfill:
	$(PYTHON) -m scripts.ingest --backfill
	$(MAKE) build

# Not in `make help` on purpose -- it is half of `pull`, and the half that leaves the
# boards showing the previous week. Reachable because it is genuinely useful when
# debugging a single source, so it says what it left undone rather than letting you find
# out from a chart.
ingest:
	$(PYTHON) -m scripts.ingest
	@echo
	@echo 'NOTE: archive updated, board_* NOT rebuilt -- Grafana still shows the old'
	@echo '      derived numbers. Run `make build`, or use `make pull` next time.'

build:
	$(PYTHON) -m scripts.build

# Against the LIVE API, so it needs network and is the one target that can fail for
# reasons outside this repo. It is also the check that catches the worst failure mode
# available here: a CFTC column that still exists but now means something else, which
# produces a plausible chart of the wrong number and raises nothing.
verify:
	$(PYTHON) -m scripts.verify_spec --rows 2000

test:
	$(PYTHON) -m pytest -q

status:
	$(PYTHON) -m scripts.status

grafana:
	$(COMPOSE) up -d
	@echo
	@echo 'Grafana starting on http://localhost:3000 (anonymous admin, no login).'
	@echo 'First start downloads the SQLite plugin, so give it ~20s.'
	@echo 'Boards: Offchart > Positioning, Offchart > Inflation expectations.'

grafana-stop:
	$(COMPOSE) down

grafana-logs:
	$(COMPOSE) logs -f

fixtures:
	$(PYTHON) -m tests.fixtures.build

# SQLite does not return freed pages to the filesystem on its own, so a rebuild that
# replaces a serving table leaves the file at its high-water mark.
compact:
	$(PYTHON) -c "import sqlite3, pathlib; \
	  p = pathlib.Path('data/offchart.sqlite'); \
	  before = p.stat().st_size; \
	  sqlite3.connect(p).execute('VACUUM'); \
	  print(f'{before/1e6:.0f} MB -> {p.stat().st_size/1e6:.0f} MB')"

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/offchart.sqlite-wal data/offchart.sqlite-shm

# Safe to do, and worth knowing that it is: nine of the ten sources are re-fetchable,
# and the tenth does not live in this file -- its history is the committed CSVs under
# data/snapshots/. So the cost of deleting the database is one `make backfill`.
reset-db:
ifndef CONFIRM
	@echo 'This deletes data/offchart.sqlite. Recovery is one `make backfill`.'
	@echo 'Re-run as: make reset-db CONFIRM=1'
	@exit 1
endif
	rm -f data/offchart.sqlite data/offchart.sqlite-wal data/offchart.sqlite-shm
	@echo 'gone. `make backfill` rebuilds it.'
