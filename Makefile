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

# Prefer a local .venv if one exists, so `make test` works without activating it.
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
COMPOSE ?= docker compose -f grafana/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help venv pull backfill build ingest verify test status grafana grafana-stop \
        grafana-logs fixtures compact clean reset-db

help:
	@echo 'offchart -- CFTC and friends into SQLite, read by Grafana.'
	@echo
	@echo '  make venv          create .venv and install requirements'
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

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo 'done -- now: make backfill'

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

ingest:
	$(PYTHON) -m scripts.ingest

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
