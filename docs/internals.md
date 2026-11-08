# How offchart is built

For whoever edits this repo next, including its author in six months.

```
lib/          db (the store), metrics, segments, schema, xlsx,
              cftc_spec, cftc_api, universe, pricemap, cohort_groups, umich_spec
sources/      one file per source, auto-discovered with pkgutil
scripts/      ingest, build (derived tables), status, verify_spec
grafana/      docker-compose.yml, provisioning/, dashboards/  — all committed
tests/        fixture.sqlite + the suite
data/         offchart.sqlite — gitignored, and nothing else lives here
docs/         this
```

Two dependencies: pandas, and pytest for the suite. Everything else — SQLite, HTTP, XLSX
parsing — is standard library. `lib/xlsx.py` is 132 lines of `zipfile` and
`ElementTree` rather than making openpyxl the third.

## The two load-bearing rules

Both are enforced by tests, not by convention, and both are the reason this repo is
shaped the way it is.

**1. Every percentile is causal.** Ranks are precomputed in pandas and never in SQL. The
four reasons are in `scripts/build.py`'s module docstring, which is authoritative
because a maintainer editing that file sees it; the shortest of them is that detecting a
contract re-specification means pulling digits out of free-text contract units, SQLite
has no REGEXP without a loadable extension, and the plugin will not load one — so the
segmentation lives in Python and the rank must follow it there.

The proof lives with the implementation: `tests/test_metrics.py` shows
`expanding_percentile` is causal **by truncation** — computing on a prefix must be
bit-identical to computing on the whole series and slicing — *and* asserts that two
deliberately non-causal one-liners fail the same check. That negative control is what
stops the harness rotting into comparing a function to itself. It has earned it: it
caught a live look-ahead bug in the min-observation **gate**, where the publish/blank
threshold was derived from the whole sample's modal window occupancy, so whether
observation *t* published depended on data arriving after it. The values were causal;
the decision to show them was not.

**2. A source whose terms forbid mirroring may have no committed artefact.** Enforced by
asking git directly — `git check-ignore` and `git ls-files` — rather than by reading
`.gitignore`. See [licensing](licensing.md).

## The build is total, never incremental

`make pull` is `ingest` then `build`, and the rebuild is unconditional. That is a
correctness requirement rather than laziness: CFTC revises already-published weeks, and
a revision at week *t* changes every expanding rank from *t* forward. Rebuilding
everything makes a stale percentile — a plausible wrong number — impossible, and at the
price it costs no invalidation logic is worth writing.

Serving tables are built into `<table>_new` and renamed inside one transaction, so
Grafana never reads a half-built table. The explicit `BEGIN IMMEDIATE` in
`scripts/build._swap_in` is load-bearing and not decoration: Python's `sqlite3` opens a
transaction implicitly before DML but **not** before DDL, so a `with con:` around a DROP
and a RENAME autocommits each statement as it runs and has nothing to roll back. A crash
between the two left no serving table at all. SQLite does DDL transactionally without
complaint; only the driver needed telling.

## Storage and durability

One SQLite file, one table per source, `STRICT` and `WITHOUT ROWID`, keyed on each
source's declared natural key.

- Writes are `ON CONFLICT DO UPDATE`, never `INSERT OR REPLACE` — REPLACE is a delete
  plus an insert, so it would silently NULL every column an incoming partial frame did
  not carry.
- Change detection uses `IS NOT`, not `<>`, because a comparison against NULL yields
  NULL and `<>` would report "unchanged" for all 81k null-bearing rows in one family.
- **Dedup is the engine's, not the application's.** The declared key is a real
  `PRIMARY KEY … WITHOUT ROWID` constraint, so a repeated `make pull` over an
  overlapping window cannot duplicate a row however the surrounding code changes. The
  incremental window overlaps on purpose — `latest stored − 56 days` — because CFTC
  revises and the upsert is what decides whether anything moved.
- Column types are typed from the **first** frame ever written and frozen by
  `CREATE TABLE IF NOT EXISTS`. `scripts/ingest` coerces declared-numeric columns before
  writing so this is deterministic, and `make status` reports any that drifted before
  that coercion existed. `lib/db.ddl_for`'s docstring has the measurement.
- Every writer hands the file back in **rollback-journal** mode, not WAL. See
  [grafana](grafana.md) for the symptom that taught this.
- **Recovery** is `rm data/offchart.sqlite && make backfill`. Nothing here is kept only
  by us.

## Adding a source

`sources/` is discovered with `pkgutil`, so there is no dispatch table to update.

1. Write `sources/yourthing.py` exposing either `build_sources()` or `SOURCE_KWARGS`,
   with a `TableSchema` declaring the columns you promise. Set `group` to the bracket it
   belongs to — sources in a bracket share a frame shape and a serving table.
2. If it feeds an existing board, add its id to that board's source tuple in
   `scripts/build.py` (`FLOW_SOURCES`, or `FAMILIES` for positioning). **This step is
   easy to miss**: without it the source ingests happily and never reaches a board.
   `build` deliberately does not import `sources` to derive the list, because it is a
   pure DB→DB transform and importing the registry would pull network-capable modules
   into it.
3. `python -m scripts.ingest --source yourthing --backfill`.

Declare `redistributable=False` if the terms forbid mirroring. The dataclass refuses to
construct such a source without `license` and `citation` — `caveats` is *not* validated,
though it should be written anyway; nothing renders it automatically, so it is the
canonical text to copy into a Grafana panel description.

A module that raises on import is reported and **fails the suite** rather than silently
vanishing, because a source that disappears looks like a design decision.

### Per-bracket uniformity

Within a bracket, every source returns the same columns. Positioning enforces this in
code: the seven CFTC sources share one `parse()` that ends in
`df[list(TIDY_COLUMNS)]`, so all seven return exactly 29 columns in fixed order and a
new family cannot quietly return a different shape.

Flows is uniform by agreement rather than by a reindex, because the four issuers differ
in exactly one column — only ProShares publishes a usable daily NAV change — and
`scripts/build` derives it from NAV for the rest. That is deliberate: *absent and
declared optional* is how `lib/schema` says "this issuer does not publish it", and
emitting an all-`None` column instead would freeze it as TEXT.

There is no ABC and no per-bracket base class. The frozen dataclass plus `TableSchema`
is the contract, validated at import, and an interface over thirteen implementations
with one `fetch` each would add indirection and catch nothing those two do not.

## The commands

```
make venv          once: create .venv with uv and install requirements
make which-python  which interpreter make will use, and why
make pull          fetch latest -> database -> rebuild board_* tables
make backfill      same, but full history (first run: ~5 min, ~900 MB)
make build         rebuild board_* only, no network
make status        what is in the database, how stale, and any type drift
make grafana       start Grafana at http://localhost:3000
make grafana-stop  stop it
make grafana-logs  follow its logs
make test          the suite (no network, ~2s)
make verify        prove the CFTC field maps against the live API
make fixtures      regenerate tests/fixtures/fixture.sqlite from the store
make compact       VACUUM
make reset-db      delete the database (CONFIRM=1 required)
make readme-table  regenerate the README's inventory table
```

There is also a bare `make ingest`, the fetch half of `make pull` without the rebuild.
It is useful for debugging one source and is deliberately **not** in `make help`,
because running it alone leaves Grafana showing the previous week's derived numbers — so
it prints a reminder rather than letting you discover that from a chart.

**Nothing schedules `make pull`.** There is no cron and no CI job that fetches; the data
is as fresh as the last time you ran it, which is what `make status` is for.

### Which interpreter

Every target runs through `$(PYTHON)`, resolved in this order: an **activated**
environment (`$VIRTUAL_ENV`), then a `.venv/` directory in the tree, then whatever
`python3` is on `PATH`.

Activation beats a directory on purpose — a leftover `.venv/` is not a statement of
intent, and having it win is how a suite ends up passing against an environment you
thought you had left behind. `make which-python` prints the choice and the reason, which
is the first thing to check when the suite behaves differently in two shells. Override
any time with `make test PYTHON=/path/to/python`.

Python itself is not a prerequisite: `make venv` runs `uv venv --clear --python 3.12`,
which fetches a standalone interpreter when the system has none. `--clear` is required
because uv refuses an existing `.venv`, which made the target work exactly once.
Nothing here has a compiled extension, so both dependencies are pure wheels from
`requirements.txt`, and there is no lock file for a dependency set of two.

## Tests

`make test` — no network, about two seconds, and no dependence on the 1.8 GB database:
the suite reads the 5 MB committed slice in `tests/fixtures/fixture.sqlite`, regenerated
by `make fixtures`. Every market code in that slice is there to make one awkward case
true — the 2023-05-02 re-basing, a market-week whose nets do not sum to zero, a null
trader count beside a nonzero position, two markets sharing one name, pre-2002
non-weekly cadence.

There are no mocks anywhere. Where a failure needs inducing, it is induced through the
real mechanism: the atomicity test subclasses `sqlite3.Connection` to raise between the
DROP and the RENAME, and the staging-leak test provokes a genuine `IntegrityError` by
letting `ddl_for` freeze an int64 column and then writing a fractional value.

Three properties are what stop the suite rotting into decoration:

- **The causality check asserts that non-causal implementations fail it** — see rule 1
  above.
- **`tests/test_grafana.py` lints the dashboards**, which is the display layer's only
  guard now that there is no rendering code to test. See [grafana](grafana.md) for the
  axis-lint trap it had to be written around.
- **The licence and durability rules are asked of git, not of a comment.** See
  [licensing](licensing.md).

CI runs the suite on every push, plus `verify_spec` against the live API — the check
that catches a CFTC column which still exists but now means something else.

## Things that look like bloat and are not

- **The repeated cohort field tables in `lib/cftc_spec.py`.** Every upstream field name
  is spelled out explicitly, which is the rename detector; a loop makes a CFTC rename
  silent instead of a `KeyError`.
- **The long docstrings.** They encode measured failure modes, not restatements of the
  code. Where one carries a number, that number was measured over the real corpus.
- **The four separate flows sources.** Their licences differ materially and `license` is
  a per-source field tests enforce, so one Source spanning three issuers would make one
  of those strings false.
- **`caveats` and `provenance`**, read by nothing today. The words are the expensive
  part and they are correct.
