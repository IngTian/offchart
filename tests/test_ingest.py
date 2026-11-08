"""The write path: what reaches the store, and what one bad source costs.

This file exists because scripts/ingest had NO test. Not thin coverage -- none: a
trace over the whole suite showed `_since_for`, `run_one` and `main` never
executing, so the gate that refuses to write on a FATAL finding, the key-column
checks and the error handling were all unexercised while 391 tests passed.

What is pinned here is the blast radius rather than the happy path. `make pull`
ingests every source on ONE connection and is followed by `make build`, so the
question that matters is what a single bad source costs the other twelve and the
rebuild. The answer has to be "itself", and three separate things had to be true
for that: a null key is caught before the engine sees it, a write error is caught
rather than propagating out of main(), and the staging table does not outlive the
failure (asserted in tests/test_db.py, because the leak is lib.db's).

Network is never touched: `fetch` is a local lambda.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from lib import db
from lib.schema import TableSchema
from scripts import ingest
from sources.base import Source

ROOT = Path(__file__).resolve().parent.parent


def _source(frame: pd.DataFrame, **over) -> Source:
    base = dict(
        id="probe",
        label="Probe",
        fetch=lambda since=None: frame,
        key=("report_date",),
        sort_key=("report_date",),
        schema=TableSchema(required=("report_date", "v"), numeric=("v",)),
        cadence="never",
    )
    base.update(over)
    return Source(**base)


@pytest.fixture
def con(tmp_path: Path):
    connection = db.connect(tmp_path / "i.sqlite")
    yield connection
    connection.close()


def test_a_null_in_a_key_column_is_refused_before_the_engine_sees_it(con) -> None:
    """lib.schema.check cannot catch this, and that is the whole point.

    `check` asks `.isna().all()` on a required column -- ALL-null only -- so ONE
    null market_code among 900,000 rows produces zero findings and passes the
    gate. The engine then refuses it, because every key column is NOT NULL in the
    generated DDL, and that IntegrityError used to escape run_one, escape the list
    comprehension in main(), and skip db.finalize_for_serving -- leaving the file
    in WAL mode, which is the state that makes Grafana draw a blank panel beside
    two full ones.
    """
    frame = pd.DataFrame({
        "report_date": ["2026-01-06", None, "2026-01-20"],
        "v": [1.0, 2.0, 3.0],
    })
    source = _source(frame)

    # First: establish that the schema gate really does let this through, so the
    # test is pinning a gap rather than duplicating a check that already exists.
    assert source.schema.check(frame) == [], (
        "schema.check now catches a partial null in a required column, so this "
        "guard may be redundant -- check before deleting it"
    )

    assert ingest.run_one(con, source) is None
    assert db.has_rows(con, "probe") is False, "a refused frame wrote rows anyway"


def test_a_good_frame_is_written(con) -> None:
    """The happy path, so the guards above cannot pass by refusing everything."""
    frame = pd.DataFrame({
        "report_date": ["2026-01-06", "2026-01-13"],
        "v": [1.0, 2.0],
    })
    report = ingest.run_one(con, _source(frame))
    assert report is not None and report["rows_total"] == 2


def test_an_all_null_numeric_column_is_stored_as_real_not_text(con) -> None:
    """Declared-numeric columns are coerced before the write.

    An all-null column arrives as `object` dtype, so lib.db.ddl_for types it TEXT
    and CREATE TABLE IF NOT EXISTS freezes that for the life of the table. That is
    how cftc_supp_cit's eight concentration columns became TEXT in the live store
    while the identical eight in all six sibling tables are REAL -- CIT returns
    them all-null. A number later landing in a TEXT column sorts LEXICALLY, which
    breaks the causal-ranking rule through the storage layer rather than through
    any ranking code.

    `make status` reports the drift for tables already frozen; this stops new ones
    happening.
    """
    frame = pd.DataFrame({
        "report_date": ["2026-01-06"],
        "v": [1.0],
        "conc": [None],
    })
    source = _source(
        frame,
        schema=TableSchema(
            required=("report_date", "v"),
            numeric=("v", "conc"),
            optional=("conc",),
        ),
    )
    assert ingest.run_one(con, source) is not None

    types = {r[1]: r[2] for r in con.execute('PRAGMA table_info("probe")')}
    assert types["conc"] == "REAL", (
        f"an all-null declared-numeric column was stored as {types['conc']}"
    )


def test_a_write_failure_is_contained_and_does_not_abort_the_pull(con) -> None:
    """A source whose write raises must cost itself and nothing else.

    Induced the way it actually happens: ddl_for types `v` from the first frame's
    int64 dtype, freezing it as STRICT INTEGER, and a later fractional value is
    refused by the engine. (The coercion above does not save this one -- it makes
    the dtype deterministic, not retroactively wider.)

    run_one must return None rather than propagate, because main() builds its
    results with a list comprehension and anything escaping it skips
    db.finalize_for_serving.
    """
    first = pd.DataFrame({"report_date": ["2026-01-06"], "v": [1000]})
    assert ingest.run_one(con, _source(first)) is not None
    assert "INTEGER" in con.execute(
        "SELECT sql FROM sqlite_master WHERE name='probe'"
    ).fetchone()[0]

    fractional = pd.DataFrame({"report_date": ["2026-01-13"], "v": [2000.5]})
    assert ingest.run_one(con, _source(fractional)) is None, (
        "the write error propagated instead of being contained"
    )

    # The table is unchanged, the connection is usable, and the file can still be
    # handed back to Grafana -- the three things the old behaviour cost.
    assert con.execute('SELECT count(*) FROM "probe"').fetchone()[0] == 1
    assert con.in_transaction is False
    assert db.finalize_for_serving(con) == "delete"


def test_the_incremental_floor_overlaps_rather_than_resuming(con) -> None:
    """A resume-from-latest pull would never see a revision.

    CFTC revises published weeks, and `make pull` is run by hand over whatever
    window the user happens to catch, so the floor reaches back
    INCREMENTAL_OVERLAP_DAYS before the newest stored row and lets the upsert
    decide what actually changed. Asserted against the stored date rather than a
    hardcoded one so the arithmetic, not the constant, is what is pinned.
    """
    frame = pd.DataFrame({"report_date": ["2026-01-06", "2026-03-03"], "v": [1.0, 2.0]})
    source = _source(frame, incremental=True)
    assert ingest.run_one(con, source) is not None

    floor = ingest._since_for(con, source, backfill=False)
    assert floor is not None
    gap = (pd.Timestamp("2026-03-03") - pd.Timestamp(floor)).days
    assert gap == ingest.INCREMENTAL_OVERLAP_DAYS

    # A backfill ignores the floor entirely, which is what makes `make backfill`
    # the way to repair a table rather than another incremental pull.
    assert ingest._since_for(con, source, backfill=True) is None
    # ...and so does a source that cannot narrow its fetch.
    assert ingest._since_for(con, _source(frame), backfill=False) is None


def test_nothing_is_written_when_the_schema_check_is_fatal(con) -> None:
    """The gate scripts/ingest exists for, which no test executed before.

    A frame missing a required column entirely is FATAL, and FATAL must mean
    nothing reaches the store -- a half-shaped table would be frozen by
    CREATE TABLE IF NOT EXISTS and outlive the bad fetch.
    """
    frame = pd.DataFrame({"report_date": ["2026-01-06"]})  # no `v`
    source = _source(frame)
    assert any(f.severity.name == "FATAL" for f in source.schema.check(frame))
    assert ingest.run_one(con, source) is None
    assert db.has_table(con, "probe") is False


def test_a_fetch_that_raises_is_contained(con) -> None:
    """An upstream outage is the most ordinary failure here and must not be fatal
    to the other sources, so it is caught in the same place as a write error."""
    def boom(since=None):
        raise sqlite3.OperationalError("upstream is down")

    assert ingest.run_one(con, _source(pd.DataFrame(), fetch=boom)) is None
