"""The SQLite store, and the round trip that proves it is faithful.

The most valuable test here is test_round_trips_every_source: for each source, load
the committed parquet fixture through the real upsert and read it back, and assert it
equals the parquet frame. While both stores exist, that equality is what licenses
deleting the parquet one. It is temporary by design and should be removed with the
parquet path, not kept as decoration.

The rest pin hazards that lib/store had solved and that a different medium can undo:
a null count becoming zero, an integer being stored as a float, LAST WINS, and a
partial frame blanking columns it does not carry.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import db  # noqa: E402
from sources import all_sources, get  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _fixture(source_id: str) -> pd.DataFrame:
    path = FIXTURE_DIR / f"{source_id}.parquet"
    if not path.exists():
        pytest.skip(f"no committed fixture for {source_id}")
    return pd.read_parquet(path)


@pytest.fixture
def con(tmp_path):
    """A fresh database per test. Creation is milliseconds, so there is no reason to
    share one and roll back -- and a shared connection could not test the schema
    creation path, which upsert exercises on every first write."""
    c = db.connect(tmp_path / "wb.sqlite")
    yield c
    c.close()


FIXTURE_SOURCES = [
    s.id for s in all_sources()
    if (FIXTURE_DIR / f"{s.id}.parquet").exists()
]


# ------------------------------------------------------- the round trip

@pytest.mark.parametrize("source_id", FIXTURE_SOURCES)
def test_round_trips_every_source(con, source_id: str) -> None:
    """Parquet in, SQLite out, same frame. The whole licence for the migration."""
    src = get(source_id)
    original = _fixture(source_id)
    report = db.upsert(con, source_id, original, src.key)

    assert report["rows_total"] == len(original)
    assert report["rows_new"] == len(original)
    assert report["rows_revised"] == 0

    back = db.read_frame(con, source_id, like=original)
    # Both sides sorted by the declared key: the database has no inherent row order
    # and neither does the parquet once it round-trips, so comparing unsorted would
    # be testing an accident.
    keys = list(src.key)
    left = original.sort_values(keys).reset_index(drop=True)
    right = back.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_like=True)


@pytest.mark.parametrize("source_id", FIXTURE_SOURCES)
def test_latest_date_agrees_with_the_frame(con, source_id: str) -> None:
    """The incremental fetch floor is computed from this, so a wrong answer silently
    re-fetches everything or, worse, skips a week."""
    src = get(source_id)
    frame = _fixture(source_id)
    db.upsert(con, source_id, frame, src.key)

    column = src.sort_key[0]
    expected = str(pd.to_datetime(frame[column].astype(str)).max().date())
    assert db.latest_date(con, source_id, column) == expected


# ------------------------------------------------------- the null hazard

def test_a_null_count_never_becomes_zero(con) -> None:
    """The hazard lib/store solved, in a new medium.

    Not hypothetical: 81,065 of 232,755 rows in cftc_tff_fut carry a null
    traders_long beside a positive long. A null that arrives as 0 asserts that nobody
    held the position, which is a different and confident claim.
    """
    frame = pd.DataFrame({
        "report_date": ["2026-01-06", "2026-01-13"],
        "market_code": ["X", "X"],
        "cohort": ["a", "a"],
        "long": pd.array([1000, 2000], dtype="Int32"),
        "traders_long": pd.array([None, 7], dtype="Int32"),
    })
    key = ("report_date", "market_code", "cohort")
    db.upsert(con, "t_nulls", frame, key)

    rows = con.execute(
        'SELECT report_date, traders_long FROM "t_nulls" ORDER BY report_date'
    ).fetchall()
    assert rows[0][1] is None, "a missing trader count must stay missing, not become 0"
    assert rows[1][1] == 7


def test_integer_columns_are_stored_as_integers(con) -> None:
    """float32 is exact for integers only to 2**24 and open interest reaches
    25,702,684. In a STRICT table the type is the guarantee rather than a check."""
    frame = pd.DataFrame({
        "d": ["2026-01-06"],
        "open_interest": pd.array([25_702_684], dtype="Int64"),
    })
    db.upsert(con, "t_ints", frame, ("d",))
    kind, value = con.execute(
        'SELECT typeof(open_interest), open_interest FROM "t_ints"'
    ).fetchone()
    assert kind == "integer"
    assert value == 25_702_684


def test_a_strict_table_refuses_a_fractional_integer(con) -> None:
    """The reason the forty lines of dtype-sniffing in lib/store can go: the engine
    refuses the narrowing instead of us detecting it."""
    frame = pd.DataFrame({"d": ["2026-01-06"], "n": pd.array([1], dtype="Int64")})
    db.upsert(con, "t_strict", frame, ("d",))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute('INSERT INTO "t_strict" (d, n) VALUES (?, ?)', ("2026-01-13", 1.5))


# ------------------------------------------------------- merge semantics

def test_last_wins_on_the_key(con) -> None:
    """CFTC revises already-published weeks, so a second arrival for the same key must
    overwrite rather than duplicate."""
    key = ("d", "code")
    first = pd.DataFrame({"d": ["2026-01-06"], "code": ["X"], "v": [1.0]})
    second = pd.DataFrame({"d": ["2026-01-06"], "code": ["X"], "v": [2.0]})
    db.upsert(con, "t_last", first, key)
    report = db.upsert(con, "t_last", second, key)

    assert report["rows_total"] == 1, "a revision must not duplicate the row"
    assert report["rows_new"] == 0
    assert report["rows_revised"] == 1
    assert report["rows_value_changed"] == 1
    assert con.execute('SELECT v FROM "t_last"').fetchone()[0] == 2.0


def test_an_identical_reingest_reports_no_value_change(con) -> None:
    """This is what the byte digest in lib/store was a proxy for, said directly.

    On a full-refresh source every row is 'revisited' on every run, so that count
    cannot tell 'upstream republished the same numbers' from 'upstream revised a
    figure'. rows_value_changed can, which is the distinction sources/umich's
    revision guard exists to report.
    """
    key = ("d",)
    frame = pd.DataFrame({"d": ["2026-01-06", "2026-01-13"], "v": [1.0, 2.0]})
    db.upsert(con, "t_same", frame, key)
    report = db.upsert(con, "t_same", frame, key)

    assert report["rows_revised"] == 2, "both rows were seen again"
    assert report["rows_value_changed"] == 0, "but nothing actually moved"
    assert report["changed"] is False


def test_a_null_valued_column_is_not_reported_as_changed(con) -> None:
    """Comparing with <> would call every null-bearing row 'changed', because a
    comparison against NULL is NULL rather than false. With 81k null trader counts in
    one family that would report a revision every single day."""
    key = ("d",)
    frame = pd.DataFrame({
        "d": ["2026-01-06"],
        "traders": pd.array([None], dtype="Int32"),
    })
    db.upsert(con, "t_nullcmp", frame, key)
    report = db.upsert(con, "t_nullcmp", frame, key)
    assert report["rows_value_changed"] == 0


def test_a_partial_frame_does_not_blank_the_columns_it_omits(con) -> None:
    """INSERT OR REPLACE is a delete plus an insert, so it would NULL every column the
    incoming frame does not carry -- blanking cftc_supp_cit's concentration columns
    the first time a partial frame arrived. DO UPDATE touches only what is present."""
    key = ("d",)
    full = pd.DataFrame({"d": ["2026-01-06"], "a": [1.0], "b": [2.0]})
    db.upsert(con, "t_partial", full, key)
    db.upsert(con, "t_partial", pd.DataFrame({"d": ["2026-01-06"], "a": [9.0]}), key)

    a, b = con.execute('SELECT a, b FROM "t_partial"').fetchone()
    assert a == 9.0, "the supplied column updates"
    assert b == 2.0, "the omitted column survives"


# ------------------------------------------------------- housekeeping

def test_an_empty_frame_is_a_no_op(con) -> None:
    report = db.upsert(con, "t_empty", pd.DataFrame(), ("d",))
    assert report["rows_total"] == 0
    assert report["changed"] is False


def test_reading_an_absent_source_is_empty_not_an_error(con) -> None:
    """Same contract as the parquet store: a source that has never been ingested reads
    as empty so the board can say 'not ingested yet' rather than crashing."""
    assert db.read_frame(con, "never_ingested").empty
    assert db.has_rows(con, "never_ingested") is False
    assert db.latest_date(con, "never_ingested") is None


def test_a_source_id_that_is_not_a_safe_identifier_is_refused() -> None:
    """The table name is interpolated into SQL, so this is the injection guard."""
    with pytest.raises(ValueError, match="not a safe table name"):
        db.table_for("evil; DROP TABLE x")


def test_the_database_file_is_gitignored() -> None:
    """It will contain umich_sca, which is licensed for use and not redistribution --
    the same rule tests/test_sources.py enforces for that source's parquet."""
    import subprocess

    rel = db.DB_PATH.relative_to(ROOT).as_posix()
    for candidate in (rel, rel + "-wal", rel + "-shm"):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", candidate], cwd=ROOT, capture_output=True
        ).returncode == 0
        assert ignored, f"{candidate} must be gitignored before any data lands in it"
