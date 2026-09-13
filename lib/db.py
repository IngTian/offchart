"""The SQLite store. Same contract as lib/store, different medium.

WHY A DATABASE AT ALL, AND WHY THIS ONE

The display layer is moving to Grafana, which reads SQL and not parquet. Of the
databases Grafana can talk to, exactly three are built in -- MSSQL, MySQL,
PostgreSQL -- and all three are servers. Every serverless option is a plugin.

SQLite is chosen over Postgres for one reason that outranks the rest: it is the only
choice that leaves NO ALWAYS-ON SERVICE ANYWHERE. Grafana becomes a stateless
container started when someone wants to look; this file is disposable and rebuildable
from the APIs; nothing holds state that must not be lost. A Postgres instance would be
the single component in the whole system holding irreplaceable state, and nursing that
is the maintenance this move exists to retire.

The cost is real and named here so nobody rediscovers it: Grafana's SQLite data source
is a signed COMMUNITY plugin (frser-sqlite-datasource), one maintainer, and its
catalog badge reads "Needs attention". It is pure Go via modernc.org/sqlite, so it
needs no cgo and runs on the default Alpine Grafana image without
allow_loading_unsigned_plugins. If it ever stops being viable the schema here is
ordinary SQL and moving these tables into Postgres is an afternoon.

Second cost: that plugin implements only one macro, $__unixEpochGroupSeconds. There is
no $__timeFilter and no $__timeGroup, so panel SQL writes its own bound --
`ts BETWEEN $__from AND $__to` -- which is why `ts` is stored as epoch MILLISECONDS
alongside the human-readable date. Grafana's $__from/$__to are milliseconds.

WHAT THIS MODULE IS NOT

It is not the serving layer. This holds the ARCHIVE: one table per source, the same
columns the parquet had, faithful to the row. The wide per-board tables Grafana
actually queries are derived from these by a separate builder, because the
percentiles they carry must be computed in pandas (see scripts/build) -- the segment
boundaries that a rank must respect come from a regex over free-text contract units,
and SQLite has no REGEXP without a loadable extension the data source will not load.

DURABILITY, WHICH IS NOT THIS FILE'S JOB

This file is deliberately NOT committed and NOT backed up. Nine of the ten sources are
backfillable=True: losing them costs one `scripts/ingest --backfill`. The tenth,
openrouter_pricing, serves only "today" and its history exists solely because we keep
it -- 4,727 rows across 11 snapshot dates, 26 KB, the entire irreplaceable asset in
this repo. That does not live here. It lives in git as one immutable CSV per snapshot
date, written before the database is touched, so a snapshot survives a corrupt
database, a migration in progress, or a host that never came back.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

#: Beside the parquet, not replacing it yet. Gitignored -- see .gitignore, and the
#: test that asserts it, because this file will contain umich_sca, which is licensed
#: for use and not for redistribution.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "watchboard.sqlite"


def table_for(source_id: str) -> str:
    """Source id to table name. They are already valid identifiers, so this is
    identity plus a guard against one arriving that is not."""
    if not source_id.replace("_", "").isalnum():
        raise ValueError(f"source id {source_id!r} is not a safe table name")
    return source_id


def connect(path: Path | str | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the store.

    WAL, because there is one writer (ingest, then the builder) and N readers
    (Grafana, the app, tests) and the default rollback journal would have them block
    each other. busy_timeout rather than an immediate failure for the same reason: a
    reader arriving mid-build should wait, not error.
    """
    if path is None:
        path = DB_PATH
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 30000")
    con.execute("PRAGMA foreign_keys = ON")
    return con


#: pandas dtype -> SQLite STRICT column type.
#:
#: STRICT is the point, not decoration. lib/store spent forty lines deciding whether a
#: float column was integral so it could avoid narrowing a count into float32 -- open
#: interest reaches 25,702,684 and float32 is exact only to 2**24. In a STRICT table
#: that reasoning becomes DDL: inserting 1.5 into an INTEGER column raises instead of
#: silently rounding. The check moves from "detect and hope" to "the engine refuses".
def _sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    return "TEXT"


def ddl_for(source_id: str, frame: pd.DataFrame, key: tuple[str, ...]) -> str:
    """CREATE TABLE for one source, typed from the frame lib.store would have stored.

    The dtypes come from `store._tighten`, which is a declared and tested
    normalisation rather than raw upstream data -- so the schema is derived from
    something with tests on it, not inferred from whatever happened to arrive today.
    WITHOUT ROWID because every table here is keyed by its natural key and the
    implicit rowid would be a second, useless index.
    """
    cols = []
    for name, dtype in frame.dtypes.items():
        null = " NOT NULL" if name in key else ""
        cols.append(f'  "{name}" {_sqlite_type(dtype)}{null}')
    pk = ", ".join(f'"{k}"' for k in key)
    return (
        f'CREATE TABLE IF NOT EXISTS "{table_for(source_id)}" (\n'
        + ",\n".join(cols)
        + f",\n  PRIMARY KEY ({pk})\n) STRICT, WITHOUT ROWID;"
    )


def ensure_table(con: sqlite3.Connection, source_id: str, frame: pd.DataFrame,
                 key: tuple[str, ...]) -> None:
    con.execute(ddl_for(source_id, frame, key))


def _rows(frame: pd.DataFrame) -> list[tuple]:
    """Frame to tuples with EVERY missing value as None.

    This is the one place the old null hazard could come back wearing a new costume.
    lib/store carried a rule that a null trader count must never become 0 -- and it is
    not hypothetical: 81,065 of 232,755 rows in cftc_tff_fut have a null traders_long
    beside a positive long. Handing pandas nullable Int32 with pd.NA straight to
    executemany routes through object or float and can land as 0.0, so every missing
    value is mapped explicitly here and a test asserts the round trip.
    """
    out = frame.astype(object).where(frame.notna(), None)
    return list(out.itertuples(index=False, name=None))


def upsert(con: sqlite3.Connection, source_id: str, frame: pd.DataFrame,
           key: tuple[str, ...]) -> dict:
    """Merge `frame` into the source's table, LAST WINS on `key`.

    Stage-then-merge in one transaction, so a failure leaves the table as it was
    rather than half-updated.

    ON CONFLICT DO UPDATE, never INSERT OR REPLACE. REPLACE is a delete followed by an
    insert, so any column the incoming frame does not carry would be silently NULLed --
    which would blank cftc_supp_cit's concentration columns the first time a partial
    frame arrived. DO UPDATE touches only the columns present.

    Returns the same report shape lib.store.upsert returns, plus rows_value_changed:
    on a full-refresh source the revisited count is every row every time and cannot
    distinguish "upstream republished identical numbers" from "upstream revised a
    figure". That distinction is the whole point of the guard in sources/umich.
    """
    table = table_for(source_id)
    if frame.empty:
        return {"source": source_id, "rows_total": _count(con, source_id),
                "rows_new": 0, "rows_revised": 0, "rows_value_changed": 0,
                "changed": False}

    ensure_table(con, source_id, frame, key)
    cols = list(frame.columns)
    quoted = ", ".join(f'"{c}"' for c in cols)
    marks = ", ".join("?" for _ in cols)
    data_cols = [c for c in cols if c not in key]
    sets = ", ".join(f'"{c}" = excluded."{c}"' for c in data_cols) or None
    on_key = ", ".join(f'"{k}"' for k in key)
    join_on = " AND ".join(f't."{k}" = s."{k}"' for k in key)

    with con:
        con.execute(f'CREATE TEMP TABLE _stage AS SELECT {quoted} FROM "{table}" WHERE 0')
        con.executemany(f"INSERT INTO _stage ({quoted}) VALUES ({marks})", _rows(frame))

        new = con.execute(
            f'SELECT count(*) FROM _stage s WHERE NOT EXISTS '
            f'(SELECT 1 FROM "{table}" t WHERE {join_on})'
        ).fetchone()[0]
        revised = len(frame) - new
        # IS NOT, never <>. A comparison against NULL yields NULL, which is not true,
        # so <> would report "unchanged" for every row where either side is null --
        # and 81k rows here have a null trader count. IS NOT compares nulls properly.
        if data_cols:
            differs = " OR ".join(f't."{c}" IS NOT s."{c}"' for c in data_cols)
            value_changed = con.execute(
                f'SELECT count(*) FROM _stage s JOIN "{table}" t ON {join_on} '
                f"WHERE {differs}"
            ).fetchone()[0]
        else:
            value_changed = 0

        if sets:
            con.execute(
                f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} FROM _stage '
                f"WHERE true ON CONFLICT ({on_key}) DO UPDATE SET {sets}"
            )
        else:
            con.execute(
                f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} FROM _stage '
                f"WHERE true ON CONFLICT ({on_key}) DO NOTHING"
            )
        con.execute("DROP TABLE _stage")

    return {
        "source": source_id,
        "rows_total": _count(con, source_id),
        "rows_new": int(new),
        "rows_revised": int(revised),
        "rows_value_changed": int(value_changed),
        "changed": bool(new or value_changed),
    }


def _count(con: sqlite3.Connection, source_id: str) -> int:
    if not has_table(con, source_id):
        return 0
    return con.execute(f'SELECT count(*) FROM "{table_for(source_id)}"').fetchone()[0]


def has_table(con: sqlite3.Connection, source_id: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_for(source_id),),
    ).fetchone()
    return row is not None


def has_rows(con: sqlite3.Connection, source_id: str) -> bool:
    return _count(con, source_id) > 0


def read_frame(con: sqlite3.Connection, source_id: str,
               columns: list[str] | None = None,
               like: pd.DataFrame | None = None) -> pd.DataFrame:
    """Read a source back, shaped like lib.store.read returned it.

    `like` is a frame whose dtypes the result should match. That argument exists
    because SQLite has five storage classes and pandas has many: a column stored as
    INTEGER comes back int64 whether it left as Int32 or as a nullable count with
    real nulls, and a category comes back as object. During the migration the caller
    has the parquet frame to hand, so matching it exactly is what makes the round-trip
    test meaningful rather than approximate.
    """
    if not has_table(con, source_id):
        return pd.DataFrame()
    what = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    frame = pd.read_sql_query(f'SELECT {what} FROM "{table_for(source_id)}"', con)
    if like is None:
        return frame
    for name, dtype in like.dtypes.items():
        if name not in frame.columns:
            continue
        if isinstance(dtype, pd.CategoricalDtype):
            # The DTYPE, not just "category". Rebuilding from the read-back values
            # gives the same 434 strings in a different order, and a CategoricalDtype
            # carries its category list as part of its identity -- so the frames
            # compare unequal while every value is right. Adopting the original dtype
            # also keeps this honest: a value in the database that is NOT in the
            # original category set becomes NaN here and the comparison fails, which
            # is what should happen.
            frame[name] = frame[name].astype(dtype)
        else:
            frame[name] = frame[name].astype(dtype)
    return frame[[c for c in like.columns if c in frame.columns]]


def finalize_for_serving(con: sqlite3.Connection) -> str:
    """Check-point the WAL and leave WAL mode. Call this when a build finishes.

    NOT cosmetic, and the failure it prevents is one panel in three showing "No data".
    Measured: with the file in WAL mode and bind-mounted into the Grafana container on
    macOS, the three panels of a board query concurrently and one loses a lock race --
    the plugin returns `database is locked (5) (SQLITE_BUSY)` and Grafana draws an
    empty panel beside two full ones. In rollback-journal mode a reader does not need
    to attach the -shm file at all, and the same three panels all return 200.

    WAL is still right for WRITING: one writer and N readers is exactly what it is for,
    and the build wants it. So the mode is a phase, not a setting -- WAL while building,
    DELETE once the file is something Grafana reads.
    """
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    mode = con.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    return str(mode)


def latest_date(con: sqlite3.Connection, source_id: str,
                date_column: str = "report_date") -> str | None:
    """Most recent stored date, or None. Drives the incremental fetch floor.

    An index seek where the parquet version read the whole column.
    """
    if not has_table(con, source_id):
        return None
    try:
        row = con.execute(
            f'SELECT max("{date_column}") FROM "{table_for(source_id)}"'
        ).fetchone()
    except sqlite3.OperationalError:
        return None  # column absent: same contract as the parquet version
    if not row or row[0] is None:
        return None
    return str(pd.to_datetime(row[0]).date())
