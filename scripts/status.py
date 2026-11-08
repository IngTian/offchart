#!/usr/bin/env python3
"""What is in the database, and how stale it is. Run: make status

The question this answers is the one a pipeline cannot answer for you: a chart drawn
from a database that stopped updating three weeks ago looks exactly like a chart drawn
from a fresh one. Grafana will happily render a flat line to the right edge of the
window. So freshness is reported here, next to the row counts, with the age of the
newest row in days -- because the failure this repo is most likely to suffer is not a
crash, it is a cron that quietly stopped.

Read-only: it opens the store with mode=ro so running it during a build cannot
interfere, and it never fetches.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sources  # noqa: E402
from lib import db  # noqa: E402

#: Tables Grafana reads. Listed separately from the sources because a stale one here
#: means something different: the fetch worked and the rebuild did not, which is the
#: exact state `make pull` exists to make unreachable.
SERVING = ("board_positioning", "board_inflation", "board_flows")


def _age_days(when: str | None) -> int | None:
    """Age in days against UTC today, not local today.

    Every date in this store is a UTC calendar date -- a CFTC report date, a daily
    close. Comparing against a local date west of Greenwich makes today's newest
    row report an age of -1, which reads as a bug in the data.
    """
    if not when:
        return None
    return (dt.datetime.now(dt.UTC).date() - dt.date.fromisoformat(when)).days


def _short(text: str, width: int = 44) -> str:
    """First clause of a cadence note. Several run to a full paragraph -- the whole
    text belongs in `sources/`, and what a status line needs is the gist."""
    head = text.split(" -- ")[0].split(". ")[0]
    return head if len(head) <= width else head[: width - 1] + "…"


def _type_drift(con) -> list[tuple[str, str, str]]:
    """Declared-numeric columns whose STORED type is not numeric.

    Reported because a stored type cannot be changed in place and nothing else
    looks at it. lib.db.ddl_for types every column from the dtypes of the FIRST
    frame written, and `CREATE TABLE IF NOT EXISTS` freezes that choice -- so a
    column typed before the parser settled keeps the old type for the life of the
    table, however the code changes.

    Only NON-NUMERIC storage is reported, which today means TEXT: a number in a
    TEXT column sorts LEXICALLY, so '9.1' outranks '85.0' and the causal-ranking
    rule fails through the storage layer rather than through any ranking code.
    That is the case where a wrong number reaches a board.

    INTEGER is deliberately NOT reported, even though the live store has 70 such
    columns where today's parser emits float. SQLite STRICT accepts a
    whole-number REAL into an INTEGER column and refuses only a fractional one,
    and every INTEGER column here is a contract count that CFTC publishes as a
    whole number -- so listing them would be 70 lines of noise about a failure
    that cannot currently occur. lib.db.ddl_for's docstring records it instead.

    The remedy is to drop the table and `make backfill` that source;
    scripts/ingest now coerces declared-numeric columns before the write, so a
    rebuilt table gets REAL even when the column arrives all-null.
    """
    out = []
    for src in sources.all_sources():
        info = list(con.execute(f'PRAGMA table_info("{src.id}")'))
        if not info:
            continue
        stored = {row[1]: row[2] for row in info}
        for col in src.schema.numeric:
            got = stored.get(col)
            if got is not None and got not in ("REAL", "INTEGER"):
                out.append((src.id, col, got))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="path to the sqlite file")
    args = ap.parse_args()

    path = Path(args.db) if args.db else db.DB_PATH
    if not path.exists():
        print(f"no database at {path}\n  ->  make backfill")
        return 1

    con = db.connect(path, read_only=True)
    try:
        size = path.stat().st_size / 1e6
        print(f"{path}  {size:,.0f} MB\n")
        print(f"  {'source':<22} {'rows':>11}  {'latest':<12} {'age':>6}  cadence")
        print(f"  {'-' * 22} {'-' * 11}  {'-' * 12} {'-' * 6}  {'-' * 12}")

        # Grouped by BRACKET rather than listed flat. Sources within a bracket share
        # a frame shape and a serving table, so they are the set a reader compares
        # against each other -- a flows source two days stale next to one five days
        # stale is a fact about the issuers, while a flows source beside a weekly
        # CFTC one is not a comparison at all.
        for group, srcs in sorted(sources.by_group().items()):
            print(f"  {group.lower()}")
            for src in srcs:
                if not db.has_table(con, src.id):
                    print(f"    {src.id:<20} {'-':>11}  {'not ingested':<12} {'':>6}  "
                          f"{_short(src.cadence)}")
                    continue
                rows = con.execute(f'SELECT count(*) FROM "{src.id}"').fetchone()[0]
                latest = db.latest_date(con, src.id, src.sort_key[0])
                age = _age_days(latest)
                flag = "" if age is None else f"{age}d"
                print(f"    {src.id:<20} {rows:>11,}  {str(latest):<12} {flag:>6}  "
                      f"{_short(src.cadence)}")

        print()
        for table in SERVING:
            if not db.has_table(con, table):
                print(f"  {table:<22} {'-':>11}  not built    ->  make build")
                continue
            rows = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            latest = con.execute(f'SELECT max(ts) FROM "{table}"').fetchone()[0]
            when = (dt.datetime.fromtimestamp(latest / 1000, dt.UTC).date().isoformat()
                    if latest else None)
            age = _age_days(when)
            print(f"  {table:<22} {rows:>11,}  {str(when):<12} "
                  f"{'' if age is None else str(age) + 'd':>6}  serving")

        drift = _type_drift(con)
        if drift:
            print(f"\n  ! {len(drift)} declared-numeric column(s) are stored as text,")
            print("    so any value in them would sort lexically. Drop the table and")
            print("    `make backfill` that source to retype it:")
            for source_id, col, got in drift:
                print(f"      {source_id}.{col}  {got}")
    finally:
        con.close()

    # Everything above is one `make backfill` away. There used to be a section here for
    # the one source that was not -- openrouter_pricing, whose history existed only as
    # committed CSVs -- and it is gone with that source.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
