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
from lib import db, snapshots  # noqa: E402

#: Tables Grafana reads. Listed separately from the sources because a stale one here
#: means something different: the fetch worked and the rebuild did not, which is the
#: exact state `make pull` exists to make unreachable.
SERVING = ("board_positioning", "board_inflation")


def _age_days(when: str | None) -> int | None:
    """Age in days against UTC today, not local today.

    Every date in this store is a UTC calendar date -- a CFTC report date, a snapshot
    taken by a UTC cron. Comparing against a local date west of Greenwich makes
    this morning's snapshot report an age of -1, which reads as a bug in the data.
    """
    if not when:
        return None
    return (dt.datetime.now(dt.UTC).date() - dt.date.fromisoformat(when)).days


def _short(text: str, width: int = 44) -> str:
    """First clause of a cadence note. Several run to a full paragraph -- the whole
    text belongs in `sources/`, and what a status line needs is the gist."""
    head = text.split(" -- ")[0].split(". ")[0]
    return head if len(head) <= width else head[: width - 1] + "…"


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

        for src in sources.all_sources():
            if not db.has_table(con, src.id):
                print(f"  {src.id:<22} {'-':>11}  {'not ingested':<12} {'':>6}  "
                      f"{_short(src.cadence)}")
                continue
            rows = con.execute(f'SELECT count(*) FROM "{src.id}"').fetchone()[0]
            latest = db.latest_date(con, src.id, src.sort_key[0])
            age = _age_days(latest)
            flag = "" if age is None else f"{age}d"
            print(f"  {src.id:<22} {rows:>11,}  {str(latest):<12} {flag:>6}  "
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
    finally:
        con.close()

    # The irreplaceable part, reported separately because it has a different recovery
    # story: everything above is one `make backfill` away, and this is not.
    print("\n  committed snapshots (the history no API can return):")
    any_snapshots = False
    for src in sources.all_sources():
        days = snapshots.dates(src.id)
        if not days:
            continue
        any_snapshots = True
        gaps = snapshots.missing_dates(src.id)
        note = f"  {len(gaps)} MISSING DATE(S)" if gaps else ""
        print(f"  {src.id:<22} {len(days):>4} dates  {days[0]} .. {days[-1]}{note}")
    if not any_snapshots:
        print("    none -- expected only if no backfillable=False source is registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
