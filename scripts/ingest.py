#!/usr/bin/env python3
"""Pull sources and upsert them into data/. This is what CI runs.

    python -m scripts.ingest                      # every source, incremental
    python -m scripts.ingest --source cftc_tff_fut
    python -m scripts.ingest --backfill           # full history, first run
    python -m scripts.ingest --list

INCREMENTAL BY DEFAULT, AND WHY

A full-history pull of all seven CFTC datasets is roughly 900 MB of JSON even
with $select projection, and the daily cron would repeat it forever against a
government API to learn about one new week. So the default asks only for reports
on or after `latest_stored - INCREMENTAL_OVERLAP_DAYS`.

The window OVERLAPS what is already stored rather than starting after it, which
is not an off-by-one -- CFTC revises already-published weeks, and the upsert is
keyed so a revision overwrites rather than duplicates. Eight weeks is enough to
catch the revisions that actually happen while keeping a daily run around 1 MB.

Use --backfill for the first run of a source, after changing a field map, or
after widening the market universe. A source with nothing stored backfills
automatically.

Exit code is non-zero if any source failed, so a broken feed is a red CI run
rather than a chart that quietly stops updating. Sources are independent: one
failure does not prevent the others from being written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import store  # noqa: E402
from lib.schema import Severity, worst  # noqa: E402
from sources import IMPORT_ERRORS, all_sources, get  # noqa: E402

#: How far back an incremental pull reaches before the newest stored report.
#: Wide enough to absorb CFTC revisions, narrow enough to stay ~1 MB.
INCREMENTAL_OVERLAP_DAYS = 56


def _since_for(source, backfill: bool) -> str | None:
    if backfill or not source.incremental:
        return None
    latest = store.latest_date(source.id, source.sort_key[0])
    if latest is None:
        return None  # nothing stored -> this IS the backfill
    floor = dt.date.fromisoformat(latest) - dt.timedelta(days=INCREMENTAL_OVERLAP_DAYS)
    return floor.isoformat()


def run_one(source, *, backfill: bool = False) -> dict | None:
    print(f"\n=== {source.id}  ({source.label})")
    print(f"    cadence: {source.cadence}")
    if not source.backfillable:
        print("    snapshot-only source -- a missed run is a permanent gap")

    since = _since_for(source, backfill)
    print(f"    mode: {'FULL HISTORY' if since is None else f'incremental since {since}'}")

    try:
        frame = source.fetch(since=since)
    except Exception:
        print(f"  ! FETCH FAILED for {source.id}")
        traceback.print_exc()
        return None

    # The row-count floor only makes sense for a full pull of a source that
    # serves history. An incremental window is legitimately a tiny fraction of
    # the store, and a snapshot source always is.
    stored_rows = None
    if since is None and source.backfillable and store.exists(source.id):
        stored_rows = len(store.read(source.id, columns=[source.key[0]]))

    findings = source.schema.check(frame, stored_rows=stored_rows)
    for f in findings:
        print(f"  {f}")
    if worst(findings) is Severity.FATAL:
        print(f"  ! {source.id} FAILED VALIDATION -- nothing written")
        return None

    missing = [c for c in source.key if c not in frame.columns]
    if missing:
        print(f"  ! {source.id} is missing declared key columns: {missing}")
        return None

    report = store.upsert(source.id, frame, source.key, sort_key=source.sort_key)
    print(
        f"  = {report['rows_total']:,} rows stored "
        f"({report['rows_new']:,} new, {report['rows_revised']:,} revisited), "
        f"{report['bytes'] / 1e6:.2f} MB"
        + ("" if report["changed"] else "  [file unchanged -- no commit needed]")
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="all", help="source id, or 'all'")
    parser.add_argument("--backfill", action="store_true", help="pull full history")
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    args = parser.parse_args()

    if IMPORT_ERRORS:
        print("!! SOME SOURCE MODULES FAILED TO IMPORT -- they are missing from the board")
        for name, tb in IMPORT_ERRORS.items():
            print(f"\n--- sources/{name}.py\n{tb}")

    if args.list:
        for s in all_sources():
            flag = "history" if s.backfillable else "SNAPSHOT-ONLY"
            inc = "incremental" if s.incremental else "full-pull"
            stored = store.read(s.id, columns=[s.key[0]])
            rows = f"{len(stored):,} rows" if not stored.empty else "empty"
            print(f"{s.id:<22} {flag:<13} {inc:<12} {rows:>14}  {s.label}")
        return 1 if IMPORT_ERRORS else 0

    targets = all_sources() if args.source == "all" else [get(args.source)]
    results = [run_one(s, backfill=args.backfill) for s in targets]

    ok = sum(1 for r in results if r)
    changed = sum(1 for r in results if r and r["changed"])
    print(f"\n{ok}/{len(results)} sources ingested, {changed} file(s) changed")
    return 0 if ok == len(results) and not IMPORT_ERRORS else 1


if __name__ == "__main__":
    raise SystemExit(main())
