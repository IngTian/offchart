#!/usr/bin/env python3
"""Pull sources and upsert them into data/. This is what CI runs.

    python -m scripts.ingest                 # every source
    python -m scripts.ingest --source cftc_tff
    python -m scripts.ingest --list

Exit code is non-zero if any source failed, so a broken feed shows up as a red
CI run rather than as a chart that quietly stops updating.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import store  # noqa: E402
from sources import all_sources, get  # noqa: E402


def run_one(source) -> dict | None:
    print(f"\n=== {source.id}  ({source.label})")
    print(f"    cadence: {source.cadence}")
    if not source.backfillable:
        print("    snapshot-only source -- a missed run is a permanent gap")
    try:
        frame = source.fetch()
    except Exception:
        print(f"  ! FETCH FAILED for {source.id}")
        traceback.print_exc()
        return None

    if frame is None or frame.empty:
        print(f"  ! {source.id} returned no rows")
        return None

    missing = [c for c in source.key if c not in frame.columns]
    if missing:
        print(f"  ! {source.id} is missing declared key columns: {missing}")
        return None

    report = store.upsert(source.id, frame, source.key)
    print(
        f"  = {report['rows_total']:,} rows stored "
        f"({report['rows_new']:,} new, {report['rows_revised']:,} revisited)"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="all", help="source id, or 'all'")
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    args = parser.parse_args()

    if args.list:
        for s in all_sources():
            flag = "history" if s.backfillable else "SNAPSHOT-ONLY"
            print(f"{s.id:<22} {flag:<13} {s.label}")
        return 0

    targets = all_sources() if args.source == "all" else [get(args.source)]
    results = [run_one(s) for s in targets]

    ok = sum(1 for r in results if r)
    print(f"\n{ok}/{len(results)} sources ingested")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
