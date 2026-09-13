"""Committed history for sources that cannot be re-fetched.

WHY THIS EXISTS, AND IT IS THE ONLY REASON

Nine of the ten sources are `backfillable=True`: their API serves history, so losing
the local database costs one `make backfill`. One is not. `openrouter_pricing` serves
"today" and nothing else -- there is no history parameter and no archive -- so a
snapshot not captured is a hole that can never be filled.

Until now that history survived because `data/*.parquet` was committed. Removing the
parquet path removes that, so the irreplaceable bytes need their own home, and this is
it: one immutable CSV per snapshot date, committed.

WHY CSV, AND WHY ONE FILE PER DATE

A file that already exists is never rewritten, so day N's blob is frozen the moment it
is created and git stores it once. That is structurally cheaper than the parquet it
replaces -- which was rewritten in full every day, meaning every run was an
opportunity to corrupt the whole history rather than one day of it -- and it needs no
churn engineering to be cheap.

It also makes gaps DETECTABLE. A missing 2026-09-07.csv is visible; a missing day
inside a monolithic parquet is not, and today nothing would notice.

ORDER MATTERS: the CSV is written BEFORE the database is touched. A snapshot then
survives a failed ingest, a half-applied migration, a corrupt database or a machine
that never came back. The irreplaceable bytes never pass through anything stateful.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

#: Committed, unlike the database. See .gitignore, and the test that asserts both.
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


def dir_for(source_id: str) -> Path:
    return SNAPSHOT_DIR / source_id


def path_for(source_id: str, date: str) -> Path:
    return dir_for(source_id) / f"{date}.csv"


def write(source_id: str, frame: pd.DataFrame, date_column: str,
          key: tuple[str, ...]) -> list[Path]:
    """One CSV per distinct date in `frame`. Returns the paths written.

    Rows are sorted by the declared key so the bytes are deterministic: re-running an
    ingest on the same day must produce a byte-identical file and therefore a git
    no-op, not a spurious commit.

    An existing file is NOT overwritten. These are immutable by construction -- a
    snapshot is what the endpoint said at that moment, and rewriting it would be
    editing history rather than recording it.
    """
    written = []
    dir_for(source_id).mkdir(parents=True, exist_ok=True)
    for date, group in frame.groupby(frame[date_column].astype(str)):
        target = path_for(source_id, str(date))
        if target.exists():
            continue
        ordered = group.sort_values(list(key))
        # QUOTE_MINIMAL with a trailing newline and no index: the smallest stable
        # rendering, so the diff for a new day is the new day and nothing else.
        ordered.to_csv(target, index=False, lineterminator="\n",
                       quoting=csv.QUOTE_MINIMAL)
        written.append(target)
    return written


def dates(source_id: str) -> list[str]:
    """Snapshot dates on disk, ascending."""
    d = dir_for(source_id)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))


def read_all(source_id: str) -> pd.DataFrame:
    """Every committed snapshot, concatenated. This is the source of truth for a
    non-backfillable source, and what rebuilds it into a fresh database."""
    files = sorted(dir_for(source_id).glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def missing_dates(source_id: str) -> list[str]:
    """Calendar dates between the first and last snapshot that have no file.

    A gap here means a day the endpoint was not asked, and it can never be recovered.
    Surfaced so it is a visible fact rather than a silent absence -- which is what it
    was for as long as the history lived inside one rewritten parquet file.
    """
    have = dates(source_id)
    if len(have) < 2:
        return []
    span = pd.date_range(have[0], have[-1], freq="D").strftime("%Y-%m-%d")
    return [d for d in span if d not in set(have)]
