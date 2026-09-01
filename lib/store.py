"""Parquet-backed store, one file per source, committed to the repo.

Why the data is versioned rather than fetched live:

- **Some sources have no history to fetch.** The OpenRouter pricing endpoint
  returns *today's* prices and nothing else. There is no backfill. For those
  sources, committing a snapshot on every run is the only mechanism that ever
  produces a time series -- which means the dataset starts the day the job
  starts, and a week not run is a week permanently missing.
- Sources that *do* serve history (CFTC) are re-pulled in full and upserted, so
  revisions to already-published rows are picked up rather than frozen.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def path_for(source_id: str) -> Path:
    return DATA_DIR / f"{source_id}.parquet"


def read(source_id: str) -> pd.DataFrame:
    p = path_for(source_id)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def upsert(source_id: str, incoming: pd.DataFrame, key: tuple[str, ...]) -> dict:
    """Merge `incoming` into the stored frame, deduping on `key` (last wins).

    Returns a small report so the ingest log says what actually changed rather
    than just "ok".
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = read(source_id)

    if existing.empty:
        merged = incoming.copy()
        added, revised = len(incoming), 0
    else:
        before = set(map(tuple, existing[list(key)].astype(str).to_numpy()))
        after = set(map(tuple, incoming[list(key)].astype(str).to_numpy()))
        added = len(after - before)
        revised = len(after & before)
        merged = pd.concat([existing, incoming], ignore_index=True)

    merged = (
        merged.drop_duplicates(subset=list(key), keep="last")
        .sort_values(list(key))
        .reset_index(drop=True)
    )
    merged.to_parquet(path_for(source_id), index=False)
    return {
        "source": source_id,
        "rows_total": len(merged),
        "rows_new": added,
        "rows_revised": revised,
    }
