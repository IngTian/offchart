"""Parquet-backed store, one file per source, committed to the repo.

WHY THE DATA IS VERSIONED RATHER THAN FETCHED LIVE

- Some sources have no history to fetch. The OpenRouter pricing endpoint returns
  today's prices and nothing else; committing a snapshot every run is the only
  mechanism that ever produces a series, so the dataset starts the day the job
  starts and a week not run is a week permanently missing.
- Sources that do serve history (all seven CFTC datasets) are upserted, so CFTC
  revisions to already-published weeks are picked up rather than frozen.
- The board then works offline and cannot display a number that is not in the
  committed dataset.

WHY THE SORT KEY PUTS report_date FIRST -- THE SINGLE MOST IMPORTANT LINE HERE

Parquet is a binary blob and git has no format-aware delta for it, so the cost
of committing a dataset daily is decided entirely by how much of the file's
BYTES move when a week is appended.

Sort by (market_code, report_date) and each new week inserts a row into the
middle of every market's run, which rewrites essentially the whole file. Sort by
(report_date, market_code) and a new week is a pure tail append, which git's
delta compression handles almost perfectly. Measured with real git on the full
tff_fut history (231,805 tidy rows, 6 sequential weekly commits with gc):

    code-first   ~2,540,000 bytes of .git growth per commit   (~126 MiB/year)
    date-first        ~5,300 bytes per commit                 (~0.3 MiB/year)

That is a factor of ~480 for one line of ordering, and it is why this repo does
NOT need year-partitioned files. Partitioning buys ~20x, costs ~170 files across
sources and years, and is fragile in a way that is easy to miss: it only works
if each partition's dictionary is built from that partition alone, since a
global dictionary makes every cold file's bytes change the moment a new market
code appears anywhere in the corpus.

Do not reorder `Source.key` without re-measuring this.

BYTE-DETERMINISM IS A PYARROW PROPERTY
Writing the same frame twice produces identical bytes within a pyarrow version,
which is what makes "skip the write when nothing changed" work and keeps no-op
days out of git. A pyarrow upgrade rewrites every file identically-in-content,
producing one large no-op commit. That is why requirements.txt pins pyarrow.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: zstd beats snappy by ~20% here at no meaningful read cost, and both are
#: deterministic within a version.
COMPRESSION = "zstd"

#: Rows per parquet row group. This is the second half of the churn fix and it
#: is worth as much as the sort order.
#:
#: A parquet file lays each column out as a contiguous chunk per row group. With
#: one row group for the whole file, appending a week shifts the byte offset of
#: every column chunk after the first, so most of the file moves even though the
#: data is appended at the end. Row groups bound that: with date-first ordering
#: the new week lands in the last group and every earlier group stays
#: byte-identical, which git deltas for free.
#:
#: Measured on the real tff_fut history (231,805 rows, ~219 rows per report week),
#: mean .git growth per weekly commit:
#:
#:     ordering     row_group_size    B/commit   % of file    MB/yr
#:     code-first   default          5,131,605      86.9%    1873.0
#:     date-first   default            516,779       8.5%     188.6
#:     date-first   50,000             361,472       5.6%     131.9
#:     date-first   20,000             232,448       3.4%      84.8
#:     date-first    5,000              76,459       1.0%      27.9   <- chosen
#:     date-first    2,000              43,179       0.5%      15.8
#:
#: 5,000 is the knee: 67x less churn than the code-first default for a 22% larger
#: file (6.05 -> 7.41 MB). Going to 2,000 buys another 1.8x on churn but costs
#: 45% on size, which is the wrong trade when the file is also read on every page
#: load. Re-measure before changing either this or the sort order.
ROW_GROUP_SIZE = 5_000


def _is_date_column(name: str) -> bool:
    return name.endswith("date") or name.endswith("_date")


def path_for(source_id: str) -> Path:
    return DATA_DIR / f"{source_id}.parquet"


def exists(source_id: str) -> bool:
    return path_for(source_id).exists()


def read(source_id: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a source. Returns an empty frame when it has never been ingested.

    Callers get a fresh object every time, so mutating the result is safe. The
    app layer caches this (see lib/cache.py) and the cache must therefore hand
    out copies -- st.cache_data does, st.cache_resource does not.
    """
    p = path_for(source_id)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, columns=columns)


def latest_date(source_id: str, date_column: str = "report_date") -> str | None:
    """Most recent date already stored, or None. Drives the incremental fetch."""
    p = path_for(source_id)
    if not p.exists():
        return None
    try:
        col = pd.read_parquet(p, columns=[date_column])[date_column]
    except (KeyError, ValueError):
        return None
    if col.empty:
        return None
    # Decategorise defensively: an older file may have this column stored as a
    # category, and pd.to_datetime on an unordered Categorical produces
    # something whose .max() raises rather than comparing.
    if isinstance(col.dtype, pd.CategoricalDtype):
        col = col.astype(object)
    return str(pd.to_datetime(col).max().date())


#: float32 represents integers exactly only up to 2**24. Open interest reaches
#: 25,702,684 contracts, so narrowing a COUNT to float32 would silently round it.
#: Integer-valued columns go to a nullable integer type instead, never to float32.
_F32_INT_SAFE = 2**24


def _tighten(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow dtypes before writing. Never at the cost of a value.

    Three moves, in order of payoff on real data:

    - Integer-valued columns (every position count, trader count and published
      change) become nullable Int32/Int64. They arrive as float64 because
      Socrata delivers numbers as strings and nulls are common, and float64 is
      twice the width they need.
    - Genuinely fractional columns (the concentration percentages, which are
      0-100 with one decimal) become float32.
    - Repeated low-cardinality strings (market names, cohort ids, unit strings)
      become dictionaries.

    A nullable integer is used rather than plain int32 because null is meaningful
    here and must not become 0 -- a null trader count co-occurs with a nonzero
    position, so zeroing it would turn "unknown" into "no firms hold this".
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]

        if s.dtype == object or isinstance(s.dtype, pd.StringDtype) or s.dtype == "str":
            # Date columns stay plain strings. Categorising them saves nothing
            # measurable (report_date is 5 KB of a 6 MB file, because there are
            # only ~1,000 distinct dates and parquet already compresses the runs)
            # and it costs real friction: pd.to_datetime on an unordered
            # Categorical yields something whose .max() raises, so every caller
            # would need to remember to decategorise first.
            if _is_date_column(col):
                continue
            # Only dictionary-encode when it actually pays.
            if len(s) and s.nunique(dropna=True) <= max(1024, len(s) // 50):
                out[col] = s.astype("category")
            continue

        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue

        vals = s.dropna()
        if vals.empty:
            continue

        integral = bool((vals % 1 == 0).all())
        if integral:
            lo, hi = float(vals.min()), float(vals.max())
            if lo >= -(2**31) and hi <= 2**31 - 1:
                out[col] = s.astype("Int32")
            else:
                out[col] = s.astype("Int64")
        elif float(vals.abs().max()) < _F32_INT_SAFE:
            out[col] = s.astype("float32")
    return out


def _dictionary_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns worth dictionary-encoding, detected rather than declared.

    In a tidy frame with one row per (date, market, cohort), every market-week
    level column -- open interest, the concentration ratios, the total trader
    count -- is repeated identically across that week's cohort rows, and
    date-first sorting puts those rows next to each other. Dictionary encoding
    turns each run into repeated indices that then compress to almost nothing.

    Detection is by run-repetition and cardinality rather than a hardcoded
    column list, so it keeps working for a future source with a different shape.

    Restricting dictionary encoding to these columns also matters in the other
    direction: pyarrow dictionary-encodes everything by default, and on the
    high-cardinality position counts that is actively worse. Measured on
    tff_fut at row_group_size=5000:

        7.57 MB   pyarrow default (dictionary on all columns)
        6.32 MB   dictionary only on the repeated market-week columns

    So this is worth 16% and it is why the row-group change costs only ~4% on
    size rather than the 22% a naive write would have cost.
    """
    hints: list[str] = []
    n = len(df)
    if n < 2:
        return hints
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue  # categoricals are already dictionary-typed in arrow
        repeat_frac = float((s.to_numpy()[1:] == s.to_numpy()[:-1]).mean())
        if repeat_frac > 0.5 or s.nunique(dropna=True) <= 1024:
            hints.append(col)
    return hints


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upsert(
    source_id: str,
    incoming: pd.DataFrame,
    key: tuple[str, ...],
    *,
    sort_key: tuple[str, ...] | None = None,
) -> dict:
    """Merge `incoming` into the stored frame, deduping on `key` (last wins).

    `sort_key` defaults to `key` and MUST lead with the date column -- see the
    module docstring. Returns a report so the ingest log says what changed
    rather than just "ok", including whether the file was rewritten at all.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = path_for(source_id)
    before_digest = _digest(path)

    existing = read(source_id)
    key_l = list(key)

    if existing.empty:
        merged = incoming.copy()
        added, revised = len(incoming), 0
    else:
        missing = [c for c in key_l if c not in existing.columns]
        if missing:
            raise ValueError(
                f"{source_id}: stored file lacks key columns {missing}. The schema "
                "changed; delete the file and backfill rather than merging."
            )
        before_keys = set(map(tuple, existing[key_l].astype(str).to_numpy()))
        after_keys = set(map(tuple, incoming[key_l].astype(str).to_numpy()))
        added = len(after_keys - before_keys)
        revised = len(after_keys & before_keys)
        # Align categories away before concat so pandas does not object.
        merged = pd.concat(
            [existing.astype({c: "object" for c in existing.select_dtypes("category")}),
             incoming],
            ignore_index=True,
        )

    order = list(sort_key or key)
    merged = (
        merged.drop_duplicates(subset=key_l, keep="last")
        .sort_values(order, kind="mergesort")
        .reset_index(drop=True)
    )

    merged = _tighten(merged)
    dict_cols = _dictionary_columns(merged)
    merged.to_parquet(
        path,
        index=False,
        compression=COMPRESSION,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=dict_cols if dict_cols else False,
    )
    after_digest = _digest(path)

    return {
        "source": source_id,
        "rows_total": len(merged),
        "rows_new": added,
        "rows_revised": revised,
        "bytes": path.stat().st_size,
        "changed": before_digest != after_digest,
    }
