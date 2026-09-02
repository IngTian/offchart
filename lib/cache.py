"""Cached reads for the display layer.

Streamlit re-executes the whole script on every widget interaction, so an
uncached read of a 30 MB parquet happens on every click. These wrappers make that
once per file version.

WHY st.cache_data AND NEVER st.cache_resource

cache_data caches serialised bytes and hands back a fresh object per call;
cache_resource caches the object itself and hands back the SAME object every
time. Panels routinely do things like `df["report_date"] = pd.to_datetime(...)`,
which under cache_resource mutates the shared cached frame and corrupts every
later reader -- verified: with cache_resource the second caller sees dtype
datetime64 where it expected a string, while with cache_data it does not.

So: cache_data everywhere here, and panels may treat the result as their own.

WHY THE CACHE KEY INCLUDES FILE MTIME AND SIZE

Otherwise a fresh ingest during a running session serves stale data forever. The
key is (source_id, mtime_ns, size), so a rewritten file invalidates naturally and
an unchanged one keeps its entry -- which matters because the ingest job rewrites
files it has not changed, and byte-identical rewrites do not bump mtime content
but do bump mtime. Including size as well makes the key robust to a filesystem
with coarse timestamps.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import store, universe


def _stat_token(source_id: str) -> tuple[float, int]:
    p = store.path_for(source_id)
    if not p.exists():
        return (0.0, 0)
    st_ = p.stat()
    return (st_.st_mtime_ns / 1e9, st_.st_size)


@st.cache_data(show_spinner=False, max_entries=16)
def _read_cached(source_id: str, _token: tuple[float, int]) -> pd.DataFrame:
    return store.read(source_id)


def read(source_id: str) -> pd.DataFrame:
    """Cached whole-source read.

    Caching the whole source rather than a filtered slice is deliberate. The
    largest file here is ~30 MB on disk, and the pushdown alternative is not the
    win it looks like: a filtered pyarrow read still decompresses every row group
    it cannot prune, so its measured peak memory is ~60 MB rather than the size of
    the result. Whole-source caching is simpler, and it makes the cross-market
    screen -- which needs every market -- free instead of pathological.
    """
    return _read_cached(source_id, _stat_token(source_id))


@st.cache_data(show_spinner=False, max_entries=16)
def _stats_cached(source_id: str, _token: tuple[float, int]) -> dict | None:
    p = store.path_for(source_id)
    if not p.exists():
        return None
    src = _source(source_id)
    col = src.sort_key[0] if src else None
    try:
        dates = store.read(source_id, columns=[col] if col else None)
    except (KeyError, ValueError):
        dates = store.read(source_id)
    if dates.empty:
        return {"rows": 0, "bytes": p.stat().st_size, "latest": None}
    series = dates[col] if col and col in dates.columns else None
    latest = None
    if series is not None:
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        latest = str(pd.to_datetime(series).max().date())
    return {"rows": len(dates), "bytes": p.stat().st_size, "latest": latest}


def file_stats(source_id: str) -> dict | None:
    """Cached {rows, bytes, latest} for a source, or None if never ingested.

    Exists so a panel never has to import lib.store to learn how big a file is.
    Column-projected and cached because app.py's sidebar calls it for every source
    on every rerun.
    """
    return _stats_cached(source_id, _stat_token(source_id))


def _source(source_id: str):
    from sources import REGISTRY

    return REGISTRY.get(source_id)


@st.cache_data(show_spinner=False, max_entries=16)
def _summary_cached(source_id: str, _token: tuple[float, int]) -> pd.DataFrame:
    df = store.read(source_id)
    if df.empty:
        return pd.DataFrame()
    return universe.assign_tiers(universe.market_summary(df))


def market_summary(source_id: str) -> pd.DataFrame:
    """Cached per-market summary with tiers. One row per market code."""
    return _summary_cached(source_id, _stat_token(source_id))
