"""Watchboard -- government and alternative data that TradingView does not carry
or does not display usefully.

    streamlit run app.py

Reads only from data/*.parquet. It never fetches, so it works offline and it is
impossible for a chart to show a number that is not in the committed dataset.
Ingest is a separate job (scripts/ingest.py, run by CI).

This file is deliberately almost empty. Boards live in panels/ and are discovered,
sources live in sources/ and are discovered, so adding either one touches no
dispatch table here. The four display rules are enforced in lib/charts.py and
lib/metrics.py rather than restated in each board.
"""
from __future__ import annotations

import streamlit as st

import panels
from lib import cache, store
from lib.ui import caveat_block, freshness  # re-exported for convenience; see lib/ui
from sources import IMPORT_ERRORS as SOURCE_IMPORT_ERRORS
from sources import REGISTRY as SOURCES
from sources import all_sources

# NOTE FOR BOARDS: import these from lib.ui, NOT from app. Streamlit runs this
# file as `__main__` and does not alias it as `app`, so `from app import ...`
# executes a second copy of this module -- sidebar and all -- and dies on a
# duplicate widget id. lib/ui.py exists precisely to give boards a safe home.
__all__ = ["caveat_block", "freshness"]

st.set_page_config(page_title="Watchboard", page_icon="◧", layout="wide")

BOARDS = panels.all_boards()


def _broken_module_warnings() -> None:
    """A module that failed to import is shown, never silently dropped.

    A board that vanishes looks like a design decision. A board that says it is
    broken gets fixed.
    """
    for label, errors in (("source", SOURCE_IMPORT_ERRORS), ("board", panels.IMPORT_ERRORS)):
        for name, tb in errors.items():
            st.error(f"The {label} module `{name}` failed to import, so it is missing from this board.")
            with st.expander(f"traceback for {name}"):
                st.code(tb, language="text")


def _sidebar() -> "panels.Board | None":
    st.sidebar.title("◧ Watchboard")
    st.sidebar.caption(
        "Government and alternative data that TradingView does not carry or does "
        "not display usefully. Numbers you cannot reproduce yourself can raise a "
        "question; they should not answer one."
    )

    if not BOARDS:
        return None

    groups: dict[str, list[panels.Board]] = {}
    for b in BOARDS:
        groups.setdefault(b.group, []).append(b)

    labels: list[str] = []
    lookup: dict[str, panels.Board] = {}
    for group in sorted(groups, key=lambda g: min(b.order for b in groups[g])):
        for b in groups[group]:
            label = f"{group} · {b.title}" if len(groups) > 1 else b.title
            labels.append(label)
            lookup[label] = b

    chosen = st.sidebar.radio("Board", labels, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.caption("**Datasets ingested**")
    # Via lib.cache, not store: this runs on every widget interaction and an
    # uncached read here would re-parse all seven parquet files on every click.
    for s in all_sources():
        stats = cache.file_stats(s.id)
        if stats is None:
            st.sidebar.caption(f"`{s.id}` — not yet ingested")
            continue
        flag = "" if s.backfillable else "  ·  snapshot-only"
        st.sidebar.caption(
            f"`{s.id}` — {stats['rows']:,} rows, {stats['bytes'] / 1e6:.1f} MB{flag}"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Ingest runs daily in CI and commits `data/`. The board never fetches, so "
        "what you see is exactly what is committed."
    )
    return lookup[chosen]


_broken_module_warnings()
board = _sidebar()

if board is None:
    st.title("◧ Watchboard")
    st.error("No boards were discovered in panels/. Nothing to show.")
else:
    # Cached and column-projected: this runs on every rerun, so "is there data"
    # must not cost a full read of a 29 MB file.
    missing = [
        sid
        for sid in board.sources
        if (cache.file_stats(sid) or {}).get("rows", 0) == 0
    ]
    st.title(board.title)
    if board.blurb:
        st.caption(board.blurb)
    if missing:
        st.info(
            "This board needs data that has not been ingested yet: "
            + ", ".join(f"`{m}`" for m in missing)
            + ".\n\nRun `python -m scripts.ingest --backfill` (first run pulls full "
            "history and takes a few minutes)."
        )
    else:
        board.render()
