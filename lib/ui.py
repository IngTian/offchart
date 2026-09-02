"""Shared UI fragments for boards.

THIS MODULE EXISTS BECAUSE `from app import ...` DOES NOT WORK.

Streamlit executes the entry script under the module name `__main__` and does not
also alias it in `sys.modules` as `app`. So a board doing
`from app import caveat_block` does not get the running module -- it IMPORTS A
SECOND COPY of app.py and runs its module body again, sidebar included, which
raises StreamlitDuplicateElementId on the second unkeyed `st.sidebar.radio` and
takes the whole board down before either helper name is even bound.

The fix is that nothing a board needs may live in the entry script. Both helpers
live here, app.py imports them from here too, and there is exactly one copy.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sources import REGISTRY as SOURCES


def caveat_block(*source_ids: str, expanded: bool = False) -> None:
    """Render the caveats that travel with these sources.

    Centralised rather than written per board so a source's limitations cannot be
    shown in one place and forgotten in another. Deduplicates, because the seven
    CFTC sources share most of their caveats and a board may read several.
    """
    seen: set[str] = set()
    with st.expander("What these numbers can and cannot tell you", expanded=expanded):
        for sid in source_ids:
            src = SOURCES.get(sid)
            if src is None:
                continue
            st.caption(f"**{src.label}**")
            st.caption(f"Source. {src.provenance}")
            st.caption(f"Cadence. {src.cadence}")
            if not src.backfillable:
                st.caption(
                    ":warning: Snapshot-only. No history is retrievable; the series "
                    "begins when the job began and missed runs are permanent gaps."
                )
            for c in src.caveats:
                if c in seen:
                    continue
                seen.add(c)
                st.caption(f"- {c}")


def freshness(dates, label: str, *, stale_days: int = 10) -> None:
    """State the age of the data rather than implying it is current.

    Pass the SOURCE's newest report date, not the selected market's. A market
    that simply did not print this week is not a stale dataset -- 52 of 146 tff
    codes skip the latest report, and calling that a stale feed both cries wolf
    and hides a genuinely stopped feed when it happens. Use
    `market_last_print()` for the per-market statement.
    """
    latest = pd.to_datetime(pd.Series(dates)).max()
    if pd.isna(latest):
        return
    age = (pd.Timestamp.today().normalize() - latest.normalize()).days
    msg = f"Latest {label}: **{latest.date()}** ({age} days ago)"
    (st.warning if age > stale_days else st.caption)(msg)


def market_last_print(market_dates, source_dates, market_label: str) -> None:
    """Say when THIS market last printed, relative to the source's latest report.

    Separate from freshness() on purpose: a market being absent from recent
    reports is a fact about the market (it may be illiquid, delisted, or simply
    not reported that week), while the source's age is a fact about the feed.
    Conflating them turns a normal gap into a false alarm about the data.
    """
    m = pd.to_datetime(pd.Series(market_dates)).max()
    s = pd.to_datetime(pd.Series(source_dates)).max()
    if pd.isna(m) or pd.isna(s):
        return
    if m >= s:
        st.caption(f"{market_label} printed on the latest report ({m.date()}).")
        return
    st.caption(
        f"{market_label} last printed **{m.date()}**, while the latest report in "
        f"this dataset is {s.date()}. Markets drop out of individual weeks and "
        "come back; this is not a stale feed."
    )
