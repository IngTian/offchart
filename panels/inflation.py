"""What households expect prices to do, and how they feel about the economy.

Two panels, because there are two units and a shared axis would let the scaling
choice manufacture a relationship:

    expected price change, % per year   year-ahead and 5-to-10-year medians
    index, 1966 Q1 = 100               sentiment, optionally its two halves

RANKS ARE NOT COMPUTED ACROSS THE 2024 MODE CHANGE, and that is the one thing on
this board that is easy to get wrong. UMich moved from cell-phone to web sampling
over April-July 2024, measured the level shift it caused at -6.6 index points, and
chose not to adjust the series -- so a chart of the LEVEL has no visible break while
a percentile of the level is meaningless across it. The pools live in
lib/umich_spec, one per column with the citation that justifies it, and the bubble
is suppressed entirely while a pool is shorter than three years. As of the August
2026 reading the web era is 26 months, so the sentiment panel shows NO rank at all.
That is the honest state of the world, not a missing feature.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import cache, charts, metrics, overlay, store, theme, umich_spec
from lib.ui import caveat_block
from panels import Board

SOURCE = "umich_sca"

#: The three index series, and whether each is on by default. Sentiment alone is
#: the default because it is the headline; its two halves are a decomposition, and
#: three lines in one panel is where a chart starts being read twice.
INDEX_SERIES = (
    ("sentiment", "Sentiment", True),
    ("current_conditions", "Current conditions", False),
    ("expectations", "Expectations", False),
)

PRICE_SERIES = (
    ("infl_exp_1y", "Next 12 months"),
    ("infl_exp_5y10y", "Next 5-10 years, per year"),
)


@st.cache_data(ttl=900, show_spinner=False)
def _load() -> pd.DataFrame:
    """The monthly era only, and that is a charting decision, not a data one.

    The parquet holds every month back to February 1951 and keeps it. But the survey
    ran quarterly or sparser until January 1978 -- counted from the file: one reading
    in 1952, two or three a year through 1959, exactly four a year 1960-1977 -- and
    the inflation questions do not exist at all before 1978. Charting from 1951 spends
    a third of the width on one sparse line above an empty panel, and puts quarterly
    and monthly observations on the same axis as though they were the same thing.
    """
    frame = store.read(SOURCE)
    if frame.empty:
        return frame
    frame = frame[frame["survey_date"] >= umich_spec.MONTHLY_FROM]
    frame = frame.sort_values("survey_date").reset_index(drop=True)
    frame["when"] = pd.to_datetime(frame["survey_date"])
    return frame


def _ranked(frame: pd.DataFrame, column: str) -> pd.Series | None:
    """A causal percentile of `column`, or None when it must not be shown.

    Two ways to get None, and both are deliberate:
      - the pool is shorter than lib.umich_spec.MIN_RANK_OBSERVATIONS, so a rank
        would carry a resolution of several percentile points per observation while
        looking exactly as authoritative as one computed off five hundred months;
      - the column has no regime at all, which would mean an unreviewed series.
    """
    if column not in umich_spec.BY_COLUMN:
        return None
    if umich_spec.rankable(frame, column) < umich_spec.MIN_RANK_OBSERVATIONS:
        return None
    pool = umich_spec.rank_pool(frame, column)
    # Ranked over the pool alone, then put back on the full index so the bubble
    # lands on the last observation. expanding_percentile is causal: each month is
    # ranked against the months up to and including itself, never the whole sample.
    ranked = metrics.expanding_percentile(pool.dropna())
    return ranked.reindex(frame.index)


def _hero(frame: pd.DataFrame) -> None:
    last = frame.iloc[-1]
    one_year = last["infl_exp_1y"]
    long_run = last["infl_exp_5y10y"]
    sentiment = last["sentiment"]

    rank = _ranked(frame, "infl_exp_1y")
    if rank is not None and pd.notna(rank.iloc[-1]):
        n = umich_spec.rankable(frame, "infl_exp_1y")
        num, suffix = charts.ordinal_parts(rank.iloc[-1])
        rank_html = (
            f'<div><span class="k">{num}<span class="k-unit">{suffix}</span></span>'
            f'<span class="v">percentile of {n:,} '
            f'{umich_spec.BY_COLUMN["infl_exp_1y"].pool_label}</span></div>'
        )
    else:
        rank_html = ""

    st.markdown(
        f"""<div class="hero">
              <div class="hero-figure">{one_year:.1f}<span class="hero-unit">% expected
                over the next year</span></div>
              <div class="hero-side">long run <strong>{long_run:.1f}%</strong>
                &middot; sentiment <strong>{sentiment:.1f}</strong></div>
            </div>
            <div class="hero-row">{rank_html}</div>""",
        unsafe_allow_html=True,
    )


def render() -> None:
    frame = _load()
    if frame.empty:
        st.info(
            f"`{SOURCE}` has no rows yet. This source is **not committed** -- the "
            "University of Michigan permits use of its public tables but not "
            "redistribution, so the parquet is gitignored and a fresh clone has to "
            "fetch it. Run `python -m scripts.ingest --source umich_sca`."
        )
        return

    st.markdown(
        '<div class="card-head"><h2>Inflation expectations &amp; the consumer</h2>'
        '<div class="card-sub">University of Michigan Surveys of Consumers &middot; '
        "monthly, final readings &middot; medians of expected change in prices in "
        "general</div></div>",
        unsafe_allow_html=True,
    )

    _stamp(frame)
    _hero(frame)

    chosen = _controls()
    fig = charts.tracks(
        [
            charts.Track(
                lines=[
                    charts.Line(
                        values=frame.set_index("when")[column].dropna(),
                        label=label,
                        decimals=1,
                        suffix="%",
                        pctile=_reindexed(_ranked(frame, column), frame, column),
                    )
                    for column, label in PRICE_SERIES
                ],
                unit="% per year, expected",
                weight=1.0,
                # Expected inflation is bounded below by zero in practice but the
                # question allows a decline, so zero is a real reference rather
                # than the edge of the scale.
                guides=(2.0,),
            ),
            charts.Track(
                lines=[
                    charts.Line(
                        values=frame.set_index("when")[column].dropna(),
                        label=label,
                        decimals=1,
                        pctile=_reindexed(_ranked(frame, column), frame, column),
                    )
                    for column, label, _ in INDEX_SERIES
                    if column in chosen
                ],
                unit="index, 1966 Q1 = 100",
                weight=1.0,
            ),
        ],
        height=620,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config=charts.png_config("watchboard-inflation"),
        key="chart-inflation",
    )
    # Slot order follows the declaration order in tracks(): the two price series
    # first, then whichever index series are switched on.
    rows = [
        overlay.Row(label, slot, 1, suffix="%")
        for slot, (_, label) in enumerate(PRICE_SERIES)
    ]
    rows += [
        overlay.Row(label, len(PRICE_SERIES) + i, 1)
        for i, (column, label, _) in enumerate(
            [s for s in INDEX_SERIES if s[0] in chosen]
        )
    ]
    overlay.crosshair("chart-inflation", rows)

    st.caption(
        "The 2 % dotted line is the Federal Reserve's stated target, for reference "
        "only -- the survey asks about prices in general, not about the price index "
        "the target is written against."
    )
    _rank_note(frame)
    caveat_block(SOURCE)


def _reindexed(rank: pd.Series | None, frame: pd.DataFrame, column: str) -> pd.Series | None:
    """Put a rank computed on the frame's integer index onto the chart's dates.

    The chart drops NaNs per line, so its index is the series' own live dates; a
    rank still keyed by row number would silently misalign and the bubble would sit
    on the wrong month.
    """
    if rank is None:
        return None
    live = frame[column].notna()
    return pd.Series(rank[live].to_numpy(), index=frame.loc[live, "when"])


def _controls() -> set[str]:
    with st.expander("Series", expanded=False):
        cols = st.columns(len(INDEX_SERIES))
        chosen = set()
        for col, (column, label, default) in zip(cols, INDEX_SERIES):
            if col.checkbox(label, value=default, key=f"umich-{column}"):
                chosen.add(column)
    return chosen or {"sentiment"}


def _stamp(frame: pd.DataFrame) -> None:
    last = frame["when"].iloc[-1]
    stats = cache.file_stats(SOURCE) or {}
    st.markdown(
        f'<div class="stamp">Latest survey month <strong>'
        f"{last:%B %Y}</strong> &middot; {stats.get('rows', 0):,} months on disk, "
        f"charted from {pd.Timestamp(umich_spec.MONTHLY_FROM):%Y} "
        "&middot; final readings only</div>",
        unsafe_allow_html=True,
    )


def _rank_note(frame: pd.DataFrame) -> None:
    """Say, per series, what its rank is computed against -- or why there is none."""
    lines = []
    for column, label, _ in INDEX_SERIES:
        n = umich_spec.rankable(frame, column)
        regime = umich_spec.BY_COLUMN[column]
        if n < umich_spec.MIN_RANK_OBSERVATIONS:
            lines.append(
                f"**{label}** — no percentile shown. Its comparable history is "
                f"{n} {regime.pool_label} and this board will not rank fewer than "
                f"{umich_spec.MIN_RANK_OBSERVATIONS}. {regime.reason}"
            )
        else:
            lines.append(f"**{label}** — ranked against {n:,} {regime.pool_label}.")
    for column, label in PRICE_SERIES:
        n = umich_spec.rankable(frame, column)
        regime = umich_spec.BY_COLUMN[column]
        lines.append(f"**{label}** — ranked against {n:,} {regime.pool_label}.")

    with st.expander("What each percentile is measured against", expanded=False):
        for line in lines:
            st.markdown(line)
        st.caption(
            "Ranks are causal: each month is ranked only against the months up to "
            "and including itself, so nothing on this chart uses information that "
            "was not available at the time."
        )


BOARD = Board(
    id="inflation",
    title="Inflation expectations",
    render=render,
    sources=(SOURCE,),
    order=20,
    blurb="What households expect prices to do, and how they feel about it.",
    group="Inflation & the consumer",
)
