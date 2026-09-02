"""Screen -- which markets are at an extreme of their own history right now.

WHAT IT ANSWERS. Pick a report family, a cohort (or a sum of cohorts) and a
ranking statistic, and this ranks every market in the chosen tier by how extreme
its CURRENT reading is against ITS OWN past. It exists because every other board
here needs you to already know which market to open, and "which market" is the
question you actually have on a Friday afternoon.

WHAT IT REFUSES TO ANSWER. Whether an extreme is worth trading. Positioning is
contemporaneous with price, this repo carries no price series, and a
cross-sectional rank has a base rate by construction -- with a 3-year window of
~157 weekly observations, SOME market is in its top decile every single week. The
number here is a level, not an edge; the base-rates board runs the conditional
test.

THREE CHOICES THAT ARE NOT COSMETIC.

1. Every statistic is computed inside the market's LAST SEGMENT (lib.segments,
   keyed on the digits of contract_units), never across its whole file. 20974+
   went from 49,531 to 255,954 contracts of open interest on 2023-05-02 because
   the contract was re-based from $100 to $20 a point. Ranking today against the
   pre-2023 rows would rank $20 observations against $100 ones and report the
   result as a percentile.
2. Markets whose current segment is shorter than universe.MIN_HISTORY reports are
   DROPPED AND COUNTED, not silently omitted. A recently re-based market has a
   perfectly computable percentile of 6 observations and it means nothing.
3. Every share is over open interest MINUS spreads. A spread is long one expiry
   and short another, so it carries no direction; leaving it in the denominator
   understates the tilt by 1/(1 - spread_share), 1.8x in 3-month SOFR.
"""
from __future__ import annotations

import sys

import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from panels import Board

#: Per family, the cohort(s) that make the screen worth opening. tff sums the two
#: speculative cohorts because asset managers and leveraged funds are the two
#: halves of the buy side and either alone is half a book; disagg and legacy have
#: a single speculative cohort each; supp's ex-index non-commercial is the closest
#: thing to legacy noncomm once the index traders are carved out.
DEFAULT_COHORTS = {
    "tff": ("asset_mgr", "lev_money"),
    "disagg": ("m_money",),
    "legacy": ("noncomm",),
    "supp": ("noncomm_nocit",),
}

TIERS = {
    "CORE": ("CORE",),
    "CORE + WIDE": ("CORE", "WIDE"),
    "everything ingested": ("CORE", "WIDE", "THIN"),
}

#: column -> (label, bar unit, charts kind, needs history, what it costs you).
#: `needs_history` is the gate: a raw share is readable off one report, a
#: percentile is not. The z-score's kind is 'purity' because a standardised score
#: is signed and dimensionless and metrics has no kind for one; ranked_bars uses
#: kind ONLY to decide diverging colour and the zero line, so the mismatch is
#: cosmetic here -- but see the library note in the module docstring of lib.metrics.
STATS: dict[str, dict] = {
    "trail_pct": dict(
        label="Trailing 3y percentile", unit="percentile of trailing 3y",
        kind="share", needs_history=True,
        help="Rank of today's level within the trailing 1,095 days of the current "
             "segment. Blank where the window holds under 60% of its usual "
             "occupancy, which is how a gappy market avoids reporting a "
             "percentile of four observations."),
    "exp_pct": dict(
        label="Expanding percentile (whole segment)", unit="percentile of segment",
        kind="share", needs_history=True,
        help="Rank against every observation in the current segment. Longer memory "
             "than the 3y window, and after twenty years that memory is dominated "
             "by regimes that no longer exist."),
    "cot": dict(
        label="COT index (3y min-max)", unit="COT index",
        kind="share", needs_history=True,
        help="100*(v-min)/(max-min) over 3 years -- what published COT commentary "
             "means by 'positioning is at 90', so it is here to check that "
             "commentary. It is a min-max, not a rank: one old print pins the "
             "scale for three years."),
    "z": dict(
        label="Trailing 3y z-score", unit="sigma vs trailing 3y",
        kind="purity", needs_history=True,
        help="Keeps resolution in the tails where a rank saturates at 0 and 100, "
             "and loses meaning exactly where a rank survives: fat tails and "
             "regime shifts make sigma unstable."),
    "share": dict(
        label="Share of directional OI", unit="% of directional OI",
        kind="share", needs_history=False,
        help="Today's level over open interest minus spreads. No history needed, so "
             "no market is dropped -- and no market is compared against its own "
             "past either. This is a size, not an extreme."),
}

#: (column, label, number format or None for text, help). Table-driven because a
#: caption that explains a column belongs next to the column, and eleven inline
#: column_config calls buried the code that decides WHICH columns to show.
TABLE_COLS = (
    ("market_label", "Market", None,
     "Latest published name plus the market_code, which is the stable key. 26-30% "
     "of codes have been renamed and CFTC shortened names wholesale on 2022-02-08, "
     "so the code is what to carry between boards."),
    ("subgroup", "Subgroup", None, "CFTC's own subgroup, or the group where it is null."),
    ("value", "{measure} (contracts)", "localized",
     "Contracts, never a percent change: a net going -100,640 to -96,727 is +3,913 "
     "contracts and LESS short, not +3.89%."),
    ("share", "% of directional OI", "%.2f",
     "Denominator is open interest minus spread positions, which carry no "
     "direction. Spreads are a median 3.2% of OI but 44.5% in 3-month SOFR."),
    ("flow_1w", "1w flow", "localized",
     "Blank where the market skipped a week. CFTC gaps are integer weeks, so a "
     "one-row diff across a hole is a two-week change wearing a one-week label."),
    ("flow_4w", "4w flow", "localized", "Same rule over 4 weeks."),
    ("open_interest", "Open interest", "localized",
     "The denominator. Read it next to the share, always -- a share can move "
     "because the cohort traded or because open interest did."),
    ("spread_pct", "Spreads % of OI", "%.1f",
     "How much of this market is calendar structure rather than direction."),
    ("reports_segment", "Reports", "%d",
     "Reports in the current contract-unit segment. The rank uses only these; "
     "earlier rows may count a different contract."),
    ("last_report", "Last report", None,
     "This market's own latest print, not the corpus latest. Markets skip weeks."),
)


def _last(s: pd.Series) -> float:
    return float(s.iloc[-1]) if len(s) else float("nan")


def _app_helpers():
    """app.py's shared renderers, WITHOUT re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not alias it in
    sys.modules as `app`, so the documented `from app import caveat_block` inside
    render imports a SECOND copy of app.py and runs its module body again --
    sidebar radio included -- which raises StreamlitDuplicateElementId before
    either helper is bound. Verified: this board dies on first render with the
    plain import. Take the already-running module and keep the import as the
    fallback for the case where app.py is not the entry point.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


@st.cache_data(show_spinner="Ranking every market in the family...", max_entries=8)
def _cross_section(source_id: str, cohorts: tuple[str, ...], measure: str) -> pd.DataFrame:
    """One row per live market: the current reading of every ranking statistic.

    Cached on (source, cohorts, measure) and computes ALL the statistics rather
    than the selected one, so flipping the ranking statistic is free. Measured on
    the worst case in the corpus -- legacy_fut, 391 live market codes -- the whole
    pass is ~2s cold and 0 thereafter.

    metrics.last_percentile is used for the expanding case on purpose: building
    the full expanding curve to read its final point is ~1,300x the work, and at
    391 markets that is the difference between a page and a coffee break.
    """
    df = cache.read(source_id)
    summary = cache.market_summary(source_id)
    if df.empty or summary.empty:
        return pd.DataFrame()

    keys = ["market_code", "report_date"]
    # Open interest, units and the concentration columns are IDENTICAL across a
    # market-week's cohort rows, so they come off .first(); summing them would
    # multiply open interest by the cohort count. The spread total is the one
    # thing that must be summed over EVERY cohort even when only one is selected:
    # the directional-OI denominator is a property of the market, not the cohort.
    week = df.groupby(keys, observed=True).agg(
        open_interest=("open_interest", "first"),
        spread_total=("spread", "sum"),
        contract_units=("contract_units", "first"),
    )
    picked = df[df["cohort"].isin(cohorts)].groupby(keys, observed=True)[["long", "short"]].sum()
    frame = week.join(picked, how="inner").reset_index()

    # A market with no print in the last 8 reports has no "this week" to screen.
    # The 8-report window rather than the latest date is deliberate: markets drop
    # out of individual weeks and come back (MICRO E-MINI DJIA carried 63,517
    # contracts three weeks before the latest report and is absent from it), so a
    # latest-date filter would churn the universe by ~5x. The table carries each
    # market's own last report date so a stale reading is visible rather than
    # implied to be current.
    live = set(summary.loc[summary["n_last8"] >= 1, "market_code"])
    frame = frame[frame["market_code"].isin(live)]
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["report_date"])

    rows = []
    for code, g in frame.groupby("market_code", observed=True, sort=False):
        g = g.sort_values("date")
        seg = g[segments.last_segment_mask(g["date"], g["contract_units"]).to_numpy()]
        if seg.empty:
            continue
        d = seg["date"]
        long_, short_ = seg["long"].astype("float64"), seg["short"].astype("float64")
        v = metrics.net(long_, short_) if measure == "net" else metrics.gross(long_, short_)
        oi = seg["open_interest"].astype("float64")
        spread = seg["spread_total"].astype("float64")
        rows.append(
            {
                "market_code": code,
                "value": _last(v),
                "previous": float(v.iloc[-2]) if len(v) > 1 else float("nan"),
                "trail_pct": _last(metrics.trailing_percentile(v, d)),
                "exp_pct": metrics.last_percentile(v),
                "cot": _last(metrics.cot_index(v, d)),
                "z": _last(metrics.trailing_zscore(v, d)),
                "share": _last(metrics.share_of(v, metrics.directional_oi(oi, spread))),
                "flow_1w": _last(metrics.flow(v, d, horizon_weeks=1)),
                "flow_4w": _last(metrics.flow(v, d, horizon_weeks=4)),
                "open_interest": _last(oi),
                "spread_pct": _last(metrics.spread_share(spread, oi)),
                "reports_segment": int(len(seg)),
                "segment_start": d.iloc[0].date().isoformat(),
                "last_report": d.iloc[-1].date().isoformat(),
            }
        )
    if not rows:
        return pd.DataFrame()

    cols = ["market_code", "market", "market_full", "subgroup", "group", "tier",
            "n_reports", "can_rank", "superseded_by"]
    return pd.DataFrame(rows).merge(summary[cols], on="market_code", how="left")


def _hover(rows: pd.DataFrame, measure: str) -> list[str]:
    """Rule 4 and rule 3 in the tooltip: raw contracts, open interest, sample size."""
    out = []
    for r in rows.itertuples():
        direction = (
            charts.signed_direction(r.previous, r.value)
            if measure == "net"
            else ("larger book" if r.value > r.previous else "smaller book")
        )
        out.append(
            f"{r.market_full}<br>"
            f"{measure} {r.value:,.0f} contracts ({direction})<br>"
            f"1-week flow {charts.fmt_change(r.flow_1w, 'flow', 'contracts')}<br>"
            f"open interest {r.open_interest:,.0f} contracts, "
            f"spreads {r.spread_pct:.1f}% of it<br>"
            f"ranked on {r.reports_segment} reports since {r.segment_start}"
        )
    return out


def _table_order(stat_key: str) -> list[str]:
    """Ranking statistic first, then the fixed columns.

    The share is in the table whatever we rank on -- but when it IS what we rank
    on, selecting it twice is a duplicate column, not a second opinion, and
    st.dataframe refuses the frame outright.
    """
    fixed = [c for c, *_ in TABLE_COLS if c != "market_label"]
    if stat_key != "share":
        fixed.insert(0, stat_key)
    return ["market_label"] + fixed


def _column_config(stat_key: str, stat: dict, measure: str) -> dict:
    cfg: dict = {}
    for col, label, fmt, help_ in TABLE_COLS:
        label = label.format(measure=measure.title())
        cfg[col] = (
            st.column_config.TextColumn(label, help=help_)
            if fmt is None
            else st.column_config.NumberColumn(label, format=fmt, help=help_)
        )
    # The ranked column carries the statistic's own caveat, and when it is the
    # share it replaces the generic share header rather than sitting beside it.
    cfg[stat_key] = st.column_config.NumberColumn(
        stat["label"], format="%.2f" if stat_key == "share" else "%.1f",
        help=stat["help"],
    )
    return cfg


def _bars(rows: pd.DataFrame, col: str, kind: str, unit: str, measure: str, key: str) -> None:
    st.plotly_chart(
        charts.ranked_bars(
            # The code travels with the name because the name is not the key: it
            # is the label, and it changes.
            [f"{r.market} ({r.market_code})" for r in rows.itertuples()],
            [float(x) for x in rows[col]],
            unit=unit,
            kind=kind,
            hover=_hover(rows, measure),
        ),
        key=key,
    )


def render() -> None:  # noqa: PLR0915 -- one screen, read top to bottom
    ids = [s.id for s in cftc_spec.SPECS]
    c1, c2 = st.columns([2, 3])
    source_id = c1.selectbox(
        "Report family", ids, index=ids.index("cftc_tff_fut"),
        format_func=lambda i: cftc_spec.spec_for(i).label,
    )
    spec = cftc_spec.spec_for(source_id)
    summary = cache.market_summary(source_id)
    if summary.empty:
        st.info(
            f"`{source_id}` has not been ingested, so this family cannot be "
            "screened. Run `python -m scripts.ingest --source " + source_id + "`. "
            "Every other family in the selector still works."
        )
        return

    labels = {c.id: c.label for c in spec.cohorts}
    cohorts = c2.multiselect(
        "Cohort — summed if you pick more than one",
        list(labels), default=list(DEFAULT_COHORTS[spec.family]),
        format_func=lambda c: labels[c],
    )
    c3, c4, c5, c6 = st.columns([1, 1.2, 2.4, 1.4])
    measure = c3.radio("Measure", ["net", "gross"], horizontal=True,
                       help="Net is direction and changes sign. Gross is book size "
                            "and is what there is to unwind.")
    tier = c4.radio("Universe", list(TIERS), horizontal=False)
    stat_key = c5.selectbox("Ranking statistic", list(STATS),
                            format_func=lambda k: STATS[k]["label"])
    per_side = c6.slider("Markets per side", 5, 30, 12)
    stat = STATS[stat_key]
    st.caption(stat["help"])

    if not cohorts:
        st.info("Pick at least one cohort. Nothing to rank until you do.")
        return

    data = _cross_section(source_id, tuple(sorted(cohorts)), measure)
    if data.empty:
        st.info("No market in this family has a print in the last 8 reports.")
        return

    pool = data[data["tier"].isin(TIERS[tier])].copy()
    if pool["superseded_by"].notna().any():
        # Not silently picked, because it is a real trade-off: the Consolidated
        # code is notionally right and carries the 2023 re-basing, the leg is
        # notionally wrong and has twenty clean years. Leaving both in the same
        # ranking is the one thing that is simply an error -- the leg's exposure
        # is already inside the Consolidated figure.
        if st.checkbox(
            "Drop E-mini / Micro legs that a Consolidated series already contains",
            value=True,
        ):
            pool = pool[pool["superseded_by"].isna()]
        st.caption(universe.CONSOLIDATED_TRADEOFF)

    n_pool = len(pool)
    if stat["needs_history"]:
        short_seg = pool["reports_segment"] < universe.MIN_HISTORY
        rebased = short_seg & pool["can_rank"].fillna(False)
        pool = pool[~short_seg]
        st.caption(
            f"Ranking {len(pool)} of {n_pool} markets in this tier. "
            f"{int(short_seg.sum())} dropped for holding under "
            f"{universe.MIN_HISTORY} reports in their CURRENT contract-unit "
            f"segment; {int(rebased.sum())} of those have enough total history but "
            "were re-based or re-issued, so their older rows count a different "
            "contract and cannot rank against today."
        )
    else:
        st.caption(
            f"Ranking {n_pool} markets. A raw share needs no history, so nothing "
            "is dropped — and nothing is compared against its own past either."
        )

    pool = pool.dropna(subset=[stat_key])
    if pool.empty:
        st.info(
            "Every market in this tier is blank on this statistic. The trailing "
            "window gates on occupancy, so a gappy market reports nothing rather "
            "than a percentile of four observations."
        )
        return

    kind = stat["kind"]
    if stat_key == "share":
        kind = "net_share" if measure == "net" else "gross_share"

    ranked = pool.sort_values(stat_key, ascending=False)
    top, bottom = ranked.head(per_side), ranked.tail(per_side).iloc[::-1]
    hi, lo = (
        ("Most stretched long", "Most stretched short")
        if measure == "net"
        else ("Largest book vs its own history", "Smallest book vs its own history")
    )
    # With fewer markets than 2x per_side the two tails overlap. Say so rather
    # than letting the same name appear on both sides as if it were two findings.
    overlap = max(0, 2 * per_side - len(ranked))
    b1, b2 = st.columns(2)
    with b1:
        st.caption(f"**{hi}** — {stat['label']}, {len(cohorts)} cohort(s) summed")
        _bars(top, stat_key, kind, stat["unit"], measure, "screen_hi")
    with b2:
        st.caption(
            f"**{lo}** — same statistic, other tail"
            + (f", and {overlap} name(s) appear on both because only "
               f"{len(ranked)} markets rank here" if overlap else "")
        )
        _bars(bottom, stat_key, kind, stat["unit"], measure, "screen_lo")

    st.caption(
        "Open interest is on screen as a COLUMN below, because these are bars and "
        "bars cannot share an x-axis with it. You need it: a share moves when the "
        "cohort trades (numerator) or when open interest does (denominator), and "
        "the share alone cannot say which. Read the open-interest and 1-week-flow "
        "columns together before calling a rank a position change."
    )

    show = ranked.copy()
    show["market_label"] = show["market"] + " (" + show["market_code"] + ")"
    show["subgroup"] = show["subgroup"].fillna(show["group"]).fillna("(ungrouped)")
    st.dataframe(
        show[_table_order(stat_key)], hide_index=True, height=420,
        column_config=_column_config(stat_key, stat, measure),
    )
    st.caption(
        "Summing cohorts adds their long and short columns before netting, so an "
        "asset manager long offset by a leveraged fund short cancels. That is the "
        "point when you want the speculative side as one book, and a distortion if "
        "you wanted either one on its own. Cohort set: "
        f"{', '.join(labels[c] for c in cohorts)}."
    )
    st.caption(
        "**Base rate first.** A cross-sectional extreme is guaranteed to exist: with "
        "~157 weekly observations in a 3-year window, some market prints in its top "
        "decile every week whether or not anything happened. Positioning is also "
        "contemporaneous with price rather than predictive, and this repo holds no "
        "price series, so no forward return is computable here at all. Take a name "
        "from this screen to the **Base rates** board before treating a rank as "
        "information."
    )
    # Resolved here rather than at the top of render so the un-ingested-family
    # branch above returns without touching app.py at all.
    caveat_block, freshness = _app_helpers()
    freshness(data["last_report"], f"{spec.label} print in this universe")
    caveat_block(source_id)


BOARD = Board(
    id="screen",
    title="Screen",
    render=render,
    sources=("cftc_tff_fut",),
    order=10,
    blurb="What is unusual this week, across every market at once.",
)
