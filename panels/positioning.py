"""The two charts this repo was built to reproduce.

Both are the same shape: one cohort's net position in one market, as a share of
that market's open interest, over its whole history, with the current reading and
where that reading sits against its own past.

They reproduce to the digit off the committed data:

    VIX          asset managers            -6.04% of OI,  4.0th pctile, n=1,014
    NASDAQ-100   asset mgr + leveraged     +9.28% of OI, 40.2nd pctile, n=846,
                                           2-week change +20.58pp

WHY THIS MEASURE NEEDS NO SEGMENTING, WHICH IS NOT OBVIOUS

CFTC re-specified the Consolidated equity index contracts on 2023-05-02 and
NASDAQ-100 open interest jumped 49,531 -> 255,954 with no change in positioning.
That break wrecks any chart of contract COUNTS, and elsewhere in this repo history
is cut at it for exactly that reason.

It does not touch these charts. Net-as-a-share-of-open-interest is dimensionless:
the re-denomination multiplies the numerator and the denominator by the same
factor, so the ratio is unchanged. Measured, the percentile moves 40.2 -> 40.8
when computed within the post-break segment only, and the entire difference is the
smaller sample (846 reports -> 174), not a scale seam. So the full history is used
here, which is also what makes the figures match the published ones.

WHAT IS DELIBERATELY ABSENT

No controls. Every widget in a Streamlit app costs a full server rerun on click,
which is what makes a board feel like a slide advancing. The two charts are fixed,
the zoom presets and the crosshair are handled by the browser, and the page
therefore never reruns while you read it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import cache, charts, metrics
from lib.ui import caveat_block, freshness
from panels import Board

SOURCE = "cftc_tff_fut"

#: The two charts, and nothing else. Keyed on market_code, never on the market
#: name -- CFTC renamed 26-30% of its codes at least once and shortened names
#: wholesale on 2022-02-08, so a name-keyed series silently splits in two there.
CHARTS = (
    {
        "code": "1170E1",
        "cohorts": ("asset_mgr",),
        "market": "VIX futures",
        "who": "Asset managers",
        "reads": (
            "Asset managers are structurally short VIX -- selling volatility is a "
            "carry trade -- so the level is almost always negative and what matters "
            "is where it sits against its own history, not its sign."
        ),
    },
    {
        "code": "20974+",
        "cohorts": ("asset_mgr", "lev_money"),
        "market": "NASDAQ-100",
        "who": "Asset managers + leveraged funds",
        "reads": (
            "The two cohorts are summed because that is how the circulating charts "
            "build it: institutional money and hedge funds together, against the "
            "dealers who take the other side. Consolidated contract, so E-mini and "
            "Micro exposure are already combined at their true 10:1 weight."
        ),
    },
)

_PCTILE_BANDS = (10.0, 90.0)


@st.cache_data(show_spinner=False, max_entries=8)
def _series(code: str, cohorts: tuple[str, ...]) -> pd.DataFrame:
    """Net share of open interest for one market, plus everything the card shows.

    Cached on (code, cohorts) so the page costs one parquet read per session
    rather than one per interaction.
    """
    df = cache.read(SOURCE)
    sub = df[df["market_code"].astype(str) == code]
    if sub.empty:
        return pd.DataFrame()

    sub = sub.copy()
    sub["report_date"] = pd.to_datetime(sub["report_date"])
    picked = sub[sub["cohort"].isin(cohorts)]

    by_date = picked.groupby("report_date")
    net = (by_date["long"].sum() - by_date["short"].sum()).astype(float)
    # .first(), not .sum(): open interest is a market-week quantity repeated on
    # every cohort row, so summing it multiplies by the cohort count.
    week = sub.groupby("report_date")
    oi = week["open_interest"].first().astype(float)

    out = pd.DataFrame({"net": net, "open_interest": oi}).dropna(subset=["open_interest"])
    out["share"] = metrics.share_of(out["net"], out["open_interest"])
    out["pctile"] = metrics.expanding_percentile(out["share"])
    dates = pd.Series(out.index)
    out["chg_2w"] = metrics.flow(
        out["share"].reset_index(drop=True), dates, horizon_weeks=2
    ).to_numpy()
    out["market"] = week["market"].first()
    return out


def _hero(frame: pd.DataFrame, who: str) -> None:
    """The reading, as a hero figure rather than a row of widgets.

    The story here is one number, and the anti-pattern for that is a chart of
    eight colours. Proportional figures, not tabular-nums: equal-width digits make
    a large standalone number look loose.
    """
    last = frame.iloc[-1]
    share, pctile, chg = last["share"], last["pctile"], last["chg_2w"]
    side = "net long" if share > 0 else "net short" if share < 0 else "flat"
    n = int(frame["share"].notna().sum())

    delta = (
        f"{chg:+.2f}pp over 2 weeks" if pd.notna(chg) else "no comparable week two back"
    )
    rank_note = (
        "lower than all but a handful of weeks on record"
        if pctile <= _PCTILE_BANDS[0]
        else "higher than all but a handful of weeks on record"
        if pctile >= _PCTILE_BANDS[1]
        else "unremarkable against its own history"
    )

    st.markdown(
        f"""<div class="hero">
              <div class="hero-figure">{share:+.2f}<span class="hero-unit">% of OI</span></div>
              <div class="hero-side">{who} are <strong>{side}</strong></div>
            </div>
            <div class="hero-row">
              <div><span class="k">{pctile:.0f}<span class="k-unit">th</span></span>
                   <span class="v">percentile of its own {n:,} weeks &mdash; {rank_note}</span></div>
              <div><span class="k-sm">{delta}</span></div>
            </div>""",
        unsafe_allow_html=True,
    )


def _card(spec: dict) -> None:
    frame = _series(spec["code"], spec["cohorts"])
    if frame.empty:
        st.info(f"{spec['market']} ({spec['code']}) is not in the ingested data yet.")
        return

    name = str(frame["market"].iloc[-1])
    st.markdown(
        f'<div class="card-head"><h2>{spec["market"]}</h2>'
        f'<div class="card-sub">{spec["who"]}, net position as a share of open '
        f"interest &nbsp;·&nbsp; {name} ({spec['code']})</div></div>",
        unsafe_allow_html=True,
    )
    _hero(frame, spec["who"])

    st.plotly_chart(
        charts.spotlight(
            frame["share"],
            frame["open_interest"],
            unit="% of open interest",
            kind="net_share",
            # No guide lines: the 10th/90th bands are percentile levels and this
            # panel is in percent-of-OI, so drawing them here would put a
            # reference line at a level it does not refer to.
        ),
        width="stretch",
        config=charts.PLOTLY_CONFIG,
    )

    st.caption(spec["reads"])
    # A tooltip must never be the only way to read a value, so every chart keeps a
    # table twin. Collapsed, because it is a fallback and not the point.
    with st.expander("The last 12 weeks, as numbers"):
        tail = frame.tail(12)[["net", "open_interest", "share", "pctile"]].copy()
        tail.index = [d.date() for d in tail.index]
        st.dataframe(
            tail.rename(
                columns={
                    "net": "net (contracts)",
                    "open_interest": "open interest",
                    "share": "% of OI",
                    "pctile": "percentile",
                }
            ).style.format(
                {
                    "net (contracts)": "{:,.0f}",
                    "open interest": "{:,.0f}",
                    "% of OI": "{:+.2f}",
                    "percentile": "{:.1f}",
                }
            ),
            width="stretch",
        )


def render() -> None:
    frames = [_series(c["code"], c["cohorts"]) for c in CHARTS]
    dated = [f for f in frames if not f.empty]
    if dated:
        freshness(
            pd.Series([f.index.max() for f in dated]),
            "CFTC report (Tuesday positions, published Friday 15:30 ET)",
        )

    for i, spec in enumerate(CHARTS):
        if i:
            st.markdown('<div class="card-gap"></div>', unsafe_allow_html=True)
        _card(spec)

    caveat_block(SOURCE)


BOARD = Board(
    id="positioning",
    title="Positioning",
    render=render,
    sources=(SOURCE,),
    order=10,
    blurb="Where the big cohorts are leaning, and how unusual that is.",
    group="CFTC",
)
