"""Crowding: how few hands hold the position, and how much of the market is
calendar structure rather than direction.

A cohort net is one number summed over every firm in the cohort, and it hides the
three things this board is about -- all three published by CFTC, none of them
plotted by an earlier version of this repo:

    concentration   CR4/CR8, gross and net, per side. A 100,000-contract net is a
                    different animal when the top four longs are 60% of the long
                    side than when they are 12%: same net, different unwind.
    trader counts   how many reporting firms stand behind that net.
    spread share    how much of open interest expresses no direction at all.

THE DENOMINATOR IS THE THING TO GET RIGHT HERE. CR4/CR8 are shares of the SIDE
TOTAL -- open interest MINUS spreads -- not of open interest, and lib/cftc_spec
.CONC_FIELDS carries the two measured disproofs of the open-interest reading: 158
tff rows report exactly 100.0 with spread > 0, impossible against a denominator
that caps them at 25.2% there, and CR4/100 * OI exceeds the entire long side in
1,623 of 46,361 market-weeks while the side-total reading never does. Reading it
against open interest OVERSTATES the contracts those four hold by 1/(1 - spread
share) -- 1.8x in SOFR-3M, the default market here. So the sentence on screen is
"the four largest longs hold X% of all long positions", with directional open
interest beside it so X converts back into contracts.

WHAT THIS BOARD REFUSES TO ANSWER. Whether crowding predicts anything: this repo
has no price series, so no forward return exists to test against, and positioning
is contemporaneous with price rather than ahead of it. And who the four are: CR is
anonymous, and "four" is four accounts as CFTC classifies them, which may be four
desks of one firm.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from lib.ui import caveat_block, freshness, market_last_print
from panels import Board

SOURCE = "cftc_tff_fut"

#: All eight published concentration columns: {gross, net} x {4, 8} x {long, short}.
CONC = tuple(f"conc_{k}_{n}_{s}" for k in ("gross", "net") for s in ("long", "short") for n in (4, 8))

#: The percentile's trailing window and the occupancy it must reach, passed
#: EXPLICITLY to metrics.trailing_percentile rather than left to its defaults so the
#: caption can state the numbers the gate used instead of restating constants that
#: drift from it. RANK_FULL is how many weekly reports fit in (t - 1095d, t] and
#: RANK_FLOOR the minimum before a percentile publishes: 157 and 94, the same floor
#: metrics derives from the cadence rather than from the sample (_causal_floor).
RANK_WINDOW_DAYS = 1095
RANK_MIN_FRAC = 0.6
RANK_FULL = int(RANK_WINDOW_DAYS // metrics.DAYS_PER_REPORT) + 1
RANK_FLOOR = math.ceil(RANK_MIN_FRAC * RANK_WINDOW_DAYS / metrics.DAYS_PER_REPORT)


def _f(s: pd.Series) -> pd.Series:
    """Nullable Int32 -> float64, NA -> NaN. `.astype(float)` raises on pd.NA."""
    return pd.Series(s.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def _fmt(x: float, suffix: str = "%", places: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{x:,.{places}f}{suffix}"


def _wow(values: pd.Series) -> str | None:
    """Latest ONE-WEEK change in percentage points, or None if the span is not a week.

    metrics.flow rather than .diff(1): CFTC skips weeks, so the previous ROW is not
    always the previous WEEK, and a two-week move wearing a weekly label is an error
    that survives review because the number looks fine.
    """
    d = metrics.flow(values, pd.Series(values.index, index=values.index)).iloc[-1]
    return None if pd.isna(d) else charts.fmt_change(d, "share", "pp")


def _p(rows: list, unit: str, title: str, breaks: tuple, height: float = 1.0, **kw) -> charts.Panel:
    """Panel from (values, label, kind, colour, dash) rows. kind drives rules 3 and 4."""
    return charts.Panel(
        series=[charts.Series(v, lbl, kind=k, color=c, dash=d) for v, lbl, k, c, d in rows],
        unit=unit, title=title, height=height, breaks=breaks, **kw,
    )


def _show(panels: list, height_per_panel: int = 205) -> None:
    st.plotly_chart(charts.stacked(panels, height_per_panel=height_per_panel), width="stretch")


def _conc_rows(wk: pd.DataFrame, kind: str) -> list:
    # Long is one colour and short the other; CR8 is the dashed member of each pair.
    # Four lines on a panel are only readable if the encoding is a rule.
    return [
        (wk[f"conc_{kind}_{n}_{side}"], f"Top {n}{' net' if kind == 'net' else ''} {side}", "share",
         charts.ACCENT if side == "long" else charts.WARN, "dash" if n == 8 else None)
        for side in ("long", "short") for n in (4, 8)
    ]


def _oi_rows(wk: pd.DataFrame) -> list:
    # Rule 4, and here also the arithmetic: every share on this board has one of these
    # two as its denominator, and a share that moved because the denominator moved
    # looks identical to one that moved because the numerator did.
    return [
        (wk["open_interest"], "Open interest", "open_interest", charts.MUTED, None),
        (wk["dir_oi"], "Directional OI (ex-spreads)", "open_interest", charts.POSITIVE, None),
    ]


def _weekly(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """One row per report date for one market, indexed by date.

    Selected on market_code, never on name: 26-30% of codes have been renamed and CFTC
    shortened names wholesale on 2022-02-08, so a name filter truncates history
    silently. Concentration, open interest and units are market-week columns, identical
    across a week's cohorts, so they take .first(); spread is genuinely per-cohort and
    sums. THE PER-COHORT TRADER COUNTS ARE NOT MARKET-WEEK COLUMNS -- they differ across
    cohorts in 94.0% of the 46,361 tff market-weeks -- so they are deliberately absent
    here, where .first() would be a silent choice of cohort. Legacy publishes
    concentration above 100% on 23 rows (worst 482.6% on 957 contracts) and tff has
    none, but the clamp runs here rather than being trusted to the family: a broken
    input should read as a gap, not as a spike.
    """
    mk = df.loc[df["market_code"] == code].copy()
    mk["report_date"] = pd.to_datetime(mk["report_date"])
    firsts = {c: (c, "first") for c in CONC + ("open_interest", "contract_units")}
    wk = mk.groupby("report_date").agg(spread_total=("spread", "sum"), **firsts).sort_index()
    for c in CONC:
        wk[c] = metrics.clamp_percentage(wk[c])
    wk["open_interest"] = _f(wk["open_interest"])
    wk["spread_total"] = _f(wk["spread_total"])
    wk["dir_oi"] = metrics.directional_oi(wk["open_interest"], wk["spread_total"])
    wk["spread_pct"] = metrics.spread_share(wk["spread_total"], wk["open_interest"])
    wk["segment"] = segments.segment_ids(pd.Series(wk.index, index=wk.index), wk["contract_units"])
    return wk


def _rank_within_segments(values: pd.Series, segment: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Trailing 3-year percentile, plus how many reports each point ranked against.

    Causal twice over: the window is (t - 3y, t], and it never reaches back across a
    re-basing. Ranking $20-per-point observations against $100-per-point ones is what
    this repo used to do, and it is not a percentile of anything.

    The count comes back with the ranks because "3-year percentile" names the WINDOW,
    not the sample: 342603's current segment fills 111 of 157 slots and still publishes,
    since the gate is RANK_FLOOR readings rather than a full window. No short-segment
    guard here on purpose -- the gate is metrics' cadence-derived floor, which is why
    the 58-report crypto segments now read n/a instead of 90 and 100.
    """
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    n_obs = pd.Series(np.nan, index=values.index, dtype="float64")
    dates = pd.Series(values.index, index=values.index)
    for sid in segment.unique():
        m = segment == sid
        out.loc[m] = metrics.trailing_percentile(
            values.loc[m], dates.loc[m],
            window_days=RANK_WINDOW_DAYS, min_obs_frac=RANK_MIN_FRAC,
        )
        by_date = pd.Series(values.loc[m].to_numpy(), index=pd.DatetimeIndex(dates.loc[m].to_numpy()))
        n_obs.loc[m] = by_date.rolling(f"{RANK_WINDOW_DAYS}D").count().to_numpy()
    return out, n_obs


def _concentration(wk: pd.DataFrame, breaks: tuple) -> None:
    st.subheader("Concentration")
    cur = wk.iloc[-1]
    rank, n_ranked = _rank_within_segments(wk["conc_gross_8_long"], wk["segment"])
    # The two readings of the same published percentage, side by side, because the
    # error is a factor of 1.8 on the default market and invisible without both.
    held_by_four = cur["conc_gross_4_long"] / 100.0 * cur["dir_oi"]
    if_oi_denominator = cur["conc_gross_4_long"] / 100.0 * cur["open_interest"]
    overstatement = cur["open_interest"] / cur["dir_oi"] if cur["dir_oi"] > 0 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 4 longs, share of the long side", _fmt(cur["conc_gross_4_long"]),
              _wow(wk["conc_gross_4_long"]), delta_color="off")
    c2.metric("= contracts held by those four", _fmt(held_by_four, "", 0))
    c3.metric("Top 8 longs, percentile in its trailing 3y window", _fmt(rank.iloc[-1], "", 0),
              f"ranked against {_fmt(n_ranked.iloc[-1], '', 0)} of {RANK_FULL} reports",
              delta_color="off")
    c4.metric("Spreads, share of open interest", _fmt(cur["spread_pct"]),
              _wow(wk["spread_pct"]), delta_color="off")
    st.caption(
        f"Read as \"the four largest longs hold {_fmt(cur['conc_gross_4_long'])} of all long "
        f"positions\": CR is a share of the SIDE TOTAL, not of open interest. On "
        f"{wk.index[-1].date()} that side total is {_fmt(cur['dir_oi'], '', 0)} contracts -- open "
        f"interest {_fmt(cur['open_interest'], '', 0)} less {_fmt(cur['spread_total'], '', 0)} of "
        f"spreads -- so multiplying that percentage by open interest instead would claim "
        f"{_fmt(if_oi_denominator, '', 0)} contracts where the four actually hold "
        f"{_fmt(held_by_four, '', 0)}: an OVERSTATEMENT of {_fmt(overstatement, 'x', 2)}, which is "
        f"1/(1 - spread share). Deltas need the previous report exactly one week back, and are "
        f"uncoloured because neither direction is good news."
    )
    _show([
        _p(_conc_rows(wk, "gross"), "% of side total",
           "Gross concentration: largest holders as a share of their own side", breaks),
        _p(_conc_rows(wk, "net"), "% of side total",
           "Net concentration: the same accounts after their own offsets", breaks, 0.9),
        _p(_oi_rows(wk), "contracts",
           "The denominator, so a share converts back to contracts", breaks, 0.9),
        _p([(rank, "Top 8 long", "share", charts.ACCENT, None)], "percentile",
           "Concentrated for THIS market? Top 8 long against its own last 3 years",
           breaks, 0.7, guides=(10.0, 90.0), y_range=(0.0, 100.0)),
    ])
    st.caption(
        "Net concentration nets each account against itself first, so it sits at or below gross; "
        "a wide gross-net gap is a top holder running both sides rather than taking a view. "
        "Neither level compares across markets -- 30% is crowded in Treasuries and loose in a "
        f"thin crypto contract -- which is what the percentile is for. It publishes only once the "
        f"trailing {RANK_WINDOW_DAYS}-day window holds {RANK_FLOOR} readings ({RANK_MIN_FRAC:.0%} "
        f"of the {RANK_FULL} a weekly cadence implies), so it stays blank until that many have "
        f"accumulated after a segment break and never appears on a shorter segment at all. "
        f"\"3-year\" names the WINDOW, not the sample: the point above ranks against "
        f"{_fmt(n_ranked.iloc[-1], '', 0)} readings, which is the whole basis it has."
    )


def _traders(df: pd.DataFrame, code: str, wk: pd.DataFrame, breaks: tuple) -> None:
    st.subheader("Reporting firms behind the position")
    labels = {c.id: c.label for c in cftc_spec.spec_for(SOURCE).cohorts if c.traders_long}
    # Spec order, with leveraged funds as the DEFAULT: the cohort whose crowding is
    # actually asked about, and the one where a handful of firms holding the whole net
    # is common.
    order = list(labels)
    cohort = st.selectbox("Cohort", order, format_func=lambda c: labels[c], key="crowd_cohort",
                          index=order.index("lev_money") if "lev_money" in labels else 0)

    g = df.loc[(df["market_code"] == code) & (df["cohort"] == cohort)].copy()
    g["report_date"] = pd.to_datetime(g["report_date"])
    g = g.sort_values("report_date").set_index("report_date")
    for col in ("long", "short", "traders_long", "traders_short"):
        g[col] = _f(g[col])
    per_long = metrics.avg_position_per_trader(g["long"], g["traders_long"])
    per_short = metrics.avg_position_per_trader(g["short"], g["traders_short"])

    # ONE definition of a hole, used by every coverage number here AND by the
    # cross-market figure below: a side that HELD a position and published no count for
    # it. The earlier both-sides-null version called 134741/dealer 421 of 422 weeks
    # covered while its long count was missing in 69 of them, and disagreed with the
    # cross-market figure in its own sentence by 3.8x. A coverage number that certifies
    # a gappy series is worse than no coverage number.
    hole_long = (g["long"].fillna(0) > 0) & g["traders_long"].isna()
    hole_short = (g["short"].fillna(0) > 0) & g["traders_short"].isna()
    n_long, n_short = int((g["long"].fillna(0) > 0).sum()), int((g["short"].fillna(0) > 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Firms long / short, latest",
              f"{_fmt(g['traders_long'].iloc[-1], '', 0)} / {_fmt(g['traders_short'].iloc[-1], '', 0)}")
    c2.metric("Contracts per long firm", _fmt(per_long.iloc[-1], "", 0))
    c3.metric("Long count published", f"{n_long - int(hole_long.sum())} / {n_long} weeks held long")
    c4.metric("Short count published",
              f"{n_short - int(hole_short.sum())} / {n_short} weeks held short")
    _show([
        _p([(g["traders_long"], "Firms long", "traders", charts.ACCENT, None),
            (g["traders_short"], "Firms short", "traders", charts.WARN, None)],
           "reporting firms", f"{labels[cohort]}: how many firms are on each side", breaks),
        _p([(per_long, "Per long firm", "long", charts.ACCENT, None),
            (per_short, "Per short firm", "short", charts.WARN, None)],
           "contracts per firm",
           "Average position per firm: 100k across 5 firms is not 100k across 60", breaks),
        _p(_oi_rows(wk), "contracts",
           "Market size, to read a rise in contracts-per-firm against", breaks, 0.7),
    ])

    # Coverage across the whole latest report, not just this market: a per-firm measure
    # that is unavailable for much of the board is a screen you cannot run, and that
    # belongs on screen rather than being discovered later. Same hole definition.
    latest = df["report_date"].max()
    snap = df.loc[(df["report_date"] == latest) & (df["cohort"].isin(order))]
    pos = np.concatenate([_f(snap["long"]).to_numpy(), _f(snap["short"]).to_numpy()])
    tr = np.concatenate([_f(snap["traders_long"]).to_numpy(), _f(snap["traders_short"]).to_numpy()])
    gap_pct = 100.0 * np.isnan(tr[pos > 0]).mean() if (pos > 0).any() else float("nan")
    miss = (snap["traders_long"].isna() & (_f(snap["long"]).fillna(0) > 0)) | (
        snap["traders_short"].isna() & (_f(snap["short"]).fillna(0) > 0))
    holes = miss.groupby(snap["market_code"], observed=True).any()
    holed = 100.0 * int(holes.sum()) / len(holes) if len(holes) else float("nan")
    # The trend is COMPUTED per cohort because the quotable number does not generalise:
    # lib/metrics cites 4.2% of 2015 rows rising to 28.7% of 2026, but that series is
    # leveraged-money LONGS. On this one-sided definition lev_money does widen while the
    # four cohorts together are flat to cyclical (23.9% in 2015, 33.9% in 2018, 27.9% in
    # 2026), so "growing across the corpus" was a cohort's trend wearing the corpus' name.
    coh = df.loc[df["cohort"] == cohort]
    coh_hole = ((_f(coh["long"]).fillna(0) > 0) & coh["traders_long"].isna()) | (
        (_f(coh["short"]).fillna(0) > 0) & coh["traders_short"].isna())
    by_year = 100.0 * coh_hole.groupby(pd.to_datetime(coh["report_date"]).dt.year).mean()
    trend = f"{_fmt(by_year.min())}-{_fmt(by_year.max())} of rows by report year"
    if len(by_year) > 1:
        trend += (f", {by_year.index[0]}: {_fmt(by_year.iloc[0])} and {by_year.index[-1]}: "
                  f"{_fmt(by_year.iloc[-1])}")
    st.caption(
        f"Coverage on the {latest} report of `{SOURCE}`: {_fmt(gap_pct)} of reportable cohort-sides "
        f"holding a position publish no trader count, and {int(holes.sum())} of {len(holes)} markets "
        f"-- {_fmt(holed, '%', 0)} of the current cross-section -- have at least one such hole, so a "
        f"table ranked on a per-firm measure is quietly missing that many. For {labels[cohort]} on "
        f"this market the long count is absent in {int(hole_long.sum())} of the {n_long} weeks it "
        f"held a long position and the short count in {int(hole_short.sum())} of {n_short}; across "
        f"every market it trades the same gap runs {trend} -- the current year included, so it is "
        f"not a legacy artifact. Gaps stay gaps: a real numerator over an unknown denominator, and "
        f"interpolating it invents firms. Non-reportables are absent from the selector because "
        f"traders_* does not exist for them in any family -- they are small BECAUSE they are under "
        f"the reporting threshold, so nobody counts them. The spec names a spread-trader column for "
        f"most reportable cohorts but the ingest stores none, so nothing here could show it: these "
        f"are the directional sides only."
    )


def _spreads(df: pd.DataFrame, wk: pd.DataFrame, summary: pd.DataFrame, breaks: tuple) -> None:
    st.subheader("Spread share: the part of the market with no direction")
    _show([
        _p([(wk["spread_pct"], "Spreads / open interest", "share", charts.WARN, None)],
           "% of open interest", "Share of open interest that is calendar structure", breaks),
        _p(_oi_rows(wk), "contracts",
           "Open interest, and what is left once spreads come out", breaks, 0.9),
    ], height_per_panel=200)
    st.caption(
        "A spread is long one expiry and short another, so it sits in neither the long nor the short "
        "column and expresses no view on the level. A market that is half spreads carries half as "
        "much directional positioning as its open interest advertises."
    )

    latest = df["report_date"].max()
    snap = (df.loc[df["report_date"] == latest].groupby("market_code", observed=True)
            .agg(spread_total=("spread", "sum"), open_interest=("open_interest", "first")))
    snap["spread_total"] = _f(snap["spread_total"])
    snap["open_interest"] = _f(snap["open_interest"])
    snap["pct"] = metrics.spread_share(snap["spread_total"], snap["open_interest"])
    liquid = snap.join(summary.set_index("market_code")[["market", "tier"]], how="left")
    liquid = liquid[liquid["tier"].isin(("CORE", "WIDE"))]
    screen = liquid.nlargest(20, "pct")
    st.plotly_chart(
        charts.ranked_bars(
            [f"{r.market}  ·  {c}" for c, r in screen.iterrows()],
            screen["pct"].tolist(), unit="% of open interest", kind="share",
            hover=[f"{r.spread_total:,.0f} spreads of {r.open_interest:,.0f} OI, "
                   f"{r.open_interest - r.spread_total:,.0f} directional"
                   for _, r in screen.iterrows()],
        ),
        width="stretch",
    )
    top = screen["pct"].max()
    correction = 1.0 / (1.0 - top / 100.0) if pd.notna(top) and top < 100.0 else float("nan")
    st.caption(
        f"Top 20 of the {len(liquid)} liquid markets on {latest}. This screen is always CORE + WIDE "
        f"whatever the Universe toggle says -- widening the universe changes the time series, not "
        f"this ranking. Across all {len(snap)} markets that week the median spread share is "
        f"{_fmt(snap['pct'].median())} and the 75th percentile {_fmt(snap['pct'].quantile(0.75))}, so "
        f"the {_fmt(screen['pct'].min())}-{_fmt(top)} band here is not typical -- it is where an "
        f"open-interest denominator is materially wrong, by 1/(1 - spread share), which at the top of "
        f"this list is {_fmt(correction, 'x', 2)}. Not one complex: the list mixes rates with the "
        f"adjusted-rate S&P contract, which lib/metrics.directional_oi names separately for that "
        f"reason. A spread total sums the cohorts that publish one: here every reportable cohort does "
        f"and non-reportables never do, but legacy commercials and disaggregated producer-merchants "
        f"have no spread column at all, so the same statistic there is a floor, not a total."
    )


def _render() -> None:
    df = cache.read(SOURCE)
    summary = cache.market_summary(SOURCE)
    freshness(df["report_date"], "financial-futures report")

    scope = st.radio("Universe", ("CORE", "CORE + WIDE", "everything ingested"),
                     horizontal=True, key="crowd_scope")
    keep = {"CORE": ("CORE",), "CORE + WIDE": ("CORE", "WIDE")}.get(scope)
    pool = (summary if keep is None else summary[summary["tier"].isin(keep)]).sort_values(
        "oi_window_max", ascending=False)
    codes = pool["market_code"].astype(str).tolist()
    names = dict(zip(codes, pool["market"].astype(str) + "  ·  " + pool["market_code"].astype(str)))
    # SOFR-3M by default because it is the market where the CR denominator matters most:
    # 44.5% of its open interest is spreads, making the wrong denominator a 1.8x error
    # rather than a rounding difference.
    default = universe.default_market(pool, prefer=("134741", "13874+"))
    code = st.selectbox("Market", codes, index=codes.index(default) if default in codes else 0,
                        format_func=lambda c: names.get(c, c), key="crowd_market")

    row = pool.loc[pool["market_code"] == code].iloc[0]
    wk = _weekly(df, code)
    segs = segments.describe_segments(pd.Series(wk.index), wk["contract_units"])
    breaks = tuple(pd.Timestamp(d) for d in segs["start"].iloc[1:])
    st.caption(f"{row['market_full']} — {row['contract_units']} — "
               f"{int(row['n_reports'])} reports from {row['first_report'].date()}, tier {row['tier']}.")
    # Every "latest" below is THIS MARKET's latest, which on a dead code is years old.
    # freshness() above is about the feed; this is about the market.
    market_last_print(pd.Series(wk.index), df["report_date"], str(row["market"]))
    if len(segs) > 1:
        st.caption(
            "Dotted verticals are segment breaks -- a contract re-specification or a hole longer than "
            "13 weeks -- and percentiles restart at each one, because a level before a re-basing is "
            "not comparable with one after it. Segments: "
            + "; ".join(f"{r.start}..{r.end}, {r.reports} reports, units {r.units}"
                        for r in segs.itertuples())
        )
    if code in universe.CONSOLIDATED or code in universe.SUPERSEDED_BY:
        st.caption(universe.CONSOLIDATED_TRADEOFF)

    _concentration(wk, breaks)
    _traders(df, code, wk, breaks)
    _spreads(df, wk, summary, breaks)

    # The market-week / per-cohort distinction, proved on the data rather than asserted:
    # reading the firms section as one market-wide number is this board's most exposed
    # misconception, since both kinds of column sit in the same table.
    per_week = df.groupby(["market_code", "report_date"], observed=True)["traders_long"].nunique()
    last = df.loc[df["market_code"] == code]
    last = last.loc[last["report_date"] == last["report_date"].max()]
    st.caption(
        "Concentration, open interest and the market-wide `traders_total` are market-week "
        "quantities, identical across the cohorts of a report, so they are read once per week "
        "rather than summed. The PER-COHORT counts in the firms section are NOT: `traders_long` "
        "and `traders_short` come from a different published column per cohort and differ across "
        f"the cohorts of a week in {_fmt(100.0 * (per_week > 1).mean())} of this file's "
        f"{len(per_week):,} market-weeks, which is what the cohort selector changes. On "
        f"{last['report_date'].iloc[0]} this market publishes "
        f"{int(_f(last['traders_long']).nunique())} distinct per-cohort long counts against a "
        f"single market-wide total of {_fmt(_f(last['traders_total']).max(), '', 0)}. Where the "
        "published columns do not reconcile exactly: " + cftc_spec.RESIDUAL_EXPLANATION
    )
    caveat_block(SOURCE)


BOARD = Board(
    id="crowding",
    title="Crowding",
    render=_render,
    sources=(SOURCE,),
    order=40,
    blurb="How few hands hold the position, and how much of the market is calendar structure "
          "rather than direction.",
)
