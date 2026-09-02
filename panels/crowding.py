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
1,623 of 46,361 market-weeks while the side-total reading never does. The error is
1/(1 - spread share): 1.8x in SOFR-3M, the default market here. So the sentence on
screen is "the four largest longs hold X% of all long positions", with directional
open interest beside it so X converts back into contracts.

WHAT THIS BOARD REFUSES TO ANSWER. Whether crowding predicts anything: this repo
has no price series, so no forward return exists to test against, and positioning
is contemporaneous with price rather than ahead of it. And who the four are: CR is
anonymous, and "four" is four accounts as CFTC classifies them, which may be four
desks of one firm.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from panels import Board

SOURCE = "cftc_tff_fut"

#: All eight published concentration columns: {gross, net} x {4, 8} x {long, short}.
CONC = tuple(f"conc_{k}_{n}_{s}" for k in ("gross", "net") for s in ("long", "short") for n in (4, 8))


def _app_helpers():
    """app.py's shared renderers, obtained without re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not alias it in sys.modules
    under its filename, so the documented `from app import caveat_block` imports a
    SECOND copy and runs app.py's module body again -- including its unkeyed sidebar
    radio, which raises StreamlitDuplicateElementId before either helper is bound.
    Verified: with the plain import this board fails on first render. So prefer the
    already-running module and keep the documented import as the fallback.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


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
    sums. Legacy publishes concentration above 100% on 23 rows (worst 482.6% on 957
    contracts) and tff has none, but the clamp runs here rather than being trusted to
    the family: a broken input should read as a gap, not as a spike.
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


def _rank_within_segments(values: pd.Series, segment: pd.Series) -> pd.Series:
    """Trailing 3-year percentile, restarted at every unit break and long hole.

    Causal twice over: the window is (t - 3y, t], and it never reaches back across a
    re-basing. Ranking $20-per-point observations against $100-per-point ones is what
    this repo used to do, and it is not a percentile of anything.
    """
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    dates = pd.Series(values.index, index=values.index)
    for sid in segment.unique():
        m = segment == sid
        if int(m.sum()) < 8:  # too short to rank; left blank rather than guessed
            continue
        out.loc[m] = metrics.trailing_percentile(values.loc[m], dates.loc[m])
    return out


def _concentration(wk: pd.DataFrame, breaks: tuple) -> None:
    st.subheader("Concentration")
    cur = wk.iloc[-1]
    rank = _rank_within_segments(wk["conc_gross_8_long"], wk["segment"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 4 longs, share of the long side", _fmt(cur["conc_gross_4_long"]),
              _wow(wk["conc_gross_4_long"]))
    c2.metric("= contracts held by those four",
              _fmt(cur["conc_gross_4_long"] / 100.0 * cur["dir_oi"], "", 0))
    c3.metric("Top 8 longs, trailing 3y percentile", _fmt(rank.iloc[-1], "", 0))
    c4.metric("Spreads, share of open interest", _fmt(cur["spread_pct"]), _wow(wk["spread_pct"]))
    st.caption(
        f"Read as \"the four largest longs hold {_fmt(cur['conc_gross_4_long'])} of all long "
        f"positions\": CR is a share of the SIDE TOTAL, not of open interest. On "
        f"{wk.index[-1].date()} that side total is {_fmt(cur['dir_oi'], '', 0)} contracts -- open "
        f"interest {_fmt(cur['open_interest'], '', 0)} less {_fmt(cur['spread_total'], '', 0)} of "
        f"spreads -- so an open-interest denominator would understate the top four by "
        f"{_fmt(cur['open_interest'] / cur['dir_oi'], 'x', 2)}. Deltas appear only where the "
        f"previous report is exactly one week back."
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
        "thin crypto contract -- which is what the percentile is for. It is blank for a segment's "
        "first weeks and wherever the trailing window is under 60% occupied."
    )


def _traders(df: pd.DataFrame, code: str, wk: pd.DataFrame, breaks: tuple) -> None:
    st.subheader("Reporting firms behind the position")
    labels = {c.id: c.label for c in cftc_spec.spec_for(SOURCE).cohorts if c.traders_long}
    # Leveraged funds first: the cohort whose crowding is actually asked about, and the
    # one where a handful of firms holding the whole net is common.
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

    usable = int((g["traders_long"].notna() | g["traders_short"].notna()).sum())
    held = (g["long"].fillna(0) > 0) | (g["short"].fillna(0) > 0)
    blind = int((held & g["traders_long"].isna() & g["traders_short"].isna()).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Firms long / short, latest",
              f"{_fmt(g['traders_long'].iloc[-1], '', 0)} / {_fmt(g['traders_short'].iloc[-1], '', 0)}")
    c2.metric("Contracts per long firm", _fmt(per_long.iloc[-1], "", 0))
    c3.metric("Weeks with a usable count", f"{usable} / {len(g)}")
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
    # that is unavailable for a third of the board is a screen you cannot run, and that
    # belongs on screen rather than being discovered later.
    latest = df["report_date"].max()
    snap = df.loc[(df["report_date"] == latest) & (df["cohort"].isin(order))]
    pos = np.concatenate([_f(snap["long"]).to_numpy(), _f(snap["short"]).to_numpy()])
    tr = np.concatenate([_f(snap["traders_long"]).to_numpy(), _f(snap["traders_short"]).to_numpy()])
    gap_pct = 100.0 * np.isnan(tr[pos > 0]).mean() if (pos > 0).any() else float("nan")
    miss = (snap["traders_long"].isna() & (_f(snap["long"]).fillna(0) > 0)) | (
        snap["traders_short"].isna() & (_f(snap["short"]).fillna(0) > 0))
    holes = miss.groupby(snap["market_code"], observed=True).any()
    st.caption(
        f"Coverage on the {latest} report of `{SOURCE}`: {_fmt(gap_pct)} of reportable cohort-sides "
        f"holding a position publish no trader count, and {int(holes.sum())} of {len(holes)} markets "
        f"have at least one such hole -- roughly a third of the current cross-section cannot support "
        f"a per-firm measure, so a table ranked on one is quietly missing them. Here the count is "
        f"absent in {blind} of {len(g)} weeks that carried a nonzero position. The null rate is "
        f"growing across the corpus (4.2% of 2015 rows to 28.7% of 2026), so it is a widening blind "
        f"spot, not a legacy artifact. Gaps stay gaps: this is a real numerator over an unknown "
        f"denominator, and interpolating it invents firms. Non-reportables are absent from the "
        f"selector because traders_* does not exist for that cohort in any family -- they are small "
        f"BECAUSE they are under the reporting threshold, so nobody counts them. Spread-trader counts "
        f"exist for some cohorts and are excluded; these are the directional sides only."
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
    st.caption(
        f"Top 20 of the {len(liquid)} liquid markets on {latest}. Across all {len(snap)} markets that "
        f"week the median spread share is {_fmt(snap['pct'].median())} and the 75th percentile "
        f"{_fmt(snap['pct'].quantile(0.75))}, so the rates complex at 40-48% is not typical -- it is "
        f"where an open-interest denominator is materially wrong, by 1/(1 - spread share). A spread "
        f"total sums the cohorts that publish one: here every reportable cohort does and "
        f"non-reportables never do, but legacy commercials and disaggregated producer-merchants have "
        f"no spread column at all, so the same statistic there is a floor rather than a total."
    )


def _render() -> None:
    caveat_block, freshness = _app_helpers()
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

    st.caption(
        "Concentration, trader counts and open interest are market-week quantities, identical across "
        "the cohorts of a report, so they are read once per week rather than summed. Where the "
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
