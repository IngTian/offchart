"""Flows -- what changed this week, and the invariants that constrain it.

Levels say where positioning sits; this board is the week-over-week delta. It
exists mainly because the flow side of this dataset carries two free
cross-checks that the level side does not, and both are run on screen:

  1. COHORT NET FLOWS SUM TO ~0, because every contract has a long and a short.
     This catches a dropped cohort, a field mapped to the wrong column and a
     misaligned join, all of which produce individually plausible series.
     NET_ZERO_TOL is the tolerance on a LEVEL; a flow differences two
     independently rounded levels, so cftc_spec.FLOW_TOL is twice it.
  2. OUR DIFF MUST EQUAL CFTC'S OWN PUBLISHED change_* COLUMNS to within
     CHANGE_FIELD_TOL, except where it provably should not.

WHAT THIS BOARD REFUSES TO ANSWER. Who traded with whom: these are net cohort
deltas, not trades, so a +10,000 against a -8,000 BOUNDS a transfer and does not
observe one -- a cohort's net also moves when it cuts the other side of its book
and when a firm is reclassified with no trade at all (July 2008 is the large
example). And whether a flow predicts a return: there is no price series here, and
positioning is contemporaneous with price rather than ahead of it.

WHAT THIS ADDS OVER metrics.flow, which guards the CALENDAR span so a skipped
report week is a gap rather than a mislabelled multi-week change. It cannot guard
a contract re-basing: 20974+ went from $100 to $20 per index point on 2023-05-02,
and an unmasked 4-week flow across that seam draws an open-interest change of
+209,517 contracts and an asset-manager net flow of +36,718, none of it a trade.
Only the OI panels can show a ~200,000 artifact -- that is the OI LEVEL jump,
49,531 -> 255,954. Cohort net flows sum to ~0 by construction, so a re-basing
cannot manufacture +200,000 of net "buying" anywhere: measured across this seam
the five cohort flows are +36,718 / +10,867 / -7,969 / -14,147 / -25,468. So flows
here are additionally nulled wherever the window crosses a lib.segments boundary,
and the boundaries are drawn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from lib.ui import caveat_block, freshness, market_last_print
from panels import Board

_HORIZONS = (1, 2, 4, 13)
_HISTORY_DAYS = 730
_UNIT = "contracts"
#: Market-wide series drawn beside positioning. `directional_oi` is the one that is
#: comparable with a cohort net, because a spread position is in neither the long
#: nor the short column; total open interest is kept next to it as the reference.
_OI_COLS = ("directional_oi", "open_interest")


def _oi_pair(directional: pd.Series, total: pd.Series, kind: str) -> list[charts.Series]:
    """The two open-interest lines, always drawn together. Same quantity, same
    unit, one panel -- no axis trick and no choice for the author to make."""
    return [
        charts.Series(directional, "Directional (ex-spread)", kind=kind, color=charts.WARN),
        charts.Series(total, "Total (includes spreads)", kind=kind, color=charts.MUTED),
    ]


def _f(s: pd.Series) -> pd.Series:
    """Nullable Int32 -> float64, NA -> NaN. `Int32.astype(float)` raises on NA,
    and these columns are genuinely null: the first report of a code has no
    published change at all."""
    return pd.Series(s.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def _one_market(source_id: str, code: str) -> pd.DataFrame:
    """One market's rows, dates parsed. Selected on market_code, never on name."""
    sub = cache.read(source_id)
    sub = sub[sub["market_code"] == code].copy()
    sub["report_date"] = pd.to_datetime(sub["report_date"])
    return sub


def _market_frame(source_id: str, code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date cohort nets, plus the market-week columns.

    open_interest and contract_units are constant across the cohorts of one
    market-week, so they are taken with .first(); summing them would multiply open
    interest by the cohort count. `spread` is the opposite: it is published PER
    COHORT and not for every cohort (never for non-reportables, in any family), so
    the market's spread total is a SUM, with min_count=1 so a market-week that
    publishes no spread at all stays NaN rather than becoming a confident zero.
    Verified: OI - sum(long) - spread_total is within 2 contracts of 0 in every
    family, so this sum really is the whole spread book and directional_oi below
    really is sum(long) == sum(short).
    """
    sub = _one_market(source_id, code)
    sub["net"] = _f(sub["long"]) - _f(sub["short"])
    sub["spread"] = _f(sub["spread"])
    keys = ["report_date", "cohort"]
    net = sub.groupby(keys, observed=True)["net"].first().unstack("cohort").sort_index()
    by_date = sub.groupby("report_date", observed=True)
    meta = by_date[["open_interest", "contract_units"]].first().sort_index()
    meta["open_interest"] = _f(meta["open_interest"])
    meta["spread_total"] = by_date["spread"].sum(min_count=1).sort_index()
    # metrics.directional_oi fills a missing spread with 0, which would draw "this
    # market has no spreads" where the truth is "no spread column". Blank instead.
    meta["directional_oi"] = metrics.directional_oi(
        meta["open_interest"], meta["spread_total"]
    ).where(meta["spread_total"].notna())
    return net, meta


def _flows(
    net: pd.DataFrame, meta: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """h-week flows per cohort and for open interest, plus segment ids.

    Two independent masks because they catch different lies: metrics.flow nulls a
    window whose calendar span is not h weeks, the segment check nulls one that
    straddles a re-basing or a multi-year hole in the code's history.
    """
    dates = pd.Series(net.index, index=net.index)
    seg = segments.segment_ids(dates, meta["contract_units"])
    same = seg.eq(seg.shift(horizon))
    flows = pd.DataFrame(
        {c: metrics.flow(net[c], dates, horizon_weeks=horizon).where(same) for c in net.columns},
        index=net.index,
    )
    oi_flows = pd.DataFrame(
        {c: metrics.flow(meta[c], dates, horizon_weeks=horizon).where(same) for c in _OI_COLS},
        index=net.index,
    )
    return flows, oi_flows, seg


def _reconcile(source_id: str, code: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Our prior-report diff against CFTC's published change_* columns.

    Deliberately `.diff(1)` -- the PRIOR REPORT IN THIS DATASET -- not a
    calendar-anchored 1-week flow. The published column is CFTC's own change
    against CFTC's own previous report, which is not always the same row: where a
    report CFTC published is missing here, or where a row is a sub-week re-issue,
    the two diffs span different intervals. That is measured rather than assumed,
    which is what the returned counts are for.
    """
    sub = _one_market(source_id, code)

    def rec(field, who, dates, level, published) -> pd.DataFrame:
        err = (_f(level).diff(1) - _f(published)).abs()
        gap = pd.Series(pd.to_datetime(dates)).diff().dt.days
        return pd.DataFrame(
            {"field": field, "who": who, "date": dates, "gap": gap.to_numpy(),
             "err": err.to_numpy()}
        )

    recs: list[pd.DataFrame] = []
    for cohort, g in sub.groupby("cohort", observed=True):
        g = g.sort_values("report_date")
        d = g["report_date"].to_numpy()
        for pos, pub in (("long", "change_long_published"), ("short", "change_short_published")):
            recs.append(rec(pos, str(cohort), d, g[pos], g[pub]))
    # Market-week columns are identical across cohorts, so open interest is
    # reconciled once per date rather than once per cohort-date.
    one = sub.groupby("report_date", observed=True)[
        ["open_interest", "oi_change_published"]
    ].first().sort_index()
    recs.append(
        rec("open_interest", "(market-wide)", one.index.to_numpy(),
            one["open_interest"], one["oi_change_published"])
    )

    tol = cftc_spec.CHANGE_FIELD_TOL
    rows = []
    for field, g in pd.concat(recs, ignore_index=True).dropna(subset=["err"]).groupby("field"):
        worst = g.loc[g["err"].idxmax()]
        weekly = g["gap"].sub(7.0).abs() <= metrics.SPAN_TOLERANCE_DAYS
        rows.append(
            {
                "field": field,
                "comparisons": len(g),
                f"agree within {tol:g}": f"{100.0 * (g['err'] <= tol).mean():.3f}%",
                "...on a 7-day prior": (
                    f"{100.0 * (g.loc[weekly, 'err'] <= tol).mean():.3f}% "
                    f"of {int(weekly.sum()):,}"
                    if weekly.any() else "no weekly prior"
                ),
                "worst gap": f"{g['err'].max():,.0f}",
                "worst on": f"{pd.Timestamp(worst['date']).date()}  {worst['who']}",
                "prior report was": f"{int(worst['gap'])} days earlier",
            }
        )

    # Which of the known mechanisms each report belongs to, counted from the
    # market-wide open-interest column so the counts are per REPORT, not per
    # cohort-field. implied_prev is the level CFTC's published change was measured
    # against; where it is not the previous level we hold, the two diffs are not
    # spanning the same interval and the error is arithmetic, not disagreement.
    oi, pub = _f(one["open_interest"]), _f(one["oi_change_published"])
    gap = pd.Series(one.index, index=one.index).diff().dt.days
    implied_prev = oi - pub
    other_base = (implied_prev - oi.shift(1)).abs() > tol
    # A sub-week re-issue repeats the PREVIOUS row's base: two rows days apart both
    # published against the same earlier report.
    reissue = other_base & (gap < 7.0 - metrics.SPAN_TOLERANCE_DAYS) & (
        (implied_prev - implied_prev.shift(1)).abs() <= tol
    )
    long_gap = gap > 7.0 + metrics.SPAN_TOLERANCE_DAYS
    counts = {
        "reports": int(len(one)),
        "missing_prior": int((other_base & ~reissue).sum()),
        "reissue": int(reissue.sum()),
        "long_gap": int(long_gap.sum()),
        "long_gap_same_base": int((long_gap & ~other_base).sum()),
    }
    return pd.DataFrame(rows), counts


def _zero_sum(flows: pd.DataFrame, net: pd.DataFrame, anchor: pd.Timestamp, labels: dict) -> None:
    """Ranked bars of each cohort's net flow, with the sum tested on screen."""
    row = flows.loc[anchor].sort_values(ascending=False, na_position="last")
    now, prev = net.loc[anchor], net.loc[anchor] - flows.loc[anchor]
    # A cohort whose flow is NaN is NOT a cohort that did not move. Plotting it as
    # a zero-length bar would be indistinguishable from a real zero, so it is
    # dropped from the chart and named in the warning instead.
    usable = row.dropna()
    missing = [labels.get(c, str(c)) for c in row.index[row.isna()]]

    chart, side = st.columns([3, 2])
    if usable.empty:
        chart.info("Nothing to plot: no cohort flow is computable at this date and horizon.")
    else:
        chart.plotly_chart(
            charts.ranked_bars(
                [labels.get(c, str(c)) for c in usable.index],
                [float(v) for v in usable],
                unit=_UNIT,
                kind="flow",
                hover=[
                    f"net {prev[c]:,.0f} to {now[c]:,.0f} "
                    f"({charts.signed_direction(prev[c], now[c])})"
                    for c in usable.index
                ],
            )
        )

    tol = cftc_spec.FLOW_TOL
    total = float(usable.sum())
    # Never "0 of 3 cohort net flows sum to +0 contracts": an empty sum is not a
    # passing test, and formatting it as one is the same lie as a zero-length bar.
    # Summed here rather than through metrics.net_flow_balances because that one's
    # row-wise sum skips NaN, so it would report a clean 0 for a row where no
    # cohort flow is computable -- which is exactly the case this branch exists for.
    side.markdown(
        f"**No cohort net flow is computable, so none of the {len(row)} are summed**"
        if usable.empty
        else f"**{len(usable)} of {len(row)} cohort net flows sum to "
             f"{charts.fmt_change(total, 'flow', _UNIT)}**"
    )
    if missing:
        side.warning(
            "No flow is computable for "
            + ", ".join(missing)
            + ", so the sum above tests nothing and those cohorts are absent from the "
            "chart rather than drawn at zero. Either this window's calendar span is not "
            "the stated number of weeks, or it crosses a contract re-basing."
        )
    elif abs(total) <= tol:
        side.caption(
            f"Invariant holds -- to within about {tol:g} contracts, not exactly: "
            f"NET_ZERO_TOL is {cftc_spec.NET_ZERO_TOL:g} on a level and a flow differences "
            "two independently rounded levels, so it carries twice the rounding."
        )
    else:
        side.error(
            f"{total:,.0f} is outside the {tol:g}-contract rounding allowance, which "
            "is not a market fact: a cohort is missing from this pivot, a field is mapped "
            "to the wrong column, or a cohort failed to report on one of the two dates."
        )
    side.caption(
        f"The underlying LEVELS on {anchor.date()} sum to {float(now.sum()):,.0f} "
        f"(allowance {cftc_spec.NET_ZERO_TOL:g}). Net cohort deltas are not trades: if "
        "asset managers add 10,000 while dealers shed 8,000 that BOUNDS a transfer "
        "between them, it does not observe one -- a cohort's net also moves when it cuts "
        "the other side of its book and when a firm is reclassified without trading. "
        "Non-reportables are a residual bucket and absorb whatever the reportable cohorts "
        "do not account for."
    )


def _render() -> None:
    ids = [s.id for s in cftc_spec.SPECS]
    c1, c2, c3, c4 = st.columns([2.2, 3, 1, 1.4])
    # Every widget is keyed: Streamlit derives an unkeyed widget's id from its type
    # and parameters, so two boards with a same-shaped widget collide.
    source_id = c1.selectbox(
        "Report", ids, index=ids.index("cftc_tff_fut"), key="flows_report",
        format_func=lambda i: cftc_spec.BY_ID[i].label,
    )
    spec = cftc_spec.spec_for(source_id)
    labels = {c.id: c.label for c in spec.cohorts}

    summary = cache.market_summary(source_id)
    if summary.empty:
        st.info(f"`{source_id}` is not ingested yet. Run `python -m scripts.ingest`.")
        return
    pool = summary[summary["tier"].isin(("CORE", "WIDE"))]
    pool = (pool if not pool.empty else summary).sort_values("oi_window_max", ascending=False)
    codes = list(pool["market_code"])
    names = dict(zip(pool["market_code"], pool["market"]))
    default = universe.default_market(summary, prefer=("13874A", "067651", "023651"))
    code = c2.selectbox(
        "Market (keyed on code -- names are not stable)",
        codes,
        index=codes.index(default) if default in codes else 0,
        format_func=lambda c: f"{names.get(c, c)}  ·  {c}",
        key="flows_market",
    )
    horizon = c3.selectbox("Horizon (weeks)", _HORIZONS, index=0, key="flows_horizon")

    # freshness() gets the SOURCE's newest report, never this market's: a market
    # that simply did not print this week is not a stale feed. The per-market
    # statement is market_last_print's job.
    source_dates = summary["last_report"]
    net, meta = _market_frame(source_id, code)
    if net.empty or len(net) <= horizon:
        st.info("Too few reports on this code to compute a flow at this horizon.")
        freshness(source_dates, "CFTC report")
        market_last_print(net.index, source_dates, names.get(code, code))
        caveat_block(source_id)
        return
    anchor = pd.Timestamp(
        c4.selectbox(
            "As of report",
            list(net.index[::-1]),
            index=0,
            format_func=lambda d: str(pd.Timestamp(d).date()),
            key="flows_anchor",
        )
    )

    freshness(source_dates, "CFTC report")
    market_last_print(net.index, source_dates, names.get(code, code))
    row = pool[pool["market_code"] == code]
    if not row.empty:
        st.caption(
            f"{row['market_full'].iloc[0]} · units {row['contract_units'].iloc[0]} · "
            f"{int(row['n_reports'].iloc[0]):,} reports from "
            f"{row['first_report'].iloc[0].date()} · tier {row['tier'].iloc[0]} · "
            f"{len(summary) - len(pool):,} of this report's {len(summary):,} codes sit below "
            "the WIDE liquidity floor and are left out of the selector only, not the data."
        )
    if code in universe.CONSOLIDATED or code in universe.SUPERSEDED_BY:
        st.caption(universe.CONSOLIDATED_TRADEOFF)

    flows, oi_flows, seg = _flows(net, meta, horizon)
    starts = pd.Series(net.index, index=net.index).groupby(seg.to_numpy()).min()
    lo = anchor - pd.Timedelta(days=_HISTORY_DAYS)
    win = slice(lo, anchor)
    breaks = tuple(pd.Timestamp(d) for d in starts.iloc[1:] if lo <= pd.Timestamp(d) <= anchor)

    st.subheader(f"Zero-sum test, {horizon}-week net flow to {anchor.date()}")
    _zero_sum(flows, net, anchor, labels)

    st.subheader("Flow history")
    cohort_series = [
        charts.Series(flows[c].loc[win], labels.get(c, str(c)), kind="flow")
        for c in net.columns
        if flows[c].loc[win].notna().any()
    ]
    if not cohort_series:
        # No early return: the reconciliation and the caveats below do not depend on
        # this chart, and dropping them would hide the source's limitations exactly
        # when the data is at its most awkward.
        st.info(
            f"No {horizon}-week flow in the two years to {anchor.date()} survives the "
            "calendar and segment checks -- this stretch of the code's history is "
            "either not weekly or is cut by a re-basing."
        )
    else:
        dir_chg = oi_flows["directional_oi"].loc[win]
        tot_chg = oi_flows["open_interest"].loc[win]
        shared = dict(unit=_UNIT, breaks=breaks)
        st.plotly_chart(
            charts.stacked([
                charts.Panel(
                    cohort_series, title=f"Cohort net flow, {horizon}-week", height=1.35, **shared
                ),
                charts.Panel(
                    _oi_pair(dir_chg, tot_chg, "flow"),
                    title=f"Open interest change, {horizon}-week", **shared,
                ),
                charts.Panel(
                    _oi_pair(meta["directional_oi"].loc[win], meta["open_interest"].loc[win],
                             "open_interest"),
                    title="Open interest, level", height=0.8, **shared,
                ),
            ])
        )
        # Measured off the two series just plotted rather than quoted, so this
        # sentence cannot drift away from the chart it describes.
        both = dir_chg.notna() & tot_chg.notna()
        opposed = ((dir_chg > 0) & (tot_chg < 0)) | ((dir_chg < 0) & (tot_chg > 0))
        sp = metrics.spread_share(meta["spread_total"], meta["open_interest"]).loc[win].dropna()
        sp_txt = (
            f"spreads are {sp.iloc[-1]:.1f}% of open interest here on {sp.index[-1].date()}"
            if not sp.empty
            else "this market publishes no spread column, so the two lines coincide"
        )
        st.caption(
            "Read the top two panels together -- that is the whole reason open interest "
            "stays on screen. A cohort adding while open interest RISES is new contracts "
            "created against someone else's new short; the same cohort adding while open "
            "interest is FLAT is contracts changing hands, and the other side is another "
            "line on this chart. Compare against the DIRECTIONAL line, not the total: a "
            "spread position is long one expiry and short another, so it is in neither the "
            "long nor the short column and cannot appear in any cohort net line here. The "
            "two are not interchangeable and the difference flips the reading -- over this "
            f"window they moved in OPPOSITE directions in {int((opposed & both).sum()):,} of "
            f"{int(both.sum()):,} computable weeks, and {sp_txt}. Across markets the spread "
            "share is a median 3.2% of open interest but 44.5% in 3-month SOFR and 47.2% in "
            "Fed Funds, where nearly half of the market is calendar structure rather than "
            f"direction. Blanks are deliberate: a flow is absent wherever the window does "
            f"not span exactly {horizon} week{'s' if horizon > 1 else ''} or crosses a dotted "
            "break (a re-basing, or a hole in this code's history)."
        )

    st.subheader("Reconciliation against the published change columns")
    table, n = _reconcile(source_id, code)
    st.dataframe(table, hide_index=True)
    st.caption(
        f"Our `.diff(1)` against CFTC's published change_* columns over all "
        f"{n['reports']:,} of this code's reports. A mismatch is not automatically our bug. "
        "Below are the classes that account for the mismatches observed in this corpus, "
        "which is not the same thing as a closed list -- the worst row's prior-report gap is "
        "in the table so you can classify it yourself instead of trusting the list. "
        "(1) A REPORT CFTC PUBLISHED IS ABSENT FROM THIS DATASET. The published change is "
        "measured against CFTC's own previous report; where we do not hold that report, OUR "
        "`.diff(1)` spans the hole and the published figure does not, so the difference "
        "measures the missing week rather than a disagreement. On this code the published "
        f"open-interest change is taken against a level this dataset does not hold on "
        f"{n['missing_prior']:,} of those reports. (2) A SUB-WEEK RE-ISSUE: two reports days "
        "apart both published against the same earlier base, so the second is not a change "
        f"from the first -- {n['reissue']:,} of them. (3) Independent rounding per column, "
        f"hence a tolerance of {cftc_spec.CHANGE_FIELD_TOL:g} rather than 0. (4) A LABELLING "
        f"difference that is no error at all: on {n['long_gap']:,} reports the previous one "
        f"is over a week back, and on {n['long_gap_same_base']:,} of those the published "
        "change is measured against exactly that report -- it agrees with our diff while "
        "covering several weeks, and our own labelled 1-week flow is deliberately blank "
        "there. (5) The July 2008 trader reclassification, where the published change "
        "applies the NEW classification against an OLD-classification prior level: legacy "
        "futures-and-options crude 067651 differs by 323,944 contracts on 2008-07-15 "
        "(commercial short), legacy futures-only by 147,755, and the disaggregated report "
        "-- on the new classification throughout -- by 0."
    )

    caveat_block(source_id)


BOARD = Board(
    id="flows",
    title="Flows",
    render=_render,
    sources=("cftc_tff_fut",),
    order=30,
    blurb="Who moved this week, and the constraint that every position has two sides.",
)
