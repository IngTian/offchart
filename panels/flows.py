"""Flows -- what changed this week, and the invariants that constrain it.

Levels say where positioning sits; this board is the week-over-week delta. It
exists mainly because the flow side of this dataset carries two free
cross-checks that the level side does not, and both are run on screen:

  1. COHORT NET FLOWS SUM TO ~0, because every contract has a long and a short.
     This catches a dropped cohort, a field mapped to the wrong column and a
     misaligned join, all of which produce individually plausible series.
     NET_ZERO_TOL is the tolerance on a LEVEL; a flow differences two
     independently rounded levels, so it carries about twice that.
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
a contract re-basing: a 4-week flow spanning 2023-05-02 in 20974+ would read
~+200,000 contracts of "buying" that is entirely a 5x unit change. So flows here
are additionally nulled wherever the window crosses a lib.segments boundary, and
the boundaries are drawn.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from panels import Board

_HORIZONS = (1, 2, 4, 13)
_HISTORY_DAYS = 730
#: A flow is a difference of two independently rounded levels.
_FLOW_TOL = 2.0 * cftc_spec.NET_ZERO_TOL
_UNIT = "contracts"


def _app_helpers():
    """app.py's shared renderers, obtained without re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not also alias it as
    `app`, so the documented `from app import caveat_block` imports a SECOND copy
    and re-runs app.py's module body -- including its sidebar radio, which dies
    with StreamlitDuplicateElementId before either helper is bound. Verified: with
    the plain import this board raises on first render.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


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
    interest by the cohort count.
    """
    sub = _one_market(source_id, code)
    sub["net"] = _f(sub["long"]) - _f(sub["short"])
    keys = ["report_date", "cohort"]
    net = sub.groupby(keys, observed=True)["net"].first().unstack("cohort").sort_index()
    meta = sub.groupby("report_date", observed=True)[
        ["open_interest", "contract_units"]
    ].first().sort_index()
    meta["open_interest"] = _f(meta["open_interest"])
    return net, meta


def _flows(
    net: pd.DataFrame, meta: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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
    oi_flow = metrics.flow(meta["open_interest"], dates, horizon_weeks=horizon).where(same)
    return flows, oi_flow, seg


def _reconcile(source_id: str, code: str) -> tuple[pd.DataFrame, int, int]:
    """Our prior-report diff against CFTC's published change_* columns.

    Deliberately `.diff(1)` -- the PRIOR REPORT -- not a calendar-anchored 1-week
    flow, because that is what the published column means. Where the prior report
    is two weeks back the published "change" is a two-week change; that is a
    labelling difference, not a disagreement, and it is counted separately.
    """
    sub = _one_market(source_id, code)

    def rec(field, who, dates, level, published) -> pd.DataFrame:
        err = (_f(level).diff(1) - _f(published)).abs()
        return pd.DataFrame({"field": field, "who": who, "date": dates, "err": err.to_numpy()})

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
        rows.append(
            {
                "field": field,
                "comparisons": len(g),
                f"agree within {tol:g}": f"{100.0 * (g['err'] <= tol).mean():.3f}%",
                "worst gap": f"{g['err'].max():,.0f}",
                "worst on": f"{pd.Timestamp(worst['date']).date()}  {worst['who']}",
            }
        )
    weeks = segments.weeks_elapsed(pd.Series(one.index, index=one.index))
    return pd.DataFrame(rows), int((weeks > 1).sum()), int(len(one))


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

    total = float(usable.sum())
    side.markdown(
        f"**{len(usable)} of {len(row)} cohort net flows sum to "
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
    elif abs(total) <= _FLOW_TOL:
        side.caption(
            f"Invariant holds -- to within about {_FLOW_TOL:g} contracts, not exactly: "
            f"NET_ZERO_TOL is {cftc_spec.NET_ZERO_TOL:g} on a level and a flow differences "
            "two independently rounded levels, so it carries twice the rounding."
        )
    else:
        side.error(
            f"{total:,.0f} is outside the {_FLOW_TOL:g}-contract rounding allowance, which "
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
    caveat_block, freshness = _app_helpers()

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

    net, meta = _market_frame(source_id, code)
    if net.empty or len(net) <= horizon:
        st.info("Too few reports on this code to compute a flow at this horizon.")
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

    freshness(net.index, "CFTC report")
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

    flows, oi_flow, seg = _flows(net, meta, horizon)
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
        oi_chg = charts.Series(
            oi_flow.loc[win], "Open interest change", kind="flow", color=charts.WARN
        )
        oi_lvl = charts.Series(
            meta["open_interest"].loc[win], "Open interest",
            kind="open_interest", color=charts.MUTED,
        )
        shared = dict(unit=_UNIT, breaks=breaks)
        st.plotly_chart(
            charts.stacked([
                charts.Panel(
                    cohort_series, title=f"Cohort net flow, {horizon}-week", height=1.35, **shared
                ),
                charts.Panel([oi_chg], title=f"Open interest change, {horizon}-week", **shared),
                charts.Panel([oi_lvl], title="Open interest, level", height=0.8, **shared),
            ])
        )
        st.caption(
            "Read the top two panels together -- that is the whole reason open interest "
            "stays on screen. A cohort adding while open interest RISES is new contracts "
            "created against someone else's new short; the same cohort adding while open "
            "interest is FLAT is contracts changing hands, and the other side is another "
            "line on this chart. Blanks are deliberate: a flow is absent wherever the "
            f"window does not span exactly {horizon} week{'s' if horizon > 1 else ''} or "
            "crosses a dotted break (a re-basing, or a hole in this code's history)."
        )

    st.subheader("Reconciliation against the published change columns")
    table, multi_week, n_reports = _reconcile(source_id, code)
    st.dataframe(table, hide_index=True)
    st.caption(
        f"Our own `.diff(1)` against CFTC's published change_* columns for this code over "
        f"all {n_reports:,} of its reports. A mismatch is not automatically our bug, and "
        "the three legitimate reasons are named rather than hand-waved. (1) The published "
        f"change compares against the PRIOR REPORT, more than a week earlier on "
        f"{multi_week:,} of this code's reports -- there the published figure is a "
        "multi-week change while our labelled 1-week flow is deliberately blank. (2) Each "
        f"column is rounded independently, hence a tolerance of {cftc_spec.CHANGE_FIELD_TOL:g} "
        "rather than 0. (3) The July 2008 trader reclassification, where the published "
        "change applies the NEW classification against an OLD-classification prior level: "
        "crude 067651 differs by 323,944 contracts on 2008-07-15 in legacy futures-and-"
        "options (commercial short) and by 147,755 in legacy futures-only, natural gas "
        "023651 by 7,705. The disaggregated report, published on the new classification "
        "throughout, has zero mismatches on either code."
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
