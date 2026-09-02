"""Market -- one market code in full: what a cohort holds, how unusual that is,
and whether the open interest underneath it was created or merely changed hands.

The drill-down. What it answers: what the selected cohorts hold, in contracts or
as a share of the DIRECTIONAL open interest (ex-spreads) that can take a side;
whether that level is unusual against its own past, in two windows -- trailing
three years and expanding-within-segment -- which disagree exactly when the
market's behaviour has changed regime; whether the position moved because the
cohort traded or because open interest moved underneath it; and where this code's
history is cut, by a hole or by a contract re-specification.

What it refuses:

  - anything forward-looking. There are no prices in this repo, so no return, no
    hit rate, no "extremes mark turns". A cohort's net is contemporaneous with
    the price it was measured against, and a board with no price series cannot
    say what happened next.
  - a percent change on a net. -100,640 -> -96,727 is "+3,913 contracts, less
    short"; "+3.89%" reads as growth while the exposure shrank toward zero.
  - a level compared across a segment boundary. 20974+ open interest went
    49,531 -> 255,954 on 2023-05-02 because CFTC re-based the contract from $100
    to $20 per index point. Nobody traded.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from panels import Board

#: A segment's first expanding percentile is 100.0 by construction (max of a
#: one-element set) and a re-basing restarts that warm-up mid-history, so blank
#: it. 26 reports is half a year of weeklies.
_MIN_EXPANDING = 26

#: Landing cohort per family -- the one whose net is conventionally read as
#: positioning. Never "all cohorts": every contract has two sides, so the sum
#: over all of them is zero by construction and carries no information.
_DEFAULT_COHORT = {"tff": "lev_money", "disagg": "m_money", "legacy": "noncomm", "supp": "cit"}

_CONTRACTS = "contracts"
_SHARE = "% of directional open interest"


def _app_helpers():
    """app.py's shared renderers, obtained without re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not also alias it as
    `app`, so a plain `from app import caveat_block` imports a SECOND copy and
    re-runs app.py's module body -- sidebar radio included, which then dies with
    StreamlitDuplicateElementId. Verified: this board raises on first render.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


def _f(s: pd.Series) -> pd.Series:
    """Nullable Int32/float32 -> float64, NA -> NaN, so plotly and rolling work."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _frame(source_id: str, code: str, cohort_ids: tuple[str, ...]) -> pd.DataFrame:
    """One row per report date: the selected cohorts' aggregate plus market columns.

    open_interest and contract_units are identical across the cohorts of a
    market-week, so they take first() -- summing them multiplies open interest by
    the cohort count, which is how a share of OI ends up five times too small. The
    spread total is the exception: a real sum, and over ALL cohorts rather than
    the selected ones, because directional OI belongs to the market.
    """
    df = cache.read(source_id)
    m = df[df["market_code"] == code]
    if m.empty:
        return pd.DataFrame()

    wk = m.groupby("report_date", observed=True).agg(
        open_interest=("open_interest", "first"),
        spread_market=("spread", "sum"),
        units=("contract_units", "first"),
    )
    sel = m[m["cohort"].isin(cohort_ids)]
    ag = sel.groupby("report_date", observed=True).agg(
        long=("long", "sum"), short=("short", "sum")
    )
    wk = wk.join(ag, how="left").sort_index()

    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(wk.index)))
    out["date"] = out.index
    for c in ("open_interest", "spread_market", "long", "short"):
        out[c] = _f(wk[c]).to_numpy()
    out["net"] = metrics.net(out["long"], out["short"])
    out["gross"] = metrics.gross(out["long"], out["short"])
    out["dir_oi"] = metrics.directional_oi(out["open_interest"], out["spread_market"])
    out["purity"] = metrics.directional_purity(out["net"], out["gross"])
    out["spread_share"] = metrics.spread_share(out["spread_market"], out["open_interest"])
    out["units"] = wk["units"].astype(object).to_numpy()
    out["segment"] = segments.segment_ids(out["date"], pd.Series(out["units"].to_numpy()))
    return out


def _ranks(f: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Trailing-3y and expanding percentiles, computed WITHIN each segment.

    lib.metrics makes both causal, but causality is not sufficient: ranking a
    $20-per-point observation against $100-per-point ones is a valid percentile
    of the wrong population.
    """
    trail = pd.Series(np.nan, index=f.index)
    expand = pd.Series(np.nan, index=f.index)
    for _, g in f.groupby(f["segment"].to_numpy(), sort=True):
        trail.loc[g.index] = metrics.trailing_percentile(g[col], g["date"])
        expand.loc[g.index] = metrics.expanding_percentile(g[col], min_history=_MIN_EXPANDING)
    return trail, expand


def _fmt_level(v: float, contracts: bool) -> str:
    if pd.isna(v):
        return "no data"
    return f"{v:,.0f}" if contracts else f"{v:,.2f}"


def _flow_metric(col, f: pd.DataFrame, name: str, kind: str, unit: str, weeks: int) -> None:
    """A change over exactly `weeks` weeks, or an explicit refusal.

    metrics.flow returns NaN when the rows do not span the claimed horizon, which
    happens on every skipped report -- and CFTC skips. A hole here is honest; a
    two-week change wearing a one-week label is not.
    """
    delta = metrics.flow(f[name], f["date"], horizon_weeks=weeks).iloc[-1]
    label = f"{weeks}-week change"
    if pd.isna(delta):
        spanned = metrics.weeks_between(f["date"], weeks).iloc[-1]
        col.metric(
            label,
            "no span",
            help=(
                f"The last {weeks + 1} reports span {spanned:,.0f} weeks, not {weeks}, so "
                "this change does not exist. Showing it anyway would mislabel the horizon "
                "-- the error that survives review because the number looks fine."
            ),
        )
        return
    prev = f[name].iloc[-1 - weeks]
    col.metric(
        label,
        charts.fmt_change(float(delta), kind, unit),
        help=(
            f"{charts.signed_direction(prev, f[name].iloc[-1]) or 'no prior level'}. "
            "Absolute units, never a percent: a ratio on a sign-changing quantity "
            "inverts its own sense somewhere in its range."
        ),
    )


def _cohort_table(source_id: str, code: str, spec: cftc_spec.ReportSpec) -> None:
    """Every cohort for the latest week, with the trader-count coverage stated.

    Per-cohort trader counts are null while the position is nonzero on a growing
    share of rows (4.2% of 2015, 28.7% of 2026), so a per-trader ranking silently
    omits about a third of any cross-section. The blank count makes that visible.
    """
    m = cache.read(source_id)
    m = m[m["market_code"] == code]
    rows = m[m["report_date"] == m["report_date"].max()].set_index("cohort")
    present = [c for c in spec.cohorts if c.id in rows.index]
    rows = rows.loc[[c.id for c in present]]

    oi = float(rows["open_interest"].iloc[0])
    lo, sh, spread = _f(rows["long"]), _f(rows["short"]), _f(rows["spread"])
    dir_oi = oi - float(spread.fillna(0).sum())
    net = metrics.net(lo, sh)
    # Share of the SIDE total, the denominator CFTC itself uses. Dividing by gross
    # open interest understates the tilt by 1/(1 - spread share): 1.8x in 3-month SOFR.
    net_share = metrics.share_of(net, pd.Series(dir_oi, index=net.index))
    per_trader_long = metrics.avg_position_per_trader(lo, rows["traders_long"])
    per_trader_short = metrics.avg_position_per_trader(sh, rows["traders_short"])
    table = pd.DataFrame(
        {
            "cohort": [c.label for c in present],
            "long": lo.to_numpy(),
            "short": sh.to_numpy(),
            "spread": spread.to_numpy(),
            "net": net.to_numpy(),
            "gross": metrics.gross(lo, sh).to_numpy(),
            "net % of dir. OI": net_share.to_numpy(),
            "traders long": _f(rows["traders_long"]).to_numpy(),
            "traders short": _f(rows["traders_short"]).to_numpy(),
            "avg long / trader": per_trader_long.to_numpy(),
            "avg short / trader": per_trader_short.to_numpy(),
        }
    )
    st.dataframe(
        table.round({"net % of dir. OI": 2, "avg long / trader": 0, "avg short / trader": 0}),
        hide_index=True,
    )

    pub = [c.id for c in present if c.traders_long is not None]
    blank = int((rows.loc[pub, "traders_long"].isna() & (lo.loc[pub] != 0)).sum())
    st.caption(
        f"Per-trader coverage: {len(pub) - blank} of {len(pub)} cohorts that publish a "
        f"trader count have one this week, {blank} blank while holding a nonzero position "
        "-- a real number over an unknown, not a 0/0. Non-reportables publish no trader "
        "count in any family, so their blank is structural, as is a blank spread (legacy "
        "commercials, disagg producer/merchants, non-reportables)."
    )
    st.caption(
        f"Cohort nets sum to {float(net.sum()):+,.0f} contracts on {dir_oi:,.0f} "
        "directional. Two sides to every contract, so this is zero up to the publisher's "
        f"independent rounding of each column: within {cftc_spec.NET_ZERO_TOL:.0f} on the "
        "futures-only reports, wider on 52.6% of supplemental rows. Rounding, not a break."
    )
    resid = oi - float(lo.sum()) - float(spread.fillna(0).sum())
    if resid != 0:
        st.caption(
            f"Open interest minus sum(long) minus sum(spread) = {resid:+,.0f}. "
            + cftc_spec.RESIDUAL_EXPLANATION
        )


def _segments_table(f: pd.DataFrame) -> None:
    """Where the history was cut and why, with contract_units on each side."""
    seg = segments.describe_segments(f["date"], pd.Series(f["units"].to_numpy()))
    why = ["start of this code's history"]
    for i in range(1, len(seg)):
        prev, cur = seg.iloc[i - 1], seg.iloc[i]
        gap = (pd.Timestamp(cur["start"]) - pd.Timestamp(prev["end"])).days
        bits = []
        if cur["unit_signature"] != prev["unit_signature"]:
            bits.append(f"contract units {prev['units']} -> {cur['units']}")
        if gap > segments.MAX_GAP_DAYS:
            bits.append(f"{gap:,}-day hole")
        why.append("; ".join(bits) or f"{gap}-day gap")
    st.dataframe(seg.assign(**{"cut because": why}), hide_index=True)
    if len(seg) == 1:
        st.caption(
            "One segment: no contract re-specification and no hole longer than "
            f"{segments.MAX_GAP_DAYS} days, so the whole history is comparable with itself."
        )
    else:
        st.caption(
            f"{len(seg)} segments, so a level in one is not comparable with a level in "
            "another and no percentile above spans a break. The signature is the DIGITS "
            "of contract_units, which is what changes the arithmetic -- a cosmetic rename "
            "does not cut the series, and 5,000 bushels versus thousand bushels does."
        )


def _market_label(r) -> str:
    where = r.subgroup if isinstance(r.subgroup, str) else r.group
    return f"{r.market} ({r.market_code}) -- {where if isinstance(where, str) else 'ungrouped'}"


def _render() -> None:
    caveat_block, freshness = _app_helpers()

    top = st.columns([3, 2])
    spec = top[0].selectbox(
        "Report family",
        cftc_spec.SPECS,
        format_func=lambda s: s.label,
        help=(
            "legacy, disaggregated and TFF are different cohort vocabularies over "
            "overlapping markets, not versions of one series: 'commercial' is not "
            "'producer + swap', and futures-only never reconciles against futures+options."
        ),
    )
    summary = cache.market_summary(spec.id)
    if summary.empty:
        st.info(f"`{spec.id}` has no ingested rows yet. Run `python -m scripts.ingest`.")
        return

    counts = summary["tier"].value_counts()
    tier = top[1].radio(
        "Liquidity tier",
        ("CORE", "WIDE", "THIN"),
        horizontal=True,
        format_func=lambda t: f"{t} ({int(counts.get(t, 0))})",
        help=(
            "Tier is the trailing 26-report MAXIMUM open interest, not the latest report: "
            "markets drop out of single weeks and come back, and a latest-report rule "
            "churns membership 5x faster for nothing. THIN is selectable but ranks noise."
        ),
    )
    pool = summary[summary["tier"] == tier]
    if pool.empty:
        st.info(f"No {tier} markets in {spec.label}.")
        return

    codes = list(pool["market_code"])
    labels = {r.market_code: _market_label(r) for r in pool.itertuples()}
    default = universe.default_market(pool, prefer=("20974+",))
    code = st.selectbox(
        "Market",
        codes,
        index=codes.index(default) if default in codes else 0,
        format_func=lambda c: labels[c],
        help=(
            "Selected by market_code, the stable key. 26-30% of codes have been renamed "
            "at least once and CFTC shortened names wholesale on 2022-02-08, so the label "
            "is the LATEST name and nothing here selects on a name."
        ),
    )
    row = pool[pool["market_code"] == code].iloc[0]

    if code in universe.CONSOLIDATED:
        st.caption(
            f"**{universe.CONSOLIDATED[code]}** is the notional sum, E-mini + Micro/10: "
            "you are taking exposure correctness over history. "
            + universe.CONSOLIDATED_TRADEOFF
        )
    elif code in universe.SUPERSEDED_BY:
        parent = universe.SUPERSEDED_BY[code]
        st.caption(
            f"A single leg, already inside {universe.CONSOLIDATED[parent]} (`{parent}`) "
            "at 10:1 -- never add it to that series or to the other leg by contract "
            "count. You are taking history over exposure correctness. "
            + universe.CONSOLIDATED_TRADEOFF
        )

    ctl = st.columns([3, 1, 2])
    all_ids = tuple(c.id for c in spec.cohorts)
    label_of = {c.id: c.label for c in spec.cohorts}
    landing = _DEFAULT_COHORT.get(spec.family, all_ids[0])
    cohort_ids = ctl[0].multiselect(
        "Cohorts (summed)",
        all_ids,
        default=[landing if landing in all_ids else all_ids[0]],
        format_func=lambda c: label_of[c],
        help=(
            "A cohort net is a sum over firms running incompatible strategies: a basis "
            "trader long cash and short futures appears here as short while holding no "
            "view at all. Summing cohorts is arithmetic, not aggregation of intent."
        ),
    )
    measure = ctl[1].radio("Measure", ("net", "gross"), horizontal=True)
    unit = ctl[2].radio("Unit", (_CONTRACTS, _SHARE), horizontal=True)
    if not cohort_ids:
        st.info("Select at least one cohort.")
        return
    if len(cohort_ids) == len(all_ids) and measure == "net":
        st.caption(
            "Every cohort selected: their nets sum to zero by construction, so the top "
            f"panel sits on the axis to within {cftc_spec.NET_ZERO_TOL:.0f} contracts. "
            "Switch to gross, or deselect, for anything informative."
        )

    f = _frame(spec.id, code, tuple(cohort_ids))
    if f.empty or f[measure].notna().sum() == 0:
        st.info(f"No rows for `{code}` with those cohorts in {spec.label}.")
        return

    contracts = unit == _CONTRACTS
    if contracts:
        f["display"] = f[measure]
        kind, unit_label = measure, "contracts"
    else:
        f["display"] = metrics.share_of(f[measure], f["dir_oi"])
        kind = "net_share" if measure == "net" else "gross_share"
        unit_label = "% of directional OI"

    trail, expand = _ranks(f, "display")
    breaks = tuple(f.loc[f["segment"] != f["segment"].shift(), "date"].iloc[1:].to_numpy())
    cohorts_txt = " + ".join(label_of[c] for c in cohort_ids)

    # Each measure gets its own panel because each has its own unit. No secondary
    # axis exists in lib.charts and none is wanted: two scales on one frame let
    # the author pick the scaling that makes a correlation look how they like.
    panels = [
        charts.Panel(
            series=[charts.Series(f["display"], f"{measure}, {cohorts_txt}", kind=kind)],
            unit=unit_label,
            title=f"{measure.capitalize()}, {cohorts_txt} -- {row['market']} ({code})",
            height=1.3,
        ),
        charts.Panel(
            series=[
                charts.Series(trail, "trailing 3 years", kind="share"),
                charts.Series(expand, "expanding, within segment", kind="share", dash="dot"),
            ],
            unit="percentile",
            title="Causal rank of the line above -- each point against its own past only",
            guides=(10.0, 90.0),
            y_range=(0.0, 100.0),
        ),
        charts.Panel(
            series=[
                charts.Series(f["open_interest"], "open interest", kind="open_interest"),
                charts.Series(
                    f["dir_oi"], "directional, ex-spreads", kind="open_interest", dash="dot"
                ),
            ],
            unit="contracts",
            title="Open interest -- a share rises if the cohort bought OR if this fell",
        ),
        charts.Panel(
            series=[charts.Series(f["spread_share"], "spread share of OI", kind="share")],
            unit="% of OI",
            title="Calendar structure, not direction: the gap between the two lines above",
            height=0.7,
        ),
        charts.Panel(
            series=[charts.Series(f["purity"], "net / gross", kind="purity")],
            unit="net / gross",
            y_range=(-1.0, 1.0),
            title="Directional purity: one-way book (+-1) or big two-sided book (0)",
            height=0.8,
        ),
    ]
    for p in panels:  # every panel, not just the top one: a break invalidates all of them
        p.breaks = breaks
    st.plotly_chart(charts.stacked(panels, height_per_panel=185))
    st.caption(
        "Dotted verticals are segment boundaries -- a contract re-specification or a hole "
        f"longer than {segments.MAX_GAP_DAYS} days. Levels compare only WITHIN a segment "
        f"and both percentiles restart at each line (expanding blank for {_MIN_EXPANDING} "
        "reports, trailing until a 3-year window is 60% occupied). In contracts a re-basing "
        "destroys the rank and only segmentation saves it; as a share of directional OI the "
        "rank is scale-free and survives a re-basing -- but nothing survives a change in "
        "what the cohort means, which is why families are not spliced."
    )

    last = f.iloc[-1]
    st.subheader(f"Reading for {last['date'].date()}")
    cols = st.columns(5)
    joiner = "minus" if measure == "net" else "plus"
    over = "" if contracts else f", over directional open interest {last['dir_oi']:,.0f}"
    seg_n = int((f["segment"] == last["segment"]).sum())
    readings = (
        (
            f"{measure.capitalize()}, {unit_label}",
            _fmt_level(last["display"], contracts),
            f"long {last['long']:,.0f} {joiner} short {last['short']:,.0f}{over}. Spread "
            "positions are in neither column -- long one expiry, short another -- so they "
            "carry no direction and leave the denominator too.",
        ),
        (
            "Trailing 3y percentile",
            _fmt_level(trail.iloc[-1], False),
            "Rank within the last 1,095 days of this segment. Blank where that window is "
            "under 60% occupied: a coverage statement, not a zero.",
        ),
        (
            "Expanding percentile",
            _fmt_level(expand.iloc[-1], False),
            f"Rank against all {seg_n} reports of this segment, out of {len(f):,} for the "
            "code. Over decades this window fills with regimes that no longer exist, so "
            "read it against the trailing one; disagreement is the information.",
        ),
    )
    for col, (lbl, val, tip) in zip(cols, readings):
        col.metric(lbl, val, help=tip)
    _flow_metric(cols[3], f, "display", kind, unit_label, 1)
    _flow_metric(cols[4], f, "display", kind, unit_label, 4)

    st.subheader("Every cohort, latest report")
    _cohort_table(spec.id, code, spec)

    st.subheader("Segments of this code's history")
    _segments_table(f)

    freshness(f["date"], f"{spec.label} report")
    caveat_block(spec.id)


BOARD = Board(
    id="market",
    title="Market",
    render=_render,
    sources=("cftc_tff_fut",),
    order=20,
    blurb=(
        "One market in full -- positioning, its rank in history, and the open interest "
        "that says whether contracts were created or changed hands."
    ),
)
