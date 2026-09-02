"""Base rates: does an extreme reading actually precede anything -- tested, not asserted.

Every other board here reports a LEVEL. This one asks whether a level is worth
conditioning on: it buckets twenty years of readings by how extreme each one was AT
THE TIME and shows the distribution of what happened next.

WHAT IT REFUSES TO ANSWER: FORWARD RETURNS. There is no price series in this repo,
so the claim COT commentary actually makes -- positioning is stretched, therefore the
market is due a reversal -- is not tested anywhere on this page and cannot be with
this data. What positioning alone CAN answer is whether an extreme reading precedes a
change in the cohort's own net, in open interest, in the cohort's share of
directional open interest, or in the reading itself. That is a question about
traders' inventories, not about prices, and the two get conflated constantly.
Extending it to returns needs one thing: a futures settlement series keyed to
market_code, the highest-value planned source in sources/__init__.py.

FOUR WAYS A BASE-RATE TABLE LIES, AND WHAT STOPS EACH HERE

1. Look-ahead in the conditioning variable -- a full-sample percentile knows where a
   level sat relative to data that did not exist yet. Only lib.metrics' causal
   statistics are offered.
2. Look-ahead in the bucket EDGES: the same error one level up, and the easier to
   miss, because fitting quintile boundaries on the sample lets the outcomes decide
   what counts as extreme. The edges here are FIXED CONSTANTS -- 10/25/75/90
   percentile points, +/-0.5 and +/-1.5 sigma -- read off no sample at all.
3. A mislabelled horizon: CFTC skips weeks, so a 13-row shift is thirteen REPORTS.
   Every outcome goes through metrics.flow, which blanks a pair whose calendar span is
   wrong, plus a segment guard for what the calendar cannot see -- a re-basing with no
   gap in the dates.
4. Overlap counted as independence. Weekly reads at 13 weeks share 12 of 13 weeks, so
   n sits beside a count of non-overlapping blocks and no p-value is computed at all.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from panels import Board

WINDOW_DAYS = 1095  # restated only to compute how much of it a horizon overlaps

PCT_EDGES = (10.0, 25.0, 75.0, 90.0)
PCT_LABELS = ("< 10", "10 - 25", "25 - 75", "75 - 90", ">= 90")
Z_EDGES = (-1.5, -0.5, 0.5, 1.5)
Z_LABELS = ("< -1.5", "-1.5 to -0.5", "-0.5 to +0.5", "+0.5 to +1.5", ">= +1.5")

STATS: dict[str, dict] = {
    "Trailing percentile, 3y window": dict(
        fn=lambda v, d: metrics.trailing_percentile(v, d, window_days=WINDOW_DAYS),
        unit="percentile", kind="share", edges=PCT_EDGES, labels=PCT_LABELS, windowed=True,
        note="Rank within the trailing three years: extreme for the current regime.",
    ),
    "Expanding percentile, all history": dict(
        fn=lambda v, d: metrics.expanding_percentile(v, min_history=52),
        unit="percentile", kind="share", edges=PCT_EDGES, labels=PCT_LABELS, windowed=False,
        note="Rank against every prior observation in the segment. Not stationary: a late "
             "observation must beat twenty years to score 90 and an early one must beat one, "
             "so the tail buckets are over-populated early in a series.",
    ),
    "COT index, 3y min-max": dict(
        fn=lambda v, d: metrics.cot_index(v, d, window_days=WINDOW_DAYS),
        unit="index", kind="share", edges=PCT_EDGES, labels=PCT_LABELS, windowed=True,
        note="What published commentary means by 'positioning is at 90'. Here so that "
             "commentary can be checked against a base rate, not because it is better: one "
             "old extreme pins the denominator for three years.",
    ),
    "Trailing z-score, 3y window": dict(
        fn=lambda v, d: metrics.trailing_zscore(v, d, window_days=WINDOW_DAYS),
        unit="sigma", kind="change", edges=Z_EDGES, labels=Z_LABELS, windowed=True,
        note="Keeps resolution in the tail, where a rank saturates at 0 and 100.",
    ),
}

OUTCOMES: dict[str, dict] = {
    "Cohort net position, change": dict(
        col="net", unit="contracts", kind="change", mode="change",
        gloss="negative means the cohort ended less long or more short, in contracts"),
    "The conditioning reading itself, level": dict(
        col="stat", mode="level",
        gloss="persistence: is extreme a state that decays or a regime that sits"),
    "Open interest, change": dict(
        col="oi", unit="contracts", kind="change", mode="change",
        gloss="whether extremes precede the market growing or shrinking"),
    "Cohort share of directional OI, change": dict(
        col="net_share", unit="pp of directional OI", kind="change", mode="change",
        gloss="net over open interest excluding spreads, in percentage points"),
}

HORIZONS = (1, 4, 13, 26, 52)

#: The cohort each family's commentary is actually about.
DEFAULT_COHORT = {"tff": "lev_money", "disagg": "m_money", "legacy": "noncomm", "supp": "cit"}


def _app_helpers():
    """app.py's shared renderers, without re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not alias it in sys.modules
    as `app`, so the documented `from app import caveat_block` imports a SECOND copy of
    app.py and runs its module body again -- sidebar radio included, which raises
    StreamlitDuplicateElementId before either name binds. Measured: this board died on
    first render with the plain import.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


def _prepare(source_id: str, codes: list[str], cohorts: list[str], stat: dict) -> pd.DataFrame:
    """Per (market, cohort) history with a causal conditioning statistic attached.

    Deliberately not memoised: the parquet read is already cached in lib.cache and the
    rest measures 0.4-0.7s on the widest selection offered, so a second cache would
    buy little and would need a file-version key only lib.store can produce -- one
    keyed on the arguments alone serves stale numbers for the session after an ingest.
    """
    df = cache.read(source_id)
    sel = df[df["market_code"].isin(codes)]
    # Spread total is the MARKET's, over all cohorts, not the selected cohort's:
    # directional open interest is a market-level denominator, and legacy comm, disagg
    # prod_merc and every nonrept publish no spread column at all.
    spread_total = (
        sel.assign(_sp=sel["spread"].astype("float64").fillna(0.0))
        .groupby(["market_code", "report_date"], observed=True)["_sp"]
        .sum()
    )
    keep = ["report_date", "market_code", "cohort", "contract_units", "long", "short",
            "open_interest"]
    d = sel[sel["cohort"].isin(cohorts)][keep].copy()
    pair = pd.MultiIndex.from_arrays([d["market_code"], d["report_date"]])
    d["spread_total"] = pair.map(spread_total)
    d["report_date"] = pd.to_datetime(d["report_date"])
    d = (d.sort_values(["market_code", "cohort", "report_date"], kind="mergesort")
          .reset_index(drop=True))

    d["net"] = metrics.net(d["long"], d["short"]).astype("float64")
    d["oi"] = d["open_interest"].astype("float64")
    doi = metrics.directional_oi(d["oi"], d["spread_total"])
    d["net_share"] = metrics.share_of(d["net"], doi)

    segs, stats = [], []
    for _, g in d.groupby(["market_code", "cohort"], observed=True, sort=False):
        seg = segments.segment_ids(g["report_date"], g["contract_units"])
        segs.append(seg)
        # Within segment, so a reading never ranks $20-per-point observations
        # against $100-per-point ones across a re-basing.
        stats.append(
            pd.concat([stat["fn"](g["net"].loc[i], g["report_date"].loc[i])
                       for _, i in seg.groupby(seg).groups.items()]).reindex(g.index)
        )
    d["segment"] = pd.concat(segs).reindex(d.index)
    d["stat"] = pd.concat(stats).reindex(d.index)
    return d


def _forward(d: pd.DataFrame, col: str, weeks: int) -> pd.Series:
    """Change in `col` over the NEXT `weeks` calendar weeks, NaN if not exactly that.

    metrics.flow looks backwards and blanks a wrong span; shifting its result up by
    the same number of rows turns it into a forward outcome and keeps the check.
    """
    out = []
    for _, g in d.groupby(["market_code", "cohort"], observed=True, sort=False):
        f = metrics.flow(g[col], g["report_date"], horizon_weeks=weeks).shift(-weeks)
        out.append(f.where(g["segment"].shift(-weeks) == g["segment"]))
    return pd.concat(out).reindex(d.index)


def _blocks(dates: pd.Series, weeks: int) -> pd.Series:
    """Which non-overlapping `weeks`-wide block of the sample each date falls in."""
    return (dates - dates.min()) // pd.Timedelta(days=7 * weeks)


def _bucket_table(ok: pd.DataFrame, stat: dict, weeks: int, hit_at: float | None) -> pd.DataFrame:
    """One row per fixed bucket: counts, non-overlapping counts, quantiles, hit rate."""
    block = _blocks(ok["report_date"], weeks)
    rows = []
    for label in stat["labels"]:
        m = ok[ok["bucket"] == label]
        if m.empty:
            rows.append({"bucket": label, "n": 0})
            continue
        q = m["fwd"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        hit = (m["fwd"] >= hit_at).mean() if hit_at is not None else (m["fwd"] < 0).mean()
        rows.append({
            "bucket": label, "n": len(m),
            "blocks": len(set(zip(m["market_code"], m["cohort"], block.loc[m.index]))),
            "p10": q[0.10], "p25": q[0.25], "median": q[0.50], "p75": q[0.75], "p90": q[0.90],
            "iqr": q[0.75] - q[0.25], "hit": 100.0 * hit, "oi_median": m["fwd_oi"].median(),
        })
    return pd.DataFrame(rows)


def _context_chart(g: pd.DataFrame, stat: dict, name: str) -> None:
    """The selected market's own history, so the buckets are not just a table.

    Open interest is on screen because one outcome is a share of it, and because an
    extreme net at flat open interest is not the same event as one at collapsing OI.
    """
    idx = g["report_date"]
    seg_desc = segments.describe_segments(g["report_date"], g["contract_units"])
    breaks = tuple(pd.to_datetime(seg_desc["start"])[1:])

    def panel(col, label, kind, unit, title, guides=()):
        return charts.Panel(
            series=[charts.Series(pd.Series(g[col].to_numpy(), index=idx), label, kind=kind)],
            unit=unit, title=title, guides=guides, breaks=breaks)

    st.plotly_chart(charts.stacked([
        panel("net", "net", "net", "contracts", f"{name} — cohort net"),
        panel("stat", "reading", stat["kind"], stat["unit"],
              "conditioning reading (causal, within segment)", tuple(stat["edges"])),
        panel("oi", "open interest", "open_interest", "contracts", "open interest"),
    ], height_per_panel=165), use_container_width=True)
    st.caption(
        "Dotted verticals are segment breaks: a contract re-basing, or a hole longer than 91 "
        "days. Readings and outcomes are computed inside a segment, never across one. "
        "Horizontal guides are the fixed bucket edges."
    )


def _fmt(v: float, kind: str, unit: str, level: bool) -> str:
    return f"{v:,.1f} {unit}" if level else charts.fmt_change(v, kind, unit)


def _wk(weeks: int) -> str:
    return "1 week" if weeks == 1 else f"{weeks} weeks"


def _verdict(tbl: pd.DataFrame, outcome: dict, edges: tuple, unit: str, kind: str,
             weeks: int, iqr: float) -> None:
    """Blunt statement of what the table in front of the user actually supports."""
    live = tbl[tbl["n"] > 0]
    if len(live) < 2 or not np.isfinite(iqr) or iqr <= 0:
        st.warning("Too few usable observations in the extreme buckets to support any verdict.")
        return
    level = outcome["mode"] == "level"
    lo, hi = live.iloc[0], live.iloc[-1]
    gap = float(lo["median"] - hi["median"])
    ratio = gap / iqr

    st.markdown(
        f"**Verdict at {_wk(weeks)}.** Median forward outcome "
        f"**{_fmt(float(hi['median']), kind, unit, level)}** after a reading in `{hi['bucket']}` "
        f"(n={int(hi['n']):,}, {int(hi['blocks']):,} non-overlapping blocks), versus "
        f"**{_fmt(float(lo['median']), kind, unit, level)}** after `{lo['bucket']}` "
        f"(n={int(lo['n']):,}, {int(lo['blocks']):,} blocks). The two medians sit "
        f"{abs(gap):,.1f} {unit} apart against a pooled inter-quartile range of "
        f"{iqr:,.1f} {unit} — **{abs(ratio):.2f} IQRs**."
    )
    if level:
        # Sticky or not is read off the number, not asserted: compare the top bucket's
        # median forward reading against that bucket's own floor and the scale midpoint.
        floor, mid = float(edges[-1]), (float(edges[1]) + float(edges[2])) / 2
        st.markdown(
            f"{float(hi['hit']):.0f}% of top-bucket observations were still above {floor:g} "
            f"{_wk(weeks)} later, and their median reading was {float(hi['median']):,.1f} against "
            f"a scale midpoint of {mid:g}. "
            + ("So an extreme is closer to a persistent regime than a self-correcting state — "
               "the opposite of what 'stretched positioning' is usually taken to imply."
               if float(hi["median"]) >= float(edges[2]) else
               "So the extreme washes out rather than persisting: the reading decays back toward "
               "the middle of its own range over this horizon.")
        )
    elif abs(ratio) < 0.25:
        st.markdown(
            "**That is inside the noise.** A median difference under a quarter of one IQR, with "
            "the two buckets' p10-p90 ranges almost entirely overlapping, is not something to "
            "condition on at this sample size."
        )
    elif ratio > 0:
        st.markdown(
            f"**The extreme does revert on the median** — high readings are followed by falls, "
            f"low ones by rises. Mind the magnitude: a {abs(ratio):.2f}-IQR tilt leaves the sign "
            f"of any single outcome unsettled ({float(hi['hit']):.0f}% of top-bucket observations "
            "were negative, against 50% for a coin)."
        )
    else:
        st.markdown(
            f"**The extreme EXTENDS rather than reverting** — high readings are followed by "
            f"higher ones, by {abs(ratio):.2f} IQRs on the median. A rule that faded a stretched "
            "reading would have been on the wrong side of the median here."
        )
    meds = pd.Series(live["median"].to_numpy())
    st.caption(
        "Bucket medians are monotone in the reading, which is stronger than the two extreme "
        "buckets alone: five ordered buckets agreeing is harder to get by luck."
        if bool(meds.is_monotonic_decreasing or meds.is_monotonic_increasing) else
        "Bucket medians are NOT monotone in the reading. Whatever the extremes show, the "
        "middle does not line up with it, which is what a spurious tail effect looks like."
    )


def render() -> None:
    caveat_block, freshness = _app_helpers()

    # Every widget below is KEYED: Streamlit derives an unkeyed widget's id from its
    # type, label and RENDERED options, so two boards with a same-shaped control share
    # one session-state entry. Measured -- the Screen board's family selector passes
    # source ids formatted to labels while this one passes the labels themselves, which
    # renders identically, so unkeyed this widget inherits 'cftc_tff_fut' from Screen,
    # a value not in its own option list. Market and cohort keys carry the family, so
    # switching family drops a selection that family has never heard of.
    labels = {s.label: s.id for s in cftc_spec.SPECS}
    c = st.columns([2.3, 1.3, 2.4, 1.0])
    source_id = labels[c[0].selectbox("Report family", list(labels), index=0, key="br_family")]
    spec = cftc_spec.spec_for(source_id)
    scope = c[1].radio("Scope", ("All CORE markets", "One market"), index=0, key="br_scope")

    summary = cache.market_summary(source_id)
    if summary.empty:
        st.info(f"`{source_id}` has no rows yet.")
        return
    pick = summary[summary["tier"].isin(["CORE", "WIDE"])]
    names = dict(zip(pick["market_code"], pick["market"]))
    opts = list(pick["market_code"])
    default = universe.default_market(summary, prefer=("13874+",))
    code = c[2].selectbox(
        "Market (used when scope is one market)", opts,
        index=opts.index(default) if default in opts else 0,
        format_func=lambda k: f"{names[k]} · {k}", key=f"br_market_{source_id}",
    )
    weeks = int(c[3].selectbox("Horizon, weeks", HORIZONS, index=2, key="br_weeks"))

    c2 = st.columns([2.4, 2.3, 2.3])
    cohort_names = {ch.id: ch.label for ch in spec.cohorts}
    picked = c2[0].multiselect(
        "Cohorts (pooled if several)", list(cohort_names),
        default=[DEFAULT_COHORT.get(spec.family, list(cohort_names)[0])],
        format_func=lambda k: cohort_names[k], key=f"br_cohorts_{spec.family}",
    )
    stat = STATS[c2[1].selectbox("Conditioning statistic, on the cohort net", list(STATS),
                                 index=0, key="br_stat")]
    oname = c2[2].selectbox("Forward outcome", list(OUTCOMES), index=0, key="br_outcome")
    outcome = OUTCOMES[oname]
    if not picked:
        st.info("Pick at least one cohort.")
        return

    pooled = scope.startswith("All")
    codes = list(summary.loc[summary["tier"] == "CORE", "market_code"]) if pooled else [code]
    d = _prepare(source_id, codes, picked, stat)
    if d.empty:
        st.info("No rows for that combination.")
        return
    freshness(d["report_date"], f"{spec.label} report")

    level = outcome["mode"] == "level"
    unit, kind = (stat["unit"], stat["kind"]) if level else (outcome["unit"], outcome["kind"])
    fwd = _forward(d, outcome["col"], weeks)
    d["fwd"] = (d["stat"] + fwd) if level else fwd
    d["fwd_oi"] = _forward(d, "oi", weeks)
    d["bucket"] = pd.cut(d["stat"], [-np.inf, *stat["edges"], np.inf],
                         labels=list(stat["labels"]), right=False)
    ok = d.dropna(subset=["stat", "fwd"])
    if ok.empty:
        st.warning(
            f"Nothing survives the {_wk(weeks)} span check here: every candidate pair either "
            "skips a report week or crosses a segment break."
        )
        return

    if not pooled:
        _context_chart(d[d["market_code"] == code], stat, names.get(code, code))
        if code in universe.CONSOLIDATED or code in universe.SUPERSEDED_BY:
            st.caption(universe.CONSOLIDATED_TRADEOFF)

    hit_at = float(stat["edges"][-1]) if level else None
    hit_col = f"share >= {stat['edges'][-1]:g}" if level else "share < 0"
    tbl = _bucket_table(ok, stat, weeks, hit_at)
    live = tbl[tbl["n"] > 0]

    rows = list(live.itertuples())
    st.plotly_chart(charts.ranked_bars(
        [f"{r.bucket}  (n={r.n:,})" for r in rows], [float(r.median) for r in rows],
        unit=f"median forward outcome — {unit}", kind=kind, height_per_bar=34,
        hover=[f"{r.blocks:,} non-overlapping blocks · p10 {r.p10:,.0f} · p90 {r.p90:,.0f} · "
               f"median forward OI change {r.oi_median:,.0f}" for r in rows],
    ), use_container_width=True)
    st.caption(
        f"Median only — read the table below for the spread, all of it in {unit}. "
        f"{outcome['gloss'].capitalize()}. Bucket edges are the fixed constants "
        f"{list(stat['edges'])} {stat['unit']}, not sample quantiles, so no boundary was fitted "
        "on the outcomes it separates; the reading is causal, ranking each date only against its "
        "own past inside its own segment. Forward open-interest change travels in every row "
        "because a share of open interest can move from either side of the ratio."
    )

    cols = {"bucket": ("bucket", "{}"), "n": ("n", "{:,}"), "blocks": ("non-overlap", "{:,}")}
    cols |= {q: (q, "{:,.1f}") for q in ("p10", "p25", "median", "p75", "p90", "iqr")}
    cols |= {"hit": (hit_col, "{:.0f}%"), "oi_median": ("median fwd OI change", "{:,.0f}")}
    st.dataframe(
        pd.DataFrame({name: live[src].map(fmt.format) for src, (name, fmt) in cols.items()}),
        hide_index=True, use_container_width=True,
    )

    iqr = float(ok["fwd"].quantile(0.75) - ok["fwd"].quantile(0.25))
    _verdict(tbl, outcome, stat["edges"], unit, kind, weeks, iqr)

    blocks = int(_blocks(ok["report_date"], weeks).nunique())
    nmkt = ok["market_code"].nunique()
    st.caption(
        f"**Overlap.** {len(ok):,} usable observations over {nmkt} market"
        f"{'' if nmkt == 1 else 's'} and {ok['report_date'].nunique():,} report dates. "
        + ("Consecutive reads do not overlap at a one-week horizon, so n is the sample size "
           "here. " if weeks == 1 else
           f"But consecutive weekly reads share {weeks - 1} of {weeks} weeks, so the effective "
           f"sample is nearer n/{weeks} ≈ {len(ok) // weeks:,} and a naive standard error is "
           f"optimistic by about sqrt({weeks}) = {weeks ** 0.5:.1f}x. ")
        + f"The non-overlap column counts distinct (market, cohort, {weeks}-week block) cells "
        f"and the sample spans only {blocks:,} such blocks in time. "
        + ("Markets pooled inside one block are NOT independent draws — a family's CORE list "
           "concentrates in a few subgroups that move together, and in tff that is mostly rate "
           "contracts — so treat the block count as an upper bound on independence. "
           if pooled else "")
        + "No p-value is offered: with overlap at every horizon past one week and "
        "cross-sectional correlation unmodelled, it would be the most confident number on the "
        "page and the least defensible."
    )
    if level and stat["windowed"]:
        shared = 100 * max(0.0, 1 - 7.0 * weeks / WINDOW_DAYS)
        st.caption(
            f"**This outcome is partly mechanical.** The readings at t and t+{_wk(weeks)} are "
            f"ranks over trailing windows sharing {shared:.0f}% of their span, so persistence "
            "of the READING overstates persistence of the position. "
            "The net-position outcome carries no such artefact: for a driftless random walk the "
            "next increment is independent of every function of the past, so a nonzero median by "
            "bucket there is not a construction effect."
        )
    st.caption(
        "**No prices, so no returns.** Every outcome here is positioning or open interest. "
        "Positioning is measured as contemporaneous with price rather than predictive and this "
        "repo carries no price series at all, so nothing above can be read as a forecast of a "
        "market. Testing the reversal claim needs a futures settlement series keyed to "
        "market_code — the highest-value planned source in sources/__init__.py."
    )
    st.caption(f"On the chosen statistic: {stat['note']}")
    caveat_block(source_id)


BOARD = Board(
    id="base_rates",
    title="Base rates",
    render=render,
    sources=("cftc_tff_fut",),
    order=50,
    blurb="does an extreme reading actually precede anything -- tested, not asserted.",
)
