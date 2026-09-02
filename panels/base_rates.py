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

FIVE WAYS A BASE-RATE TABLE LIES, AND WHAT STOPS EACH HERE

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
5. NO NULL. A number is only evidence if something with no structure in it would not
   produce the same number, and the loudest reading on this page -- an extreme percentile
   is still extreme a quarter later -- is one a DRIFTLESS RANDOM WALK reproduces at least
   as strongly, because a cohort net is close to a unit root. Measured on the pooled tff
   default at 13 weeks: the null's top-bucket median forward reading is 93 with 61% still
   at or above the 90th percentile, against 84 and 37% in the real data. So the null is
   simulated live -- through this board's own statistic, horizon and bucket edges, so it
   cannot drift away from the controls -- and printed next to the observed figure whenever
   the outcome is a function of the conditioning series itself (the reading's own level,
   or the net whose rank it is). IID noise is simulated beside it, because the usual
   explanation for the persistence, two ranking windows sharing 92% of their span, gives
   a top-bucket median forward reading of 50 and is therefore not the mechanism.
   A verdict is also refused outright below a minimum bucket size: one observation used
   to be enough to print a bolded 9.25-IQR claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from lib.ui import caveat_block, freshness, market_last_print
from panels import Board

WINDOW_DAYS = 1095  # the trailing window every windowed statistic below is given

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

#: `down` and `up` say what a negative and a positive forward move MEAN for this
#: quantity, in its own terms. They exist because one shared sentence -- "the extreme
#: reverts, high readings are followed by falls" -- was printed for all four outcomes,
#: including open interest, where the thing that fell was the size of the market and
#: nothing reverted at all.
OUTCOMES: dict[str, dict] = {
    "Cohort net position, change": dict(
        col="net", unit="contracts", kind="change", mode="change",
        down="the cohort ending less long or more short",
        up="the cohort ending more long or less short",
        gloss="negative means the cohort ended less long or more short, in contracts"),
    "The conditioning reading itself, level": dict(
        col="stat", mode="level", down="a lower reading", up="a higher reading",
        gloss="persistence: is extreme a state that decays or a regime that sits"),
    "Open interest, change": dict(
        col="oi", unit="contracts", kind="change", mode="change",
        down="the market shrinking", up="the market growing",
        gloss="whether extremes precede the market growing or shrinking"),
    "Cohort share of directional OI, change": dict(
        col="net_share", unit="pp of directional OI", kind="net_share", mode="change",
        down="the cohort holding a smaller share of directional OI",
        up="the cohort holding a larger share of directional OI",
        gloss="net over open interest excluding spreads, in percentage points"),
}

HORIZONS = (1, 4, 13, 26, 52)

#: A bucket must hold this much before any verdict is written off it. Both halves are
#: needed: 20 consecutive weekly reads of one market at a 52-week horizon are 20 rows
#: and one independent block. Measured before the floor existed: single-market 146LM3 at
#: 52 weeks printed a bolded 0.59-IQR reversion verdict and "100% of top-bucket
#: observations were negative" off n=1, and 344606 at 1 week printed 9.25 IQRs off one
#: observation in the low bucket while its own table showed the top bucket UP.
MIN_BUCKET_N = 20
MIN_BUCKET_BLOCKS = 3

#: Size of the simulated null. Series count is fixed; the length tracks the sample.
#: Simulated on every new selection rather than quoted from a measurement, so the null
#: cannot drift away from the statistic, horizon and edges actually on screen.
NULL_SERIES = 24
NULL_TRIALS = 3
NULL_MIN_POINTS, NULL_MAX_POINTS = 400, 1200

#: The cohort each family's commentary is actually about.
DEFAULT_COHORT = {"tff": "lev_money", "disagg": "m_money", "legacy": "noncomm", "supp": "cit"}


@st.cache_data(show_spinner=False, max_entries=64)
def _simulated_null(stat_name: str, weeks: int, n_points: int, hit_at: float | None,
                    level: bool, walk: bool) -> dict | None:
    """The same table computed on data with nothing in it, through the same machinery.

    Memoised, unlike _prepare: this reads no file, so there is no file version to key on
    and no way to serve a stale number -- the arguments and a fixed seed determine the
    result completely. It costs ~0.17s uncached (0.34s for the persistence outcome, which
    runs both nulls), which is worth not paying on every widget click.

    Two nulls, and the pair is the point:

    walk=True -- a DRIFTLESS RANDOM WALK. Its next increment is independent of every
    function of its past, so it has nothing to trade on, but its LEVEL is as persistent
    as a level can be. A cohort net is close to a unit root, so this is the null that
    the persistence outcome has to beat to mean anything.

    walk=False -- IID NOISE. No persistence anywhere, so this isolates what the ranking
    machinery, the fixed edges and the overlap between the window ending at t and the
    one ending at t+h can manufacture on their own.

    A shuffle of the real series is deliberately not offered: it destroys the level's
    persistence and so tests a hypothesis nobody holds.

    The simulated calendar is unbroken weekly, so metrics.flow blanks nothing but the
    ends. CFTC's real holes only DELETE rows from the observed table, they do not bias a
    bucket, so the null does not need to reproduce them.

    Returns None when a trial cannot be scored (empty tail bucket, degenerate IQR).
    """
    stat = STATS[stat_name]
    edges, labels = list(stat["edges"]), list(stat["labels"])
    dates = pd.Series(pd.date_range("2006-01-03", periods=n_points, freq="7D"))
    meds, hits, ratios = [], [], []
    for trial in range(NULL_TRIALS):
        rng = np.random.default_rng(trial)
        frames = []
        for _ in range(NULL_SERIES):
            steps = rng.standard_normal(n_points)
            v = pd.Series(np.cumsum(steps) if walk else steps)
            s = stat["fn"](v, dates)
            src = s if level else v
            fwd = metrics.flow(src, dates, horizon_weeks=weeks).shift(-weeks)
            frames.append(pd.DataFrame({"stat": s, "fwd": (s + fwd) if level else fwd}))
        r = pd.concat(frames, ignore_index=True).dropna(subset=["stat", "fwd"])
        if r.empty:
            continue
        b = pd.cut(r["stat"], [-np.inf, *edges, np.inf], labels=labels, right=False)
        top, bot = r[b == labels[-1]], r[b == labels[0]]
        iqr = float(r["fwd"].quantile(0.75) - r["fwd"].quantile(0.25))
        if top.empty or bot.empty or not np.isfinite(iqr) or iqr <= 0:
            continue
        meds.append(float(top["fwd"].median()))
        hits.append(100.0 * float((top["fwd"] >= hit_at).mean() if level
                                  else (top["fwd"] < 0).mean()))
        ratios.append(float(bot["fwd"].median() - top["fwd"].median()) / iqr)
    if not ratios:
        return None
    return {
        "median": float(np.mean(meds)), "hit": float(np.mean(hits)),
        "ratio": float(np.mean(ratios)), "worst": float(np.max(np.abs(ratios))),
        "trials": len(ratios), "series": NULL_SERIES, "points": n_points,
    }


def _prepare(source_id: str, codes: list[str], cohorts: list[str], stat: dict) -> pd.DataFrame:
    """Per (market, cohort) history with a causal conditioning statistic attached.

    Deliberately not memoised: the parquet read is already cached in lib.cache and the
    rest measures 0.3-1.0s on the widest selection offered, so a second cache would
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
            # blocks=0 explicitly: a partial dict makes the whole column float, and
            # "{:,}".format(1.0) then renders one non-overlapping block as "1.0".
            rows.append({"bucket": label, "n": 0, "blocks": 0})
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
    extreme net at flat open interest is not the same event as one at collapsing OI. It
    is on screen UNCONDITIONALLY -- rule 4 here is satisfied by construction, not by
    charts.stacked() happening to raise: the percentile statistics are tagged kind=
    "share", which trips the share-of-OI test for the wrong reason (a percentile is not
    a share of anything), and lib.metrics owns the vocabulary of kinds, so a truer tag
    would have to be added there rather than invented here.
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
    ], height_per_panel=165), width="stretch")
    st.caption(
        "Dotted verticals are segment breaks: a contract re-basing, or a hole longer than 91 "
        "days. Readings and outcomes are computed inside a segment, never across one. "
        "Horizontal guides are the fixed bucket edges."
    )


def _fmt(v: float, kind: str, unit: str, level: bool) -> str:
    return f"{v:,.1f} {unit}" if level else charts.fmt_change(v, kind, unit)


def _wk(weeks: int) -> str:
    return "1 week" if weeks == 1 else f"{weeks} weeks"


def _bucket_row(tbl: pd.DataFrame, label: str) -> pd.Series | None:
    m = tbl[(tbl["bucket"] == label) & (tbl["n"] > 0)]
    return None if m.empty else m.iloc[0]


def _thick(row: pd.Series | None) -> bool:
    """Is this bucket big enough to write a sentence about? Both tests are needed."""
    return (row is not None and int(row["n"]) >= MIN_BUCKET_N
            and int(row["blocks"]) >= MIN_BUCKET_BLOCKS)


def _side(outcome: dict, v: float) -> str:
    """What the sign of a median forward move MEANS for this particular quantity."""
    if not np.isfinite(v) or v == 0:
        return "no median change"
    return outcome["up"] if v > 0 else outcome["down"]


def _verdict(tbl: pd.DataFrame, outcome: dict, stat: dict, unit: str, kind: str, weeks: int,
             iqr: float, base_hit: float, null: dict | None) -> None:
    """Blunt statement of what the table in front of the user actually supports.

    Three things it refuses to do, each of which it used to do. It will not write a
    verdict off a bucket thinner than MIN_BUCKET_N. It will not call two buckets "the
    extremes" unless they are the two TAIL buckets -- comparing `75 - 90` against
    `>= 90`, both of them 100% negative, and printing "low ones by rises" describes a
    contrast that was never measured. And it says what moved in the selected outcome's
    own words, because "the extreme reverts" is a claim about a reading, not about the
    size of a market.
    """
    live = tbl[tbl["n"] > 0]
    if len(live) < 2 or not np.isfinite(iqr) or iqr <= 0:
        st.warning("Fewer than two populated buckets, or no spread in the outcome — there is "
                   "nothing here to compare, so no verdict is written.")
        return
    labels = list(stat["labels"])
    lo_row, hi_row = _bucket_row(tbl, labels[0]), _bucket_row(tbl, labels[-1])
    tails = _thick(lo_row) and _thick(hi_row)
    lo, hi = (lo_row, hi_row) if tails else (live.iloc[0], live.iloc[-1])
    level = outcome["mode"] == "level"
    hi_med, lo_med = float(hi["median"]), float(lo["median"])
    gap = lo_med - hi_med
    ratio = gap / iqr

    st.markdown(
        f"**{'Verdict' if tails else 'The two outermost populated buckets'} at {_wk(weeks)}.** "
        f"Median forward outcome **{_fmt(hi_med, kind, unit, level)}** after a reading in "
        f"`{hi['bucket']}` (n={int(hi['n']):,}, {int(hi['blocks']):,} non-overlapping blocks), "
        f"versus **{_fmt(lo_med, kind, unit, level)}** after `{lo['bucket']}` "
        f"(n={int(lo['n']):,}, {int(lo['blocks']):,} blocks). The two medians sit "
        f"{abs(gap):,.1f} {unit} apart against a pooled inter-quartile range of "
        f"{iqr:,.1f} {unit} — **{abs(ratio):.2f} IQRs**."
    )
    if not tails:
        counts = ", ".join(
            f"`{lb}` n={0 if r is None else int(r['n']):,}/"
            f"{0 if r is None else int(r['blocks']):,} blocks"
            for lb, r in ((labels[0], lo_row), (labels[-1], hi_row))
        )
        st.warning(
            f"**No verdict.** A tail-to-tail statement needs both tail buckets to hold at least "
            f"{MIN_BUCKET_N} observations in at least {MIN_BUCKET_BLOCKS} non-overlapping blocks; "
            f"here {counts}. The two rows compared above are simply the outermost buckets that "
            "have anything in them, which is not the same comparison and cannot be read as one."
        )
        return

    if level:
        floor = float(stat["edges"][-1])
        mid = (float(stat["edges"][1]) + float(stat["edges"][2])) / 2
        st.markdown(
            f"{float(hi['hit']):.0f}% of `{hi['bucket']}` observations were still at or above "
            f"{floor:g} {stat['unit']} {_wk(weeks)} later, against {base_hit:.0f}% of every "
            f"usable observation in this table, and their median forward reading was "
            f"{hi_med:,.1f} against a scale midpoint of {mid:g}."
        )
        if null is None:
            st.warning(
                "No null could be simulated for this selection, so the persistence above is a "
                "number with nothing to be measured against. Read it as description only."
            )
            return
        d_med, d_hit = hi_med - null["median"], float(hi["hit"]) - null["hit"]
        st.markdown(
            f"**Against a null with nothing in it.** A driftless random walk pushed through this "
            f"same statistic, horizon and bucket edges ({null['series']} series x "
            f"{null['points']} weekly points, {null['trials']} draws) gives a top-bucket median "
            f"forward reading of {null['median']:,.1f} and {null['hit']:.0f}% still at or above "
            f"{floor:g} — against {hi_med:,.1f} and {float(hi['hit']):.0f}% here, i.e. "
            f"{d_med:+,.1f} on the median and {d_hit:+.0f} points on the share. "
            + ("So the reading persists MORE than a random walk in the underlying level does, on "
               "both counts, which is the only version of this result that says anything about "
               "traders."
               if d_med > 0 and d_hit > 0 else
               "So the observed persistence does not clear the null on both counts. It is a "
               "property of ranking a near-unit-root level, NOT evidence that an extreme is a "
               "regime traders sit in — and it supports neither the 'stretched positioning "
               "reverts' claim nor its opposite.")
        )
        return

    if abs(ratio) < 0.25:
        st.markdown(
            "**That is inside the noise.** A median difference under a quarter of one IQR, with "
            "the two buckets' p10-p90 ranges almost entirely overlapping, is not something to "
            "condition on at this sample size."
        )
    elif (hi_med > 0) == (lo_med > 0):
        st.markdown(
            f"**Both tails point the same way — {_side(outcome, hi_med)} — and only the size "
            f"differs**, by {abs(gap):,.1f} {unit} ({abs(ratio):.2f} IQRs), the "
            f"{'larger' if abs(hi_med) > abs(lo_med) else 'smaller'} after `{hi['bucket']}`. "
            "There is no sign flip here, so nothing reversed: read this as a difference in "
            "degree, not a change of direction."
        )
    elif ratio > 0:
        st.markdown(
            f"**A high reading precedes {_side(outcome, hi_med)}; a low reading precedes "
            f"{_side(outcome, lo_med)}** — a {abs(ratio):.2f}-IQR contrast on the median, the "
            "direction a reversion story predicts. The magnitude leaves the sign of any single "
            "outcome unsettled."
        )
    else:
        st.markdown(
            f"**The contrast runs the other way**: a high reading precedes "
            f"{_side(outcome, hi_med)} and a low reading precedes {_side(outcome, lo_med)}, by "
            f"{abs(ratio):.2f} IQRs on the median. A rule that faded a stretched reading would "
            "have been on the wrong side of the median here."
        )
    st.markdown(
        f"**The benchmark is not a coin.** {float(hi['hit']):.0f}% of `{hi['bucket']}` outcomes "
        f"were negative ({outcome['down']}), against {base_hit:.0f}% across every usable "
        f"observation in this table. 50% would only be the null if this outcome were driftless, "
        f"and at {base_hit:.0f}% unconditional it is not."
    )
    if null is not None:
        st.caption(
            f"**Against a null with nothing in it.** The same driftless random walk, ranked by "
            f"the same statistic, gives a tail-to-tail contrast of {null['ratio']:+.2f} IQRs on "
            f"this outcome ({null['trials']} draws, largest {null['worst']:.2f} in absolute "
            f"terms) against {ratio:+.2f} here. A random walk's next increment is independent of "
            "every function of its past, so whatever the null prints is the machinery talking."
        )
    thick = live[(live["n"] >= MIN_BUCKET_N) & (live["blocks"] >= MIN_BUCKET_BLOCKS)]
    if len(thick) < len(labels):
        st.caption(
            f"Shape across the buckets is not read here: {len(thick)} of {len(labels)} buckets "
            f"clear {MIN_BUCKET_N} observations in {MIN_BUCKET_BLOCKS} non-overlapping blocks, "
            "and two or three points are monotone too easily to mean anything."
        )
        return
    meds = pd.Series(thick["median"].to_numpy())
    st.caption(
        f"Bucket medians are monotone in the reading across all {len(thick)} buckets, which is "
        "stronger than the two tails alone: ordered buckets agreeing is harder to get by luck."
        if bool(meds.is_monotonic_decreasing or meds.is_monotonic_increasing) else
        "Bucket medians are NOT monotone in the reading. Whatever the tails show, the middle "
        "does not line up with it, which is what a spurious tail effect looks like."
    )


def render() -> None:
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
    stat_name = c2[1].selectbox(
        "Conditioning statistic, on the cohort net", list(STATS), index=0, key="br_stat")
    stat = STATS[stat_name]
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
    # freshness() gets the SOURCE's newest report date, never the selection's: a market
    # that simply did not print this week is not a stale feed, and saying so would both
    # cry wolf and hide a genuinely stopped feed. The per-market statement is
    # market_last_print(), below, where a single market is on screen.
    latest = (cache.file_stats(source_id) or {}).get("latest")
    source_dates = pd.Series([latest] if latest else summary["last_report"].to_numpy())
    freshness(source_dates, f"{spec.label} report")

    level = outcome["mode"] == "level"
    unit, kind = (stat["unit"], stat["kind"]) if level else (outcome["unit"], outcome["kind"])
    fwd = _forward(d, outcome["col"], weeks)
    d["fwd"] = (d["stat"] + fwd) if level else fwd
    d["fwd_oi"] = _forward(d, "oi", weeks)
    d["bucket"] = pd.cut(d["stat"], [-np.inf, *stat["edges"], np.inf],
                         labels=list(stat["labels"]), right=False)
    ok = d.dropna(subset=["stat", "fwd"])

    # Drawn BEFORE the emptiness guard, not after: when nothing survives the span check,
    # this market's own history with its segment breaks on it is the one artefact that
    # answers "why nothing", and it used to be the thing withheld.
    if not pooled:
        mine = d[d["market_code"] == code]
        market_last_print(mine["report_date"], source_dates, names.get(code, code))
        _context_chart(mine, stat, names.get(code, code))
        if code in universe.CONSOLIDATED or code in universe.SUPERSEDED_BY:
            st.caption(universe.CONSOLIDATED_TRADEOFF)

    if ok.empty:
        st.warning(
            f"Nothing survives the {_wk(weeks)} span check here: every candidate pair either "
            "skips a report week or crosses a segment break."
        )
        return

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
    ), width="stretch")
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
        hide_index=True, width="stretch",
    )

    iqr = float(ok["fwd"].quantile(0.75) - ok["fwd"].quantile(0.25))
    # The unconditional version of the `hit` column, over exactly the rows in the table.
    # This, not 50%, is what a bucket's hit rate has to beat: the outcomes have drift
    # (open interest trends up), and a coin is the null only for a driftless quantity.
    base_hit = 100.0 * float((ok["fwd"] >= hit_at).mean() if level else (ok["fwd"] < 0).mean())
    # A null is only offered where the outcome is a function of the conditioning series
    # itself -- the reading's own level, or the net it is a rank of. For open interest and
    # for the share, the simulated series is not the outcome series and a random walk
    # would be answering a different question.
    null = noise = None
    if outcome["col"] in ("stat", "net"):
        points = int(np.clip(ok["report_date"].nunique(), NULL_MIN_POINTS, NULL_MAX_POINTS))
        null = _simulated_null(stat_name, weeks, points, hit_at, level, True)
        if level:
            noise = _simulated_null(stat_name, weeks, points, hit_at, level, False)
    _verdict(tbl, outcome, stat, unit, kind, weeks, iqr, base_hit, null)

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
    if level and noise is not None:
        shared = (f"{100 * max(0.0, 1 - 7.0 * weeks / WINDOW_DAYS):.0f}% of their span"
                  if stat["windowed"] else "every observation before t")
        st.caption(
            f"**Where the persistence does not come from.** The readings at t and t+{_wk(weeks)} "
            f"are ranks over windows sharing {shared}, which is the obvious explanation and is "
            f"not the operative one. The same machinery on IID NOISE — no persistence "
            f"anywhere — gives a top-bucket median forward reading of {noise['median']:,.1f}, "
            f"{noise['hit']:.0f}% of them still in the top bucket, so the window and the fixed "
            "edges manufacture close to nothing by themselves. The persistence belongs to the "
            "LEVEL being ranked, which is why the null is run for the expanding statistic too."
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
