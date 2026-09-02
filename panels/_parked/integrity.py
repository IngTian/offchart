"""Integrity -- the accounting identities, checked over the whole committed dataset.

WHAT THIS BOARD ANSWERS. Whether the numbers every other board draws are
internally consistent, measured over all 1,046,369 market-weeks of the seven CFTC
datasets rather than over the one week that happens to be on screen. Three
identities are checkable with no outside data at all:

    1  sum(net) over cohorts   == 0                    every contract has two sides
    2  sum(long) == sum(short) == open_interest - sum(spread)
    3  our week-over-week diff == the published change_* columns

plus the two things that break a feed rather than an identity: a source that has
stopped updating, and a contract that was re-specified so levels either side of
the seam are not the same quantity.

WHAT IT REFUSES TO ANSWER. Whether the data is CORRECT. These are consistency
checks -- CFTC could publish a self-consistent set of wrong numbers and every
panel here would pass. What they do catch is a dropped cohort, a wrong field map
(the disagg old-crop-year trap fails identity 2 loudly), a misaligned join, a
stalled feed and a re-based contract. Nothing beyond that.

A DROPPED COHORT MUST NOT READ AS A PASS. Both identities need every cohort row
of a market-week to mean anything, so an incomplete market-week is excluded from
them. That exclusion is itself the loudest failure this board can see, and it is
reported as its own counter in the verdict strip: "no nonzero residual" and
"nothing was measurable" are different sentences here, never the same one.

WHY EVERY CHECK CARRIES A TOLERANCE. The previous version tested one displayed week
for exact equality and printed BROKEN on any nonzero residual. Both identities fail
an exact test routinely, because CFTC rounds each column independently. An exact
test is therefore not a strict test but a broken one: it fires on the publisher's
rounding and says nothing about the data. Tolerances live in lib/cftc_spec -- and
the two level tolerances are set one contract wider than the corpus maximum, so a
zero count there is a statement about calibration, not a test that passed. The
board says so where the counts are displayed, and STOPS saying so the moment one
of those counters is nonzero: the calibration sentence is gated on the zero it
explains, because a breach means the tolerance no longer bounds the corpus and
"nothing here can breach it" would be the one caption on this board that is
strictly false.

AND A TOLERANCE IS NOT A SPAN TEST. Invariant 3 compares our week-over-week diff
against the published change_* columns, so it needs to know which row pairs are a
week apart. round(days/7) == 1 is not that test -- round(4/7) is 1 as well -- and
using it made this board manufacture 141 of its own 228 headline exceptions out of
147 four-day re-issues. The gate is metrics.SPAN_TOLERANCE_DAYS, read off the
library rather than restated, and the pairs it drops are reported as a named
exclusion count with their spans measured rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments
from lib.ui import caveat_block, freshness
from panels import Board

#: A per-market gap this long makes a code's history discontinuous. lib.segments
#: cuts at a quarter; a year is used here so the count is dominated by code reuse.
HOLE_DAYS = 365

#: What a weekly publisher actually emits. Any other gap is where round(days/7)
#: can mislabel a horizon.
WEEKLY_GAPS = (6, 7, 8)

#: Exception tables are evidence, not a data dump: the worst few rows carry the
#: argument, and keeping the whole frame would put 288,151 rows in the cache the
#: moment an identity really broke.
FAIL_SHOW = 12

FAIL_COLS = ["report_date", "market_code", "market", "oi", "value"]

BAD_COLS = ["report_date", "market_code", "market", "cohort", "gap_days", "err"]


def _num(x: float | None, fmt: str = ",.0f", suffix: str = "") -> str:
    """Format, or say the statistic does not exist -- never render "nan".

    Every summary here is NaN when nothing was measurable, and "nan contracts"
    beside a bold zero reads as a pass. "not measurable" does not.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "not measurable"
    return f"{x:{fmt}}{suffix}"


def _token(df: pd.DataFrame) -> tuple[int, str]:
    """Cache key derived from the frame lib.cache has already validated.

    lib.cache invalidates on file mtime and size; these aggregates then key off
    (rows, latest report date), which moves on any ingest that added or revised a
    week and costs no second stat of the file.
    """
    return (len(df), str(df["report_date"].iloc[-1])) if len(df) else (0, "")


def _residual(mw: pd.DataFrame, col: str, tol: float) -> dict:
    """Everything both invariants need from one signed per-market-week residual.

    Shared because the checks differ only in what they subtract. Correlation with
    open interest is in here because a rounding error has none while a genuinely
    missing position bucket would. `n` is the number of market-weeks that were
    actually measurable and every caller gates its wording on it.
    """
    v = mw[col].dropna()
    buckets = v.round().astype("int64")
    buckets = buckets[buckets != 0].value_counts().sort_index()
    # assign BEFORE filtering: in pandas 3.0 an empty frame .assign(series)
    # adopts the series' index and resurrects every row.
    fail = mw.assign(value=mw[col]).loc[mw[col].abs() > tol, FAIL_COLS]
    return {
        "n": int(len(v)),
        "max": float(v.abs().max()) if len(v) else float("nan"),
        "lo": float(v.min()) if len(v) else float("nan"),
        "hi": float(v.max()) if len(v) else float("nan"),
        "exact": float((v == 0).mean() * 100) if len(v) else float("nan"),
        "within": float((v.abs() <= tol).mean() * 100) if len(v) else float("nan"),
        "neg": int((v < 0).sum()),
        "pos": int((v > 0).sum()),
        "zero": int((v == 0).sum()),
        "corr": float(np.corrcoef(v, mw["oi"][v.index])[0, 1]) if v.std() > 0 else float("nan"),
        "buckets": {int(k): int(n) for k, n in buckets.items()},
        "n_fail": int(len(fail)),
        "fail": fail.sort_values("value", key=lambda s: s.abs(), ascending=False).head(FAIL_SHOW),
    }


@st.cache_data(show_spinner=False, max_entries=16)
def _scan(source_id: str, token: tuple[int, str]) -> dict:
    """Every check for one source in one pass, returning only small objects.

    Vectorised groupby over the whole source: 4.1 million rows across all seven in
    ~2.4 s cold, then cached. What comes back is per-report-date (~1,900 rows), a
    handful of counters, and exception tables truncated to the worst FAIL_SHOW
    rows, so the cache holds tens of kilobytes however badly the data breaks.
    """
    df = cache.read(source_id)  # cache hit: the caller has already read it
    spec = cftc_spec.spec_for(source_id)
    n_co = len(spec.cohorts)
    df = df.assign(_d=pd.to_datetime(df["report_date"]))

    # ---- identities 1 and 2, one row per market-week ------------------------
    mw = (
        df.groupby(["report_date", "market_code"], observed=True, sort=True)
        # spread is NULL for cohorts the report gives no spread column (legacy
        # comm, disagg prod_merc, nonrept everywhere) and the sum skips them, which
        # is right: absent is not zero-with-a-value. open_interest is constant
        # across a market-week's cohort rows, so .first() and never .sum().
        .agg(long=("long", "sum"), short=("short", "sum"), spread=("spread", "sum"),
             oi=("open_interest", "first"), market=("market", "last"),
             rows=("cohort", "size"), n_long=("long", "count"),
             n_oi=("open_interest", "count"))
        .reset_index()
    )
    for col in ("long", "short", "spread", "oi"):
        mw[col] = mw[col].astype("float64")
    # A market-week missing a cohort row says nothing about rounding, so it is
    # excluded from the identity statistics and counted separately. n_incomplete
    # is the dropped-cohort detector: it is a headline counter, not a footnote.
    complete = (mw["rows"] == n_co) & (mw["n_long"] == n_co) & (mw["n_oi"] > 0)
    mw["netsum"] = (mw["long"] - mw["short"]).where(complete)
    mw["resid"] = (mw["oi"] - mw["long"] - mw["spread"]).where(complete)

    by_date = (
        mw.assign(_d=pd.to_datetime(mw["report_date"]),
                  nz=mw["netsum"].abs() > 0, ar=mw["resid"].abs())
        .groupby("_d", sort=True)
        .agg(markets=("oi", "size"), nz_net=("nz", "sum"),
             worst_resid=("ar", "max"), max_oi=("oi", "max"))
    )

    # ---- identity 3: our diff vs the published change_* columns -------------
    # metrics.flow and metrics.weeks_between take one series and there is no
    # grouped variant, so the same rule is applied within (market_code, cohort)
    # here -- and it is metrics.flow's rule, read off the library constant rather
    # than restated, so the gate cannot drift from the function it mirrors: a span
    # is one week only within SPAN_TOLERANCE_DAYS of seven, never
    # round(days/7) == 1, because round(4/7) is 1 too and a four-day re-issue is
    # not a week. Those pairs are excluded and counted as `relabel`, which is the
    # difference between a named exclusion and a manufactured failure: admitting
    # them put 141 of legacy_fut's 186 headline exceptions into the data itself.
    sub = df.sort_values(["market_code", "cohort", "_d"], kind="mergesort")
    grp = sub.groupby(["market_code", "cohort"], observed=True, sort=False)
    days = grp["_d"].diff().dt.days
    err = pd.concat(
        [(grp["long"].diff() - sub["change_long_published"]).abs(),
         (grp["short"].diff() - sub["change_short_published"]).abs()],
        axis=1,
    ).max(axis=1)
    cand = err.notna() & days.notna()
    ok = cand & ((days - 7.0).abs() <= metrics.SPAN_TOLERANCE_DAYS)
    # The relabel set is what the ROUNDED rule would have accepted and this one
    # refuses: 4, 5, 9 and 10 days. Which of those actually occur is a property of
    # the data, so `gaps` is carried through and every caption reads it off.
    relabel = cand & ~ok & ((days / 7.0).round() == 1)
    tagged = sub.assign(err=err.astype("float64"), gap_days=days)
    e = err[ok].astype("float64")
    re_ = err[relabel].astype("float64")
    bad = tagged.loc[ok & (err > cftc_spec.CHANGE_FIELD_TOL), BAD_COLS] \
        .sort_values("err", ascending=False)
    chg = {
        "n": int(len(e)),
        "exact": float((e == 0).mean() * 100) if len(e) else float("nan"),
        "within": float((e <= cftc_spec.CHANGE_FIELD_TOL).mean() * 100) if len(e) else float("nan"),
        "max": float(e.max()) if len(e) else float("nan"),
        "n_bad": int(len(bad)),
        "bad": bad.head(FAIL_SHOW),
        "bad_dates": {str(k): int(v) for k, v in bad["report_date"].value_counts().items()},
        "relabel": {
            "n": int(len(re_)),
            "n_bad": int((re_ > cftc_spec.CHANGE_FIELD_TOL).sum()),
            "gaps": {int(k): int(v) for k, v in days[relabel].value_counts().sort_index().items()},
            "rows": tagged.loc[relabel & (err > cftc_spec.CHANGE_FIELD_TOL), BAD_COLS]
            .sort_values("err", ascending=False).head(FAIL_SHOW),
        },
    }

    # ---- feed health. nonrept is the only cohort in all seven families, and
    # every market-week column is constant across cohorts, so one is enough.
    one = df[df["cohort"] == "nonrept"].sort_values(["market_code", "_d"], kind="mergesort")
    codes = one["market_code"].to_numpy()
    units = one["contract_units"].astype(object)
    sig = units.map({u: segments.unit_signature(u) for u in units.dropna().unique()})
    # lib.segments' rule exactly: only a change between two KNOWN signatures is a
    # break, so a null unit string does not manufacture two of them.
    known = sig.where(sig != "").groupby(codes).ffill()
    unit_break = (known.ne(known.groupby(codes).shift()) & known.groupby(codes).shift().notna())
    code_gap = one.groupby("market_code", observed=True)["_d"].diff().dt.days

    # ---- cadence, over report dates rather than markets ---------------------
    dates = pd.Series(sorted(df["_d"].unique()))
    gaps = dates.diff().dt.days.dropna().astype("int64")
    dev = (gaps - 7 * (gaps / 7.0).round()).abs()
    early = dates < pd.Timestamp(segments.LEGACY_WEEKLY_FROM)
    off = (
        pd.DataFrame({"previous": dates.shift(), "report_date": dates,
                      "gap_days": gaps, "days_off_a_week": dev,
                      "rounds to (weeks)": (gaps / 7.0).round()})
        .dropna(subset=["gap_days"])
        # Off-table means outside the span tolerance the library actually applies,
        # so the classification below and metrics.flow agree on what a week is.
        .loc[dev > metrics.SPAN_TOLERANCE_DAYS]
        .assign(previous=lambda f: f["previous"].dt.date,
                report_date=lambda f: f["report_date"].dt.date,
                gap_days=lambda f: f["gap_days"].astype("int64"),
                days_off_a_week=lambda f: f["days_off_a_week"].astype("int64"),
                **{"rounds to (weeks)": lambda f: f["rounds to (weeks)"].astype("int64")})
    )

    return {
        "label": spec.label,
        "n_rows": int(len(df)),
        "n_mw": int(len(mw)),
        "n_incomplete": int((~complete).sum()),
        "oi_max": float(mw["oi"].max()),
        "net": _residual(mw, "netsum", cftc_spec.NET_ZERO_TOL),
        "resid": _residual(mw, "resid", cftc_spec.OI_RESIDUAL_TOL),
        "chg": chg,
        "by_date": by_date,
        "n_codes": int(one["market_code"].nunique()),
        "n_unit_break": int(unit_break.groupby(codes).any().sum()),
        "n_holes": int((code_gap.groupby(codes).max() > HOLE_DAYS).sum()),
        "n_reports": int(len(dates)),
        "n_early": int(early.sum()),
        "n_early_long": int((gaps[early.reindex(gaps.index, fill_value=False)] > max(WEEKLY_GAPS)).sum()),
        "early_weekdays": int(dates[early].dt.day_name().nunique()),
        "first": dates.iloc[0].date(),
        "last": dates.iloc[-1].date(),
        "gap_counts": {int(k): int(v) for k, v in gaps.value_counts().items()},
        "gaps_off": off,
        "worst_dev": int(dev.max()) if len(dev) else 0,
    }


def scan(source_id: str) -> dict | None:
    df = cache.read(source_id)
    return None if df.empty else _scan(source_id, _token(df))


def _bars(res: dict, empty: str, key: str) -> None:
    """The nonzero buckets of a residual; zero is dropped and stated in words.

    Zero is 49-100% of every source, so plotting it flattens the only part that
    carries an argument: the shape and symmetry of the nonzero tail. Three
    outcomes, never two -- a tail, no tail because every measurable market-week
    was exactly zero, or NOTHING MEASURABLE, which is a coverage failure and must
    not borrow the wording of a pass.
    """
    if not res["n"]:
        st.warning(
            "Nothing to plot and nothing to claim: no market-week in this source "
            "had a complete set of cohort rows, so this identity was never "
            "evaluated. That is a coverage failure -- most likely a cohort "
            "dropped in ingest -- not a clean result."
        )
        return
    if not res["buckets"]:
        st.caption(empty)
        return
    total = sum(res["buckets"].values())
    st.plotly_chart(
        charts.ranked_bars(
            [f"{k:+d} contracts" for k in res["buckets"]],
            [float(v) for v in res["buckets"].values()],
            unit="market-weeks",
            kind="gross",  # a count of market-weeks; the sign lives in the label
            hover=[f"{100.0 * v / total:.1f}% of the nonzero ones" for v in res["buckets"].values()],
            height_per_bar=26,
        ),
        key=key,
    )


def _fail_table(n_fail: int, frame: pd.DataFrame, tol: float) -> None:
    if not n_fail:
        return
    st.warning(
        f"{n_fail:,} market-weeks clear the {tol:g}-contract tolerance"
        + (f"; the worst {len(frame)} are shown." if n_fail > len(frame) else ":")
    )
    st.dataframe(frame, hide_index=True)


def _history(sid: str, sc: dict) -> None:
    bd = sc["by_date"]
    brk = ((pd.Timestamp(segments.LEGACY_WEEKLY_FROM),)
           if pd.Timestamp(segments.LEGACY_WEEKLY_FROM) > bd.index[0] else ())
    rows = [
        ([charts.Series(bd["markets"], "markets reporting", kind="gross"),
          charts.Series(bd["nz_net"], "of which cohort nets do not sum to zero",
                        kind="gross", color=charts.WARN)],
         "markets", "Coverage each week, and how much of it the publisher's rounding touches", ()),
        ([charts.Series(bd["worst_resid"], "worst |OI - sum long - sum spread|", kind="gross")],
         "contracts",
         f"Worst open-interest residual that week, against the "
         f"{cftc_spec.OI_RESIDUAL_TOL:g}-contract tolerance",
         (cftc_spec.OI_RESIDUAL_TOL,)),
        ([charts.Series(bd["max_oi"], "largest open interest in the source", kind="open_interest")],
         "contracts", "Scale for the panel above: the biggest market that week", ()),
    ]
    fig = charts.stacked(
        [charts.Panel(series=s, unit=u, title=t, guides=g, breaks=brk) for s, u, t, g in rows],
        height_per_panel=165,
    )
    st.plotly_chart(fig, key=f"history_{sid}")
    # Three states, and the rounding conclusion is asserted in exactly one of
    # them. The middle panel showing a residual larger than the tolerance is the
    # same picture as the middle panel showing rounding noise, at a different
    # scale, so the caption must not carry the conclusion when the tolerance broke.
    if not sc["resid"]["n"]:
        body = (
            "The middle panel is empty and the argument of invariant 2 cannot be made on "
            f"this data: none of the {sc['n_mw']:,} market-weeks has a complete set of "
            "cohort rows, so no residual was computable. Treat the flat panel as missing "
            "evidence, not as a reconciliation."
        )
    elif sc["resid"]["n_fail"]:
        body = (
            "Read the middle and bottom panels together -- but not as invariant 2 "
            f"passing. Open interest in this source reaches {sc['oi_max']:,.0f} contracts "
            f"and the worst residual in it is {sc['resid']['max']:,.0f}, which clears the "
            f"{cftc_spec.OI_RESIDUAL_TOL:g}-contract tolerance on "
            f"{sc['resid']['n_fail']:,} of the {sc['resid']['n']:,} measurable "
            "market-weeks. That is a break to explain, not rounding to characterise."
        )
    else:
        body = (
            "Read the middle and bottom panels together -- that pairing is the whole "
            "argument of invariant 2. Open interest in this source reaches "
            f"{sc['oi_max']:,.0f} contracts while the worst residual anywhere in it is "
            f"{sc['resid']['max']:,.0f}. A missing position bucket would scale with the "
            "market; rounding does not."
        )
    st.caption(
        body
        + (f" Dotted line: {segments.LEGACY_WEEKLY_FROM}, before which legacy "
           "reporting is not weekly." if brk else "")
    )


def _exact_split(scans: dict[str, dict], key: str) -> dict[str, tuple[int, int]]:
    """(market-weeks failing an EXACT test, market-weeks measured), by basis.

    Computed from what is on screen rather than quoted from lib/cftc_spec, so the
    sentence cannot drift away from the data it describes.
    """
    out = {"futures-only": [0, 0], "combined-basis": [0, 0]}
    for sid, v in scans.items():
        grp = "combined-basis" if cftc_spec.spec_for(sid).is_combined else "futures-only"
        out[grp][0] += v[key]["n"] - v[key]["zero"]
        out[grp][1] += v[key]["n"]
    return {k: (a, b) for k, (a, b) in out.items() if b}


def _relabel_spans(scans: dict[str, dict]) -> dict[int, int]:
    """gap in days -> excluded pairs, summed over sources.

    The exclusion filter admits 4, 5, 9 and 10 days -- everything round(days/7)
    calls one week and the span tolerance does not. Which of those the corpus
    contains is a fact about the data (today: 147 pairs, all four-day, all in
    legacy_fut), so every caption naming the spans reads them off here rather than
    hard-coding "4-day", which would go quietly wrong on the next ingest.
    """
    out: dict[int, int] = {}
    for v in scans.values():
        for gap, k in v["chg"]["relabel"]["gaps"].items():
            out[gap] = out.get(gap, 0) + k
    return out


def _invariant_net(sc: dict, scans: dict[str, dict]) -> None:
    st.subheader("Invariant 1. Cohort nets sum to zero")
    net = sc["net"]
    c = st.columns(4)
    c[0].metric("market-weeks measured", f"{net['n']:,} of {sc['n_mw']:,}")
    c[1].metric("max |sum of nets| (contracts)", _num(net["max"]))
    c[2].metric("exactly zero", _num(net["exact"], ".2f", "%"))
    c[3].metric(f"within {cftc_spec.NET_ZERO_TOL:g}", _num(net["within"], ".2f", "%"))
    if not net["n"]:
        st.caption(
            f"There is no pass condition to report: all {sc['n_mw']:,} market-weeks in "
            "this source are missing a cohort row, so the identity was never evaluated. "
            "The tolerance, the rounding argument and the bucket chart all need a "
            "complete market-week to mean anything."
        )
    else:
        split = _exact_split(scans, "net")
        st.caption(
            f"Pass condition: the identity holds to within {cftc_spec.NET_ZERO_TOL:g} "
            f"contracts on {net['within']:.2f}% of the market-weeks that could be "
            "measured. It is not exact because CFTC rounds each cohort column "
            "independently, so the sum carries one rounding error per cohort, signed "
            "either way -- which is why the nonzero buckets come in near-equal pairs. "
            "Measured across every ingested source, an exact test fails on "
            + " and ".join(f"{a:,} of {b:,} {k} market-weeks" for k, (a, b) in split.items())
            + ", so the exact test is the bug rather than the finding."
        )
    _bars(net, "Every measurable market-week in this source sums to exactly zero, so "
               "there is no nonzero bucket to plot.", key="net_buckets")
    _fail_table(net["n_fail"], net["fail"], cftc_spec.NET_ZERO_TOL)
    if sc["n_incomplete"]:
        st.warning(
            f"{sc['n_incomplete']:,} of {sc['n_mw']:,} market-weeks are missing a "
            "cohort row and are excluded from every figure above -- a coverage "
            "problem, not a rounding one, and the one this board would expect to see "
            "if a cohort were dropped in ingest."
        )


def _invariant_oi(sc: dict, scans: dict[str, dict]) -> None:
    st.subheader("Invariant 2. Longs, shorts and spreads reconcile to open interest")
    res = sc["resid"]
    st.caption(
        "sum(long) == sum(short) == open_interest - sum(spread). A spread position "
        "is long one expiry and short another, so it belongs to neither side's "
        "column and open interest counts it separately."
    )
    c = st.columns(4)
    c[0].metric("residual range",
                "not measurable" if not res["n"]
                else f"{_num(res['lo'])} to {_num(res['hi'])}")
    c[1].metric("negative / positive", f"{res['neg']:,} / {res['pos']:,}")
    c[2].metric("corr with open interest",
                "undefined" if np.isnan(res["corr"]) else f"{res['corr']:+.4f}")
    c[3].metric("largest open interest", f"{sc['oi_max']:,.0f}")
    _bars(res, "The residual is exactly zero on every measurable market-week in this "
               "source, which is the strongest form this evidence takes.",
          key="resid_buckets")
    st.info(cftc_spec.RESIDUAL_EXPLANATION)
    # Rendered verbatim above, figures included, and its figures are a measurement
    # of an earlier corpus. If the current one breaches the tolerance then that
    # paragraph is the most authoritative-looking false statement on the board, so
    # it gets contradicted here rather than silently outranked by the table below.
    res_fails = sum(v["resid"]["n_fail"] for v in scans.values())
    if res_fails:
        st.warning(
            f"The paragraph above is lib/cftc_spec's record of an earlier measurement, and "
            f"the ingested data no longer matches it: {res_fails:,} market-week(s) clear "
            f"the {cftc_spec.OI_RESIDUAL_TOL:g}-contract tolerance, so the range it quotes "
            "does not bound what is on screen. The rounding reading is unavailable until "
            "that is explained."
        )
    # Every number in the correction below is read off `scans` rather than quoted,
    # including the ones cftc_spec states in words: a caption that recomputes
    # cannot contradict the table printed under it.
    live = [v["resid"] for v in scans.values() if v["resid"]["n"]]
    # The sentence below claims the residual GOES NEGATIVE, so this counts sources
    # with a negative residual and not sources with any nonzero one. The two agree
    # on today's data (6 and 6) and separate the moment one source's residual is
    # pushed wholly positive -- which is precisely the case where the sign
    # argument, the disproof of the unpublished-spread reading, has stopped
    # holding and must not be reported as if it still did.
    neg_src = [r for r in live if r["neg"]]
    nonzero = [r for r in live if r["neg"] or r["pos"]]
    neg_bigger = sum(1 for r in neg_src if r["neg"] > r["pos"])
    all_zero = [k for k, v in scans.items()
                if v["resid"]["n"] and not (v["resid"]["neg"] or v["resid"]["pos"])]
    head = (
        ":warning: **Correction.** An earlier version of this repo rendered "
        "\"residual = unpublished non-reportable spread\" beside these same, correct "
        "numbers -- a chart that is wrong while every figure in it is right, which is "
        "the worst failure mode available because nothing looks broken. It is "
        "rounding noise. "
    )
    if not live:
        st.caption(
            head + "None of the four measurements that disprove the spread reading can "
            "be reproduced on the currently ingested data: no source has a single "
            "market-week with a complete set of cohort rows, so the residual was never "
            "computed. What is written above is lib/cftc_spec's record of an earlier "
            "measurement, not evidence from this screen."
        )
    else:
        lo = min(r["lo"] for r in live)
        hi = max(r["hi"] for r in live)
        corrs = [abs(r["corr"]) for r in live if not np.isnan(r["corr"])]
        st.caption(
            head + "The table below measures the disproof source by source. A "
            "non-negative missing spread could only push the residual positive, and "
            + (f"in {len(neg_src)} of the {len(live)} measurable sources the residual goes "
               f"negative at all, with the negative side the larger of the two in "
               f"{neg_bigger} of those {len(neg_src)}. "
               if neg_src else
               f"in all {len(live)} measurable sources the residual is identically "
               "zero, so a missing spread would have to be identically zero too. "
               if not nonzero else
               f"on this data it never does: the residual is non-negative in all "
               f"{len(live)} measurable sources, which is also what a missing spread "
               "would look like, so the sign half of the disproof is unavailable here "
               "and only the range, the correlation and the exactly-zero sources stand "
               "against the spread reading. ")
            + f"The residual stays inside {lo:,.0f}..{hi:,.0f} across every source while "
              f"open interest reaches {max(v['oi_max'] for v in scans.values()):,.0f} "
              "contracts, where a real position bucket would scale with the market."
            + (f" Its correlation with open interest never exceeds {max(corrs):.4f} in "
               "absolute value." if corrs else "")
            + (" And it is exactly zero on every market-week of "
               + ", ".join(f"`{k}` ({scans[k]['resid']['n']:,} market-weeks)"
                           for k in all_zero)
               + " -- market-weeks, not rows, which is the unit cftc_spec's wording "
                 "elides; the disaggregated futures-only report is the commodity dataset "
                 "where small-trader calendar spreads would surface first."
               if all_zero else "")
        )
    st.dataframe(
        pd.DataFrame([
            {"source": k, "market-weeks": v["n_mw"], "measured": v["resid"]["n"],
             "residual range": "n/a" if not v["resid"]["n"]
             else f"{v['resid']['lo']:,.0f} .. {v['resid']['hi']:,.0f}",
             "negative": v["resid"]["neg"], "positive": v["resid"]["pos"],
             "exactly zero": v["resid"]["zero"],
             # A string column, because None in a float column renders as a raw
             # "NaN" beside a measured row count and reads as a broken cell rather
             # than as an absent statistic. The two absences are different and both
             # are named: nothing measured at all, versus a residual with no
             # variance for a correlation to exist over (disagg_fut, exactly zero
             # on all 184,229 of its market-weeks).
             "corr with OI": ("n/a" if not v["resid"]["n"]
                              else "undefined" if np.isnan(v["resid"]["corr"])
                              else f"{v['resid']['corr']:+.4f}"),
             "largest OI": f"{v['oi_max']:,.0f}"}
            for k, v in scans.items()
        ]),
        hide_index=True,
    )
    _fail_table(res["n_fail"], res["fail"], cftc_spec.OI_RESIDUAL_TOL)


def _invariant_change(sc: dict, scans: dict[str, dict]) -> None:
    st.subheader("Invariant 3. Our week-over-week change vs the published change_* columns")
    chg = sc["chg"]
    c = st.columns(4)
    c[0].metric("comparable pairs", f"{chg['n']:,}")
    c[1].metric("exact agreement", _num(chg["exact"], ".2f", "%"))
    c[2].metric(f"within {cftc_spec.CHANGE_FIELD_TOL:g}", _num(chg["within"], ".3f", "%"))
    c[3].metric("largest disagreement (contracts)", _num(chg["max"]))
    off_by_one = {
        "combined-basis": [], "futures-only": [],
    }
    for sid, v in scans.items():
        if v["chg"]["n"]:
            grp = "combined-basis" if cftc_spec.spec_for(sid).is_combined else "futures-only"
            off_by_one[grp].append(100.0 - v["chg"]["exact"])
    rule = (
        "Only pairs whose calendar span is within a day of seven are compared. CFTC "
        "skips weeks, so a one-row diff is not a one-week change -- and round(days/7) "
        "is not the test either, because round(4/7) is 1 and a four-day holiday span "
        "is not a week. "
    )
    # The rates are a join over whatever groups have comparisons, so it can be
    # empty -- a one-report-date corpus has no pair spanning a week anywhere. That
    # state used to render "measured here, , and all but the exceptions...", and
    # the sentence it was inside claimed a tolerance held over nothing.
    rates = " and ".join(
        f"{grp} datasets fall short of exact on "
        f"{min(vals):.1f}-{max(vals):.1f}% of comparisons"
        for grp, vals in off_by_one.items() if vals
    )
    st.caption(
        rule
        + ("Exact agreement is not the pass test: measured here, " + rates
           + f", and all but the exceptions counted below stay inside the "
             f"{cftc_spec.CHANGE_FIELD_TOL:g}-contract tolerance. Same mechanism as "
             "invariant 1 -- independently rounded columns"
           + (", and the combined-basis reports carry more of it because the "
              "delta-equivalent option figures are rounded independently of the levels "
              "they are added to." if off_by_one["combined-basis"] else ".")
           if rates else
           "No source on this data holds a single pair of rows spanning a week, so there "
           "is no exact-agreement rate to quote and nothing at all behind this "
           "invariant's counters: it was not evaluated. An empty comparison set is a "
           "coverage statement, never a pass.")
    )
    if chg["n_bad"]:
        dates = chg["bad_dates"]
        top = max(dates, key=lambda d: dates[d])
        st.warning(
            f"{chg['n_bad']} comparisons clear +/-{cftc_spec.CHANGE_FIELD_TOL:g} "
            f"contracts, on {len(dates)} report date(s); {dates[top]} of them on {top} "
            "alone. These are revisions rather than parsing faults: the published change "
            "compares a new classification or a revised prior level against the level we "
            "differenced, and it clusters on whole report dates rather than on markets."
            + (" The largest cluster is the July 2008 trader reclassification, worst on "
               "WTI crude 067651 and natural gas 023651."
               if top.startswith("2008-07") else "")
            + (f" The worst {len(chg['bad'])} are shown."
               if chg["n_bad"] > len(chg["bad"]) else "")
        )
        st.dataframe(chg["bad"], hide_index=True)
    else:
        # "Nothing clears the tolerance" is also true of nothing at all, so which
        # of the two it is has to be on screen beside the empty metrics.
        st.caption("No comparison in this source clears the tolerance."
                   if chg["n"] else
                   "There is nothing here to clear it: this source holds no pair of rows "
                   "a week apart, so nothing was compared.")
    rel = chg["relabel"]
    if rel["n"]:
        st.caption(
            f"Excluded, not failed: {rel['n']} further pairs span "
            f"{'/'.join(str(g) for g in rel['gaps'])} days, which round(days/7) would "
            f"have called one week. {rel['n_bad']} of them disagree with the published "
            "change columns, which is exactly why they are dropped rather than "
            "relabelled -- see Cadence."
        )
        if len(rel["rows"]):
            st.dataframe(rel["rows"], hide_index=True)


def _feeds(scans: dict[str, dict]) -> None:
    st.subheader("Feed health and cadence, all sources")
    today = pd.Timestamp.today().normalize()
    # Via lib.cache.file_stats, which is cached and column-projected. A panel does
    # not import lib.store and does not stat a file itself: the size on screen has
    # to come from the same cache key that decided whether to re-read the file, or
    # the two can disagree about which version of it is being described.
    on_disk = {k: (cache.file_stats(k) or {}).get("bytes", 0) for k in scans}
    st.dataframe(
        pd.DataFrame([
            {"source": k, "rows": v["n_rows"], "MB": round(on_disk[k] / 1e6, 1),
             "market-weeks": v["n_mw"], "incomplete mw": v["n_incomplete"],
             "markets": v["n_codes"], "reports": v["n_reports"],
             "first": v["first"], "latest": v["last"],
             "age (d)": (today - pd.Timestamp(v["last"])).days,
             "unit-break codes": v["n_unit_break"],
             f"holes over {HOLE_DAYS}d": v["n_holes"],
             "net fails": v["net"]["n_fail"], "OI fails": v["resid"]["n_fail"],
             "change fails": v["chg"]["n_bad"],
             "gaps excluded": v["chg"]["relabel"]["n"]}
            for k, v in scans.items()
        ]),
        hide_index=True,
    )
    st.caption(
        "`age` is days since the latest REPORT, not since the last ingest: CFTC "
        "publishes Friday 15:30 ET and the job runs daily, so 3-9 days is normal and "
        "one source climbing past ~10 while the others stay low is a stopped feed. "
        "`incomplete mw` counts market-weeks missing a cohort row -- the dropped-cohort "
        "detector; a nonzero value there invalidates the identity columns beside it "
        "rather than joining them. `unit-break codes` counts codes whose "
        "contract_units digit signature changes at least once, by lib.segments' rule "
        "-- those levels are not one comparable series and a percentile must not span "
        "the seam. `holes` counts codes with a gap over a year, which includes outright "
        "code reuse. `gaps excluded` is the pairs invariant 3 refused to compare "
        "because their span is not within a day of a week. `MB` is the committed "
        "parquet, from lib.cache.file_stats -- a file that stops growing while its age "
        "column climbs is the other shape a stopped feed takes."
    )
    other = cache.read("openrouter_pricing")
    if not other.empty:
        latest = pd.to_datetime(other["snapshot_date"]).max()
        st.caption(
            f"`openrouter_pricing` has a different shape and no identity to check: "
            f"{len(other):,} rows, latest snapshot {latest.date()} "
            f"({(today - latest.normalize()).days} days ago). It is snapshot-only, so "
            "a missed run is a permanent hole rather than something a backfill "
            "repairs -- its age matters more than any CFTC source's."
        )
    table = pd.DataFrame(
        {k.replace("cftc_", ""): pd.Series(v["gap_counts"]) for k, v in scans.items()}
    ).fillna(0).astype("int64").sort_index()
    table.index.name = "gap (days)"
    st.dataframe(table)
    st.caption(
        "Gaps between consecutive report dates. Weekly publication shows up as 6, 7 "
        "and 8 -- the 6s and 8s are holiday shifts, not missing weeks."
    )


def _cadence(sc: dict, scans: dict[str, dict]) -> None:
    off = sc["gaps_off"]
    n_gaps = sc["n_reports"] - 1
    # The same column the table below shows, so the prose cannot classify a gap
    # differently from the row the reader is looking at.
    weeks = off["rounds to (weeks)"]
    # Partitioned on the rounded week count, which is exhaustive over `off`: every
    # off-table gap rounds to zero, to one, or to two or more. An earlier taxonomy
    # keyed on gap_days < min(WEEKLY_GAPS) instead, which put the three-day gaps in
    # with the four-day ones (5 "hazards" where there are 2) and left anything that
    # was neither early nor short -- the 11-day 2001-12-28 pair -- in no class at all.
    mislabelled = off[weeks == 1]  # 4-5 and 9-10 days: round(days/7) says one week
    too_short = off[weeks == 0]  # 1-3 days: rounds to zero weeks
    multi = off[weeks >= 2]
    if not n_gaps:
        st.caption(
            "**Integer-week check, selected source.** This source holds a single report "
            "date, so there is no gap to measure and the cadence here is unmeasured "
            "rather than confirmed."
        )
    else:
        st.caption(
            f"**Integer-week check, selected source.** {n_gaps - len(off):,} of "
            f"{n_gaps:,} gaps are within {metrics.SPAN_TOLERANCE_DAYS:g} day of a whole "
            f"number of weeks; worst deviation {sc['worst_dev']} day(s). That "
            f"{metrics.SPAN_TOLERANCE_DAYS:g} is metrics.SPAN_TOLERANCE_DAYS, which is "
            "what lets flow accept a 6- or an 8-day gap as one week and refuse every "
            "other span."
            + (" Every gap in this source clears that, so there is nothing to classify "
               "below."
               if not len(off) else
               f" The {len(off)} that do not split three ways: {len(multi)} round to two or "
               f"more weeks and {len(too_short)} round to zero -- both harmless, because "
               "metrics.flow(horizon_weeks=1) returns NaN across a span outside its "
               f"tolerance -- while {len(mislabelled)} round to exactly one week without "
               "being one, and those are the ones a rounded week count gets wrong.")
        )
    if len(off):
        st.dataframe(off.sort_values("report_date", ascending=False), hide_index=True)
    if sc["n_early"]:
        long_gaps = (
            f", {sc['n_early_long']} of them more than a week apart -- legacy is "
            "effectively semi-monthly before 1993" if sc["n_early_long"] else ""
        )
        st.caption(
            f"Harmless: {sc['n_early']} reports precede {segments.LEGACY_WEEKLY_FROM} "
            f"on {sc['early_weekdays']} different weekdays{long_gaps}. A gap that "
            "rounds to two or more weeks"
            + (f" -- or, like the {len(too_short)} sub-week gap(s) in the table above, "
               "to none --" if len(too_short) else "")
            + " cannot be mislabelled as one week, so lib.metrics.flow returns NaN "
              "across it rather than a wrong horizon."
        )
    if len(mislabelled):
        st.caption(
            f":warning: The rounding trap: {len(mislabelled)} gap(s) of "
            f"{'/'.join(str(int(g)) for g in sorted(set(mislabelled['gap_days'])))} days, "
            "the spans this source actually contains out of the 4, 5, 9 and 10 that "
            "round(days/7) calls one week. None of them is a week -- CFTC's own published "
            "change compares against the previous weekly print seven days back -- so "
            f"metrics.flow refuses them, being more than {metrics.SPAN_TOLERANCE_DAYS:g} "
            "day off seven, and invariant 3 excludes them. metrics.weeks_between still "
            "rounds and would label them one week, which is why that function is "
            "documented for labelling a span and never for validating one."
            + (f" The {len(too_short)} gap(s) of "
               f"{'/'.join(str(int(g)) for g in sorted(set(too_short['gap_days'])))} days "
               "are NOT in this count: they round to zero weeks, so both rules already "
               "return NaN across them." if len(too_short) else "")
        )
    # Corroboration is a corpus-level fact, so it is summed over sources rather
    # than read off the selected one -- only legacy_fut has comparable pairs across
    # a short gap at all, and quoting the selected source would print "0 of 0".
    n = sum(v["chg"]["relabel"]["n"] for v in scans.values())
    bad = sum(v["chg"]["relabel"]["n_bad"] for v in scans.values())
    tot_n = sum(v["chg"]["n"] for v in scans.values())
    tot_bad = sum(v["chg"]["n_bad"] for v in scans.values())
    if n:
        st.caption(
            f"Invariant 3 measures the cost of that mislabelling independently, across "
            f"every source: of the {n} pairs it EXCLUDED because their span rounds to a "
            f"week without being one, {bad} ({100.0 * bad / n:.0f}%) disagree with the "
            f"published change columns, against {tot_bad} of the {tot_n:,} it compared"
            # A rate over zero comparisons is not a rate. A corpus with a short gap
            # and no weekly one is contrived, but it costs one branch to not divide.
            + (f" ({100.0 * tot_bad / tot_n:.4f}%)" if tot_n else "")
            + ". Two unrelated checks landing on the same rows is the strongest signal "
              "this board produces, and what it says is that a span of "
            + "/".join(str(g) for g in sorted(_relabel_spans(scans)))
            + " days must be dropped rather than relabelled -- which is why those "
              f"{bad} rows appear nowhere in the failure counts above."
        )


def _render() -> None:
    scans = {s.id: sc for s in cftc_spec.SPECS if (sc := scan(s.id)) is not None}
    if not scans:
        st.info("No CFTC source is ingested yet, so there is nothing to reconcile.")
        return

    stale_id, stale = min(scans.items(), key=lambda kv: pd.Timestamp(kv[1]["last"]))
    n_incomplete = sum(s["n_incomplete"] for s in scans.values())
    net_fail = sum(s["net"]["n_fail"] for s in scans.values())
    oi_fail = sum(s["resid"]["n_fail"] for s in scans.values())
    chg_bad = sum(s["chg"]["n_bad"] for s in scans.values())
    chg_n = sum(s["chg"]["n"] for s in scans.values())
    rel_n = sum(s["chg"]["relabel"]["n"] for s in scans.values())
    rel_bad = sum(s["chg"]["relabel"]["n_bad"] for s in scans.values())
    dates: dict[str, int] = {}
    for s in scans.values():
        for d, k in s["chg"]["bad_dates"].items():
            dates[d] = dates.get(d, 0) + k
    worst_net = max((s["net"]["max"] for s in scans.values()
                     if not np.isnan(s["net"]["max"])), default=float("nan"))
    res_lo = min((s["resid"]["lo"] for s in scans.values()
                  if not np.isnan(s["resid"]["lo"])), default=float("nan"))
    res_hi = max((s["resid"]["hi"] for s in scans.values()
                  if not np.isnan(s["resid"]["hi"])), default=float("nan"))

    # The denominator of each level counter, so a zero is never read against the
    # wrong population: with a cohort dropped, market-weeks reconciled stays at
    # 1,046,369 while the measured count falls, and the pair of numbers says so in
    # the same glance that shows the zero.
    # Counted separately rather than shared: the two identities need different
    # columns of a market-week, so a null open interest on an otherwise complete
    # week leaves the residual unmeasurable while the net sum is fine.
    net_n = sum(s["net"]["n"] for s in scans.values())
    res_n = sum(s["resid"]["n"] for s in scans.values())
    c = st.columns(5)
    c[0].metric("market-weeks reconciled", f"{sum(s['n_mw'] for s in scans.values()):,}")
    c[1].metric("not checkable (cohort row missing)", f"{n_incomplete:,}")
    c[2].metric(f"|sum net| over {cftc_spec.NET_ZERO_TOL:g}",
                f"{net_fail:,} of {net_n:,}")
    c[3].metric(f"|OI residual| over {cftc_spec.OI_RESIDUAL_TOL:g}",
                f"{oi_fail:,} of {res_n:,}")
    c[4].metric(f"change fields off by over {cftc_spec.CHANGE_FIELD_TOL:g}",
                f"{chg_bad:,} of {chg_n:,}")
    st.caption(
        "Every market-week of every ingested CFTC source, not the week on screen -- and "
        "the five counters are not the same kind of statement, so read them differently."
    )
    # The calibration claim is a claim about a corpus whose maximum is inside the
    # tolerance. It is therefore gated on the counters being zero, not merely
    # narrated beside them: forcing a break used to leave "nothing in the current
    # data can breach them by construction" underneath two counters reading 46,361.
    if net_fail or oi_fail:
        st.caption(
            f":warning: **The two level counters are findings here, not calibration.** "
            f"The tolerances ({cftc_spec.NET_ZERO_TOL:g} and "
            f"{cftc_spec.OI_RESIDUAL_TOL:g} contracts) were set one contract wider than "
            "the widest value the corpus held when they were measured, so anything "
            f"clearing them is new: {net_fail:,} market-week(s) over the net tolerance, "
            f"{oi_fail:,} over the residual one, worst |sum of nets| "
            f"{_num(worst_net)} and a residual spanning {_num(res_lo)}..{_num(res_hi)}. "
            "Rounding does not do that. Treat the per-source failure tables below as the "
            "finding and the rounding explanation as suspended until they are explained."
        )
    else:
        st.caption(
            f"**The two level counters are calibration, not a result.** The tolerances "
            f"({cftc_spec.NET_ZERO_TOL:g} and {cftc_spec.OI_RESIDUAL_TOL:g} contracts) are "
            "set one contract wider than the widest value this corpus contains"
            + (f" -- measured here, the worst |sum of nets| is {worst_net:,.0f} and the "
               f"residual spans {res_lo:,.0f}..{res_hi:,.0f} -- so nothing in the current "
               "data can breach them by construction. A zero there says the tolerance is "
               "calibrated, not that a test passed; they are forward-looking tripwires for "
               "the next ingest."
               if not np.isnan(worst_net) and not np.isnan(res_lo) else
               ", so a zero there never was a test that passed. On this data it is not even "
               "that: no market-week had a complete set of cohort rows, so neither identity "
               "was evaluated and both zeros mean NOT TESTED. See the coverage counter.")
        )
    if chg_bad:
        top = max(dates, key=lambda d: dates[d])
        change_txt = (
            f"{chg_bad} comparisons of {chg_n:,} disagree by more than "
            f"{cftc_spec.CHANGE_FIELD_TOL:g} contracts, clustered on {len(dates)} report "
            f"date(s) with {dates[top]} of them on {top} alone. Clustering by report date "
            "is what a reclassification or a revised prior level looks like, not a "
            "parsing fault."
            + (" The largest cluster is the July 2008 trader reclassification, where the "
               "published change compares a new classification against an "
               "old-classification prior level." if top.startswith("2008-07") else "")
        )
    elif chg_n:
        change_txt = (
            f"No comparison in any ingested source disagrees by more than "
            f"{cftc_spec.CHANGE_FIELD_TOL:g} contracts, over {chg_n:,} comparisons."
        )
    else:
        # Zero of zero. "No comparison disagrees" is true of an empty set and reads
        # as a pass, so the sentence has to name the emptiness instead.
        change_txt = (
            "On this data it has nothing to count: no source holds a single pair of rows "
            "spanning a week, so the identity was never evaluated and the zero beside it "
            "means NOT TESTED rather than agreement."
        )
    if rel_n:
        spans = "/".join(str(g) for g in sorted(_relabel_spans(scans)))
        change_txt += (
            f" Separately, {rel_n} pairs whose span rounds to a week without being one "
            f"are EXCLUDED from that count rather than added to it -- spans of {spans} "
            "days here, out of the 4, 5, 9 and 10 that round(days/7) admits; "
            f"{rel_bad} of them disagree, which is a horizon-labelling artifact and not "
            "a reclassification. See Cadence."
        )
    st.caption(
        "**The change-field counter is the only one of the three that is a real count.** "
        + change_txt
    )
    st.caption(
        "**The coverage counter is the one that would fire on a dropped cohort.** "
        + (f"{n_incomplete:,} market-weeks are missing a cohort row and are therefore "
           "excluded from both level identities, so every zero above is a statement only "
           "about the market-weeks that remained."
           if n_incomplete else
           "It is zero here: every market-week has a full set of cohort rows, so the "
           "identities above were evaluated on all of them.")
        + f" Stalest feed: `{stale_id}`, latest report {stale['last']}."
    )

    sid = st.selectbox("Source to examine in detail", list(scans),
                       format_func=lambda s: f"{scans[s]['label']}  ({s})")
    sc = scans[sid]
    freshness(sc["by_date"].index, f"{sid} report")
    _history(sid, sc)
    _invariant_net(sc, scans)
    _invariant_oi(sc, scans)
    _invariant_change(sc, scans)
    _feeds(scans)
    _cadence(sc, scans)
    caveat_block(*scans)


BOARD = Board(
    id="integrity",
    title="Integrity",
    render=_render,
    # Deliberately empty: this board reconciles whatever is present and says so
    # when nothing is. Naming one source made app.py suppress the whole board when
    # that single file was absent, even though the other six were reconcilable.
    sources=(),
    order=60,
    blurb="the accounting identities, checked across the whole committed dataset.",
)
