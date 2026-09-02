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

WHY EVERY CHECK CARRIES A TOLERANCE. The previous version tested one displayed week
for exact equality and printed BROKEN on any nonzero residual. Both identities fail
an exact test routinely, because CFTC rounds each column independently. An exact
test is therefore not a strict test but a broken one: it fires on the publisher's
rounding and says nothing about the data. Tolerances live in lib/cftc_spec.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, segments, store
from panels import Board

#: A per-market gap this long makes a code's history discontinuous. lib.segments
#: cuts at a quarter; a year is used here so the count is dominated by code reuse.
HOLE_DAYS = 365

#: What a weekly publisher actually emits. Any other gap is where round(days/7)
#: can mislabel a horizon.
WEEKLY_GAPS = (6, 7, 8)

FAIL_COLS = ["report_date", "market_code", "market", "oi", "value"]


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
    missing position bucket would.
    """
    v = mw[col].dropna()
    buckets = v.round().astype("int64")
    buckets = buckets[buckets != 0].value_counts().sort_index()
    return {
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
        # assign BEFORE filtering: in pandas 3.0 an empty frame .assign(series)
        # adopts the series' index and resurrects every row.
        "fail": mw.assign(value=mw[col]).loc[mw[col].abs() > tol, FAIL_COLS],
    }


@st.cache_data(show_spinner=False, max_entries=16)
def _scan(source_id: str, token: tuple[int, str]) -> dict:
    """Every check for one source in one pass, returning only small objects.

    Vectorised groupby over the whole source: 4.1 million rows across all seven in
    ~2.4 s cold, then cached. What comes back is per-report-date (~1,000 rows) or a
    list of exceptions (under 200), so the cache holds kilobytes.
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
    # excluded from the identity statistics and counted separately.
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
    # grouped variant, so the same rule -- calendar span rounded to weeks, never a
    # row-position diff -- is applied within (market_code, cohort) here.
    sub = df.sort_values(["market_code", "cohort", "_d"], kind="mergesort")
    grp = sub.groupby(["market_code", "cohort"], observed=True, sort=False)
    days = grp["_d"].diff().dt.days
    err = pd.concat(
        [(grp["long"].diff() - sub["change_long_published"]).abs(),
         (grp["short"].diff() - sub["change_short_published"]).abs()],
        axis=1,
    ).max(axis=1)
    ok = ((days / 7.0).round() == 1) & err.notna()
    e = err[ok].astype("float64")
    chg = {
        "n": int(len(e)),
        "exact": float((e == 0).mean() * 100) if len(e) else float("nan"),
        "within": float((e <= cftc_spec.CHANGE_FIELD_TOL).mean() * 100) if len(e) else float("nan"),
        "max": float(e.max()) if len(e) else float("nan"),
        "bad": sub.assign(err=err.astype("float64"), gap_days=days)
        .loc[ok & (err > cftc_spec.CHANGE_FIELD_TOL),
             ["report_date", "market_code", "market", "cohort", "gap_days", "err"]]
        .sort_values("err", ascending=False),
        "by_gap": pd.DataFrame({"gap_days": days[ok], "bad": e > cftc_spec.CHANGE_FIELD_TOL})
        .groupby("gap_days")
        .agg(comparisons=("bad", "size"), disagreements=("bad", "sum")),
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
                      "gap_days": gaps, "days_off_a_week": dev})
        .dropna(subset=["gap_days"])
        .loc[dev > 1]
        .assign(previous=lambda f: f["previous"].dt.date,
                report_date=lambda f: f["report_date"].dt.date,
                gap_days=lambda f: f["gap_days"].astype("int64"),
                days_off_a_week=lambda f: f["days_off_a_week"].astype("int64"))
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
        # Metadata, not data: lib.cache exposes no path or size accessor.
        "bytes": store.path_for(source_id).stat().st_size,
    }


def scan(source_id: str) -> dict | None:
    df = cache.read(source_id)
    return None if df.empty else _scan(source_id, _token(df))


def _bars(res: dict, empty: str, key: str) -> None:
    """The nonzero buckets of a residual; zero is dropped and stated in words.

    Zero is 49-100% of every source, so plotting it flattens the only part that
    carries an argument: the shape and symmetry of the nonzero tail.
    """
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
    st.caption(
        "Read the middle and bottom panels together -- that pairing is the whole "
        "argument of invariant 2. Open interest grows by orders of magnitude across "
        "the sample while the worst residual never leaves a few contracts. A missing "
        "position bucket would scale with the market; rounding does not."
        + (f" Dotted line: {segments.LEGACY_WEEKLY_FROM}, before which legacy "
           "reporting is not weekly." if brk else "")
    )


def _invariant_net(sc: dict) -> None:
    st.subheader("Invariant 1. Cohort nets sum to zero")
    net = sc["net"]
    c = st.columns(4)
    c[0].metric("market-weeks", f"{sc['n_mw']:,}")
    c[1].metric("max |sum of nets|", f"{net['max']:,.0f} contracts")
    c[2].metric("exactly zero", f"{net['exact']:.2f}%")
    c[3].metric(f"within {cftc_spec.NET_ZERO_TOL:g}", f"{net['within']:.2f}%")
    st.caption(
        f"Pass condition: the identity holds to within {cftc_spec.NET_ZERO_TOL:g} "
        f"contracts on {net['within']:.2f}% of market-weeks. It is not exact because "
        "CFTC rounds each cohort column independently, so the sum carries one "
        "rounding error per cohort, signed either way -- which is why the nonzero "
        "buckets come in near-equal pairs. An exact test fails on 2,292 of 518,741 "
        "futures-only market-weeks and on about half the supplemental ones, so the "
        "exact test is the bug rather than the finding."
    )
    _bars(net, "Every market-week in this source sums to exactly zero, so there is "
               "no nonzero bucket to plot.", key="net_buckets")
    if len(net["fail"]):
        st.warning(f"{len(net['fail'])} market-weeks clear the tolerance:")
        st.dataframe(net["fail"], hide_index=True)
    if sc["n_incomplete"]:
        st.warning(
            f"{sc['n_incomplete']:,} market-weeks are missing a cohort row and are "
            "excluded above -- a coverage problem, not a rounding one."
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
    c[0].metric("residual range", f"{res['lo']:,.0f} to {res['hi']:,.0f}")
    c[1].metric("negative / positive", f"{res['neg']:,} / {res['pos']:,}")
    c[2].metric("corr with open interest",
                "undefined" if np.isnan(res["corr"]) else f"{res['corr']:+.4f}")
    c[3].metric("largest open interest", f"{sc['oi_max']:,.0f}")
    _bars(res, "The residual is exactly zero on every market-week in this source, "
               "which is the strongest form this evidence takes.", key="resid_buckets")
    st.info(cftc_spec.RESIDUAL_EXPLANATION)
    st.caption(
        ":warning: **Correction.** An earlier version of this repo rendered "
        "\"residual = unpublished non-reportable spread\" beside these same, correct "
        "numbers -- a chart that is wrong while every figure in it is right, which is "
        "the worst failure mode available because nothing looks broken. It is "
        "rounding noise. The table measures the disproof per source: a non-negative "
        "missing spread could only push the residual positive, yet the negative "
        "column is never empty and is usually the larger of the two; the residual "
        "stays inside -4..+4 on markets running to 25.7 million contracts; it is "
        "uncorrelated with open interest; and it is exactly zero on all 184,229 rows "
        "of disagg_fut, the commodity dataset where small-trader calendar spreads "
        "would surface first."
    )
    st.dataframe(
        pd.DataFrame([
            {"source": k, "market-weeks": v["n_mw"],
             "residual range": f"{v['resid']['lo']:,.0f} .. {v['resid']['hi']:,.0f}",
             "negative": v["resid"]["neg"], "positive": v["resid"]["pos"],
             "exactly zero": v["resid"]["zero"],
             "corr with OI": None if np.isnan(v["resid"]["corr"]) else round(v["resid"]["corr"], 4),
             "largest OI": f"{v['oi_max']:,.0f}"}
            for k, v in scans.items()
        ]),
        hide_index=True,
    )
    if len(res["fail"]):
        st.warning(f"{len(res['fail'])} market-weeks clear the tolerance:")
        st.dataframe(res["fail"], hide_index=True)


def _invariant_change(sc: dict) -> None:
    st.subheader("Invariant 3. Our week-over-week change vs the published change_* columns")
    chg = sc["chg"]
    c = st.columns(4)
    c[0].metric("comparable pairs", f"{chg['n']:,}")
    c[1].metric("exact agreement", f"{chg['exact']:.2f}%")
    c[2].metric(f"within {cftc_spec.CHANGE_FIELD_TOL:g}", f"{chg['within']:.3f}%")
    c[3].metric("largest disagreement", f"{chg['max']:,.0f} contracts")
    st.caption(
        "Only pairs spanning one week by the calendar are compared, per "
        "lib.metrics.flow: CFTC skips weeks, so a one-row diff is not a one-week "
        "change. Exact agreement is not the pass test -- combined-basis datasets "
        "(`_futopt`, `supp_cit`) differ by exactly +/-1 on a tenth to a half of "
        "comparisons, against 0-2% in the futures-only reports, because the "
        "delta-equivalent option figures are rounded independently of the levels they "
        "are added to. Same mechanism as invariant 1."
    )
    if len(chg["bad"]):
        st.warning(
            f"{len(chg['bad'])} comparisons clear +/-{cftc_spec.CHANGE_FIELD_TOL:g} "
            "contracts. Two named causes, both real rather than a parsing fault: the "
            "July 2008 trader reclassification, where the published change compares a "
            "new classification against an old-classification prior level (worst on "
            "WTI crude 067651 and natural gas 023651, reaching 323,944 contracts in "
            "legacy_futopt), and the legacy weeks whose gap is not a whole number of "
            "weeks -- see Cadence."
        )
        st.dataframe(chg["bad"].head(12), hide_index=True)
    else:
        st.caption("No comparison in this source clears the tolerance.")


def _feeds(scans: dict[str, dict]) -> None:
    st.subheader("Feed health and cadence, all sources")
    today = pd.Timestamp.today().normalize()
    st.dataframe(
        pd.DataFrame([
            {"source": k, "rows": v["n_rows"], "market-weeks": v["n_mw"],
             "markets": v["n_codes"], "reports": v["n_reports"],
             "first": v["first"], "latest": v["last"],
             "age (d)": (today - pd.Timestamp(v["last"])).days,
             "MB": round(v["bytes"] / 1e6, 1),
             "unit-break codes": v["n_unit_break"],
             f"holes over {HOLE_DAYS}d": v["n_holes"],
             "net fails": len(v["net"]["fail"]), "OI fails": len(v["resid"]["fail"]),
             "change fails": len(v["chg"]["bad"])}
            for k, v in scans.items()
        ]),
        hide_index=True,
    )
    st.caption(
        "`age` is days since the latest REPORT, not since the last ingest: CFTC "
        "publishes Friday 15:30 ET and the job runs daily, so 3-9 days is normal and "
        "one source climbing past ~10 while the others stay low is a stopped feed. "
        "`unit-break codes` counts codes whose contract_units digit signature changes "
        "at least once, by lib.segments' rule -- those levels are not one comparable "
        "series and a percentile must not span the seam. `holes` counts codes with a "
        "gap over a year, which includes outright code reuse."
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


def _odd_gap_comparisons(sc: dict) -> tuple[int, int]:
    """(comparisons, disagreements) for pairs whose gap is not a real weekly gap."""
    bg = sc["chg"]["by_gap"]
    odd = bg.drop(index=[g for g in bg.index if int(g) in WEEKLY_GAPS], errors="ignore")
    return int(odd["comparisons"].sum()), int(odd["disagreements"].sum())


def _cadence(sc: dict, scans: dict[str, dict]) -> None:
    off = sc["gaps_off"]
    n_gaps = sc["n_reports"] - 1
    st.caption(
        f"**Integer-week check, selected source.** {n_gaps - len(off):,} of "
        f"{n_gaps:,} gaps are within one day of a whole number of weeks; worst "
        f"deviation {sc['worst_dev']} day(s). That is what makes round(days/7) exact, "
        "and it is what lib.metrics.flow relies on to label a horizon. Where it "
        "fails, it fails in two ways and only one of them is harmless."
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
            "rounds to two or more weeks cannot be mislabelled as one, so "
            "lib.metrics.flow returns NaN across it rather than a wrong horizon."
        )
    short = off[off["gap_days"] < min(WEEKLY_GAPS)]
    if len(short):
        st.caption(
            f":warning: Not harmless: {len(short)} gap(s) of "
            f"{'/'.join(str(int(g)) for g in sorted(set(short['gap_days'])))} days, at "
            "holiday weeks. round(4/7) is 1, so a four-day change enters "
            "lib.metrics.flow labelled as a one-week change, while CFTC's own published "
            "change compares against the previous weekly print seven days back."
        )
    # Corroboration is a corpus-level fact, so it is summed over sources rather
    # than read off the selected one -- only legacy_fut has comparable pairs across
    # a short gap at all, and quoting the selected source would print "0 of 0".
    odd = [_odd_gap_comparisons(v) for v in scans.values()]
    n, bad = sum(x[0] for x in odd), sum(x[1] for x in odd)
    tot_n = sum(v["chg"]["n"] for v in scans.values())
    tot_bad = sum(len(v["chg"]["bad"]) for v in scans.values())
    if n:
        st.caption(
            f"Invariant 3 finds that independently, across every source: {bad} of {n} "
            "comparisons whose gap is NOT 6-8 days disagree with the published change "
            f"columns, against {tot_bad - bad} of {tot_n - n:,} that do. Two unrelated "
            "checks landing on the same rows is the strongest signal this board "
            "produces, and what it says is that a four-day span should be dropped "
            "rather than relabelled."
        )


def _app_helpers():
    """app.py's shared renderers, obtained without re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not alias it as `app`,
    so the documented `from app import caveat_block` imports a SECOND copy and
    re-runs app.py's body -- including its unkeyed sidebar radio, which raises
    StreamlitDuplicateElementId before either helper is bound. Verified: with the
    plain import this board fails on first render. Every other board carries the
    same shim, so the fix belongs in app.py.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


def _render() -> None:
    caveat_block, freshness = _app_helpers()
    scans = {s.id: sc for s in cftc_spec.SPECS if (sc := scan(s.id)) is not None}
    if not scans:
        st.info("No CFTC source is ingested yet, so there is nothing to reconcile.")
        return

    stale_id, stale = min(scans.items(), key=lambda kv: pd.Timestamp(kv[1]["last"]))
    c = st.columns(4)
    c[0].metric("market-weeks reconciled", f"{sum(s['n_mw'] for s in scans.values()):,}")
    c[1].metric(f"|sum net| over {cftc_spec.NET_ZERO_TOL:g}",
                f"{sum(len(s['net']['fail']) for s in scans.values()):,}")
    c[2].metric(f"|OI residual| over {cftc_spec.OI_RESIDUAL_TOL:g}",
                f"{sum(len(s['resid']['fail']) for s in scans.values()):,}")
    c[3].metric(f"change fields off by over {cftc_spec.CHANGE_FIELD_TOL:g}",
                f"{sum(len(s['chg']['bad']) for s in scans.values()):,} of "
                f"{sum(s['chg']['n'] for s in scans.values()):,}")
    st.caption(
        "Every market-week of every ingested CFTC source, not the week on screen. The "
        "tolerances are one contract wider than the corpus maximum, so anything that "
        "clears them is a genuine reclassification rather than rounding -- worth "
        f"reading rather than assuming. Stalest feed: `{stale_id}`, latest report "
        f"{stale['last']}."
    )

    sid = st.selectbox("Source to examine in detail", list(scans),
                       format_func=lambda s: f"{scans[s]['label']}  ({s})")
    sc = scans[sid]
    freshness(sc["by_date"].index, f"{sid} report")
    _history(sid, sc)
    _invariant_net(sc)
    _invariant_oi(sc, scans)
    _invariant_change(sc)
    _feeds(scans)
    _cadence(sc, scans)
    caveat_block(*scans)


BOARD = Board(
    id="integrity",
    title="Integrity",
    render=_render,
    sources=("cftc_tff_fut",),
    order=60,
    blurb="the accounting identities, checked across the whole committed dataset.",
)
