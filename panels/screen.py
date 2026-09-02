"""Screen -- which markets are at an extreme of their own history right now.

WHAT IT ANSWERS. Pick a report family, a cohort (or a sum of cohorts) and a
ranking statistic, and this ranks every market in the chosen tier by how extreme
its CURRENT reading is against ITS OWN past. It exists because every other board
here needs you to already know which market to open, and "which market" is the
question you actually have on a Friday afternoon.

WHAT IT REFUSES TO ANSWER. Whether an extreme is worth trading. Positioning is
contemporaneous with price, this repo carries no price series, and a
cross-sectional rank has a base rate by construction -- with a 3-year window of
~157 weekly observations, SOME market is in its top decile every single week. The
number here is a level, not an edge; the base-rates board runs the conditional
test.

THREE CHOICES THAT ARE NOT COSMETIC.

1. Every statistic is computed inside the market's LAST SEGMENT (lib.segments,
   keyed on the digits of contract_units), never across its whole file. 20974+
   went from 49,531 to 255,954 contracts of open interest on 2023-05-02 because
   the contract was re-based from $100 to $20 a point. Ranking today against the
   pre-2023 rows would rank $20 observations against $100 ones and report the
   result as a percentile.
2. Markets whose current segment is shorter than universe.MIN_HISTORY reports are
   DROPPED AND COUNTED, not silently omitted. A recently re-based market has a
   perfectly computable percentile of 6 observations and it means nothing. The
   same accounting covers the second way a market leaves the ranking: a
   statistic that is UNDEFINED for it. A cohort flat at zero for the whole
   window has no min-max span and no sigma, so its COT index and its z-score are
   blank -- three disagg markets are in that state right now -- and the count
   and the reason are printed rather than left as a silent gap between the
   caption and the table.
3. Every share is over open interest MINUS spreads. A spread is long one expiry
   and short another, so it carries no direction; leaving it in the denominator
   understates the tilt by 1/(1 - spread_share) -- 1.8x in 3-month SOFR
   futures-only and 3.4x in the futures-and-options version, where spreads are
   70.3% of open interest. The denominator stays ONE-SIDED for the gross measure
   too, which is why a gross share is a share out of 200 rather than 100; see
   SHARE_GROSS, which is what the screen says when the measure is gross.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, segments, universe
from lib.ui import caveat_block, freshness
from panels import Board

#: Per family, the cohort(s) that make the screen worth opening. tff sums the two
#: speculative cohorts because asset managers and leveraged funds are the two
#: halves of the buy side and either alone is half a book; disagg and legacy have
#: a single speculative cohort each; supp's ex-index non-commercial is the closest
#: thing to legacy noncomm once the index traders are carved out.
DEFAULT_COHORTS = {
    "tff": ("asset_mgr", "lev_money"),
    "disagg": ("m_money",),
    "legacy": ("noncomm",),
    "supp": ("noncomm_nocit",),
}

TIERS = {
    "CORE": ("CORE",),
    "CORE + WIDE": ("CORE", "WIDE"),
    "everything ingested": ("CORE", "WIDE", "THIN"),
}

#: column -> (label, bar unit, charts kind, needs history, why it can be blank,
#: what it costs you). `needs_history` is the gate: a raw share is readable off one
#: report, a percentile is not. `blank_when` is the sentence the drop accounting
#: prints when the statistic is undefined for a market that cleared the history
#: gate -- one per statistic, because they do not fail for the same reason. The
#: z-score's kind is 'purity' because a standardised score is signed and
#: dimensionless and metrics has no kind for one; ranked_bars uses kind ONLY to
#: decide diverging colour and the zero line, so the mismatch is cosmetic here --
#: but see the library note in the module docstring of lib.metrics.
STATS: dict[str, dict] = {
    "trail_pct": dict(
        label="Trailing 3y percentile", unit="percentile of trailing 3y",
        kind="share", needs_history=True,
        blank_when="the trailing 1,095-day window holds fewer than 60% of the ~157 "
                   "weekly reports it expects (~94), which is how a gappy market "
                   "avoids reporting a percentile of four observations",
        help="Rank of today's level within the trailing 1,095 days of the current "
             "segment. Blank unless that window holds ~94 of the ~157 weekly "
             "reports it expects, which is how a gappy market avoids reporting a "
             "percentile of four observations."),
    "exp_pct": dict(
        label="Expanding percentile (whole segment)", unit="percentile of segment",
        kind="share", needs_history=True,
        blank_when="every observation in the current segment is missing, so there "
                   "is nothing to rank the last one against",
        help="Rank against every observation in the current segment. Longer memory "
             "than the 3y window, and after twenty years that memory is dominated "
             "by regimes that no longer exist."),
    "cot": dict(
        label="COT index (3y min-max)", unit="COT index",
        kind="share", needs_history=True,
        blank_when="the trailing 3-year min and max are EQUAL, so the min-max span "
                   "is zero -- this cohort has held an identical position, in "
                   "practice flat at zero, for the whole window -- or that window "
                   "holds under ~94 reports",
        help="100*(v-min)/(max-min) over 3 years -- what published COT commentary "
             "means by 'positioning is at 90', so it is here to check that "
             "commentary. It is a min-max, not a rank: one old print pins the "
             "scale for three years."),
    "z": dict(
        label="Trailing 3y z-score", unit="sigma vs trailing 3y",
        kind="purity", needs_history=True,
        blank_when="the trailing 3-year standard deviation is zero -- the same "
                   "position every week, in practice flat at zero -- so there is "
                   "nothing to standardise by, or the window holds under ~94 reports",
        help="Keeps resolution in the tails where a rank saturates at 0 and 100, "
             "and loses meaning exactly where a rank survives: fat tails and "
             "regime shifts make sigma unstable."),
    "share": dict(
        label="Share of directional OI", col_label="% of directional OI",
        # The SELECTOR's label may not depend on the measure -- the widget would
        # rename its own options under the reader when the measure flips -- so it
        # stays the one wording that is true of both cases. Everything downstream
        # (header, bar unit, help) comes from _stat_for and does depend on it.
        menu_label="Share of one-sided open interest",
        unit="% of directional OI",
        kind="share", needs_history=False,
        blank_when="open interest minus spread positions is zero or unpublished, "
                   "leaving no denominator",
        help="Today's net over open interest minus spreads. No history needed, so a "
             "short segment costs nothing here -- and no market is compared against "
             "its own past either. This is a size, not an extreme."),
}

#: The gross measure's share is the same ratio with a numerator that counts BOTH
#: sides, so it needs its own labels rather than sharing the net ones.
#:
#: The denominator stays one-sided (open interest minus spreads) on purpose, and
#: the consequence has to be on screen rather than inferred: a market's cohort
#: gross shares sum to exactly 200%, not 100%, because the published longs sum to
#: OI-minus-spreads and so do the shorts (verified on legacy futures-only 044601,
#: 2026-08-25: comm 150.17 + noncomm 36.12 + nonrept 13.71 = 200.00). So a value
#: over 100% is a cohort holding more than one side's worth of contracts, not a
#: bug -- 20 of the 272 markets on the legacy futures-only CORE+WIDE screen are
#: above it, topped by NANO XRP (176LM1) at 198.4%.
#:
#: Halving it by dividing by 2x directional OI would make the cohorts partition
#: 100%, at the cost of halving every ONE-SIDED market's number relative to its
#: own net share -- flipping the measure toggle would then change the scale as
#: well as the quantity. Keeping the denominator and stating the scale is the
#: honest half of that trade.
SHARE_GROSS = dict(
    label="Gross vs one-sided OI (cohorts sum to 200)",
    col_label="% of one-sided OI (sums to 200)",
    unit="% of one-sided OI (200 = both sides in full)",
    help="Long+short over open interest minus spreads. The numerator counts both "
         "sides and the denominator counts one, so this is a share out of 200, not "
         "100: a market's cohorts sum to exactly 200% and anything over 100% is a "
         "cohort holding more than one side's worth of contracts. Read it as "
         "comparable book size across markets, never as a fraction of the market.",
)


def _stat_for(stat_key: str, measure: str) -> dict:
    """The statistic's display metadata, adjusted where the MEASURE changes it.

    Only the share needs this, and it needs it badly: 'Share of directional OI'
    describes the net case and misdescribes the gross one by a factor of two.
    """
    stat = dict(STATS[stat_key])
    if stat_key == "share" and measure == "gross":
        stat.update(SHARE_GROSS)
    return stat


#: (column, label, number format or None for text, help). Table-driven because a
#: caption that explains a column belongs next to the column, and eleven inline
#: column_config calls buried the code that decides WHICH columns to show.
TABLE_COLS = (
    ("market_label", "Market", None,
     "Latest published name plus the market_code, which is the stable key. 26-30% "
     "of codes have been renamed and CFTC shortened names wholesale on 2022-02-08, "
     "so the code is what to carry between boards."),
    ("subgroup", "Subgroup", None, "CFTC's own subgroup, or the group where it is null."),
    ("value", "{measure} (contracts)", "localized",
     "Contracts, never a percent change: a net going -100,640 to -96,727 is +3,913 "
     "contracts and LESS short, not +3.89%."),
    # Header and help both depend on the measure and on the pool: '{share}' comes
    # from the share statistic (net and gross are on different scales) and the
    # spread sentence is measured off the markets actually on screen, because the
    # median spread share runs from 2.0% to 36% across the seven families and the
    # three tiers and any constant here would contradict the column beside it.
    ("share", "{share}", "%.2f",
     "Denominator is open interest minus spread positions, which carry no "
     "direction."),
    ("flow_1w", "1w flow", "localized",
     "Blank where the market skipped a week. CFTC gaps are integer weeks, so a "
     "one-row diff across a hole is a two-week change wearing a one-week label."),
    ("flow_4w", "4w flow", "localized", "Same rule over 4 weeks."),
    ("open_interest", "Open interest", "localized",
     "The denominator. Read it next to the share, always -- a share can move "
     "because the cohort traded or because open interest did."),
    ("spread_pct", "Spreads % of OI", "%.1f",
     "How much of this market is calendar structure rather than direction."),
    ("reports_segment", "Reports", "%d",
     "Reports in the current contract-unit segment. The rank uses only these; "
     "earlier rows may count a different contract."),
    ("last_report", "Last report", None,
     "This market's own latest print, not the corpus latest. Markets skip weeks."),
)


def _last(s: pd.Series) -> float:
    return float(s.iloc[-1]) if len(s) else float("nan")


@st.cache_data(show_spinner="Ranking every market in the family...", max_entries=8)
def _cross_section(source_id: str, cohorts: tuple[str, ...], measure: str) -> pd.DataFrame:
    """One row per live market: the current reading of every ranking statistic.

    Cached on (source, cohorts, measure) and computes ALL the statistics rather
    than the selected one, so flipping the ranking statistic is free. Measured on
    the worst case in the corpus -- legacy_fut, 391 live market codes -- the whole
    pass is ~2s cold and 0 thereafter.

    metrics.last_percentile is used for the expanding case on purpose: building
    the full expanding curve to read its final point is ~1,300x the work, and at
    391 markets that is the difference between a page and a coffee break.
    """
    df = cache.read(source_id)
    summary = cache.market_summary(source_id)
    if df.empty or summary.empty:
        return pd.DataFrame()

    keys = ["market_code", "report_date"]
    # Open interest, units and the concentration columns are IDENTICAL across a
    # market-week's cohort rows, so they come off .first(); summing them would
    # multiply open interest by the cohort count. The spread total is the one
    # thing that must be summed over EVERY cohort even when only one is selected:
    # the directional-OI denominator is a property of the market, not the cohort.
    week = df.groupby(keys, observed=True).agg(
        open_interest=("open_interest", "first"),
        spread_total=("spread", "sum"),
        contract_units=("contract_units", "first"),
    )
    picked = df[df["cohort"].isin(cohorts)].groupby(keys, observed=True)[["long", "short"]].sum()
    frame = week.join(picked, how="inner").reset_index()

    # A market with no print in the last 8 reports has no "this week" to screen.
    # The 8-report window rather than the latest date is deliberate: markets drop
    # out of individual weeks and come back (MICRO E-MINI DJIA carried 63,517
    # contracts three weeks before the latest report and is absent from it), so a
    # latest-date filter would churn the universe by ~5x. The table carries each
    # market's own last report date so a stale reading is visible rather than
    # implied to be current.
    live = set(summary.loc[summary["n_last8"] >= 1, "market_code"])
    frame = frame[frame["market_code"].isin(live)]
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["report_date"])

    rows = []
    for code, g in frame.groupby("market_code", observed=True, sort=False):
        g = g.sort_values("date")
        seg = g[segments.last_segment_mask(g["date"], g["contract_units"]).to_numpy()]
        if seg.empty:
            continue
        d = seg["date"]
        long_, short_ = seg["long"].astype("float64"), seg["short"].astype("float64")
        v = metrics.net(long_, short_) if measure == "net" else metrics.gross(long_, short_)
        oi = seg["open_interest"].astype("float64")
        spread = seg["spread_total"].astype("float64")
        rows.append(
            {
                "market_code": code,
                "value": _last(v),
                "previous": float(v.iloc[-2]) if len(v) > 1 else float("nan"),
                "trail_pct": _last(metrics.trailing_percentile(v, d)),
                "exp_pct": metrics.last_percentile(v),
                "cot": _last(metrics.cot_index(v, d)),
                "z": _last(metrics.trailing_zscore(v, d)),
                "share": _last(metrics.share_of(v, metrics.directional_oi(oi, spread))),
                "flow_1w": _last(metrics.flow(v, d, horizon_weeks=1)),
                "flow_4w": _last(metrics.flow(v, d, horizon_weeks=4)),
                "open_interest": _last(oi),
                "spread_pct": _last(metrics.spread_share(spread, oi)),
                "reports_segment": int(len(seg)),
                "segment_start": d.iloc[0].date().isoformat(),
                "last_report": d.iloc[-1].date().isoformat(),
            }
        )
    if not rows:
        return pd.DataFrame()

    # can_rank_full_history rather than can_rank: universe.assign_tiers computes
    # can_rank off the CURRENT segment, which is the gate this panel already
    # applies itself via reports_segment. The whole-file count is what tells a
    # short-segment drop apart from a genuinely young market, which is the only
    # thing it is used for here.
    cols = ["market_code", "market", "market_full", "subgroup", "group", "tier",
            "n_reports", "can_rank_full_history", "superseded_by"]
    return pd.DataFrame(rows).merge(summary[cols], on="market_code", how="left")


def _hover(rows: pd.DataFrame, measure: str) -> list[str]:
    """Rule 4 and rule 3 in the tooltip: raw contracts, open interest, sample size.

    EVERY FIELD HERE CAN BE MISSING, and a tooltip is the last place a NaN should
    reach a reader. Two of them used to:

    - `charts.fmt_change(nan, 'flow', ...)` formats as '+nan contracts', which is
      what the flow column's own help promises not to do ("blank where the market
      skipped a week"). A market whose previous print is not exactly one week back
      has no one-week flow, so the tooltip says that instead.
    - `r.value > r.previous` is False when `previous` is NaN, so the comparison
      quietly returned "smaller book" for a market with ONE observation in its
      current segment -- a direction of travel asserted by a NaN, not measured.
      The net branch already refused this via signed_direction's guard, which is
      the behaviour mirrored here, and an empty direction drops its parentheses
      rather than rendering '()'.
    """
    out = []
    for r in rows.itertuples():
        if measure == "net":
            direction = charts.signed_direction(r.previous, r.value)
        elif pd.isna(r.previous) or pd.isna(r.value):
            direction = ""
        elif r.value == r.previous:
            direction = "unchanged"
        else:
            direction = "larger book" if r.value > r.previous else "smaller book"

        level = f"{r.value:,.0f} contracts" if pd.notna(r.value) else "not published"
        flow_txt = (
            charts.fmt_change(r.flow_1w, "flow", "contracts")
            if pd.notna(r.flow_1w)
            else "no comparable week (no print exactly one week back)"
        )
        oi_txt = (
            f"open interest {r.open_interest:,.0f} contracts"
            if pd.notna(r.open_interest)
            else "open interest not published"
        )
        n = int(r.reports_segment)
        out.append(
            f"{r.market_full}<br>"
            f"{measure} {level}"
            + (f" ({direction})" if direction else "")
            + f"<br>1-week flow {flow_txt}<br>"
            + oi_txt
            + (f", spreads {r.spread_pct:.1f}% of it" if pd.notna(r.spread_pct) else "")
            + f"<br>{n} report{'' if n == 1 else 's'} in its current contract-unit "
            f"segment, from {r.segment_start}"
        )
    return out


def _table_order(stat_key: str) -> list[str]:
    """Ranking statistic first, then the fixed columns.

    The share is in the table whatever we rank on -- but when it IS what we rank
    on, selecting it twice is a duplicate column, not a second opinion, and
    st.dataframe refuses the frame outright.
    """
    fixed = [c for c, *_ in TABLE_COLS if c != "market_label"]
    if stat_key != "share":
        fixed.insert(0, stat_key)
    return ["market_label"] + fixed


def _spread_note(pool: pd.DataFrame) -> str:
    """The spread share of open interest, MEASURED ON THE MARKETS ON SCREEN.

    This used to be the constant "a median 3.2% of OI but 44.5% in 3-month SOFR",
    which is the tff futures-only figure and is wrong everywhere else: the median
    spread share of the pool runs 2.0% (legacy futures-only, everything ingested)
    to 36.0% (disagg futures-and-options, CORE) to 31.3% (supplemental), where the
    same caption sat next to a Spreads % of OI column showing ten times its claim.
    A number recomputed from the rendered rows cannot drift away from them.
    """
    sp = pool["spread_pct"].dropna()
    if sp.empty:
        return "No spread figure is published for the markets ranked here."
    top = pool.loc[sp.idxmax()]
    note = (
        f"Measured on the {len(sp)} markets ranked here, not quoted from a "
        f"constant: spreads are a median {sp.median():.1f}% of open interest, up "
        f"to {sp.max():.1f}% in {top['market']} ({top['market_code']})"
    )
    # 1/(1 - s) is the factor by which leaving spreads in the denominator would
    # understate the tilt. Stated only where it is finite -- a published spread
    # total at or above open interest would make it meaningless, not large.
    if sp.max() < 100.0:
        note += f", where the correction to the tilt is {1 / (1 - sp.max() / 100):.1f}x"
    return note + "."


def _index_tradeoff(data: pd.DataFrame, codes: list[str]) -> str:
    """State the Consolidated-vs-leg trade-off IN THE NUMBERS OF THIS FAMILY.

    universe.CONSOLIDATED_TRADEOFF states it in tff futures-only numbers (846
    reports from 2010-06-15, 1,055 from 2006-06-13), and those are wrong in the
    legacy families, where the same E-mini leg carries 1,506 reports from
    1997-09-16. So the sentence is built from the rows on screen instead.
    """
    bits = []
    for code in codes:
        c = data[data["market_code"] == code]
        if c.empty:
            continue
        row = c.iloc[0]
        legs = data[data["superseded_by"] == code].sort_values(
            "reports_segment", ascending=False
        )
        bit = (
            f"{row['market']} ({code}) ranks on {int(row['reports_segment']):,} of "
            f"its {int(row['n_reports']):,} reports, its current contract unit "
            f"starting {row['segment_start']}"
        )
        if not legs.empty:
            leg = legs.iloc[0]
            seg, tot = int(leg["reports_segment"]), int(leg["n_reports"])
            leg_txt = (
                f"all {seg:,} of its reports, no unit break"
                if seg == tot
                else f"{seg:,} of its {tot:,}"
            )
            bit += (
                f", against {leg_txt} for the {leg['market']} leg "
                f"({leg['market_code']}) from {leg['segment_start']}"
            )
        bits.append(bit)
    text = (
        "A Consolidated series is notionally correct (E-mini + Micro/10, verified "
        "against the published figures to under one contract) and it carries the "
        "worst scale break in the dataset; the single legs are notionally wrong to "
        "add up — a Micro is a tenth of an E-mini per contract — and they have the "
        "unbroken history. Never rank both at once: the leg's exposure is already "
        "inside the Consolidated figure."
    )
    if bits:
        text += " " + ". ".join(bits) + "."
    return text


def _column_config(stat_key: str, stat: dict, measure: str, spread_note: str) -> dict:
    cfg: dict = {}
    for col, label, fmt, help_ in TABLE_COLS:
        # The share's HEADER depends on the measure: a net share is out of 100 and
        # a gross share of the same denominator is out of 200.
        label = label.format(
            measure=measure.title(), share=_stat_for("share", measure)["col_label"]
        )
        if col == "share":
            help_ = f"{help_} {spread_note}"
        cfg[col] = (
            st.column_config.TextColumn(label, help=help_)
            if fmt is None
            else st.column_config.NumberColumn(label, format=fmt, help=help_)
        )
    # The ranked column carries the statistic's own caveat, and when it is the
    # share it replaces the generic share header rather than sitting beside it.
    ranked_help = stat["help"]
    if stat_key == "share":
        ranked_help = f"{ranked_help} {spread_note}"
    cfg[stat_key] = st.column_config.NumberColumn(
        stat["label"], format="%.2f" if stat_key == "share" else "%.1f",
        help=ranked_help,
    )
    return cfg


def _bars(rows: pd.DataFrame, col: str, kind: str, unit: str, measure: str, key: str) -> None:
    st.plotly_chart(
        charts.ranked_bars(
            # The code travels with the name because the name is not the key: it
            # is the label, and it changes.
            [f"{r.market} ({r.market_code})" for r in rows.itertuples()],
            [float(x) for x in rows[col]],
            unit=unit,
            kind=kind,
            hover=_hover(rows, measure),
        ),
        key=key,
    )


def render() -> None:  # noqa: PLR0915 -- one screen, read top to bottom
    ids = [s.id for s in cftc_spec.SPECS]
    c1, c2 = st.columns([2, 3])
    source_id = c1.selectbox(
        "Report family", ids, index=ids.index("cftc_tff_fut"),
        format_func=lambda i: cftc_spec.spec_for(i).label,
    )
    spec = cftc_spec.spec_for(source_id)
    summary = cache.market_summary(source_id)
    if summary.empty:
        st.info(
            f"`{source_id}` has not been ingested, so this family cannot be "
            "screened. Run `python -m scripts.ingest --source " + source_id + "`. "
            "Every other family in the selector still works."
        )
        return

    labels = {c.id: c.label for c in spec.cohorts}
    cohorts = c2.multiselect(
        "Cohort — summed if you pick more than one",
        list(labels), default=list(DEFAULT_COHORTS[spec.family]),
        format_func=lambda c: labels[c],
    )
    c3, c4, c5, c6 = st.columns([1, 1.2, 2.4, 1.4])
    measure = c3.radio("Measure", ["net", "gross"], horizontal=True,
                       help="Net is direction and changes sign. Gross is book size "
                            "and is what there is to unwind.")
    tier = c4.radio("Universe", list(TIERS), horizontal=False)
    stat_key = c5.selectbox("Ranking statistic", list(STATS),
                            format_func=lambda k: STATS[k].get("menu_label")
                            or STATS[k]["label"])
    per_side = c6.slider("Markets per side", 5, 30, 12)
    # _stat_for, not STATS[...]: the share's label, unit and help all change with
    # the measure, because a gross over a one-sided denominator is out of 200.
    stat = _stat_for(stat_key, measure)
    st.caption(stat["help"])

    if not cohorts:
        st.info("Pick at least one cohort. Nothing to rank until you do.")
        return

    data = _cross_section(source_id, tuple(sorted(cohorts)), measure)
    if data.empty:
        st.info("No market in this family has a print in the last 8 reports.")
        return

    pool = data[data["tier"].isin(TIERS[tier])].copy()
    # The trade-off is surfaced whenever EITHER side of it is in the tier, not only
    # when a leg is. On the landing view (CORE) only the Consolidated codes are
    # here -- universe.assign_tiers keeps a superseded leg out of CORE -- so
    # triggering on the leg meant the default view silently picked the short,
    # seamed series and offered no way to see the long one.
    consolidated_here = [c for c in universe.CONSOLIDATED if (pool["market_code"] == c).any()]
    swapped_in_legs = False
    if consolidated_here or pool["superseded_by"].notna().any():
        if st.checkbox(
            "Equity indices: rank the Consolidated series (uncheck to rank the "
            "E-mini / Micro legs instead, which have the longer unbroken history)",
            value=True,
        ):
            pool = pool[pool["superseded_by"].isna()]
        else:
            # Swap, never union: the legs come in from the whole family (they sit
            # in WIDE, so a CORE screen cannot reach them otherwise) and the
            # Consolidated codes go out, so no exposure is counted twice.
            legs = data[data["superseded_by"].isin(consolidated_here)]
            swapped_in_legs = not legs.empty
            pool = pd.concat(
                [pool[~pool["market_code"].isin(consolidated_here)], legs]
            ).drop_duplicates(subset="market_code")
        st.caption(_index_tradeoff(data, consolidated_here))

    # ONE accounting for BOTH ways a market leaves the ranking, and it is computed
    # before anything is dropped. The caption used to be printed above the
    # dropna(), so a market whose statistic is undefined vanished from the charts
    # and the table with no accounting at all -- e.g. disagg futures-only,
    # everything ingested, COT index: caption "201 of 290", table 198 rows.
    n_pool = len(pool)
    short_seg = (
        pool["reports_segment"] < universe.MIN_HISTORY
        if stat["needs_history"]
        else pd.Series(False, index=pool.index)
    )
    rebased = short_seg & pool["can_rank_full_history"].fillna(False)
    undefined = ~short_seg & pool[stat_key].isna()
    blanked = pool[undefined]
    pool = pool[~(short_seg | undefined)]

    where = "in this tier plus the index legs you asked for" if swapped_in_legs else "in this tier"
    lines = [f"Ranking {len(pool)} of {n_pool} markets {where}."]
    if stat["needs_history"] and int(short_seg.sum()):
        lines.append(
            f"{int(short_seg.sum())} dropped for holding under "
            f"{universe.MIN_HISTORY} reports in their CURRENT contract-unit "
            f"segment; {int(rebased.sum())} of those have {universe.MIN_HISTORY}+ "
            "reports in the file but were re-based or re-issued, so their older "
            "rows count a different contract and cannot rank against today."
        )
    elif stat["needs_history"]:
        lines.append(
            f"None dropped for short history: every market here holds "
            f"{universe.MIN_HISTORY}+ reports in its current contract-unit segment."
        )
    if not blanked.empty:
        named = ", ".join(
            f"{r.market} ({r.market_code})" for r in blanked.head(3).itertuples()
        )
        lines.append(
            f"{len(blanked)} more dropped because {stat['label']} is UNDEFINED "
            f"there, not because they are short of history: {stat['blank_when']}. "
            + named
            + (" among them." if len(blanked) > 3 else ".")
        )
    elif not stat["needs_history"]:
        lines.append(
            "A raw share needs no history, so nothing is dropped for being young — "
            "and nothing is compared against its own past either."
        )
    st.caption(" ".join(lines))

    if pool.empty:
        st.info(
            f"No market in this tier has a usable {stat['label']}: "
            f"{stat['blank_when']}. Try another statistic — a raw share is readable "
            "off a single report."
        )
        return

    kind = stat["kind"]
    if stat_key == "share":
        kind = "net_share" if measure == "net" else "gross_share"

    ranked = pool.sort_values(stat_key, ascending=False)
    top, bottom = ranked.head(per_side), ranked.tail(per_side).iloc[::-1]
    hi, lo = (
        ("Most stretched long", "Most stretched short")
        if measure == "net"
        else ("Largest book vs its own history", "Smallest book vs its own history")
    )
    # With fewer markets than 2x per_side the two tails overlap. Say so rather
    # than letting the same name appear on both sides as if it were two findings.
    # min(per_side, n) first: head(n) and tail(n) both return the WHOLE pool once
    # per_side passes the pool size, so the overlap saturates at n. Without the
    # clamp the supplemental default (9 markets, slider 12) printed "15 name(s)
    # appear on both because only 9 markets rank here".
    shown = min(per_side, len(ranked))
    overlap = max(0, 2 * shown - len(ranked))
    b1, b2 = st.columns(2)
    with b1:
        st.caption(f"**{hi}** — {stat['label']}, {len(cohorts)} cohort(s) summed")
        _bars(top, stat_key, kind, stat["unit"], measure, "screen_hi")
    with b2:
        st.caption(
            f"**{lo}** — same statistic, other tail"
            + (f", and the two tails overlap: {overlap} of the {len(ranked)} ranked "
               "markets are on both sides" if overlap else "")
        )
        _bars(bottom, stat_key, kind, stat["unit"], measure, "screen_lo")

    if stat_key == "share" and measure == "gross":
        over = int((ranked["share"] > 100).sum())
        st.caption(
            f"**This share is out of 200, not 100.** Gross is long+short and the "
            f"denominator is one-sided (open interest minus spreads), so a market's "
            f"cohorts sum to 200% and {over} of the {len(ranked)} markets here print "
            f"above 100% — the largest is {ranked['share'].max():.1f}% "
            f"({ranked.loc[ranked['share'].idxmax(), 'market_code']}), a cohort "
            "holding more than one side's worth of contracts. Switch the measure to "
            "net for a share of one side that is out of 100."
        )

    st.caption(
        "Open interest is on screen as a COLUMN below, because these are bars and "
        "bars cannot share an x-axis with it. You need it: a share moves when the "
        "cohort trades (numerator) or when open interest does (denominator), and "
        "the share alone cannot say which. Read the open-interest and 1-week-flow "
        "columns together before calling a rank a position change."
    )

    show = ranked.copy()
    show["market_label"] = show["market"] + " (" + show["market_code"] + ")"
    show["subgroup"] = show["subgroup"].fillna(show["group"]).fillna("(ungrouped)")
    st.dataframe(
        show[_table_order(stat_key)], hide_index=True, height=420,
        column_config=_column_config(stat_key, stat, measure, _spread_note(ranked)),
    )
    st.caption(
        "Summing cohorts adds their long and short columns before netting, so an "
        "asset manager long offset by a leveraged fund short cancels. That is the "
        "point when you want the speculative side as one book, and a distortion if "
        "you wanted either one on its own. Cohort set: "
        f"{', '.join(labels[c] for c in cohorts)}."
    )
    st.caption(
        "**Base rate first.** A cross-sectional extreme is guaranteed to exist: with "
        "~157 weekly observations in a 3-year window, some market prints in its top "
        "decile every week whether or not anything happened. Positioning is also "
        "contemporaneous with price rather than predictive, and this repo holds no "
        "price series, so no forward return is computable here at all. Take a name "
        "from this screen to the **Base rates** board before treating a rank as "
        "information."
    )
    # freshness() gets the SOURCE's newest report date -- the whole market summary,
    # not the screened pool and not one market. A market that simply did not print
    # this week is not a stale feed; the per-market date is the last_report column.
    freshness(summary["last_report"], f"{spec.label} report")
    caveat_block(source_id)


BOARD = Board(
    id="screen",
    title="Screen",
    render=render,
    sources=("cftc_tff_fut",),
    order=10,
    blurb="What is unusual this week, across every market at once.",
)
