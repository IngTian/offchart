"""Splitting one market code's history into stretches that are safe to compare.

A CFTC market code is a stable primary key -- (code, report_date) has zero
duplicate groups in all seven datasets -- but it is NOT a stable *series*. Three
things happen to a code over 40 years, and each one makes a level computed
before the event incomparable with a level computed after it:

CONTRACT RE-SPECIFICATION. The contract multiplier changes and every position
count re-bases. The one that matters most is 2023-05-02, when CFTC re-based its
Consolidated equity index series from the big contract to the E-mini:

    20974+  NASDAQ-100 Consolidated   OI 49,531 -> 255,954   (x5.17)
            contract_units  '(NASDAQ 100 INDEX X $100)' -> '(NASDAQ 100 INDEX X $20)'

That is not a positioning change, it is a unit change. This repo previously
defaulted to that exact market and ran an expanding percentile straight across
the seam, so its headline "40th percentile" reading was ranking $20-per-point
observations against $100-per-point ones.

Counted by THIS MODULE'S OWN RULE over the committed data -- which is the number
that matters, since an earlier docstring quoted a different rule's count and drifted:

    cftc_legacy_fut      96 of 951 codes      cftc_legacy_futopt   91 of 940
    cftc_disagg_fut      56 of 652            cftc_disagg_futopt   56 of 682
    cftc_tff_fut         13 of 146            cftc_tff_futopt      13 of 146
    cftc_supp_cit         0 of  13

CODE REUSE. The same code is retired and later re-issued for a different
contract:

    191691  ALUMINUM - COMMODITY EXCHANGE  '(CONTRACTS OF 40,000 POUNDS)'
            1986-01-15 .. 1989-02-28, then a 12,341-day hole, then the same
            code and name again from 2022-12-13 with '(25 Metric Tons)'

Concatenating those into one series and taking a diff across the hole is
meaningless. 146 legacy codes, 108 disagg codes and 13 tff codes have a hole
longer than a year.

CADENCE CHANGE. Legacy is not weekly before 2002-01-08 -- 646 report dates
precede it with a Tue/Fri/Wed/Mon/Thu mix and 161 gaps of 11-18 days. Even
after it, gaps of 3, 4, 6 and 8 days occur. So a one-row diff is not a one-week
change and must never be labelled as one.

The response is to segment: percentiles, ranges and z-scores are computed WITHIN
a segment, and a chart draws a visible break between segments rather than
joining them with a line that implies continuity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: A gap longer than this ends a segment. 13 weeks is deliberately generous --
#: CFTC skips the odd week for holidays and government shutdowns, and a
#: three-month hole is qualitatively different from a missed print.
MAX_GAP_DAYS = 91

#: Report dates before this are not weekly in the legacy datasets.
LEGACY_WEEKLY_FROM = "2002-01-08"


def unit_signature(units: str | None) -> str:
    """The part of contract_units that changes the arithmetic: its digits.

    contract_units is free text and churns cosmetically -- exchange renames,
    capitalisation, punctuation. What matters is whether the NUMBERS changed:

        '(NASDAQ 100 INDEX X $100)'  ->  '100'
        '(NASDAQ 100 INDEX X $20)'   ->  '20'      break
        '(CONTRACTS OF 5,000 BUSHELS)' -> '5000'
        '(THOUSAND BUSHELS)'         ->  ''        break, and a real one:
            legacy grains are denominated in thousand bushels before 1998-01-06
            and in 5,000-bushel contracts after, a ~5x level discontinuity.

    Commas are stripped so '5,000' and '5000' agree.
    """
    if not units:
        return ""
    return "".join(ch for ch in str(units).replace(",", "") if ch.isdigit())


def segment_ids(
    dates: pd.Series,
    units: pd.Series | None = None,
    *,
    max_gap_days: int = MAX_GAP_DAYS,
) -> pd.Series:
    """Label each observation with a segment number, starting at 0.

    A new segment starts when either the unit signature changes or the gap since
    the previous report exceeds `max_gap_days`. Input must be sorted by date;
    the result is aligned to `dates.index`.
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if d.empty:
        return pd.Series([], dtype="int64", index=dates.index)

    gap_break = d.diff().dt.days.fillna(0) > max_gap_days

    if units is None:
        unit_break = pd.Series(False, index=d.index)
    else:
        sig = pd.Series(units).reset_index(drop=True).map(unit_signature)
        # Only a change between two KNOWN signatures is a break. contract_units
        # is null on some rows and a null must not manufacture two breaks (one
        # into the gap and one out of it).
        known = sig.replace("", np.nan).ffill()
        unit_break = known.ne(known.shift()) & known.shift().notna()

    ids = (gap_break | unit_break).cumsum().astype("int64")
    ids.index = dates.index
    return ids


def last_segment_mask(dates: pd.Series, units: pd.Series | None = None) -> pd.Series:
    """Boolean mask selecting only the most recent segment.

    This is what a "current reading" panel should use. Ranking today's level
    against a pre-re-basing regime is not a percentile of anything.
    """
    ids = segment_ids(dates, units)
    if ids.empty:
        return pd.Series([], dtype=bool, index=dates.index)
    return ids == ids.max()


def weeks_elapsed(dates: pd.Series) -> pd.Series:
    """Whole weeks since the previous report, per observation.

    CFTC publishes weekly, and every hole is an integer number of weeks: across
    all ~46,000 gaps in tff_fut the largest deviation from a multiple of 7 days
    is ONE day, including the 3,255-day E-mini Russell hole. So round(days/7) is
    exact and is the right way to say "how long a change actually covers".

    Use this instead of .diff() whenever a change is labelled with a horizon. A
    one-row diff over a skipped week is a two-week change, and calling it weekly
    is the kind of error that survives review because the number looks fine.
    """
    d = pd.to_datetime(pd.Series(dates))
    days = d.diff().dt.days
    return (days / 7.0).round()


def describe_segments(dates: pd.Series, units: pd.Series | None = None) -> pd.DataFrame:
    """One row per segment: bounds, length, unit signature. For the UI."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    u = (
        pd.Series(units).reset_index(drop=True)
        if units is not None
        else pd.Series([None] * len(d))
    )
    ids = segment_ids(d, u)
    rows = []
    for sid, idx in pd.Series(range(len(d))).groupby(ids.to_numpy()):
        sl = idx.to_numpy()
        rows.append(
            {
                "segment": int(sid),
                "start": d.iloc[sl[0]].date(),
                "end": d.iloc[sl[-1]].date(),
                "reports": len(sl),
                "units": u.iloc[sl[-1]],
                "unit_signature": unit_signature(u.iloc[sl[-1]]),
            }
        )
    return pd.DataFrame(rows)
