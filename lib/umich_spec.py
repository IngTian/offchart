"""Regime boundaries for the Michigan survey, with a citation per boundary.

WHY THIS FILE EXISTS AT ALL

The payload will never tell you. tbmics.csv is a bare `Month,YYYY,ICS_ALL` with no
mode column, no footnote and no flag; tables.html carries no caveat either. Every
discontinuity below is documented only in UMich's own technical PDFs, so it has to
be written down somewhere in this repo or it is lost. This is the same job
lib/cftc_spec.py does for contract re-specifications, and for the same reason: a
level that changes basis without changing appearance corrupts every rank computed
across it while leaving the chart looking fine.

THE ONE THAT MATTERS MOST: PHONE -> WEB, 2024

UMich moved from cell-phone RDD to address-based web sampling over four months and
QUANTIFIED the level shift it caused:

    "we observe average method effects of -6.6 percentage points for [the Index of
    Consumer Sentiment and -13.6 for the Current Conditions Index]"
    -- data.sca.isr.umich.edu/fetchdoc.php?docid=75437

and then deliberately did NOT adjust the series:

    "Because the magnitude of the method effects varies across questions, we have
    opted to transition to the new methodology over the course of four months
    rather than implement a different adjustment factor for each question going
    forward."
    -- sca.isr.umich.edu/files/methodtransitionannouncement2024.pdf

Read those two together and the consequence is precise: the four-month blend buys
the absence of a visible STEP, at the cost of leaving the level displacement inside
the published series unflagged. A chart of the level is unaffected -- there is
genuinely no jump to see. A PERCENTILE of the level is destroyed. Measured here with
lib.metrics.expanding_percentile on the ingested file, August 2026 sentiment of 51.7
is the 1.2th percentile of the full 676-month history and the 19.2th of the 26-month
web era. Those are different claims about the world and only the second is
meaningful: the first mostly says "web readings are lower than phone readings",
which is a fact about the instrument, not about consumers.

The decisive evidence that this is UMich's own reading, not ours: when they want to
compare a current web-era level against history, they abandon the published
phone-era series and use their parallel web collection instead --

    "The May 2026 reading was 44.8 and, as many noticed, lower than the official
    June 2022 reading of 50.0. It is also lower than the June 2022 web reading that
    was 46.3 ... even a methodologically consistent comparison [shows]"
    -- sca.isr.umich.edu/files/partisaneconomy202607_revised.pdf

WHAT IS DELIBERATELY *NOT* HERE

- No synthetic "subtract 6.6 from the phone era" bridge. It would restore hundreds
  of observations to the rank pool and it is tempting for exactly that reason. But
  UMich refused to publish such an adjustment precisely because the effect varies
  by question, and it is not even constant within a question: the phone-web gap in
  June 2022 was 3.7 points against a -6.6 average. A bridge would be our number
  wearing their authority.
- No partisan-regime segmentation. Their three partisanship reports converge on the
  aggregate being sound -- national estimates track independents on level and
  trend, the groups co-move at ~0.85, and the 2025 Republican-share decline
  post-dates the mode transition by a year. The partisan swing is signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Below this many observations a percentile is not reported at all.
#:
#: The web era had 26 monthly readings as of August 2026, which is a resolution of
#: about 3.8 percentile points per observation -- one month of tariff news moves the
#: rank by four points, and the extremes of the pool are set by two or three months.
#: A chip that says "15th percentile" off 26 observations reads exactly as
#: authoritative as one off 500 and is not. 36 is three years of monthly data.
MIN_RANK_OBSERVATIONS = 36


@dataclass(frozen=True)
class Regime:
    """The window a series may be ranked within, and why it starts there."""

    #: Column in the ingested frame.
    column: str
    #: Human label for the pool, shown next to the rank.
    pool_label: str
    #: First survey month admitted to the rank pool, ISO.
    start: str
    #: Months to exclude outright, ISO -- mixed-instrument readings that belong to
    #: no population. They are still PLOTTED; they just cannot be ranked.
    excluded: tuple[str, ...] = ()
    #: Verbatim reason, and the URL it came from.
    reason: str = ""
    citation: str = ""


#: Apr/May/Jun 2024 are 75/25, 50/50 and 25/75 mixtures of two instruments, so each
#: is an average of two different populations and belongs to neither.
BLEND_MONTHS = ("2024-04-01", "2024-05-01", "2024-06-01")

#: The mode periodization, verbatim from docid 75437:
#:   "January - March 2024: RDD Cell Telephone Surveys
#:    April - June 2024: RDD Cell Telephone Surveys and ABS Web Surveys
#:    July 2024 onward: ABS Web Surveys"
WEB_ERA_START = "2024-07-01"

_METHOD_DOC = "https://data.sca.isr.umich.edu/fetchdoc.php?docid=75437"
_PRICE_DOC = "https://data.sca.isr.umich.edu/fetchdoc.php?docid=75433"

REGIMES: tuple[Regime, ...] = (
    Regime(
        column="sentiment",
        pool_label="web-era months",
        start=WEB_ERA_START,
        excluded=BLEND_MONTHS,
        reason=(
            "Collection mode changed from cell-phone RDD to address-based web over "
            "April-July 2024. UMich measured the level shift at -6.6 index points "
            "and chose not to adjust the series, so the displacement is inside the "
            "published numbers with no flag."
        ),
        citation=_METHOD_DOC,
    ),
    Regime(
        column="current_conditions",
        pool_label="web-era months",
        start=WEB_ERA_START,
        excluded=BLEND_MONTHS,
        reason=(
            "Same mode change, and this is the WORST affected series: UMich measured "
            "-13.6 index points, over twice the effect on the headline index."
        ),
        citation=_METHOD_DOC,
    ),
    Regime(
        column="expectations",
        pool_label="web-era months",
        start=WEB_ERA_START,
        excluded=BLEND_MONTHS,
        reason=(
            "Same mode change. UMich published no method effect for this component; "
            "the index identity implies roughly -2 points, the mildest of the three."
        ),
        citation=_METHOD_DOC,
    ),
    Regime(
        column="infl_exp_1y",
        pool_label="months since the 1982 wording change",
        # NOT segmented at the 2024 mode change: UMich report "few differences in
        # medians between interviews conducted via phone and web" -- the ~+2pp
        # method effect they measured is on the MEAN, which this repo does not
        # ingest. The binding break for the median is much older.
        start="1982-03-01",
        reason=(
            "A \"same probe\" was added in March 1982 to separate respondents who "
            "expected the price LEVEL to hold from those who expected the RATE to "
            "hold; one-third to one-half of prior \"same\" answers had been "
            "misclassified. January 1978 to February 1982 was then retroactively "
            "model-adjusted by regression, moving an average 7.8pp from \"same\" to "
            "\"up\", so that stretch is imputed rather than measured."
        ),
        citation=_PRICE_DOC,
    ),
    Regime(
        column="infl_exp_5y10y",
        pool_label="months since the series became continuous",
        start="1990-04-01",
        reason=(
            "The 5-to-10-year question was not asked every month until 1990. "
            "Counted from the live file: present in 1 month of 1979, 3 of 1980, "
            "about 6 a year through 1985, 4 in 1986 and 1987, NONE in 1988 or 1989, "
            "9 in 1990 and every month from 1991. An expanding window over the "
            "1980s would rank a monthly reading against an irregular sample."
        ),
        citation=_PRICE_DOC,
    ),
)

BY_COLUMN = {r.column: r for r in REGIMES}

#: The whole-series break that applies to the INDEX columns regardless of mode:
#: before January 1978 the survey was in-person and quarterly or sparser, so those
#: readings are not comparable in frequency even though they are comparable in
#: level. Kept separate from Regime because it bounds the CHART, not just the rank.
MONTHLY_FROM = "1978-01-01"


def rank_pool(frame: pd.DataFrame, column: str) -> pd.Series:
    """`frame[column]` restricted to the months that may be ranked together.

    Returns the series with excluded months and pre-regime months set to NaN rather
    than dropped, so the result stays aligned to the frame's index and a caller can
    hand it straight to a causal-percentile function without re-indexing.
    """
    regime = BY_COLUMN.get(column)
    values = frame[column].copy()
    if regime is None:
        return values
    dates = pd.to_datetime(frame["survey_date"])
    values[dates < pd.Timestamp(regime.start)] = float("nan")
    for month in regime.excluded:
        values[dates == pd.Timestamp(month)] = float("nan")
    return values


def rankable(frame: pd.DataFrame, column: str) -> int:
    """How many observations the pool actually has. Below MIN_RANK_OBSERVATIONS the
    caller must not print a rank."""
    return int(rank_pool(frame, column).notna().sum())
