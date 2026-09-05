"""University of Michigan Surveys of Consumers -- sentiment and inflation expectations.

Three public tables from https://www.sca.isr.umich.edu/tables.html, joined into one
monthly frame:

    tbmics.csv      Month,YYYY,ICS_ALL        Index of Consumer Sentiment, from 1952
    tbmiccice.csv   Month,YYYY,ICC,ICE        current conditions / expectations, 1951
    tbmpx1px5.csv   Month,YYYY,PX_MD,PX5_MD   median expected inflation, 1978

Headers above are verbatim from the live files, not from documentation.

WHY THIS SOURCE IS NOT COMMITTED, WHICH IS A FIRST FOR THIS REPO
----------------------------------------------------------------
Every other source here is US federal work or an openly published API, so mirroring
the fetched values into data/ was free. This one is not, and the reason is worth
stating because the two grants involved contradict each other:

  https://data.sca.isr.umich.edu/agreement.php
      "The data and other materials available through the Surveys of Consumers
      website are the property of the University of Michigan ... You agree not to
      reproduce, retransmit, distribute, sell, publish, or broadcast the data and
      materials from the Surveys of Consumers website without the express written
      consent of the University of Michigan."

  https://data.sca.isr.umich.edu/faq.php
      "Any data, tables, or charts available to the public on our main website can
      be used without permission. We ask that you recreate charts you wish to use
      and cite our data as 'University of Michigan, Survey Research Center,
      Surveys of Consumers.'"

So USE and RE-CHARTING of these three public tables is expressly permitted, and
redistribution is expressly not. A committed parquet in a public repo is
redistribution however you label it. The corroborating evidence is FRED, which
carries the same series as a licensed party -- "Reprinted with permission ... At
the request of the source, the data is delayed by 1 month"
(https://fred.stlouisfed.org/series/UMCSENT). UMich grants redistribution case by
case; it is not open.

The resolution keeps the repo's architecture intact rather than working around it.
The board still reads ONLY from data/*.parquet and never fetches at render time --
that invariant is what makes it impossible for a chart to show a number that is not
on disk. The parquet is simply gitignored (see .gitignore, and the test in
tests/test_sources.py that enforces it from the `redistributable` flag rather than
from a comment). Cost: a fresh clone has no sentiment data until it ingests, and
app.py already renders the "not ingested yet" notice for exactly that case.

If you want it committed, the agreement's own escape hatch is "express written
consent". UMich does not publish a permissions channel; the general contact address
on sca.isr.umich.edu/contact.html is umsurvey@umich.edu. With consent in hand, flip
`redistributable` and drop the .gitignore line.

Two things NOT to conclude from the above. The data site is not login-gated -- it is
publicly readable, and what sponsors buy is early access, the public copy lagging by
a four-week embargo. And this is a reading of two conflicting public statements, not
legal advice.

WHAT THE NUMBERS ARE
--------------------
ICS is an index, 1966 Q1 = 100, built from five survey questions. ICC and ICE are
its two halves: appraisals of the CURRENT situation and EXPECTATIONS for the
future. ICE is the component that feeds the Conference Board-style leading-indicator
reading; ICC tracks what people are experiencing now.

PX_MD and PX5_MD are MEDIAN expected change in "prices in general", in percent. The
questions, verbatim from UMich's questionnaire documentation
(data.sca.isr.umich.edu/fetchdoc.php?docid=75433):

    "During the next 12 months, do you think that prices in general will go up, or
    go down, or stay where they are now?" then "By about what percent do you expect
    prices to go (up/down) on the average, during the next 12 months?"

    "What about the outlook for prices over the next 5 to 10 years? ..." then "By
    about what percent PER YEAR do you expect prices to go (up/down) on the
    average, during the next 5 to 10 years?"

Three consequences, each of which a careless label would get wrong:

  IT IS NOT A CPI EXPECTATION. The question says "prices in general" and never
  mentions the CPI or the respondent's own cost of living -- the survey asks about
  own income, gasoline and home prices as separate items. UMich's own research
  found only about one in five respondents knew the most recently published CPI
  change and a third had never heard of it. So this is not a forecast of CPI by
  people trying to forecast CPI.

  THE LONG HORIZON IS 5 TO 10 YEARS, PER YEAR. "Five-year" is UMich's own
  convenience label -- they say so explicitly -- and the number is an annual rate,
  not a cumulative change over the period.

  MEDIAN, NOT MEAN, AND THE GAP IS NOT A CONSTANT. UMich publishes both and states
  "the median is the headline and preferred measure ... the mean is not our
  preferred measure due to its sensitivity to extremely high responses". The
  published mean is Winsorized (capped at +50%/-10%) so it depends on a truncation
  rule the median does not; and the wedge moves with the regime -- about +1.2pp
  year-ahead over 1978-1995, +0.7pp over 2010-2019, and roughly +2.9pp in 2026.
  Quoting a mean against this chart would look like a different, much worse number.

THE CADENCE IS NOT MONTHLY FOR MOST OF THE HISTORY
--------------------------------------------------
Counted from the live file rather than assumed:

    1952          1 observation
    1953-1959     2-3 a year
    1960-1977     4 a year   (quarterly)
    1978-present  12 a year  (monthly)

So "the monthly table" is quarterly for its first quarter-century. Drawing it as a
continuous monthly line would imply observations that were never taken, which is
why the chart leaves gaps rather than joining across them, and why any rank of the
level is computed over the monthly era. PX5_MD is additionally blank for 105 early
rows -- the 5-to-10-year question was not always asked.
"""
from __future__ import annotations

import csv
import io
import urllib.request

import pandas as pd

from lib.schema import TableSchema

BASE = "https://www.sca.isr.umich.edu/files"

#: table file -> the columns we take from it, as {upstream name: our name}.
#: Names are the verbatim CSV headers. A rename upstream shows up as a missing
#: column in the schema check rather than as a silently all-null series.
TABLES = {
    "tbmics.csv": {"ICS_ALL": "sentiment"},
    "tbmiccice.csv": {"ICC": "current_conditions", "ICE": "expectations"},
    "tbmpx1px5.csv": {"PX_MD": "infl_exp_1y", "PX5_MD": "infl_exp_5y10y"},
}

#: Month name -> number, explicitly, NOT strptime("%B").
#: %B is locale-dependent: under a non-English LC_TIME every row fails to parse and
#: the whole series silently becomes NaT. The upstream files are English-only and
#: this mapping says so.
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}

UA = {"User-Agent": "watchboard research (personal, low volume; charts recreated per sca.isr.umich.edu FAQ)"}

#: The month monthly interviewing began. Before this the survey ran quarterly or
#: less, so a rank of the level across the boundary would compare a monthly
#: observation against a sparser era.
MONTHLY_FROM = "1978-01-01"


def _read_table(name: str, columns: dict[str, str]) -> pd.DataFrame:
    """One upstream CSV as a frame indexed by survey month.

    Parsed with the csv module rather than handed straight to pandas so that a
    changed header is a KeyError naming the column, at the boundary, instead of a
    column of NaN that charts as a flatline.
    """
    url = f"{BASE}/{name}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        body = r.read().decode("utf-8-sig")

    rows = list(csv.DictReader(io.StringIO(body)))
    if not rows:
        raise RuntimeError(f"{name}: no rows")

    header = set(rows[0])
    # Parenthesised deliberately: `-` binds tighter than `|`, so writing
    # {"Month","YYYY"} | set(columns) - header puts Month and YYYY in `missing`
    # unconditionally and the fetch raises every time, even on a healthy file.
    missing = ({"Month", "YYYY"} | set(columns)) - header
    if missing:
        raise RuntimeError(
            f"{name}: expected columns {sorted(missing)} are gone. Header is now "
            f"{sorted(header)}. The upstream table changed shape -- fix the map in "
            "sources/umich.py rather than letting it write nulls."
        )

    out = []
    for row in rows:
        month = MONTHS.get((row.get("Month") or "").strip())
        year = (row.get("YYYY") or "").strip()
        if month is None or not year.isdigit():
            # A footnote line, not an observation. Skipped, but counted below so a
            # file that becomes mostly footnotes cannot pass quietly.
            continue
        rec = {"survey_date": f"{int(year):04d}-{month:02d}-01"}
        for upstream, ours in columns.items():
            raw = (row.get(upstream) or "").strip()
            rec[ours] = float(raw) if raw else None
        out.append(rec)

    kept = len(out)
    if kept < 0.9 * len(rows):
        raise RuntimeError(
            f"{name}: only {kept} of {len(rows)} rows parsed as observations. Either "
            "the month names changed or the file is no longer a plain table."
        )
    print(f"  . {name:<16} {kept:>4} months, {out[0]['survey_date']} .. {out[-1]['survey_date']}")
    return pd.DataFrame.from_records(out).set_index("survey_date")


def fetch(since: str | None = None) -> pd.DataFrame:
    """All three tables, outer-joined on the survey month.

    `since` is accepted and ignored. The three files together are about 40 KB and
    always carry full history, so there is nothing to narrow and no reason to risk
    an incremental window drifting out of step with a revision. Full history every
    run also means the preliminary-to-final revision of the newest month is picked
    up automatically: the upsert is keyed on survey_date, so the final value
    overwrites the preliminary one rather than appending a second row.
    """
    del since

    frame = None
    for name, columns in TABLES.items():
        part = _read_table(name, columns)
        # OUTER join: the three tables start in different years (1951, 1952, 1978),
        # so an inner join would silently discard a quarter-century of sentiment
        # history for want of an inflation reading that was never collected.
        frame = part if frame is None else frame.join(part, how="outer")

    frame = frame.sort_index().reset_index()
    frame["year"] = frame["survey_date"].str.slice(0, 4).astype(int)

    print(
        f"  . joined {len(frame)} months, {frame['survey_date'].iloc[0]} .. "
        f"{frame['survey_date'].iloc[-1]}"
    )
    return frame


SOURCE_KWARGS = dict(
    id="umich_sca",
    label="Consumer sentiment & inflation expectations (U. Michigan)",
    fetch=fetch,
    key=("survey_date",),
    sort_key=("survey_date",),
    schema=TableSchema(
        required=("survey_date", "sentiment"),
        numeric=(
            "sentiment",
            "current_conditions",
            "expectations",
            "infl_exp_1y",
            "infl_exp_5y10y",
            "year",
        ),
        optional=("year",),
    ),
    group="Inflation & the consumer",
    cadence=(
        "Monthly. Preliminary reading mid-month, final at month end, 10:00 ET. "
        "The table carries whichever is current, so the preliminary value is "
        "revised in place."
    ),
    # The files carry full history every time, so a missed run costs nothing.
    backfillable=True,
    incremental=False,
    provenance=(
        "sca.isr.umich.edu/files/{tbmics,tbmiccice,tbmpx1px5}.csv -- the public "
        "monthly tables linked from sca.isr.umich.edu/tables.html. ICS_ALL is the "
        "Index of Consumer Sentiment (1966 Q1 = 100); ICC/ICE are its current-"
        "conditions and expectations halves; PX_MD and PX5_MD are MEDIAN expected "
        "price change over the next year and next 5-10 years, in percent."
    ),
    license=(
        "Copyright The Regents of the University of Michigan. Public tables may be "
        "USED without permission and charts must be recreated, per "
        "data.sca.isr.umich.edu/faq.php; redistribution requires written consent, "
        "per data.sca.isr.umich.edu/agreement.php. Hence redistributable=False."
    ),
    redistributable=False,
    citation="University of Michigan, Survey Research Center, Surveys of Consumers",
    caveats=(
        "Not affiliated with or endorsed by the University of Michigan. Charts here "
        "are recreated from the published public tables, as the source asks.",
        "THE SERIES IS NOT MONTHLY BEFORE 1978. Counted from the file: one reading "
        "in 1952, two or three a year through 1959, quarterly 1960-1977, monthly "
        "from January 1978. Gaps are drawn as gaps, and ranks of the level are "
        "computed over the monthly era only.",
        "This is expected change in PRICES IN GENERAL, not in the CPI. The question "
        "never mentions the CPI, and UMich's own research found only about one in "
        "five respondents knew the most recently published CPI change. Do not read "
        "it as households forecasting the CPI print.",
        "MEDIANS, not means. UMich calls the median \"the headline and preferred "
        "measure\" because the mean is sensitive to extreme answers and depends on "
        "a capping rule. The gap between them is not a constant -- roughly +1.2pp "
        "year-ahead in 1978-1995, +0.7pp in 2010-2019, about +2.9pp in 2026 -- so a "
        "mean quoted against this chart is a different and much worse number.",
        "The long-run measure is the expected rate PER YEAR over the next 5 TO 10 "
        "years. \"Five-year\" is the source's own convenience label; it is not a "
        "cumulative five-year change.",
        "The 5-to-10-year question was not always asked -- 105 early months are "
        "blank -- so the long-run series starts later than the year-ahead one.",
        "The newest month may be the PRELIMINARY reading and can be revised at the "
        "final release. The ingest re-pulls full history and overwrites by survey "
        "month, so a revision replaces rather than duplicates.",
    ),
)
