"""Market-wide leverage from the Federal Reserve's Z.1 Financial Accounts.

Endpoint: federalreserve.gov/releases/z1/data/FRB_Z1_csv.zip (no key, honest
User-Agent accepted, no robots.txt on the host). 64 MB zipped, one member Z1.csv,
long format, 7.78 million rows covering every Z.1 series. Quarterly, released about
ten weeks after quarter end; 2026:Q2 was published 2026-09-11.

WHY THIS AND NOT FINRA, STATED PLAINLY

What everyone means by "margin debt" is FINRA's monthly series: debit balances in
customers' securities margin accounts, reported by member firms under Rule 4521,
back to January 1997. It is the better series and it is not here, for one reason:
FINRA's Terms of Use prohibit the way a program would have to get it.

Their grant is real -- content "shall be used ONLY for your own non-commercial
personal or professional use" -- but the restrictions carve out the mechanism:

  (d) "develop or create a database of data using the FINRA Website"
  (e) "use any process to monitor or copy the FINRA Website in bulk", or any
      "data mining, scraping or harvesting tools (including robots)"
  (m) use any portion of the site "in the development of any software program"
  (o) any purpose "for which you have not been granted written permission"

Those are not redistribution clauses, so "this repo stores no data and the user
fetches it locally" is not an answer to them -- it is a description of them. Reading
the page yourself is fine and inside the grant; shipping a fetcher that automates
(d), (e) and (m) in a public repository is not. FINRA also states the data "is not
distributed elsewhere and no data feeds exist", so there is no licensed mirror.

Federal Reserve data has no equivalent clause. It is US Government work, the host
serves no robots.txt, and it is the PUBLISHER rather than a mirror -- FRED carries
the same numbers but is a redistribution of this file, and its CSV endpoint
additionally refuses an honest User-Agent while serving a request with no UA at all,
which is not something to build on.

WHAT THIS SERIES IS, AND IS NOT

FL663067003 is "Security brokers and dealers; receivables due from customers (margin
loans and other receivables)". It is NOT FINRA's margin debt and must never be
labelled as such. Measured at the same date, 2026:Q2:

    FINRA margin debt                     $1,502,072 M
    Z.1 broker-dealer receivables           $742,321 M     ~49%

and the ratio drifts rather than being a scalar: ~0.55 at the 2021 peak, ~0.66 at the
2022 trough, ~0.49 now. The Fed's "Security brokers and dealers" is a national-
accounts SECTOR, so customer margin lending booked in bank-affiliated entities lands
elsewhere in Z.1. Drawdowns differ too: 2021 peak to 2022 trough was about -38% on
FINRA against -26% here.

So this is a different measure of the same phenomenon. It agrees on direction -- both
inflate in manias and collapse in deleveragings, both fell roughly half through the
GFC -- and disagrees on level and severity. Use it for regime, not for the number
anyone quotes.

THE DENOMINATOR IS PART OF THE SOURCE

A level from 1945 grows 593x, almost all of which is the economy getting larger, so a
raw line says little beyond "number goes up" -- lib/metrics' third rule is that a
normaliser must be stated rather than assumed. FL153064105 (households and nonprofit
organizations; corporate equities; asset) is fetched alongside it as the thing the
borrowing is secured against, so the ratio is a stock over a stock rather than a stock
over an annualised flow like GDP.
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile

import pandas as pd

from lib.schema import TableSchema

ENDPOINT = "https://www.federalreserve.gov/releases/z1/data/FRB_Z1_csv.zip"
UA = {"User-Agent": "offchart research (personal, low volume)"}

#: Z.1 series name -> the column it becomes. Quarterly only: the `.A` annual twins
#: carry the same numbers at lower resolution and would duplicate the key.
#:
#: Values arrive in MILLIONS of dollars (unit_mult column) and are scaled to dollars
#: here, so nothing downstream has to remember the unit -- the same decision
#: sources/leveraged_etf_flows makes for share counts.
SERIES = {
    "Z1/Z1/FL663067003.Q": "broker_receivables",
    "Z1/Z1/FL153064105.Q": "household_equities",
    # The DECOMPOSITION, and it is not optional colour. The aggregate above is the
    # sum of these three to the dollar, and its composition changed completely:
    #
    #   $bn              2000-03  2007-06  2009-03  2021-12  2026-06
    #   households           265      299      120      147      227
    #   hedge funds dom        0        0        0       88      131
    #   hedge funds for        0        0        0      283      384
    #   = aggregate          265      299      120      518      742
    #
    # So before about 2010 the aggregate IS household margin debt, and today it is
    # 69% hedge funds. Reading the long series as one thing compares households in
    # 2000 against households-plus-hedge-funds now. The Fed added hedge funds as a
    # sector in this release (2026:Q2, with private credit vehicles), which is also
    # why the earlier quarters read zero rather than small.
    "Z1/Z1/FL153167005.Q": "household_margin",
    "Z1/Z1/FL623167003.Q": "hedge_fund_margin_domestic",
    "Z1/Z1/FL623167063.Q": "hedge_fund_margin_foreign",
    # A different INSTRUMENT -- bank loans, not broker accounts -- so it is outside
    # the aggregate entirely. Carried because it is where household margin borrowing
    # actually went: zero in 2000, $493bn now. Any read of "household leverage" that
    # looks only at broker accounts misses it.
    "Z1/Z1/FL153168015.Q": "household_margin_from_banks",
}

#: The member inside the zip. One file, long format.
MEMBER = "Z1.csv"


def fetch(since: str | None = None) -> pd.DataFrame:
    """Both series, every quarter. `since` is ignored.

    The endpoint is one static bulk file with no parameters, so there is nothing to
    narrow and incremental=False. 64 MB is streamed and filtered rather than read
    whole: Z1.csv is 3.3 GB uncompressed across 7.78 million rows, and the two series
    wanted here are about 600 of them.

    Z.1 revises history with every release -- this one incorporated private credit
    vehicles and hedge funds -- so the full series is re-read each time and
    lib.db.upsert overwrites on the key. A revision is a corrected number, not a new
    one.
    """
    req = urllib.request.Request(ENDPOINT, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()

    archive = zipfile.ZipFile(io.BytesIO(raw))
    wide: dict[str, dict[str, float]] = {}
    with archive.open(MEMBER) as handle:
        reader = csv.DictReader(
            io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
        )
        for record in reader:
            column = SERIES.get(record.get("series_name") or "")
            if column is None:
                continue
            value = (record.get("value") or "").strip()
            if not value:
                continue
            # Z.1 dates are quarter-END calendar dates (1945-12-31 is 1945:Q4).
            period = (record.get("date") or "").strip()
            if not period:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            mult = (record.get("unit_mult") or "").strip().lower()
            if mult == "millions":
                number *= 1e6
            elif mult == "billions":
                number *= 1e9
            else:
                # Refuse rather than guess: a silent unit change would rescale the
                # whole series by a thousand and still draw a plausible chart.
                raise ValueError(
                    f"{record['series_name']} arrived with unit_mult {mult!r}, which "
                    "this source does not know how to scale. Check the Z.1 release "
                    "notes before widening this."
                )
            wide.setdefault(period, {})[column] = number

    if not wide:
        return pd.DataFrame()

    rows = [{"period_date": period, **values} for period, values in sorted(wide.items())]
    frame = pd.DataFrame(rows)
    for column in SERIES.values():
        if column not in frame:
            frame[column] = pd.NA
    return frame.sort_values("period_date").reset_index(drop=True)


SOURCE_KWARGS = dict(
    id="fed_z1_leverage",
    label="Market leverage (Federal Reserve Z.1)",
    fetch=fetch,
    key=("period_date",),
    sort_key=("period_date",),
    schema=TableSchema(
        required=("period_date", "broker_receivables"),
        numeric=(
            "broker_receivables", "household_equities", "household_margin",
            "hedge_fund_margin_domestic", "hedge_fund_margin_foreign",
            "household_margin_from_banks",
        ),
        # Only the aggregate is required. The denominator and the components are
        # optional because a Z.1 release that reorganised a sector should degrade
        # the decomposition panel rather than fail the whole ingest -- and this
        # release reorganised two sectors, so that is not hypothetical.
        optional=(
            "household_equities", "household_margin", "hedge_fund_margin_domestic",
            "hedge_fund_margin_foreign", "household_margin_from_banks",
        ),
    ),
    cadence="Quarterly, released about ten weeks after quarter end (2026:Q2 on 2026-09-11)",
    incremental=False,
    redistributable=True,
    license=(
        "US Government work, Board of Governors of the Federal Reserve System, Z.1 "
        "Financial Accounts of the United States. No copyright notice, licence, "
        "attribution requirement or anti-automation clause on the release page, and "
        "www.federalreserve.gov serves no robots.txt."
    ),
    citation=(
        "Board of Governors of the Federal Reserve System, Z.1 Financial Accounts of "
        "the United States."
    ),
    group="Leverage",
    provenance=(
        "federalreserve.gov/releases/z1/data/FRB_Z1_csv.zip, member Z1.csv, series "
        "Z1/Z1/FL663067003.Q (broker-dealer receivables due from customers) and "
        "Z1/Z1/FL153064105.Q (households and nonprofit organizations; corporate "
        "equities; asset). Both in millions of dollars, scaled to dollars on read."
    ),
    caveats=(
        "This is NOT FINRA's margin debt and must not be labelled as such. At "
        "2026:Q2 it reads $742bn against FINRA's $1,502bn -- about 49% -- and the "
        "ratio drifts (~0.55 at the 2021 peak, ~0.66 at the 2022 trough). The Fed's "
        "'Security brokers and dealers' is a national-accounts sector, so margin "
        "lending booked in bank-affiliated entities sits elsewhere in Z.1.",
        "The concept is 'margin loans AND OTHER RECEIVABLES', so it is broader than "
        "margin lending even within the sector it covers.",
        "Quarterly and published about ten weeks late, so it describes a quarter that "
        "has already ended. It is a regime measure, never a timing one.",
        "Z.1 revises history with every release; the 2026:Q2 release incorporated "
        "private credit vehicles and hedge funds, which can move older quarters.",
        "The level alone is not comparable across decades -- it grows 593x from 1945 "
        "mostly because the economy did. Read the share of household equity holdings.",
        "THE AGGREGATE CHANGES COMPOSITION. It is households plus domestic and "
        "foreign hedge funds, summing to the dollar, and the mix inverted: 100% "
        "households through 2009, 69% hedge funds by 2026:Q2 ($515bn of $742bn). "
        "Comparing today's reading to 2000's compares two different populations, so "
        "read the decomposition rather than the total.",
        "Household margin borrowing also moved OUTSIDE this series. Margin loans "
        "from depository institutions are a different Z.1 instrument -- $0 in 2000, "
        "$493bn at 2026:Q2 -- and are carried separately here rather than folded in.",
    ),
)
