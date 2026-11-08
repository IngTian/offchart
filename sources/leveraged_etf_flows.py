"""ProShares leveraged ETF flows -- the money that has to trade whatever it thinks.

Endpoint: accounts.profunds.com/etfdata/historical_nav.csv (no key, no cookie, no
User-Agent games; 51.9 MB, 532,018 rows, 163 funds as of 2026-11-07). Linked from
proshares.com/resources/data-downloads. Overnight T+1: today's file carries through
the previous close, last-modified ~01:15 GMT.

WHY THIS SOURCE, NEXT TO CFTC

CFTC tells you what discretionary money chose to hold, weekly, as of a Tuesday,
published Friday. This tells you what a mandate FORCES, daily, with no view at all.
A leveraged fund must rebalance to its multiple every single day, so a 1% index move
sets off a known quantity of buying or selling that no participant decided on. That
is a different object from positioning and the board had nothing like it.

It also answers the washout question directly, which positioning cannot: on
2026-11-05 TQQQ's NAV fell 19.10% and $837.7M flowed IN. Whether money arrives or
leaves on the worst day is a fact about conviction, and it is daily.

FULL HISTORY, WHICH IS WHY THIS SOURCE AND NOT ITS COMPETITOR

Measured, not assumed: TQQQ and SQQQ from 2010-02-09, UPRO/SPXU from 2009-06-23,
SSO/QLD from 2006-06-19, USD from 2007-01-29, all through the latest close with no
gaps beyond US market holidays. Direxion (SOXL/SOXS) was considered and rejected:
its only usable endpoint serves one day, `?date=` and `?asOfDate=` are silently
ignored (byte-identical responses, same ETag), so its history would begin the day
collection began -- and this repo deleted the machinery for collect-as-you-go
sources, deliberately. A source that answers nothing on day one is not a source yet.

THE FLOW FORMULA IS NOT THE OBVIOUS ONE

`diff(shares_outstanding) * nav` is wrong here, and wrong on exactly the funds a
washout chart is about. See lib.metrics.etf_flow for the measurements; briefly, the
file is retroactively split-adjusted, SQQQ's cumulative reverse-split factor pushes
early share counts below the published column's two decimals of thousands, and 140
SQQQ rows read exactly 0.0. flow comes from the unrounded assets column instead.

`shares_outstanding` is still carried, because it is the honest denominator for
"how big is this fund in units" and because its disagreement with assets is itself
diagnostic -- but it is declared OPTIONAL so that a column of rounded zeros can
never satisfy a required-and-not-all-null check as though it meant something.

TERMS

ProShares' Terms and Conditions of Use grant use and forbid mirroring, the same
tension as umich_sca: "No permission is granted to sell, copy, publish, reproduce,
distribute, modify, post, 'mirror' on any server, transfer or frame, in whole or in
part, any of the contents", and "provided for information, education and
non-commercial purposes only". So redistributable=False -- the rows live in the
gitignored store and a fresh clone fetches them itself. There is no anti-automation
clause, and accounts.profunds.com serves no robots.txt.
"""
from __future__ import annotations

import csv
import io
import urllib.request

import pandas as pd

from lib.schema import TableSchema

ENDPOINT = "https://accounts.profunds.com/etfdata/historical_nav.csv"
UA = {"User-Agent": "offchart research (personal, low volume)"}

#: ticker -> (index family, multiple). NEITHER is in the file: the leverage lives
#: only inside the marketing name, where "UltraPro" means 3x and "Ultra" means 2x,
#: and the index is implied by the name too. Both are load-bearing -- the rebalancing
#: measure scales with the SQUARE of the multiple -- so they are an explicit map that
#: fails loudly on an unknown ticker rather than a regex over a product name.
#:
#: Deliberately small. These are the funds with a liquid index and a bull/bear pair,
#: which is what makes the pair-adds-demand arithmetic worth showing. Adding a fund
#: is one line, but check its prospectus for the multiple first.
FUNDS: dict[str, tuple[str, float]] = {
    "TQQQ": ("NDX", 3.0),    # UltraPro QQQ
    "SQQQ": ("NDX", -3.0),   # UltraPro Short QQQ
    "QLD": ("NDX", 2.0),     # Ultra QQQ
    "UPRO": ("SPX", 3.0),    # UltraPro S&P500
    "SPXU": ("SPX", -3.0),   # UltraPro Short S&P500
    "SSO": ("SPX", 2.0),     # Ultra S&P500
    "SDS": ("SPX", -2.0),    # UltraShort S&P500
    "USD": ("SOX", 2.0),     # Ultra Semiconductors -- the only semis exposure here
}

#: Columns as the publisher spells them, so a rename upstream fails on the KeyError
#: rather than silently producing an all-null column.
COL_DATE = "Date"
COL_NAME = "ProShares Name"
COL_TICKER = "Ticker"
COL_NAV = "NAV"
COL_NAV_PCT = "NAV Change (%)"
COL_SHARES = "Shares Outstanding (000)"
COL_ASSETS = "Assets Under Management"


def _num(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch(since: str | None = None) -> pd.DataFrame:
    """Every row for the mapped funds. `since` is ignored, and that is not laziness.

    The endpoint takes no parameters: one URL serves inception-to-date for all 163
    funds, so there is nothing to narrow and incremental=False. The whole 52 MB is
    re-read each run, which costs seconds and keeps the upsert the only thing that
    decides what changed.

    Re-reading everything also matters for correctness rather than only simplicity:
    historical rows CHANGE when a fund splits, because the file is retroactively
    adjusted. lib.db.upsert is ON CONFLICT DO UPDATE on (trade_date, ticker), so a
    re-published history overwrites rather than being skipped as already-present.
    """
    req = urllib.request.Request(ENDPOINT, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read()

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="replace")))
    rows = []
    for row in reader:
        ticker = (row.get(COL_TICKER) or "").strip()
        if ticker not in FUNDS:
            continue
        family, multiple = FUNDS[ticker]
        # MM/DD/YYYY in the file. Parsed explicitly rather than left to pandas,
        # which would read 11/06/2026 as a day-first date in some locales.
        month, day, year = row[COL_DATE].strip().split("/")
        # THOUSANDS in the file. Scaled once, here, so the stored column is in
        # shares and no reader has to remember the unit.
        shares_k = _num(row.get(COL_SHARES))
        rows.append({
            "trade_date": f"{int(year):04d}-{int(month):02d}-{int(day):02d}",
            "ticker": ticker,
            "fund_name": (row.get(COL_NAME) or "").strip(),
            "index_family": family,
            "multiple": multiple,
            "nav": _num(row.get(COL_NAV)),
            "nav_change_pct": _num(row.get(COL_NAV_PCT)),
            "shares_outstanding": None if shares_k is None else shares_k * 1000.0,
            "assets": _num(row.get(COL_ASSETS)),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


SOURCE_KWARGS = dict(
    id="proshares_flows",
    label="ProShares leveraged ETF flows",
    fetch=fetch,
    key=("trade_date", "ticker"),
    sort_key=("trade_date", "ticker"),
    schema=TableSchema(
        # nav and assets are required because both are load-bearing for flow.
        # shares_outstanding is NOT: see the module docstring -- 140 SQQQ rows round
        # to 0.0, and requiring it would bless that as data.
        required=("trade_date", "ticker", "nav", "assets"),
        numeric=("nav", "nav_change_pct", "shares_outstanding", "assets", "multiple"),
        optional=("shares_outstanding", "nav_change_pct", "fund_name", "index_family"),
    ),
    cadence="Daily, overnight T+1 -- the file carries through the previous close",
    incremental=False,
    redistributable=False,
    license=(
        "ProShares Terms and Conditions of Use "
        "(proshares.com/legal/terms-and-conditions): use granted, mirroring "
        'forbidden -- "No permission is granted to sell, copy, publish, reproduce, '
        "distribute, modify, post, 'mirror' on any server, transfer or frame, in "
        'whole or in part, any of the contents", and the site is "provided for '
        'information, education and non-commercial purposes only". No '
        "anti-automation clause; accounts.profunds.com serves no robots.txt."
    ),
    citation="Source: ProShares. © ProShares, all rights reserved.",
    group="Flows",
    provenance=(
        "accounts.profunds.com/etfdata/historical_nav.csv, linked from "
        "proshares.com/resources/data-downloads. Columns Date, Ticker, NAV, "
        "Shares Outstanding (000), Assets Under Management. Retroactively "
        "split-adjusted on both NAV and shares. Flow is computed from assets, not "
        "shares -- see lib.metrics.etf_flow."
    ),
    caveats=(
        "Flow is creation and redemption by authorised participants, which is the "
        "money entering or leaving the fund -- not a measure of who is right. A fund "
        "can take inflows all the way down.",
        "Rebalancing demand is MANDATE-IMPLIED, never an observed trade. It is "
        "executed in swap and futures notional, and because assets are struck at the "
        "close, a given day's figure is what a move on the NEXT day would force.",
        "A bull/bear pair's rebalancing demand ADDS rather than offsetting: both buy "
        "into a rally, because the inverse fund's short exposure grows as the index "
        "falls. Never net TQQQ against SQQQ.",
        "(multiple^2 - multiple) means an inverse fund rebalances twice as hard as "
        "its bull twin -- coefficient 12 for -3x against 6 for +3x.",
        "Per-index-family only. $1 of NDX demand and $1 of SPX demand are not "
        "interchangeable, and summing them assumes the two indices move together.",
        "ProShares is one issuer. Direxion's leveraged semiconductor funds (SOXL, "
        "SOXS) are far larger in the chip complex and are NOT here, because Direxion "
        "publishes only the latest day. USD is the only semis exposure on this board "
        "and it is a 2x, not a 3x.",
    ),
)
