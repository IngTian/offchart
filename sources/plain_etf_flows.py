"""Unleveraged ETF flows -- the ordinary money, next to the mechanical kind.

Two funds from two issuers, both unleveraged, both with full history:

  SPY  State Street (SSGA)   2006-06-01 ->   the S&P 500 benchmark
  SMH  VanEck                2011-12-21 ->   semiconductors

WHY THESE TWO AND NOT THE OBVIOUS FOUR

QQQ (Invesco) and SOXX (iShares) were rejected, measured rather than assumed: both
serve only the latest day, and this repo deliberately has no collect-as-you-go
machinery, so a latest-only source has no history to chart on the day it ships. For
SOXX specifically, iShares' own two surfaces disagree -- the holdings CSV lags the
product page by one business day, which showed up as an 800,000-share / $109M gap.
Within a single surface the numbers reconcile to 5,272 shares, so the hazard is
crossing surfaces rather than bad data, and neither surface carries history anyway.

WHAT THEY ADD

A denominator. The leveraged board says TQQQ must trade $1,747M for a 1% NDX move;
that figure means nothing without knowing what the ordinary money in the same index
looks like. And SMH closes the semiconductor gap: $60.7bn unlevered, against a board
whose only other semis exposure was ProShares USD, a 2x.

They also make the rebalancing column read correctly by arithmetic rather than by
special case. (multiple^2 - multiple) is 0 at multiple=1, so an unleveraged fund shows
exactly zero forced trading -- which is the true answer, not a missing value.

SHARES OUTSTANDING IS PUBLISHED BY ONE AND NOT THE OTHER

SSGA publishes it. VanEck does not, anywhere -- not in the price history and not in
the holdings workbook -- so it is derived as assets / NAV. Measured over the last 300
trading days that derivation lands within 42.5 shares of a round hundred (median
26.4), and successive differences fall on SMH's 50,000-share creation-unit grid, so it
recovers the real share count rather than approximating it. It is still a derived
figure and the column is declared optional for both, because flow here comes from
assets and NAV (see lib.metrics.etf_flow) and never from the share count.

NEITHER PUBLISHES A USABLE DAILY NAV CHANGE

SSGA does not publish one at all. VanEck publishes '% Change', and it is TOTAL RETURN
on the one day a year SMH distributes: measured, exactly 14 rows across 4,106 disagree
with the NAV-derived change by more than 0.1pp, one per December, worst 2.22pp. So
neither issuer's change column is carried, and scripts/build computes the change from
NAV where a source does not supply one.
"""
from __future__ import annotations

import urllib.request
from http.cookiejar import CookieJar

import pandas as pd

from lib import xlsx
from lib.schema import TableSchema

UA = {"User-Agent": "offchart research (personal, low volume)"}

#: SSGA publishes one NAV-history workbook per SPDR, at a predictable path.
#: Verified live: spy, gld, dia, xlk, gldm and mdy all return 200. Non-SPDRs do
#: not exist here -- navhist-us-en-qqq.xlsx is a 404, which is why the Nasdaq-100
#: has no unleveraged fund on this board.
SSGA_URL = ("https://www.ssga.com/library-content/products/fund-data/etfs/us/"
            "navhist-us-en-{ticker}.xlsx")

#: SPDR ticker -> (fund name, index family). Kept small deliberately: each entry is
#: a fund someone will actually read beside its leveraged counterparts.
SSGA_FUNDS = {
    "SPY": ("SPDR S&P 500 ETF Trust", "SPX"),
    # The unleveraged side of gold, and two orders of magnitude larger than either
    # ProShares leveraged gold fund. Unlike those, GLD holds bullion rather than
    # futures, so its NAV is a spot-backed price and not a rolling futures one.
    "GLD": ("SPDR Gold Shares", "GOLD"),
}
#: The header SSGA's workbook must still have. Anchored on text, never on a row
#: offset: the header sits at the third row today, which is true rather than
#: promised, and disclosure prose occupies the rows around it.
SSGA_HEADER = ("Date", "NAV", "Shares Outstanding", "Total Net Assets")

VANECK_URL = ("https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/"
              "downloads/fundhistoprices/")
VANECK_HEADER = ("Date", "NAV", "AUM")


def _num(raw: str | None) -> float | None:
    """Float, or None. Handles '1.060932116E9' and SSGA's '-' placeholder alike."""
    if raw is None:
        return None
    text = raw.strip().replace(",", "").replace("$", "")
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _date(raw: str | None) -> str | None:
    """'05-Nov-2026' or '11/6/2026' -> ISO. Both issuers write text, not Excel serials.

    Parsed explicitly rather than handed to pandas, which reads an ambiguous
    '11/06/2026' as day-first under some locales and would silently relabel half a
    year of flows.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    for fmt in ("%d-%b-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d %b %Y", "%b %d, %Y"):
        try:
            return pd.to_datetime(text, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


def _fetch_bytes(url: str, with_cookies: bool = False) -> bytes:
    """GET the workbook. `with_cookies` is VanEck's requirement, not a nicety.

    VanEck answers a cookie-less request with 302 -> ?cken=true and then
    302 -> /row/disabled-cookies/ with an empty body, regardless of User-Agent --
    verified, including with a browser UA, so the gate is cookies and not UA
    sniffing. One stdlib opener with a jar satisfies it; no spoofing is needed
    anywhere here, and both hosts answer an honest User-Agent.
    """
    request = urllib.request.Request(url, headers=UA)
    if with_cookies:
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        with opener.open(request, timeout=300) as response:
            return response.read()
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def _table(url: str, header: tuple[str, ...], with_cookies: bool = False):
    raw = _fetch_bytes(url, with_cookies=with_cookies)
    rows = xlsx.rows(raw)
    start = xlsx.find_header(rows, header)
    names = [c.strip().lower() for c in rows[start]]
    return names, rows[start + 1:]


def _frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def _fetch_ssga(ticker: str) -> pd.DataFrame:
    """One SPDR's whole NAV history. No parameters, so no `since` to honour.

    Rows arrive newest-first and are sorted on the way out. SPY's shares outstanding
    is the literal '-' for 2003-12-01..2006-05-30 (628 rows) and numeric from
    2006-05-31, so NAV exists three years before a flow can be computed; those rows
    are kept with a null share count and the first computable flow is 2006-06-01.
    """
    fund_name, family = SSGA_FUNDS[ticker]
    names, body = _table(SSGA_URL.format(ticker=ticker.lower()), SSGA_HEADER)
    i_date, i_nav = names.index("date"), names.index("nav")
    i_so, i_tna = names.index("shares outstanding"), names.index("total net assets")

    out = []
    for row in body:
        if len(row) <= i_tna:
            continue
        day = _date(row[i_date])
        nav = _num(row[i_nav])
        if day is None or nav is None:
            continue
        out.append({
            "trade_date": day,
            "ticker": ticker,
            "fund_name": fund_name,
            "index_family": family,
            "multiple": 1.0,
            "nav": nav,
            "shares_outstanding": _num(row[i_so]),
            "assets": _num(row[i_tna]),
        })
    return _frame(out)


def fetch_spy(since: str | None = None) -> pd.DataFrame:
    return _fetch_ssga("SPY")


def fetch_gld(since: str | None = None) -> pd.DataFrame:
    return _fetch_ssga("GLD")


def fetch_smh(since: str | None = None) -> pd.DataFrame:
    """SMH's whole NAV history. `since` ignored: one file, regenerated per request.

    Non-trading days are PRESENT and carried forward -- 364 rows whose NAV and assets
    both repeat the previous row, being 332 weekends and 32 NYSE closures. They are
    kept exactly as published rather than filtered: lib.metrics.etf_flow computes
    identically zero on them (verified, max |flow| == 0.0), so they cost nothing, and
    dropping them would mean this source deciding which of the publisher's rows are
    real.
    """
    names, body = _table(VANECK_URL, VANECK_HEADER, with_cookies=True)
    i_date, i_nav, i_aum = names.index("date"), names.index("nav"), names.index("aum")

    out = []
    for row in body:
        if len(row) <= max(i_date, i_nav, i_aum):
            continue
        day = _date(row[i_date])
        nav = _num(row[i_nav])
        assets = _num(row[i_aum])
        if day is None or nav is None:
            continue
        out.append({
            "trade_date": day,
            "ticker": "SMH",
            "fund_name": "VanEck Semiconductor ETF",
            "index_family": "SOX",
            "multiple": 1.0,
            "nav": nav,
            # Derived, not published. See the module docstring for the measurement
            # that says the derivation recovers the real count.
            "shares_outstanding": (
                None if not assets or not nav else assets / nav
            ),
            "assets": assets,
        })
    return _frame(out)


def _kwargs(**over) -> dict:
    base = dict(
        key=("trade_date", "ticker"),
        sort_key=("trade_date", "ticker"),
        schema=TableSchema(
            required=("trade_date", "ticker", "nav", "assets"),
            numeric=("nav", "shares_outstanding", "assets", "multiple"),
            optional=("shares_outstanding", "fund_name", "index_family"),
        ),
        incremental=False,
        group="Flows",
        redistributable=False,
    )
    base.update(over)
    return base


def build_sources():
    from sources.base import Source

    return [
        Source(**_kwargs(
            id="spy_flows",
            label="SPY flows (State Street)",
            fetch=fetch_spy,
            cadence="Daily, the file is overwritten in place with the latest close",
            license=(
                "SSGA Terms and Conditions (ssga.com/us/en/footer/terms-and-conditions): "
                '"You may not copy, reproduce, duplicate, modify, adopt or lend, sell, '
                "disseminate or otherwise transfer, in whole or in part, any of the "
                "information contained in this Site, except for the purpose of accessing "
                'the Site and producing print-outs for your own personal use". Nothing '
                "in the terms or in robots.txt restricts automated access; "
                "/library-content/products/ is not disallowed."
            ),
            citation="Source: State Street Global Advisors (SPDR).",
            provenance=(
                "ssga.com/library-content/products/fund-data/etfs/us/"
                "navhist-us-en-spy.xlsx, columns Date / NAV / Shares Outstanding / "
                "Total Net Assets. Total Net Assets equals NAV x shares to the penny. "
                "Served from Azure blob storage with no bot gate; no cookie, referer "
                "or User-Agent required."
            ),
            caveats=(
                "No flow before 2006-06-01. NAV runs from 2003-12-01 but shares "
                "outstanding is the literal '-' until 2006-05-31, so the first "
                "computable flow is the day after that.",
                "The file is overwritten in place each day. It carries the whole "
                "23-year history today; if SSGA ever truncated it to a rolling window "
                "this repo has no collect-as-you-go machinery to fall back on.",
                "SSGA's terms bar dissemination of site content, so this is "
                "redistributable=False: fetching and rendering locally is fine, "
                "shipping a snapshot or a derived table is not.",
            ),
        )),
        Source(**_kwargs(
            id="gld_flows",
            label="GLD flows (State Street)",
            fetch=fetch_gld,
            cadence="Daily, the file is overwritten in place with the latest close",
            license=(
                "SSGA Terms and Conditions (ssga.com/us/en/footer/terms-and-conditions): "
                '"You may not copy, reproduce, duplicate, modify, adopt or lend, sell, '
                "disseminate or otherwise transfer, in whole or in part, any of the "
                "information contained in this Site, except for the purpose of accessing "
                'the Site and producing print-outs for your own personal use". Nothing '
                "in the terms or in robots.txt restricts automated access."
            ),
            citation="Source: State Street Global Advisors (SPDR).",
            provenance=(
                "ssga.com/library-content/products/fund-data/etfs/us/"
                "navhist-us-en-gld.xlsx, same four columns as the SPY workbook."
            ),
            caveats=(
                "GLD holds BULLION, while ProShares' leveraged gold funds hold "
                "futures. So GLD's NAV is a spot-backed price and theirs track a "
                "rolling futures position -- comparable as flows, not as prices.",
                "SSGA's terms bar dissemination of site content, so this is "
                "redistributable=False.",
            ),
        )),
        Source(**_kwargs(
            id="smh_flows",
            label="SMH flows (VanEck)",
            fetch=fetch_smh,
            cadence="Daily, through the previous close; non-trading days carried forward",
            license=(
                "VanEck legal terms (vaneck.com/us/en/legal/): \"No information "
                "contained on this site may be reproduced, transmitted, displayed, "
                "distributed, published or otherwise used for commercial purposes "
                "without the prior consent of VanEck. You may, however, print or "
                "electronically store copies of the information for your own personal "
                'use." The restriction is on COMMERCIAL use and personal electronic '
                "storage is expressly permitted -- but the grant is personal, so "
                "republication is not available and this stays redistributable=False. "
                "robots.txt declares Crawl-delay: 25 and disallows only PDF-modal "
                "paths."
            ),
            citation="Source: VanEck.",
            provenance=(
                "vaneck.com/us/en/investments/semiconductor-etf-smh/downloads/"
                "fundhistoprices/ -- an XLSX named SMH_asof_<date>.xlsx, columns Date / "
                "NAV / Change / %% Change / Last Trade / Volume / Premium-Discount / "
                "%% Premium-Discount / AUM / Index Level. A cookie jar is required; the "
                "workbook is regenerated per request so its bytes are not stable."
            ),
            caveats=(
                "Shares outstanding is DERIVED as assets / NAV, because VanEck "
                "publishes it nowhere. Measured over 300 trading days it lands within "
                "42.5 shares of a round hundred and its differences fall on the "
                "50,000-share creation-unit grid, so it recovers rather than "
                "approximates -- but it is still derived, and flow does not use it.",
                "Non-trading days are present and carried forward: 364 rows repeating "
                "the previous NAV and assets, being 332 weekends and 32 NYSE closures. "
                "Flow computes to exactly zero on them. Blank Volume is NOT a clean "
                "filter for them -- 34 weekday rows also have it.",
                "VanEck's published '%% Change' column is TOTAL RETURN on SMH's annual "
                "December distribution date and disagrees with the NAV-derived change "
                "by up to 2.22 points on 14 of 4,106 rows. It is not carried.",
                "Early history is violent but real: 125 days show single-day implied "
                "share moves above 25%%, 124 of them before 2020, and every one is an "
                "exact creation-unit multiple. SMH was a $140-500M fund then against "
                "$60.7bn now, so one basket moved ~10%% of it. A chart over the full "
                "history is dominated by a fund 1/400th of today's size.",
                "Only one gap over four days in fifteen years: 2012-10-26 to "
                "2012-10-31, the Hurricane Sandy closure.",
            ),
        )),
    ]
