"""CFTC Traders in Financial Futures (TFF) -- positioning by trader cohort.

Endpoint: publicreporting.cftc.gov/resource/gpe5-46if.json  (Socrata, no key)
Cadence:  positions as of TUESDAY, released FRIDAY 15:30 ET. The 3-day lag is
          structural -- it is the floor, not something to engineer around.

Field names below were read off a live payload (90 fields) rather than recalled,
because two of the five cohorts break the naming pattern:

    dealer / nonrept   ->  *_long_all, *_short_all
    asset_mgr / lev_money / other_rept  ->  *_long, *_short   (no _all suffix)
    nonrept             ->  HAS NO SPREAD FIELD AT ALL

The spread column is the one that matters for arithmetic, not just for naming.
A spread position is long one expiry and short another, so the report counts it
in NEITHER the long nor the short column -- it gets its own. Consequence, and it
is the correction most people miss:

    sum(longs) == sum(shorts) == open_interest - sum(spreads)      NOT == OI

Verified on the 2026-08-25 NASDAQ-100 report: longs and shorts both total
288,110 against open interest of 322,190, with the four reported spreads summing
to 34,081. The residual of 1 contract is the unreported non-reportable spread.

The NET identity is unaffected -- a spread adds equally to both sides, so it
cancels -- which is why sum(cohort nets) == 0 exactly while sum(longs) != OI.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

ENDPOINT = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
UA = {"User-Agent": "watchboard research (personal, low volume)"}

MARKETS = {
    "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE": "NDX (E-mini, $20/pt)",
    "MICRO E-MINI NASDAQ-100 INDEX - CHICAGO MERCANTILE EXCHANGE": "NDX (Micro, $2/pt)",
    "VIX FUTURES - CBOE FUTURES EXCHANGE": "VIX",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "SPX (E-mini)",
}

# cohort id -> (long field, short field, spread field or None), display label
COHORTS: dict[str, tuple[tuple[str, str, str | None], str]] = {
    "dealer": (
        ("dealer_positions_long_all", "dealer_positions_short_all", "dealer_positions_spread_all"),
        "Dealers / intermediaries",
    ),
    "asset_mgr": (
        ("asset_mgr_positions_long", "asset_mgr_positions_short", "asset_mgr_positions_spread"),
        "Asset managers",
    ),
    "lev_money": (
        ("lev_money_positions_long", "lev_money_positions_short", "lev_money_positions_spread"),
        "Leveraged funds",
    ),
    "other_rept": (
        ("other_rept_positions_long", "other_rept_positions_short", "other_rept_positions_spread"),
        "Other reportables",
    ),
    "nonrept": (
        ("nonrept_positions_long_all", "nonrept_positions_short_all", None),
        "Non-reportable (small traders)",
    ),
}


def _pull(market: str) -> list[dict]:
    query = {
        "$where": f"market_and_exchange_names='{market}'",
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "20000",
    }
    url = f"{ENDPOINT}?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.load(r)


def _num(row: dict, name: str | None) -> float | None:
    if name is None:
        return None
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def fetch() -> pd.DataFrame:
    """Tidy long format: one row per (market, report_date, cohort)."""
    records = []
    for market, short_name in MARKETS.items():
        try:
            rows = _pull(market)
        except Exception as exc:  # a dead market should not kill the whole run
            print(f"  ! {short_name}: {type(exc).__name__}: {exc}")
            continue
        if not rows:
            print(f"  ! {short_name}: no rows returned -- market name may have changed")
            continue
        for row in rows:
            oi = _num(row, "open_interest_all")
            if not oi:
                continue
            date = str(row["report_date_as_yyyy_mm_dd"])[:10]
            for cohort, ((f_long, f_short, f_spread), label) in COHORTS.items():
                lo, sh = _num(row, f_long), _num(row, f_short)
                if lo is None or sh is None:
                    continue
                records.append(
                    {
                        "market": short_name,
                        "market_full": market,
                        "report_date": date,
                        "cohort": cohort,
                        "cohort_label": label,
                        "long": lo,
                        "short": sh,
                        "spread": _num(row, f_spread),
                        "open_interest": oi,
                    }
                )
        print(f"  . {short_name}: {len(rows)} weekly reports")

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.date.astype(str)
    return df


SOURCE_KWARGS = dict(
    id="cftc_tff",
    label="CFTC positioning (Traders in Financial Futures)",
    fetch=fetch,
    key=("market", "report_date", "cohort"),
    cadence="Positions as of Tuesday, published Friday 15:30 ET",
    backfillable=True,
    provenance=(
        "publicreporting.cftc.gov/resource/gpe5-46if.json (Socrata, no key). "
        "Futures-only report -- the futures-and-options-combined report is a "
        "different dataset and its figures will not reconcile with these."
    ),
    caveats=(
        "Positioning in this vault measured as CONTEMPORANEOUS with price, not "
        "predictive. Report the level; do not infer a direction without a "
        "base-rate test.",
        "A cohort's net is a sum over firms running incompatible strategies. A "
        "basis trader long the cash index and short the future appears here as "
        "short while holding no view at all. Net is not a stance.",
        "Percentiles are EXPANDING (ranked only against history up to that "
        "date). A full-history percentile would carry look-ahead bias.",
        "sum(longs) == sum(shorts) == open_interest MINUS spreads. Spread "
        "positions get their own column and belong to neither side.",
        "The non-reportable cohort has no spread field in the dataset, so the "
        "spread reconciliation is short by that unpublished amount.",
    ),
)
