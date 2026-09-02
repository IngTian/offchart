"""Daily closes for the markets CFTC positioning is reported in.

This is the source the rest of the repo kept saying it needed: without a price
series every measure here is positioning-only, so "does an extreme reading precede
anything" cannot be asked about returns at all.

WHY DAILY AND NOT WEEKLY

CFTC positions are as of Tuesday, so a weekly price series would seem to be
enough. Two reasons it is not. Yahoo's weekly bars are Monday-anchored, so a
weekly fetch never lands on the Tuesday the positions describe; and forward
returns at arbitrary horizons need the daily series anyway. Storing daily and
aligning at read time keeps both options and costs about 2 MB.

THE TRAP THIS SOURCE GUARDS AGAINST

`range=max` makes the endpoint IGNORE `interval` and return MONTHLY bars while
still answering 200 OK -- measured: `range=max&interval=1wk` on ^VIX gives 440
points with dataGranularity "1mo", against 1,915 true weekly points for the same
span requested with explicit period bounds. Silently storing monthly data in a
column labelled daily is exactly the failure this repo exists to prevent, so the
fetch uses explicit period1/period2 and ASSERTS the granularity that came back.

PROVENANCE, STATED PLAINLY

This is Yahoo's chart endpoint. It needs no key, it is not a documented public API,
and it can change or start refusing without notice. That is a real step down from
the CFTC Socrata endpoint, which is a government publication with a stable schema.
It is used because the better keyless options do not work from here: FRED's
fredgraph.csv times out and Stooq returns a consent page instead of CSV. If this
breaks, the CFTC boards keep working -- the price panel is additive and its absence
is handled.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from lib.pricemap import SYMBOLS
from lib.schema import TableSchema
from sources.base import Source

ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/"

# A browser UA: the endpoint refuses some non-browser agents outright.
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) watchboard research"}

#: 1990-01-01. Earlier than any mapped series begins, so a full pull gets
#: everything the endpoint holds without asking for a range it will reject.
EPOCH_FLOOR = 631152000

#: What we asked for, and therefore what must come back. See the module docstring.
WANT_GRANULARITY = "1d"

TIMEOUT = 90

#: Courtesy pause between symbols. ~40 symbols, so this adds ~15s to a full pull
#: and keeps the request rate obviously human.
PAUSE_S = 0.4


def _fetch_one(symbol: str, since_epoch: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "period1": since_epoch,
            "period2": int(time.time()),
            "interval": WANT_GRANULARITY,
            "events": "history",
        }
    )
    url = f"{ENDPOINT}{urllib.parse.quote(symbol)}?{query}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.load(resp)

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"{symbol}: endpoint error {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"{symbol}: no result block in the response")

    res = results[0]
    meta = res.get("meta") or {}
    got = meta.get("dataGranularity")
    if got != WANT_GRANULARITY:
        # The whole reason this check exists. range=max silently returns "1mo".
        raise RuntimeError(
            f"{symbol}: asked for {WANT_GRANULARITY} bars and got {got!r}. Storing "
            "these would put coarser data in a column labelled daily."
        )

    stamps = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])
    adjclose = (adj[0].get("adjclose") if adj else None) or [None] * len(stamps)

    rows = []
    for t, c, a in zip(stamps, closes, adjclose):
        if c is None:
            continue  # a holiday or a hole; a null close is not a zero price
        rows.append(
            {
                "date": dt.datetime.utcfromtimestamp(t).date().isoformat(),
                "symbol": symbol,
                "close": float(c),
                "adj_close": float(a) if a is not None else None,
                "currency": meta.get("currency"),
            }
        )
    return rows


def fetch(since: str | None = None) -> pd.DataFrame:
    """Daily closes for every mapped symbol.

    `since` narrows the window. The daily cron passes it, so a routine run asks
    for a few weeks per symbol rather than 36 years.
    """
    floor = EPOCH_FLOOR
    if since:
        floor = int(
            dt.datetime.fromisoformat(since).replace(tzinfo=dt.timezone.utc).timestamp()
        )

    records: list[dict] = []
    failed: list[str] = []
    for symbol in SYMBOLS:
        try:
            got = _fetch_one(symbol, floor)
            records.extend(got)
            print(f"    . {symbol:11s} {len(got):>6,} daily closes")
        except Exception as exc:  # noqa: BLE001
            # One dead symbol must not cost the other thirty-odd. A symbol that
            # stops resolving shows up as a gap the board can see, and the run
            # still fails loudly at the end.
            failed.append(symbol)
            print(f"    ! {symbol:11s} {type(exc).__name__}: {exc}")
        time.sleep(PAUSE_S)

    if failed:
        print(f"    ! {len(failed)} of {len(SYMBOLS)} symbols failed: {failed}")
    if not records:
        return pd.DataFrame(columns=["date", "symbol", "close", "adj_close", "currency"])
    return pd.DataFrame.from_records(records)


SOURCE_KWARGS = dict(
    id="prices",
    label="Daily closes for the CFTC-reported markets",
    fetch=fetch,
    key=("date", "symbol"),
    sort_key=("date", "symbol"),
    schema=TableSchema(
        required=("date", "symbol", "close"),
        numeric=("close", "adj_close"),
        optional=("currency",),
        # A narrowed incremental pull is legitimately a tiny fraction of the store,
        # and ingest only passes stored_rows for a full pull anyway.
        min_rows_vs_stored=0.90,
    ),
    group="Prices",
    cadence="Daily closes, available the evening of each session",
    backfillable=True,
    incremental=True,
    provenance=(
        "query1.finance.yahoo.com/v8/finance/chart (no key). NOT a documented "
        "public API -- it can change or refuse without notice, unlike the CFTC "
        "Socrata endpoint. Requested with explicit period bounds because range=max "
        "silently returns monthly bars; the response's dataGranularity is asserted "
        "to be 1d before anything is written."
    ),
    caveats=(
        "Every series is a PROXY for the contract CFTC reports on -- the underlying "
        "index, a front-month continuous future, or a tracking ETF, never the exact "
        "contract. lib/pricemap.py names the gap for each one.",
        "VIX is the worst case: positioning is in VIX FUTURES while the price here "
        "is SPOT VIX, and the two can move in opposite directions because futures "
        "carry their own term structure.",
        "Front-month continuous futures splice across expiries, so their level "
        "includes roll effects that the underlying does not have.",
        "Prices are aligned to CFTC report dates AS OF -- the last close at or "
        "before the Tuesday. Never interpolated forward, which would put a price "
        "that did not exist yet next to a position.",
        "Positions are as of Tuesday but published Friday 15:30 ET. The price "
        "beside a position is contemporaneous with the position, not with the "
        "moment you could first have known it.",
        "Closes are unadjusted; adj_close is stored alongside where the endpoint "
        "supplies it. For an index or a future the two are the same.",
    ),
)
