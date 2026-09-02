"""Socrata client for publicreporting.cftc.gov. No API key required.

Three things here are load-bearing and none of them are obvious.

$SELECT IS NOT AN OPTIMISATION, IT IS THE DRIFT DETECTOR.
Projecting to the ~40 columns we actually use costs 10x less on the wire and 9x
less peak memory than taking all 133-194 (measured on legacy_fut full history:
1,343 MB of JSON and 3,764 MB of json.loads peak unprojected, versus 130 MB and
421 MB projected). But the reason it is mandatory is that Socrata answers a
$select naming a column that no longer exists with HTTP 400 and the column name
in the body. That turns an upstream rename from "charts silently go null" into
"the ingest job dies at the network boundary and names the column". There is no
cheaper or earlier place to catch it.

THE '+' IN THREE MARKET CODES.
12460+, 13874+ and 20974+ are the CFTC Consolidated series for DJIA, S&P 500
and NASDAQ-100. A raw '+' in a query string decodes to a space, so a $where
built by f-string concatenation silently matches nothing and returns zero rows.
urllib.parse.urlencode percent-encodes it correctly. Never hand-build the query.

SOCRATA OMITS NULL KEYS.
A row does not contain a key whose value is null -- it simply lacks the key.
Per-row key counts in tff_fut range 65..90 against a 90-key union. So a missing
key means null, not schema drift, and len(row) says nothing about the schema.
`num()` treats absent and null identically and returns None for both.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from lib import cftc_spec

BASE = "https://publicreporting.cftc.gov/resource"
UA = {"User-Agent": "watchboard research (personal, low volume)"}

#: Socrata imposes no $limit ceiling on these datasets -- $limit=300000 returned
#: all 288,151 legacy_fut rows in a single 7.2s response -- so there is no need
#: to page. We still assert we came in under the ceiling, because a silently
#: truncated pull is indistinguishable from a market delisting.
HARD_LIMIT = 500_000

TIMEOUT = 300


def _get(url: str) -> Any:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


def fetch_rows(
    spec: cftc_spec.ReportSpec,
    *,
    since: str | None = None,
    limit: int = HARD_LIMIT,
    order: str = "report_date_as_yyyy_mm_dd",
) -> list[dict]:
    """Pull rows for one dataset, projected to the spec'd columns.

    `since` is an inclusive ISO date floor on report_date. Passing it is what
    makes the daily cron cheap: a full-history pull of all seven datasets is
    ~900 MB, while an 8-week incremental window is ~1 MB. Because CFTC revises
    already-published weeks, the window has to overlap what we already store
    rather than start after it -- see scripts/ingest.INCREMENTAL_OVERLAP_DAYS.
    """
    query = {
        "$select": ",".join(spec.select_fields()),
        "$order": order,
        "$limit": str(limit),
    }
    if since:
        query["$where"] = f"report_date_as_yyyy_mm_dd >= '{since}'"

    url = f"{BASE}/{spec.dataset}.json?{urllib.parse.urlencode(query)}"
    rows = _get(url)
    # Only meaningful when we asked for everything. A caller that passes a small
    # explicit limit (a probe, a fixture) wants exactly that many rows and
    # hitting the number is the expected outcome, not a truncation.
    if limit >= HARD_LIMIT and len(rows) >= limit:
        raise RuntimeError(
            f"{spec.id}: got {len(rows):,} rows at $limit={limit:,} -- the pull was "
            "probably truncated. Raise HARD_LIMIT rather than trusting this."
        )
    return rows


def normalise_keys(row: dict) -> dict:
    """Lowercase every key.

    Needed because the supplemental dataset's payload uses MixedCase for 16 of
    its 60 columns while the Socrata *metadata* for those same columns is
    lowercase -- and the split runs per side within a cohort, so
    comm_positions_long_all_nocit is lowercase while
    Comm_Positions_Short_All_NoCIT is not. Selecting by the metadata name works;
    reading the response by that name does not. Lowercasing both sides removes
    the whole class of problem.

    Raises on a genuine case collision rather than silently dropping a column.
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        lk = k.lower()
        if lk in out and out[lk] != v:
            raise ValueError(f"case collision on {lk!r}: {out[lk]!r} vs {v!r}")
        out[lk] = v
    return out


def num(row: dict, field: str | None) -> float | None:
    """Read a numeric field. Absent, null and unparseable all give None.

    Absent and null are the same condition here (Socrata drops null keys), and
    both are genuinely different from zero: a null per-cohort trader count
    co-occurs with a NONZERO position, so coercing it to 0 would manufacture a
    division by zero in avg_position_per_trader rather than an honest NaN.
    """
    if field is None:
        return None
    try:
        return float(row[field.lower()])
    except (KeyError, TypeError, ValueError):
        return None


def text(row: dict, field: str | None) -> str | None:
    if field is None:
        return None
    v = row.get(field.lower())
    if v is None:
        return None
    # Market names carry inconsistent internal whitespace across datasets --
    # 'GOLD -1 TROY OUNCE - COINBASE' in legacy versus a double space in the
    # same name in disagg. Collapse it so the same market matches itself.
    return " ".join(str(v).split()) or None
