"""OpenRouter model pricing -- the price half of the inference-revenue question.

Endpoint: openrouter.ai/api/v1/models  (no key, no auth, 419 models as of
2026-09-01). Prices are US dollars PER TOKEN, delivered as strings.

WHY THIS SOURCE IS SNAPSHOT-ONLY, AND WHY THAT MATTERS
------------------------------------------------------
The endpoint returns *today's* prices. There is no history parameter and no
archive. So this dataset does not exist until the job starts writing it, and
every run that does not happen is a hole that can never be filled. That is the
opposite of the CFTC source, which serves 16 years of history on demand.

WHAT IT IS FOR
--------------
The bull/bear crux on AI inference is a claim about a PRODUCT:

    inference revenue = tokens  x  price per token

If volume grows more slowly than price falls, revenue shrinks while a "token
consumption is booming" headline stays literally true. This source measures the
second factor. The first factor (volume) has no clean free endpoint -- see the
stub in sources/__init__.py.

MEASUREMENT CAVEAT, AND IT IS LARGE
-----------------------------------
OpenRouter routes for independent developers and small shops. It carries almost
none of the enterprise or first-party OpenAI / Anthropic / Google traffic, which
is where the volume actually is. Use this for DIRECTION and MIX -- is the
frontier price falling, which models are gaining share -- never for LEVEL.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.request

import pandas as pd

from lib.schema import TableSchema

ENDPOINT = "https://openrouter.ai/api/v1/models"
UA = {"User-Agent": "watchboard research (personal, low volume)"}

PRICE_FIELDS = ("prompt", "completion", "input_cache_read", "input_cache_write")


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch(since: str | None = None) -> pd.DataFrame:
    # `since` is accepted and ignored: the endpoint has no history to narrow.
    # Every run returns exactly today's cross-section.
    del since
    with urllib.request.urlopen(urllib.request.Request(ENDPOINT, headers=UA), timeout=90) as r:
        payload = json.load(r)

    today = dt.date.today().isoformat()
    records = []
    for model in payload.get("data", []):
        pricing = model.get("pricing") or {}
        row = {
            "snapshot_date": today,
            "model_id": model.get("id"),
            "model_name": model.get("name"),
            "vendor": (model.get("id") or "/").split("/")[0],
            "context_length": _f(model.get("context_length")),
        }
        for f in PRICE_FIELDS:
            row[f"usd_per_token_{f}"] = _f(pricing.get(f))
        # Per-million is the unit everyone actually quotes prices in.
        for f in ("prompt", "completion"):
            v = row[f"usd_per_token_{f}"]
            row[f"usd_per_mtok_{f}"] = None if v is None else v * 1_000_000
        records.append(row)

    df = pd.DataFrame.from_records(records)
    print(f"  . {len(df)} models priced, snapshot {today}")
    return df


SOURCE_KWARGS = dict(
    id="openrouter_pricing",
    label="Token prices (OpenRouter)",
    fetch=fetch,
    key=("snapshot_date", "model_id"),
    sort_key=("snapshot_date", "model_id"),
    schema=TableSchema(
        required=("snapshot_date", "model_id"),
        numeric=(
            "context_length",
            "usd_per_token_prompt",
            "usd_per_token_completion",
            "usd_per_token_input_cache_read",
            "usd_per_token_input_cache_write",
            "usd_per_mtok_prompt",
            "usd_per_mtok_completion",
        ),
        optional=("model_name", "vendor"),
        # A snapshot source legitimately returns ~400 rows against a store that
        # accumulates thousands, so the row-count guard must not apply. Ingest
        # only passes stored_rows for full pulls of backfillable sources.
        min_rows_vs_stored=0.0,
    ),
    group="Token economics",
    cadence="Live -- one snapshot per ingest run",
    backfillable=False,
    incremental=False,
    provenance="openrouter.ai/api/v1/models, pricing.{prompt,completion} in USD/token",
    caveats=(
        "SNAPSHOT-ONLY: no history is available from the API. The series starts "
        "the day this job starts and a missed run is a permanent gap.",
        "OpenRouter routes for indie developers and small shops -- not "
        "enterprise, not first-party OpenAI/Anthropic/Google traffic. Good for "
        "DIRECTION and MIX, never for LEVEL.",
        "A falling price per token is not falling revenue. Revenue is price x "
        "volume, and volume is measured by a different source.",
        "Model IDs are recycled and re-versioned; a vendor's cheapest model on "
        "two dates may not be the same product.",
    ),
)
