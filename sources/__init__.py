"""Source registry.

Add a source by writing a module that exposes `SOURCE_KWARGS`, then registering
it below. Nothing else in the codebase needs to change -- ingest and the app both
iterate this registry.

PLANNED SOURCES (not yet implemented -- each needs its endpoint verified first,
per the "primary source, verified payload" rule)

  token volume        OpenRouter /rankings. HTML only, no JSON endpoint found.
                      Scraping it is brittle; needs a schema-change guard that
                      fails loudly rather than silently writing zeros.
  AI adoption         Ramp AI Index (ramp.com/data/ai-index), monthly, % of US
                      businesses paying for AI tools, from card spend.
  compute / releases  Epoch AI (epoch.ai/data) -- downloadable datasets.
  company financials  SEC XBRL (data.sec.gov/api/xbrl/) -- shares outstanding,
                      revenue, net income, debt straight from the filing. This
                      is also the route to a market-cap series, since market cap
                      = cover-page share count x price.
  semis cycle         SIA monthly billings; TSMC monthly revenue (~10th of the
                      month); Korea 20-day exports.
  credit / vol        FRED (BAMLH0A0HYM2, DFII10) -- needs a free API key.

OPEN QUESTION for "model market cap": two readings, both buildable.
  (a) market cap of the AI-exposed companies  -> SEC XBRL shares x price
  (b) share of routed token volume per model  -> OpenRouter /rankings
Reading (b) is the one with no substitute anywhere else; (a) is available from
any broker screen. Worth deciding which was meant before building either.
"""
from __future__ import annotations

from . import cftc_tff, openrouter_pricing
from .base import Source

REGISTRY: dict[str, Source] = {}


def _register(module) -> None:
    source = Source(**module.SOURCE_KWARGS)
    REGISTRY[source.id] = source


for _module in (cftc_tff, openrouter_pricing):
    _register(_module)


def get(source_id: str) -> Source:
    if source_id not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown source {source_id!r}; known: {known}")
    return REGISTRY[source_id]


def all_sources() -> list[Source]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]
