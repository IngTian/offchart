"""Source registry, discovered rather than hand-listed.

Any module in this package that exposes either `build_sources() -> list[Source]`
or `SOURCE_KWARGS: dict` is registered automatically. Adding a data source is
therefore one new file and no edits here.

A module that RAISES on import is recorded in `IMPORT_ERRORS` and surfaced in the
UI and in `scripts/ingest --list`, never swallowed. A source that silently
vanishes from the board is worse than one that is visibly broken: the board keeps
rendering and the missing panel looks like a design choice.

PLANNED SOURCES -- each needs its endpoint verified against a live payload first,
per the "primary source, verified payload" rule that the rest of this repo obeys.

  token volume        OpenRouter /rankings. HTML only, no JSON endpoint found.
                      Scraping needs a schema-change guard that fails loudly
                      rather than silently writing zeros.
  AI adoption         Ramp AI Index (ramp.com/data/ai-index), monthly, share of
                      US businesses paying for AI tools, from card spend.
  compute / releases  Epoch AI (epoch.ai/data) -- downloadable datasets.
  company financials  SEC XBRL (data.sec.gov/api/xbrl/) -- share count, revenue,
                      net income, debt straight from the filing. Also the route
                      to a market-cap series, since market cap is the cover-page
                      share count times price.
  semis cycle         SIA monthly billings; TSMC monthly revenue (~10th of the
                      month); Korea 20-day exports.
  credit / vol        FRED (BAMLH0A0HYM2, DFII10) -- needs a free API key.
  PRICES              The one addition that would change what this board can
                      answer. Every CFTC measure here is positioning-only, so
                      the forward-outcome board can condition forward POSITIONING
                      on today's percentile but not forward RETURNS. A futures
                      settlement series keyed to cftc_contract_market_code is
                      what unlocks the latter.
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback

from .base import Source

REGISTRY: dict[str, Source] = {}

#: module name -> formatted traceback, for modules that failed to import.
IMPORT_ERRORS: dict[str, str] = {}


def _register(source: Source) -> None:
    if source.id in REGISTRY:
        raise ValueError(f"duplicate source id {source.id!r}")
    REGISTRY[source.id] = source


def _discover() -> None:
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name.startswith("_") or name == "base":
            continue
        try:
            module = importlib.import_module(f"{__name__}.{name}")
            if hasattr(module, "build_sources"):
                for src in module.build_sources():
                    _register(src)
            elif hasattr(module, "SOURCE_KWARGS"):
                _register(Source(**module.SOURCE_KWARGS))
        except Exception:  # noqa: BLE001 -- a bad module must not hide the rest
            IMPORT_ERRORS[name] = traceback.format_exc()


_discover()


def get(source_id: str) -> Source:
    if source_id not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown source {source_id!r}; known: {known}")
    return REGISTRY[source_id]


def all_sources() -> list[Source]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def by_group() -> dict[str, list[Source]]:
    out: dict[str, list[Source]] = {}
    for s in all_sources():
        out.setdefault(s.group, []).append(s)
    return out
