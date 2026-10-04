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

REJECTED, recorded so it is not rebuilt: AI token PRICE and token VOLUME.

  openrouter_pricing shipped and was removed. /api/v1/models collapses a model's
  price across every host that serves it into ONE number by an unstated rule --
  measured 2026-10-04 across 8 multi-hosted models, the per-host spread had a
  median of 20.7x and reached 150x, and the reported figure was the cheapest host
  in only 3 of 8. So the series mostly records which host was selected that
  morning, not a price. Provider-level prices need /api/v1/models/{id}/endpoints,
  which is a real ingest rather than one row per model.

  Token volume has no credible free series at all, which is the bigger problem:
  "inference revenue = tokens x price" needs both factors, and only the hyperscalers
  know the first. Without it a price series answers nothing on its own.
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
