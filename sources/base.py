"""The Source contract. Adding a data source means adding one of these.

A source is deliberately thin: it knows how to fetch a tidy DataFrame and it
carries the caveats that must travel with its numbers. The caveats are a field
rather than documentation because the display layer renders them next to the
chart -- a number whose limitation is only in a README gets read without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Source:
    id: str
    label: str
    fetch: Callable[[], pd.DataFrame]

    #: Columns that uniquely identify a row, used for upsert dedupe.
    key: tuple[str, ...]

    #: Human description of when new data appears, shown in the UI.
    cadence: str

    #: True  -> the API serves history, so a missed run costs nothing.
    #: False -> the API serves only "now", so a missed run is a permanent hole.
    backfillable: bool

    #: Where the numbers come from, verbatim enough to re-derive them.
    provenance: str = ""

    #: Limitations rendered alongside every chart built from this source.
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError(f"source {self.id!r} must declare a dedupe key")
