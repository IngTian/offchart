"""The Source contract. Adding a data source means adding one of these.

A source is deliberately thin: it knows how to fetch a tidy DataFrame, it
declares the shape that frame must have, and it carries the caveats that travel
with its numbers.

The caveats are a FIELD rather than documentation because the display layer
renders them next to the chart. A limitation that lives only in a README gets
read without it, which is the same as not existing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from lib.schema import TableSchema


@dataclass(frozen=True)
class Source:
    id: str
    label: str

    #: fetch(since=None) -> tidy frame. `since` is an ISO date floor; a source
    #: that cannot fetch incrementally ignores it and returns everything.
    fetch: Callable[..., pd.DataFrame]

    #: Columns that uniquely identify a row, used for upsert dedupe.
    key: tuple[str, ...]

    #: Write order. MUST lead with the date column -- see lib/store, where the
    #: measured difference is ~480x on git growth.
    sort_key: tuple[str, ...]

    #: Declared shape, checked between fetch and write.
    schema: TableSchema

    #: Human description of when new data appears, shown in the UI.
    cadence: str

    #: True  -> the API serves history, so a missed run costs nothing.
    #: False -> the API serves only "now", so a missed run is a permanent hole.
    backfillable: bool

    #: True -> fetch(since=...) meaningfully narrows the pull. Sources that
    #: serve full history cheaply can leave this False and always pull in full.
    incremental: bool = False

    #: Where the numbers come from, verbatim enough to re-derive them.
    provenance: str = ""

    #: Limitations rendered alongside every chart built from this source.
    caveats: tuple[str, ...] = field(default_factory=tuple)

    #: Grouping for the sidebar, e.g. "CFTC positioning".
    group: str = "Other"

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError(f"source {self.id!r} must declare a dedupe key")
        if not self.sort_key:
            raise ValueError(f"source {self.id!r} must declare a sort key")
        # The whole git-churn argument depends on this, so assert it rather than
        # trusting that nobody reorders the tuple later.
        lead = self.sort_key[0]
        if not (lead.endswith("date") or lead.endswith("_date")):
            raise ValueError(
                f"source {self.id!r} sort_key must lead with the date column so a "
                f"weekly append is a tail append, got {lead!r}. See lib/store."
            )
