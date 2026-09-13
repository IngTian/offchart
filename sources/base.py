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

    #: WHAT THE UPSTREAM ALLOWS. Not decoration -- `redistributable` decides
    #: whether this source may have a COMMITTED artefact at all, and that is
    #: enforced by a test rather than left to whoever adds the next source.
    #:
    #: Every source in this repo up to now has been US federal work (CFTC) or an
    #: openly published API, so "commit the data" was free and the question never
    #: came up. It is not free in general: the University of Michigan asserts
    #: copyright over the Surveys of Consumers tables, grants permission-free USE
    #: of the public ones, and separately prohibits redistribution without written
    #: consent. Those two grants are in tension, and the honest reading is that
    #: charting is allowed and mirroring is not. Such a source lands only in the
    #: ignored database: a fresh clone has to fetch it for itself.
    license: str = "unspecified"

    #: False -> this source may have NO committed artefact, which today means no
    #: directory under data/snapshots/. The database is never committed either
    #: way, so this flag is about what reaches version control. See license
    #: above, and note the separate and opposite rule on `backfillable`: a source
    #: that cannot be re-fetched MUST have committed snapshots. A source that was
    #: both backfillable=False and redistributable=False would be unstorable, and
    #: tests/test_sources.py would fail on it from both directions at once.
    redistributable: bool = True

    #: Attribution the upstream requires, verbatim. Rendered next to the charts.
    citation: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError(f"source {self.id!r} must declare a dedupe key")
        if not self.sort_key:
            raise ValueError(f"source {self.id!r} must declare a sort key")
        # A source we may not mirror is one whose terms we have had to read, so
        # there is no excuse for not recording them -- and the UI cannot attribute
        # what it was not told. Fail at import, where it is unmissable.
        if not self.redistributable:
            if self.license == "unspecified":
                raise ValueError(
                    f"source {self.id!r} is marked not redistributable but declares no "
                    "license. Record the terms that led to that decision."
                )
            if not self.citation:
                raise ValueError(
                    f"source {self.id!r} may not be redistributed, which means the "
                    "upstream is asserting rights, which means it almost certainly "
                    "requires attribution. Declare `citation`."
                )
        # The whole git-churn argument depends on this, so assert it rather than
        # trusting that nobody reorders the tuple later.
        lead = self.sort_key[0]
        if not (lead.endswith("date") or lead.endswith("_date")):
            raise ValueError(
                f"source {self.id!r} sort_key must lead with the date column so a "
                f"weekly append is a tail append, got {lead!r}. See lib/store."
            )
