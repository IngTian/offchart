"""Board registry, discovered rather than hand-listed.

Adding a board is one new file in this package exposing a module-level `BOARD`.
Nothing else in the repo changes -- app.py iterates this registry. That is the
half of the design that makes a new alternative-data source cheap: a source is a
fetcher plus a schema, a board is a render function, and neither one requires
editing a dispatch table someone has to remember to update.

A module that raises on import lands in IMPORT_ERRORS and is shown as broken in
the sidebar, never silently dropped.
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass, field
from typing import Callable

REGISTRY: dict[str, "Board"] = {}
IMPORT_ERRORS: dict[str, str] = {}


@dataclass(frozen=True)
class Board:
    id: str
    title: str
    render: Callable[[], None]

    #: Source ids this board reads. app.py checks they have data before calling
    #: render, so a board never has to hand-write an "empty" branch.
    sources: tuple[str, ...] = ()

    #: Sidebar ordering; lower sorts first.
    order: int = 100

    #: One line under the board title explaining what question it answers.
    blurb: str = ""

    #: Sidebar grouping.
    group: str = "CFTC"


def _discover() -> None:
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
            board = getattr(module, "BOARD", None)
            if board is not None:
                if board.id in REGISTRY:
                    raise ValueError(f"duplicate board id {board.id!r}")
                REGISTRY[board.id] = board
        except Exception:  # noqa: BLE001
            IMPORT_ERRORS[info.name] = traceback.format_exc()


_discover()


def all_boards() -> list[Board]:
    return sorted(REGISTRY.values(), key=lambda b: (b.order, b.title))
