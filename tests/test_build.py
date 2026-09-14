"""scripts/build's serving-table swap: the promise that Grafana never sees a gap.

The metrics are proven in tests/test_metrics.py and the archive write in tests/test_db.py.
What is covered here is the one moment the serving layer does not exist -- between
dropping the old board table and renaming the freshly built one into its place.

build calls that swap atomic, and whether it is depends on the sqlite3 DRIVER rather
than on SQLite. SQLite supports transactional DDL perfectly well; Python's legacy
transaction handling opens a transaction before DML but NOT before DDL, so a `with con:`
wrapped around DROP + ALTER autocommits each statement as it runs and has nothing to
roll back. The failure is invisible in normal operation -- the window is two statements
wide and the rebuild is weekly -- which is exactly why it needs a test rather than a
reading.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import db  # noqa: E402
from scripts import build  # noqa: E402

KEY = ("ts",)


class _RaisesOnRename(sqlite3.Connection):
    """Fails at exactly one point: after the DROP, before the RENAME.

    Injecting the failure there is the only way to observe whether the swap is one
    unit. A crash anywhere else is indistinguishable from a crash before the swap
    began, which is already safe.
    """

    def execute(self, sql, *args):  # type: ignore[override]
        if sql.lstrip().upper().startswith("ALTER TABLE"):
            raise RuntimeError("interrupted between DROP and RENAME")
        return super().execute(sql, *args)


def _seed(con: sqlite3.Connection) -> None:
    db.upsert(con, "board_x", pd.DataFrame({"ts": [1], "v": [10.0]}), KEY)


def _tables(con: sqlite3.Connection) -> list[str]:
    return [
        r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


def test_a_failed_swap_leaves_the_previous_board_table_serving(tmp_path: Path) -> None:
    """The property the docstring claims: a rebuild that dies mid-swap costs freshness,
    never availability.

    Grafana reads this file continuously and `allowUiUpdates: false` means every panel
    is a committed query against a table name. If the swap is not one unit, a crash
    between the two statements leaves NO serving table at all -- every panel on the
    board reads `no such table: board_positioning` until someone notices and reruns the
    build. Last week's numbers surviving is strictly better than that, and it is what
    the docstring promises.
    """
    con = sqlite3.connect(tmp_path / "t.sqlite", factory=_RaisesOnRename)
    _seed(con)

    with pytest.raises(RuntimeError, match="interrupted"):
        build._swap_in(con, "board_x", pd.DataFrame({"ts": [1, 2], "v": [99.0, 98.0]}), KEY)

    assert "board_x" in _tables(con), (
        "the serving table is GONE after a swap that failed partway: the DROP committed "
        "on its own, so `with con:` had nothing to roll back. Open the transaction "
        "explicitly -- Python's legacy driver does not BEGIN for DDL, though SQLite "
        f"itself is happy to. Tables left behind: {_tables(con)}"
    )
    assert con.execute("SELECT ts, v FROM board_x").fetchall() == [(1, 10.0)], (
        "the previous week's rows must survive a failed rebuild untouched"
    )


def test_a_successful_swap_replaces_the_rows_and_leaves_no_staging_table(
    tmp_path: Path,
) -> None:
    """The other half, so the fix above cannot be 'never swap at all'."""
    con = sqlite3.connect(tmp_path / "t.sqlite")
    _seed(con)

    build._swap_in(con, "board_x", pd.DataFrame({"ts": [2, 3], "v": [98.0, 97.0]}), KEY)

    assert con.execute("SELECT ts, v FROM board_x ORDER BY ts").fetchall() == [
        (2, 98.0),
        (3, 97.0),
    ], "a successful swap serves exactly the newly built rows, not a merge with the old"
    assert "board_x_new" not in _tables(con), (
        f"the staging table outlived the swap: {_tables(con)}"
    )


def test_two_swaps_then_the_serving_handoff_leave_no_transaction_open(
    tmp_path: Path,
) -> None:
    """scripts.build.main's real sequence: swap both boards, then hand the file to Grafana.

    A guard on the failure mode the explicit BEGIN could introduce rather than one it
    fixes, and both halves are reachable: a swap that left its transaction open makes
    the next `BEGIN IMMEDIATE` raise "cannot start a transaction within a transaction",
    and makes the handoff raise "cannot change out of wal mode from within a
    transaction" -- leaving the file in WAL, the one state where Grafana loses its lock
    race and draws a blank panel beside two full ones. db.connect opens in WAL, so this
    exercises that transition rather than a no-op.
    """
    con = db.connect(tmp_path / "t.sqlite")
    for table in ("board_positioning", "board_inflation"):
        db.upsert(con, table, pd.DataFrame({"ts": [1], "v": [10.0]}), KEY)
        build._swap_in(con, table, pd.DataFrame({"ts": [2], "v": [20.0]}), KEY)

    assert con.in_transaction is False, "a swap left its transaction open"
    assert db.finalize_for_serving(con) == "delete", (
        "the file was not handed back in rollback-journal mode, so Grafana would race "
        "on the -shm file"
    )
    for table in ("board_positioning", "board_inflation"):
        assert con.execute(f"SELECT ts, v FROM {table}").fetchall() == [(2, 20.0)]
