"""The source contract: what may be committed, and what MUST be.

Two failure modes, opposite in direction, both of which leave the pipeline working.

  Publishing what we may not. A source whose terms forbid redistribution gets
  ingested and then committed because nobody remembered. The database is gitignored
  so the ordinary path is safe; the risk is a snapshot directory, which IS committed.

  Losing what cannot be re-fetched. A `backfillable=False` source's history exists
  only because we keep it. The database is gitignored and disposable, so if that
  source has no snapshot directory its history depends on one machine's disk.

A comment in .gitignore catches neither. `git check-ignore` does, so these tests ask
git rather than trusting the file to say what it means.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sources  # noqa: E402
from lib import db, snapshots  # noqa: E402
from lib.schema import TableSchema  # noqa: E402
from sources.base import Source  # noqa: E402


def _dummy_kwargs(**over):
    base = dict(
        id="dummy",
        label="Dummy",
        fetch=lambda since=None: None,
        key=("report_date",),
        sort_key=("report_date",),
        schema=TableSchema(required=("report_date",)),
        cadence="never",
        backfillable=True,
    )
    base.update(over)
    return base


def test_every_source_module_imported() -> None:
    """A source that fails to import is invisible on the board, which reads as a
    design choice rather than as a breakage."""
    assert not sources.IMPORT_ERRORS, f"source modules failed to import: {sources.IMPORT_ERRORS}"


def test_sources_were_discovered() -> None:
    assert sources.all_sources(), "no sources discovered at all"


def _is_ignored(rel: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", rel], cwd=ROOT, capture_output=True
    ).returncode == 0


def test_the_database_is_gitignored() -> None:
    """It holds every source, including ones licensed for use but not redistribution,
    so it must never be committable -- and it does not need to be, because nine of ten
    sources can be re-fetched."""
    rel = db.DB_PATH.relative_to(ROOT).as_posix()
    assert _is_ignored(rel), f"{rel} must be gitignored"


def test_the_snapshot_directory_is_NOT_gitignored() -> None:
    """The inverse, and it is the one that loses data if it breaks. data/snapshots/ is
    the only committed copy of the history that no API can return."""
    rel = snapshots.SNAPSHOT_DIR.relative_to(ROOT).as_posix()
    assert not _is_ignored(rel), (
        f"{rel} is gitignored -- that is the committed home of the one source that "
        "cannot be re-fetched, and ignoring it makes a missed day permanent"
    )


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_a_source_we_may_not_redistribute_has_no_committed_snapshots(source: Source) -> None:
    """Snapshots are committed, so a restricted source having one would publish it --
    the exact thing its licence forbids."""
    if source.redistributable:
        return
    directory = snapshots.dir_for(source.id)
    assert not directory.exists() or not list(directory.glob("*.csv")), (
        f"{source.id} may not be redistributed ({source.license}) but has committed "
        f"snapshots in {directory}"
    )


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_a_source_that_cannot_be_refetched_has_committed_snapshots(source: Source) -> None:
    """The durability rule, and the reason lib/snapshots exists.

    A backfillable source loses nothing when the database is deleted -- one refetch
    restores it. This one cannot be refetched at all, so if it has no committed
    snapshot its history lives on exactly one disk.
    """
    if source.backfillable:
        return
    files = sorted(snapshots.dir_for(source.id).glob("*.csv"))
    assert files, (
        f"{source.id} is backfillable=False, so its history cannot be recovered from "
        f"the API. It must have committed snapshots in "
        f"{snapshots.dir_for(source.id)}; run scripts.ingest to write them."
    )
    gaps = snapshots.missing_dates(source.id)
    assert not gaps, (
        f"{source.id} is missing {len(gaps)} snapshot date(s) between its first and "
        f"last: {gaps[:8]}. These are unrecoverable; if the gap is known and accepted, "
        "record it here rather than deleting this assertion."
    )


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_restricted_source_declares_its_terms_and_attribution(source: Source) -> None:
    if source.redistributable:
        return
    assert source.license != "unspecified", "record the terms that forbade mirroring"
    assert source.citation, "an upstream asserting rights will want attribution"
    assert source.caveats, "a restricted source must carry its limitations"


def test_restricted_source_cannot_omit_its_license() -> None:
    with pytest.raises(ValueError, match="declares no license"):
        Source(**_dummy_kwargs(redistributable=False, citation="someone"))


def test_restricted_source_cannot_omit_its_citation() -> None:
    with pytest.raises(ValueError, match="Declare `citation`"):
        Source(**_dummy_kwargs(redistributable=False, license="all rights reserved"))


def test_open_source_needs_neither() -> None:
    """The rule must not become paperwork for the federal data that is genuinely
    public domain."""
    assert Source(**_dummy_kwargs()).redistributable is True


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_sort_key_leads_with_a_date(source: Source) -> None:
    """Already asserted in __post_init__; pinned here so the reason survives.

    The git-churn justification expired with the parquet store. Two reasons remain:
    it is the column scripts/ingest reads to compute the incremental fetch floor, and
    it is the column that becomes `ts` in the serving tables.
    """
    assert source.sort_key[0].endswith("date")
