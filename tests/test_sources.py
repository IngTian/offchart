"""The source contract, and the licence rule in particular.

Everything here is about a failure mode that leaves the pipeline working: a source
whose terms forbid redistribution, ingested into a parquet that then gets committed
because nobody remembered. A comment in .gitignore cannot catch that. `git
check-ignore` can, so these tests ask git.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sources  # noqa: E402
from lib import store  # noqa: E402
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


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_non_redistributable_data_is_gitignored(source: Source) -> None:
    """The licence rule, enforced by git rather than by good intentions.

    A source marked not-redistributable must have its parquet ignored, or an ingest
    run followed by `git add -A` publishes data we were not licensed to publish. The
    converse matters too: silently ignoring a source we ARE allowed to commit would
    quietly break the offline guarantee the whole repo rests on.
    """
    path = store.path_for(source.id)
    rel = path.relative_to(ROOT).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", rel],
        cwd=ROOT, capture_output=True,
    ).returncode == 0

    if source.redistributable:
        assert not ignored, (
            f"{rel} is gitignored but {source.id} is marked redistributable. "
            "Committing the data is how snapshot sources acquire history and how the "
            "board works offline."
        )
    else:
        assert ignored, (
            f"{source.id} is marked NOT redistributable ({source.license}) but {rel} "
            "is not gitignored. An ingest run plus `git add -A` would publish it. "
            "Add the path to .gitignore."
        )


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_restricted_source_declares_its_terms_and_attribution(source: Source) -> None:
    if source.redistributable:
        return
    assert source.license != "unspecified", "record the terms that forbade mirroring"
    assert source.citation, "an upstream asserting rights will want attribution"
    assert source.caveats, "a restricted source must say so next to its charts"


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

    Date-first write order is what makes a weekly append a tail append instead of a
    rewrite of every row group -- measured at ~480x on git growth.
    """
    assert source.sort_key[0].endswith("date")
