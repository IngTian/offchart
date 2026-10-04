"""The source contract: what may be committed.

One failure mode, and it is the only irreversible one in the repo. A source whose
terms forbid redistribution gets ingested and then committed because nobody
remembered, and a public repo's history is not something you can take back. The
database is gitignored so the ordinary path is safe; the risk is any OTHER file
anyone is tempted to add -- a CSV export, a cached fetch, a convenience dump.

There used to be a second, opposite rule here: a source whose API served only
"today" could not be re-fetched, so it HAD to have a committed artefact or its
history lived on one disk. That source (openrouter_pricing) is gone, along with the
`backfillable` flag and the snapshot machinery. Every source now serves history, so
the only artefact question left is whether committing is ALLOWED -- never whether it
is required, which means no test here can be satisfied by writing data into the repo.

A comment in .gitignore catches none of this. `git check-ignore` and `git ls-files`
do, so these tests ask git rather than trusting the file to say what it means.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sources  # noqa: E402
from lib import db  # noqa: E402
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
    so it must never be committable -- and it does not need to be, because every
    source can now be re-fetched."""
    rel = db.DB_PATH.relative_to(ROOT).as_posix()
    assert _is_ignored(rel), f"{rel} must be gitignored"


#: Every script takes `--db` (scripts/ingest.py, scripts/build.py, scripts/status.py),
#: so a store holding umich_sca can legitimately be written to any of these. The rule
#: is about the SHAPE of a store path, not about one default: `data/*.sqlite` matched a
#: single directory level and a single extension, and when this test was written six of
#: the paths below were reported NOT ignored by git -- untracked-visible, taken by
#: `git add -A`, and green under all 357 tests.
_STORE_PATHS_THAT_MUST_BE_IGNORED = (
    "data/offchart.sqlite",       # the default, already covered
    "data/offchart.sqlite-wal",
    "data/offchart.sqlite-shm",
    "data/offchart.db",           # the same store under an ordinary extension
    "data/offchart.sqlite3",
    "data/scratch.sqlite",
    "data/sub/offchart.sqlite",   # one level deeper than `data/*.sqlite` reaches
    "offchart.sqlite",            # `--db` run from the repo root
    "scratch.sqlite",
    "scratch.db",
)


@pytest.mark.parametrize("rel", _STORE_PATHS_THAT_MUST_BE_IGNORED)
def test_no_store_path_inside_the_repo_is_committable(rel: str) -> None:
    """The redistribution rule stated as a property instead of as one path.

    A store contains umich_sca, which is licensed for use and not for redistribution,
    so the question is never "is the default path ignored" but "can a store be
    committed at all". Asserting the constant answers a narrower question than the
    rule makes, and the gap is not hypothetical: the scripts accept `--db`, so
    `make backfill --db data/offchart.db` produced a committable file holding every
    restricted row while every test stayed green.
    """
    assert _is_ignored(rel), (
        f"{rel} is NOT gitignored, but a store written there holds umich_sca, which "
        "may not be redistributed. Widen the pattern in .gitignore rather than "
        "narrowing this list -- a store is defined by what it contains, not by where "
        "someone happened to put it."
    )


def test_the_committed_test_fixture_is_still_committable() -> None:
    """The converse, and the reason the ignore pattern cannot simply be `*`.

    tests/fixtures/fixture.sqlite is the corpus the whole suite runs against and it
    MUST stay committed -- which is safe only because its restricted slice is
    fabricated rather than sampled (tests/fixtures/build.py). Without this assertion,
    broadening the store pattern to close the hole above would silently un-commit the
    fixture and every test would start depending on a file nobody has.
    """
    rel = (ROOT / "tests" / "fixtures" / "fixture.sqlite").relative_to(ROOT).as_posix()
    assert not _is_ignored(rel), (
        f"{rel} is gitignored, so the suite's own corpus would drop out of the repo. "
        "The store pattern needs an exception for it."
    )


def _tracked(pathspec: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", pathspec], cwd=ROOT, capture_output=True, text=True
    )
    return [p for p in out.stdout.splitlines() if p]


def test_nothing_under_data_is_committed() -> None:
    """data/ is where every fetched row lands, so nothing in it may reach git.

    This replaces a pair of narrower tests that asked only about data/snapshots/, and
    it is strictly stronger: the question is not whether one known directory is
    committed but whether ANY data file is. The store itself is covered by the ignore
    tests above; this catches the other half -- a CSV export, a debug dump, a cached
    fetch dropped next to it, any of which could carry umich_sca rows into a public
    history that cannot be rewritten after the fact.

    If a committed data artefact is ever genuinely wanted, it has to be argued for
    here rather than appearing as an untracked file somebody ran `git add -A` over.
    """
    tracked = _tracked("data")
    assert not tracked, (
        f"{len(tracked)} file(s) under data/ are tracked by git: {tracked[:8]}. "
        "Everything there is fetched from an API and re-fetchable, and some of it "
        "(umich_sca) may not be redistributed at all."
    )


@pytest.mark.parametrize("source", sources.all_sources(), ids=lambda s: s.id)
def test_a_source_we_may_not_redistribute_has_no_committed_artefact(source: Source) -> None:
    """The licence rule, asked of the whole tree rather than one directory.

    A restricted source's rows must not reach any committed file, wherever someone
    puts it. Matching on the source id is a heuristic -- it catches
    `umich_sca_export.csv` and not `survey_dump.csv` -- so it is a backstop behind
    the ignore rules, not the primary guard. The primary guard is that the only place
    these rows ever land is the gitignored store.
    """
    if source.redistributable:
        return
    hits = [p for p in _tracked("*") if source.id in Path(p).name
            and Path(p).suffix in {".csv", ".parquet", ".json", ".sqlite", ".tsv"}]
    assert not hits, (
        f"{source.id} may not be redistributed ({source.license}) but these committed "
        f"files are named for it: {hits}"
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
