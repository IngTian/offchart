"""The Michigan survey's regime boundaries, and the rank they are there to protect.

The tests that matter here are about ONE failure: ranking a web-era reading against
phone-era history. It produces a number that looks like a percentile, is wrong by
roughly eighteen points, and cannot be spotted by eye because UMich deliberately
smeared the break so the level line has no step.

Mostly synthetic frames, because data/umich_sca.parquet is gitignored (see
sources/umich) and these must pass on a fresh clone. The one data-backed test skips
when the file is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import metrics, store, umich_spec  # noqa: E402
from sources import umich  # noqa: E402


def _frame(months: list[str], **columns) -> pd.DataFrame:
    data = {"survey_date": months}
    data.update(columns)
    return pd.DataFrame(data)


def _month_range(start: str, periods: int) -> list[str]:
    return [d.strftime("%Y-%m-01") for d in pd.date_range(start, periods=periods, freq="MS")]


# ---------------------------------------------------------------- the rank pool

def test_rank_pool_drops_everything_before_the_mode_change() -> None:
    months = _month_range("2024-01-01", 12)
    frame = _frame(months, sentiment=[float(i) for i in range(12)])
    pool = umich_spec.rank_pool(frame, "sentiment")

    kept = [m for m, v in zip(months, pool) if pd.notna(v)]
    assert kept == _month_range("2024-07-01", 6), (
        "the pool must start at the first fully-web month and exclude the blend"
    )


def test_rank_pool_excludes_the_blended_months_specifically() -> None:
    """April, May and June 2024 are 75/25, 50/50 and 25/75 mixtures of two
    instruments, so each averages two populations and belongs to neither."""
    frame = _frame(umich_spec.BLEND_MONTHS, sentiment=[77.2, 69.1, 68.2])
    assert umich_spec.rank_pool(frame, "sentiment").notna().sum() == 0


def test_rank_pool_keeps_the_index_aligned_rather_than_dropping_rows() -> None:
    """Callers hand the result straight to a causal percentile keyed on the frame's
    own index; a shorter series would silently misalign."""
    months = _month_range("2023-01-01", 40)
    frame = _frame(months, sentiment=[1.0] * 40)
    pool = umich_spec.rank_pool(frame, "sentiment")
    assert len(pool) == len(frame)
    assert list(pool.index) == list(frame.index)


@pytest.mark.parametrize(
    "column, first_allowed, last_refused",
    [
        ("sentiment", "2024-07-01", "2024-06-01"),
        ("current_conditions", "2024-07-01", "2024-06-01"),
        ("expectations", "2024-07-01", "2024-06-01"),
        # NOT the mode change: UMich report "few differences in medians" between
        # phone and web, and the +2pp effect they measured is on the mean, which is
        # not ingested. The binding break for the median is the 1982 wording change.
        ("infl_exp_1y", "1982-03-01", "1982-02-01"),
        ("infl_exp_5y10y", "1990-04-01", "1990-03-01"),
    ],
)
def test_each_boundary_is_where_the_documentation_says(
    column: str, first_allowed: str, last_refused: str
) -> None:
    frame = _frame([last_refused, first_allowed], **{column: [1.0, 2.0]})
    pool = umich_spec.rank_pool(frame, column)
    assert pd.isna(pool.iloc[0]), f"{last_refused} must be outside the {column} pool"
    assert pd.notna(pool.iloc[1]), f"{first_allowed} must be inside the {column} pool"


def test_inflation_medians_are_not_segmented_at_the_mode_change() -> None:
    """The mode change moved the MEAN, which this repo does not ingest. Segmenting
    the median there too would throw away four decades of a pool for nothing."""
    for column in ("infl_exp_1y", "infl_exp_5y10y"):
        assert umich_spec.BY_COLUMN[column].start < umich_spec.WEB_ERA_START
        assert not umich_spec.BY_COLUMN[column].excluded


def test_every_regime_cites_a_source() -> None:
    """A boundary with no citation is folklore, and the payload will never confirm
    it -- the CSV has no mode column, no flag and no footnote."""
    for r in umich_spec.REGIMES:
        assert r.reason, f"{r.column} regime has no stated reason"
        assert r.citation.startswith("http"), f"{r.column} regime has no citation URL"
        assert r.pool_label, f"{r.column} regime has no label for its pool"


# ------------------------------------------------------- the thin-segment guard

def test_a_thin_segment_is_not_rankable() -> None:
    frame = _frame(_month_range("2024-07-01", 26), sentiment=[50.0] * 26)
    assert umich_spec.rankable(frame, "sentiment") == 26
    assert umich_spec.rankable(frame, "sentiment") < umich_spec.MIN_RANK_OBSERVATIONS


def test_the_guard_opens_once_the_segment_is_long_enough() -> None:
    frame = _frame(_month_range("2024-07-01", 40), sentiment=[50.0] * 40)
    assert umich_spec.rankable(frame, "sentiment") >= umich_spec.MIN_RANK_OBSERVATIONS


def test_the_minimum_is_at_least_three_years() -> None:
    """Not a magic number: below three years of months a single news cycle moves the
    rank by several points and two or three months set the extremes."""
    assert umich_spec.MIN_RANK_OBSERVATIONS >= 36


# -------------------------------------------------- what the segmentation buys

def test_ranking_across_the_mode_change_materially_misstates_the_rank() -> None:
    """The whole point, on a synthetic series with the shift UMich measured.

    Phone era sits around 80, web era around 80 - 6.6. Nothing about consumers
    changed. Ranked across the break, every web month looks near-record-low.
    """
    phone_months = _month_range("2014-01-01", 123)          # .. 2024-03
    web_months = _month_range("2024-07-01", 26)
    months = phone_months + web_months
    # Phone era oscillates 80..84. Web era is the same series shifted down by the
    # 6.6 points UMich measured, so it oscillates 73.4..77.4. The FINAL web value is
    # 75.4, deliberately the middle of its own era -- an unremarkable month. It is
    # also below every phone reading, which is the whole trap.
    values = [80.0 + (i % 5) for i in range(len(phone_months))]
    values += [73.4 + (i % 5) for i in range(len(web_months) - 1)] + [75.4]
    frame = _frame(months, sentiment=values)

    across = metrics.expanding_percentile(frame["sentiment"]).iloc[-1]
    within = metrics.expanding_percentile(
        umich_spec.rank_pool(frame, "sentiment").dropna()
    ).iloc[-1]

    assert across < 25, f"ranked across the break the reading looks extreme ({across})"
    assert within > 50, f"ranked within its own era it is ordinary ({within})"
    assert within - across > 30, "the segmentation must change the claim materially"


# --------------------------------------------------------------- the parser

def test_month_names_are_mapped_explicitly_not_via_strptime() -> None:
    """strptime("%B") is locale-dependent: under a non-English LC_TIME every row
    fails and the series silently becomes empty."""
    assert set(umich.MONTHS) == {
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    }
    assert umich.MONTHS["December"] == 12
    # Comments stripped first: the module documents WHY it avoids strptime("%B"),
    # and a scan that cannot tell code from an explanation of the hazard would fail
    # on the very comment warning about it.
    src = (ROOT / "sources" / "umich.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#") and not ln.lstrip().startswith("#:")
    )
    assert "%B" not in code, "locale-dependent month parsing must not come back"


def test_the_source_declares_itself_unmirrorable_with_its_terms() -> None:
    from sources import get

    source = get("umich_sca")
    assert source.redistributable is False
    assert "agreement.php" in source.license
    assert source.citation == (
        "University of Michigan, Survey Research Center, Surveys of Consumers"
    )


def test_caveats_state_the_two_things_most_likely_to_mislead() -> None:
    from sources import get

    blob = " ".join(get("umich_sca").caveats).lower()
    assert "median" in blob, "median-vs-mean must be stated"
    assert "prices in general" in blob, "it is not a CPI expectation and must say so"
    assert "-6.6" in blob or "6.6" in blob, "the measured mode-change shift must be stated"


# ------------------------------------------------ against the real file, if present

@pytest.mark.skipif(
    not store.exists("umich_sca"),
    reason="data/umich_sca.parquet is gitignored; run scripts.ingest --source umich_sca",
)
def test_boundaries_match_the_real_payload() -> None:
    """Each boundary was read out of a UMich document. The documents can be right
    about history and still not describe the file that shipped."""
    frame = store.read("umich_sca")
    dates = pd.to_datetime(frame["survey_date"])

    px5_after = frame.loc[dates >= pd.Timestamp("1990-04-01"), "infl_exp_5y10y"]
    assert px5_after.isna().sum() == 0, (
        "the 5-10 year series is meant to be continuous from April 1990"
    )
    px5_before = frame.loc[dates < pd.Timestamp("1990-04-01"), "infl_exp_5y10y"]
    assert px5_before.isna().sum() > 0, "and sporadic before it, or the boundary is wrong"

    px1_after = frame.loc[dates >= pd.Timestamp("1982-03-01"), "infl_exp_1y"]
    assert px1_after.isna().sum() == 0

    monthly = pd.PeriodIndex(dates.dt.to_period("M"))
    era = monthly[monthly >= pd.Period("1978-01", "M")]
    assert len(era) == len(pd.period_range(era.min(), era.max(), freq="M")), (
        "1978 onward must have no missing months"
    )
    pre = monthly[monthly < pd.Period("1978-01", "M")]
    assert len(pre) < len(pd.period_range(pre.min(), pre.max(), freq="M")), (
        "before 1978 the survey was quarterly or sparser, so months must be missing"
    )
