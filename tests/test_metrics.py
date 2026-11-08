"""Tests for lib/metrics.py and lib/segments.py -- the arithmetic layer.

The centrepiece is causality by truncation. Every ranking function here claims to
rank observation t against observations at or before t, and that claim has
exactly one mechanical consequence: computing on a PREFIX must give bit-identical
output to computing on the whole series and slicing to the prefix. Otherwise the
function saw the future. `_assert_prefix_stable` is that check.

A causality test that cannot fail is not a test, so the same check is applied to
two deliberately non-causal implementations -- a full-sample `.rank(pct=True)`
and a centred rolling mean -- and asserted to FAIL. Those are the two mistakes
this repo has actually made.

Causality is necessary and nowhere near sufficient: it constrains which data a
window may see, not what is computed from it, so an inverted COT index and a
population sigma are both perfectly causal. Every ranker therefore also has
hand-computed values, and the holes get their own section -- a NaN observation
must come out NaN, since a percentile of 0.0 reads as "the most extreme low in
three years" and is worse than a blank for looking actionable.

What these tests refuse to do: touch data/, read the live store, or hit the
network. Every fixture is either a literal from the corpus
(the 2023-05-02 NASDAQ re-basing, the 191691 aluminium code reuse, the negative
1998 non-reportable legacy row) or a seeded generator, because a test whose
verdict changes when the weekly ingest lands is not a test of the code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from lib import metrics as M
from lib import segments as S

#: A 1095-day window holds 157 weekly reports (156 prior plus the current one).
WINDOW_REPORTS = 157

#: 420 clean weekly reports: long enough that the trailing window is saturated
#: well before the first truncation point. That matters -- see
#: test_trailing_gate_is_not_prefix_stable.
N_WEEKS = 420


def weekly(n: int = N_WEEKS, start: str = "2015-01-02") -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq="7D"))


def walk(seed: int, n: int = N_WEEKS) -> pd.Series:
    """A random walk, not iid noise. Positioning series are highly persistent,
    and a rank of an iid series is uniform by construction -- which would hide
    bugs that only bite on a trending window."""
    return pd.Series(np.random.default_rng(seed).normal(size=n).cumsum())


#: name -> callable(values, dates, **window_kwargs) for every function that
#: produces a ranking. The dates and kwargs are accepted and ignored, so the
#: causality harness below has one signature; expanding_percentile has no window.
#: Trailing-window rankers lived here too and were deleted -- see the note on
#: test_non_causal_implementations_fail_the_truncation_check.
RANKERS = {
    "expanding_percentile": lambda v, d, **kw: M.expanding_percentile(v),
}

#: Plausible-looking one-liners that see the whole sample. Both must fail.
NON_CAUSAL = {
    "full_sample_rank_pct": lambda v, d: v.rank(pct=True) * 100.0,
    "centred_rolling_mean": lambda v, d: v.rolling(5, center=True, min_periods=1).mean(),
}


def _assert_prefix_stable(fn, values: pd.Series, dates: pd.Series, k: int) -> None:
    """THE causality check. Raises AssertionError if `fn` can see past index k."""
    assert_series_equal(
        fn(values.iloc[:k], dates.iloc[:k]),
        fn(values, dates).iloc[:k],
        check_exact=True,  # bit-identical, not approximately causal
        check_names=False,  # the series name is not the claim under test
    )


# --------------------------------------------------------------------------- #
# Causality by truncation -- the most important test here
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
@pytest.mark.parametrize("k", [1, 2, 5, 37, WINDOW_REPORTS, 300])
def test_expanding_percentile_causal_by_truncation(seed: int, k: int) -> None:
    """Causal at EVERY truncation point including the warm-up, because nothing
    about it depends on how long the series turns out to be."""
    _assert_prefix_stable(RANKERS["expanding_percentile"], walk(seed), weekly(), k)


@pytest.mark.parametrize("name", sorted(NON_CAUSAL))
@pytest.mark.parametrize("k", [37, 200, 350])
def test_non_causal_implementations_fail_the_truncation_check(name: str, k: int) -> None:
    """The check has teeth. Without this it could be comparing a function to
    itself and passing forever.

    This matters more now than when the harness ranked four functions. The
    trailing-window rankers were deleted for having no production caller, so
    expanding_percentile is the only real implementation left under test -- and a
    one-entry harness is exactly the shape that can rot into a tautology. These
    two deliberately-wrong one-liners are what stops that.
    """
    with pytest.raises(AssertionError):
        _assert_prefix_stable(NON_CAUSAL[name], walk(0), weekly(), k)



# --------------------------------------------------------------------------- #
# expanding_percentile against an obviously-correct reference
# --------------------------------------------------------------------------- #
def naive_expanding_percentile(values, min_history: int = 1) -> np.ndarray:
    """O(n^2) python double loop. Slow, and transparently the definition."""
    out, history = [], []
    for x in values:
        if pd.isna(x):
            out.append(np.nan)
            continue
        history.append(float(x))
        n = len(history)
        out.append(
            100.0 * sum(1 for h in history if h <= x) / n
            if n >= max(min_history, 1)
            else np.nan
        )
    return np.asarray(out, dtype=float)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("min_history", [1, 3, 12])
def test_expanding_percentile_matches_naive_reference(seed: int, min_history: int) -> None:
    """The fast version walks a Fenwick tree over a global argsort -- a real
    algorithm with real off-by-one risk, so it is pinned to the loop it replaced.
    A small integer alphabet on purpose: it forces ties, which is where a rank
    implementation goes wrong."""
    rng = np.random.default_rng(seed)
    a = pd.Series(rng.integers(0, 5, size=30).astype(float))
    a.iloc[[3, 10, 29 if seed % 2 else 11]] = np.nan
    np.testing.assert_array_equal(  # exact, not allclose
        M.expanding_percentile(a, min_history=min_history).to_numpy(),
        naive_expanding_percentile(a.to_numpy(), min_history),
    )


NAN = float("nan")

#: (id, values, min_history, expected). Each row is one documented property.
_PCTILE_CASES = [
    # The first point is the maximum of a one-element set: always 100, always
    # meaningless, which is the whole reason min_history exists.
    ("first-observation-is-100", [-9_999.0, 0.0], 1, [100.0, 100.0]),
    ("min_history-blanks-warmup", [3.0, 1.0, 2.0, 5.0], 3, [NAN, NAN, 200 / 3, 100.0]),
    # A hole must not enlarge the denominator: 1 is the smaller of the two REAL
    # observations, so 50.0 -- not the 33.3 that counting the NaN gives. The same
    # property is asserted for the other three rankers under "Holes" below.
    ("nan-skipped-not-ranked", [5.0, NAN, 1.0], 1, [100.0, NAN, 50.0]),
    ("monotone-is-100-throughout", [1.0, 2.0, 3.0, 4.0], 1, [100.0] * 4),
    # A repeated level sits at the top of its own history, not below it.
    ("ties-count-as-le", [2.0, 1.0, 1.0], 1, [100.0, 50.0, 200 / 3]),
]


@pytest.mark.parametrize(
    ("values", "min_history", "expected"),
    [c[1:] for c in _PCTILE_CASES],
    ids=[c[0] for c in _PCTILE_CASES],
)
def test_expanding_percentile_properties(values, min_history, expected) -> None:
    np.testing.assert_allclose(
        M.expanding_percentile(pd.Series(values, dtype=float), min_history=min_history).to_numpy(),
        np.asarray(expected, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Holes. A missing observation is not a reading, in ANY ranker.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(RANKERS))
def test_no_ranker_ranks_a_missing_observation(name: str) -> None:
    """A NaN at the labelled point must produce NaN, not a number.

    This was a live display bug, not a hypothetical: comparing against NaN yields
    False, so `(w <= w[-1]).sum() / len(w)` published exactly 0.0 for a hole --
    10,720 week-readings across 97 (market, cohort) series, every one of them 0.0.
    And 0.0 reads as "the most extreme low on record" where the truth is "no
    reading", which is the worst direction for the error to point.
    """
    v = pd.Series([10.0, 20.0, 30.0, 40.0, NAN, 50.0])
    out = RANKERS[name](v, weekly(6))
    assert pd.isna(out.iloc[4])  # the hole -- NaN, never 0.0
    assert out.notna().iloc[[3, 5]].all()  # ...and only the hole


def test_a_hole_does_not_enlarge_the_denominator() -> None:
    """The second half of the same defect. [5, NaN, 1]: 1 is the smaller of the
    two REAL observations, so 50.0. Counting the NaN gives 33.3 and shrinks every
    rank toward zero in proportion to how gappy the series is."""
    v = pd.Series([5.0, NAN, 1.0])
    assert M.expanding_percentile(v).iloc[2] == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# Level measures
# --------------------------------------------------------------------------- #
def test_net_and_gross() -> None:
    long, short = pd.Series([100, 20]), pd.Series([40, 60])
    assert M.net(long, short).tolist() == [60, -40]  # signed
    assert M.gross(long, short).tolist() == [140, 80]  # long + short, nothing more


def test_gross_is_negative_on_the_pre_1999_legacy_nonrept_rows() -> None:
    """gross is non-negative by CONSTRUCTION and not in the published data.

    Measured over all seven CFTC datasets, long + short is NEGATIVE on 46 rows --
    29 in legacy_fut (1986-01-15..1997-12-19) and 17 in legacy_futopt
    (1998-07-07..1998-12-29), every one the `nonrept` cohort, worst -228,000
    contracts on code 005601 in 1986. The row below is published 148776 /
    1998-12-22 / nonrept: long -3,999 and short -3,726 against an open interest of
    957. 46 of 4,087,353 rows is 0.001%, but legacy is the only pre-2006 history,
    so long-history charts cross them.

    Left as published rather than clamped, and this test is what says so
    deliberately: a clamp would hide a bad input instead of showing it, and the
    board's own panel description carries the caveat. The knock-on is that a share
    derived from such a row is out of range too -- -417.87% below -- which is why
    share_of is not range-guarded either.
    """
    assert M.gross(pd.Series([-3999.0]), pd.Series([-3726.0])).iloc[0] == -7725.0
    share = M.share_of(pd.Series([-3999.0]), pd.Series([957.0]))
    assert share.iloc[0] == pytest.approx(-417.87, abs=0.01)


def test_directional_oi_treats_null_spread_as_zero() -> None:
    """legacy comm and disagg prod_merc have no spread column at all. NaN there
    means 'not reported', so directional OI is the whole OI -- not NaN, which
    would blank every share on those cohorts."""
    assert M.directional_oi(pd.Series([1000, 1000]), pd.Series([100.0, np.nan])).tolist() == [
        900.0,
        1000.0,
    ]


def test_share_of_guards_the_denominator() -> None:
    """Zero, negative and null denominators all become NaN, never inf -- an inf
    sorts to the top of any cross-market ranking."""
    out = M.share_of(pd.Series([10.0] * 4), pd.Series([100.0, 0.0, -5.0, np.nan]))
    assert out.iloc[0] == 10.0
    assert out.iloc[1:].isna().all()


def test_share_of_min_whole_drops_small_markets() -> None:
    out = M.share_of(pd.Series([10.0, 10.0]), pd.Series([500.0, 5000.0]), min_whole=1000)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.2)


def test_spread_share() -> None:
    out = M.spread_share(pd.Series([np.nan, 445.0]), pd.Series([1000.0, 1000.0]))
    assert out.iloc[0] == 0.0  # absent spread column is 0% structure, not unknown
    assert out.iloc[1] == pytest.approx(44.5)  # the 3-month SOFR order of magnitude


def test_directional_purity_is_signed_and_bounded() -> None:
    out = M.directional_purity(pd.Series([-5.0, 10.0, 0.0]), pd.Series([10.0, 10.0, 0.0]))
    assert out.iloc[0] == -0.5  # keeps "short"
    assert out.iloc[1] == 1.0
    assert pd.isna(out.iloc[2])  # flat because empty, not flat because hedged


@pytest.mark.parametrize("dtype", ["Int32", "float64", "object"])
def test_avg_position_per_trader_null_and_zero_traders_give_nan(dtype: str) -> None:
    """A null trader count is a real position over an UNKNOWN divisor. It must be
    NaN, must not be inf, and must not emit a divide warning. Three dtypes because
    the column arrives as nullable Int32 from the store and as float or object after
    a reindex.

    Naming the population, because metrics.py's "about a third of the board" is one
    cohort-column: the 4.2 / 12.6 / 22.0 / 28.7% rates for 2015 / 2018 / 2025 /
    2026 and the "30 of 94 markets" are cftc_tff_fut, cohort lev_money, column
    traders_long, on the 2026-08-25 report. Board-wide it is worse, not better --
    65 of those 94 markets carry at least one null trader count across the four
    reportable cohorts."""
    positions = pd.Series([100.0, 100.0, 100.0])
    traders = pd.Series([4, 0, None], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = M.avg_position_per_trader(positions, traders)
    assert out.iloc[0] == 25.0
    assert out.iloc[1:].isna().all()
    assert np.isfinite(out.dropna().to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------- #
# asof -- look-ahead entering through a join rather than a statistic
# --------------------------------------------------------------------------- #
def test_asof_is_strictly_backward() -> None:
    """Sampling a daily series at a weekly date must never reach forward.

    This is the price-beside-positioning join. Positions are as of a Tuesday; if
    the join picked the NEAREST close it would sometimes pick Wednesday's, putting
    a price that did not exist yet beside the position. No ranking function is
    involved, which is exactly why this variety of look-ahead survives review.
    """
    # Prices on Mon/Wed only, so Tuesday has no same-day close and the direction
    # of the fallback is unambiguous: Monday (before) not Wednesday (after).
    price_dates = pd.to_datetime(["2026-01-05", "2026-01-07", "2026-01-12", "2026-01-14"])
    prices = pd.Series([100.0, 999.0, 200.0, 888.0])
    at = pd.Series(pd.to_datetime(["2026-01-06", "2026-01-13"]))  # two Tuesdays

    got = M.asof(pd.Series(price_dates), prices, at)
    # Monday's close, never Wednesday's. 999 and 888 are the future.
    assert list(got.to_numpy()) == [100.0, 200.0]
    assert 999.0 not in set(got.dropna())

    # Before the series starts there is nothing to sample, and NaN is the only
    # honest answer -- a backfill here would invent a price.
    early = M.asof(pd.Series(price_dates), prices, pd.Series(pd.to_datetime(["2025-12-30"])))
    assert early.isna().all()


def test_asof_forward_join_would_leak_and_is_not_what_we_do() -> None:
    """The negative control: prove the wrong direction is detectably wrong.

    Without this, `asof` could quietly become a forward join and every other
    assertion here would still pass on data where the two agree.
    """
    dates = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-07"]))
    vals = pd.Series([1.0, 2.0])
    at = pd.Series(pd.to_datetime(["2026-01-06"]))

    ours = M.asof(dates, vals, at)
    leaky = pd.merge_asof(
        pd.DataFrame({"_at": at}), pd.DataFrame({"_on": dates, "_v": vals}),
        left_on="_at", right_on="_on", direction="forward",
    )["_v"]
    assert float(ours.iloc[0]) == 1.0
    assert float(leaky.iloc[0]) == 2.0, "forward join must differ, or this proves nothing"


def test_asof_handles_empty_inputs() -> None:
    empty = pd.Series([], dtype="datetime64[ns]")
    at = pd.Series(pd.to_datetime(["2026-01-06"]))
    assert M.asof(empty, pd.Series([], dtype=float), at).isna().all()
    assert M.asof(at, pd.Series([1.0]), empty).empty


# --------------------------------------------------------------------------- #
# Flow: never mislabel a horizon
# --------------------------------------------------------------------------- #
def test_flow_returns_nan_across_a_skipped_week() -> None:
    """2026-01-13 -> 2026-01-27 is a fortnight. `.diff(1)` would report its
    25-contract change under a weekly label; a gap in the chart is honest."""
    dates = pd.Series(pd.to_datetime(["2026-01-06", "2026-01-13", "2026-01-27", "2026-02-03"]))
    out = M.flow(pd.Series([10.0, 20.0, 45.0, 50.0]), dates)
    assert pd.isna(out.iloc[0])  # no predecessor
    assert out.iloc[1] == 10.0
    assert pd.isna(out.iloc[2])  # the skipped week, NOT 25.0
    assert out.iloc[3] == 5.0


def test_flow_at_a_longer_horizon_uses_the_calendar_not_the_row_count() -> None:
    dates = pd.Series(pd.to_datetime(["2026-01-06", "2026-01-13", "2026-01-20", "2026-02-03"]))
    out = M.flow(pd.Series([10.0, 20.0, 45.0, 50.0]), dates, horizon_weeks=2)
    assert out.iloc[2] == 35.0  # 2 rows back AND 2 weeks back
    assert pd.isna(out.iloc[3])  # 2 rows back is 3 weeks back


# --------------------------------------------------------------------------- #
# segments: unit signatures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("units", "signature"),
    [
        ("(NASDAQ 100 INDEX X $100)", "100100"),
        ("(NASDAQ 100 INDEX X $20)", "10020"),
        ("(CONTRACTS OF 5,000 BUSHELS)", "5000"),  # a comma is not a digit
        ("(Contracts of 1,000 Bushels)", "1000"),  # case is not arithmetic either
        ("(THOUSAND BUSHELS)", ""),  # a real break vs a 1,000- or 5,000-bushel contract
        ("(25 Metric Tons)", "25"),
        ("(CONTRACTS OF 40,000 POUNDS)", "40000"),
        (None, ""),
        ("", ""),
        (float("nan"), ""),
    ],
)
def test_unit_signature(units, signature: str) -> None:
    assert S.unit_signature(units) == signature


def test_unit_signature_ignores_cosmetic_churn_in_contract_units() -> None:
    """contract_units churns without the contract changing, so a signature that
    tracked the WORDS would cut a spurious segment on every churn. All four
    spellings below are real corpus strings, and the churn is observed rather than
    hypothetical: code 209742 (NASDAQ MINI) reported
    '( NASDAQ 100 STOCK INDEX X $20)' from 1999-06-22 and the same string without
    the leading space from 2008-09-30. The quoted form is how legacy ships it, and
    the word STOCK comes and goes. None of that is arithmetic.

    (The story this test used to tell -- CFTC's 2022-02-08 wholesale name
    shortening -- is not the mechanism and cannot be: segment_ids is never handed
    the market name, and of the 263 legacy_fut codes reporting on both 2022-02-01
    and 2022-02-08, zero changed contract_units and zero changed name.)
    """
    same_unit = [
        "(NASDAQ 100 STOCK INDEX X $20)",
        "( NASDAQ 100 STOCK INDEX X $20)",  # 209742 before 2008-09-30
        "'(NASDAQ 100 STOCK INDEX X $20)'",  # quoted, as legacy ships it
        "(NASDAQ 100 INDEX X $20)",
    ]
    assert {S.unit_signature(u) for u in same_unit} == {"10020"}
    # ...while the digits still break, or this test would pass on a constant.
    assert S.unit_signature("(NASDAQ 100 INDEX X $100)") == "100100"


# --------------------------------------------------------------------------- #
# segments: segment_ids
# --------------------------------------------------------------------------- #
def test_segment_ids_splits_the_2023_05_02_nasdaq_rebasing() -> None:
    """20974+ open interest went 49,531 -> 255,954 (x5.17) with no calendar gap;
    only the unit signature reveals it. An expanding percentile across this seam
    ranks $20-per-point observations against $100-per-point ones, which is what
    this repo used to ship."""
    dates = pd.Series(
        pd.to_datetime(["2023-04-11", "2023-04-18", "2023-04-25", "2023-05-02", "2023-05-09"])
    )
    units = pd.Series(["(NASDAQ 100 INDEX X $100)"] * 3 + ["(NASDAQ 100 INDEX X $20)"] * 2)
    assert S.segment_ids(dates, units).tolist() == [0, 0, 0, 1, 1]
    assert S.segment_ids(dates).tolist() == [0] * 5  # invisible without units
    # The signatures the split turns on, so a change to the parser shows up here
    # rather than as a seam that silently stops being detected.
    assert [S.unit_signature(u) for u in units] == ["100100"] * 3 + ["10020"] * 2


def test_segment_ids_splits_the_191691_code_reuse() -> None:
    """Same code: 40,000-pound aluminium to 1989, then 25-metric-ton aluminium
    from 2022 across a 12,341-day hole. The gap alone is sufficient, so code
    reuse is caught even where contract_units is unhelpful."""
    dates = pd.Series(
        pd.to_datetime(["1989-02-14", "1989-02-21", "1989-02-28", "2022-12-13", "2022-12-20"])
    )
    units = pd.Series(["(CONTRACTS OF 40,000 POUNDS)"] * 3 + ["(25 Metric Tons)"] * 2)
    assert S.segment_ids(dates, units).tolist() == [0, 0, 0, 1, 1]
    assert S.segment_ids(dates).tolist() == [0, 0, 0, 1, 1]


@pytest.mark.parametrize(("gap_days", "n_segments"), [(7, 1), (56, 1), (91, 1), (92, 2), (365, 2)])
def test_segment_ids_gap_threshold_is_91_days_inclusive(gap_days: int, n_segments: int) -> None:
    """13 weeks is deliberately generous: CFTC skips weeks for holidays and
    shutdowns, and a missed print is not a re-specification."""
    base = pd.Timestamp("2020-01-03")
    dates = pd.Series([base, base + pd.Timedelta(days=gap_days)])
    assert S.segment_ids(dates).nunique() == n_segments


def test_segment_ids_consecutive_identical_units_give_one_segment() -> None:
    dates, units = weekly(20, "2024-01-05"), pd.Series(["(CONTRACTS OF 5,000 BUSHELS)"] * 20)
    assert S.segment_ids(dates, units).tolist() == [0] * 20


@pytest.mark.parametrize("null", [None, np.nan, ""])
def test_a_null_unit_string_does_not_manufacture_two_breaks(null) -> None:
    """contract_units is null on 537 legacy_fut rows across 6 codes. Treating
    null as its own signature would break INTO the hole and again OUT of it --
    two spurious segments where nothing changed. It is forward-filled instead."""
    units = pd.Series(["(X $100)", "(X $100)", null, "(X $100)", "(X $100)"])
    assert S.segment_ids(weekly(5, "2024-01-05"), units).tolist() == [0] * 5


def test_a_leading_null_unit_does_not_break() -> None:
    """Nothing to forward-fill from, so the first known signature is not a change
    from anything."""
    units = pd.Series([None, "(X $100)", "(X $100)", "(X $20)"])
    assert S.segment_ids(weekly(4, "2024-01-05"), units).tolist() == [0, 0, 0, 1]


def test_segment_ids_handles_categorical_units_with_nan() -> None:
    """store.read returns contract_units as a Categorical, whose nulls are NaN
    inside a category dtype -- a different code path from an object column of
    None, and the one production actually takes."""
    units = pd.Series(["(X $100)", None, "(X $100)", "(X $20)"], dtype="category")
    assert S.segment_ids(weekly(4, "2024-01-05"), units).tolist() == [0, 0, 0, 1]


def test_segment_ids_preserves_the_caller_index() -> None:
    """Panels segment a filtered slice of a big frame, so the result must align
    back to that slice's index, not to a fresh 0..n range."""
    dates = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-10", "2021-06-04"]), index=[11, 22, 33])
    assert S.segment_ids(dates).index.tolist() == [11, 22, 33]


def test_a_long_gap_opens_a_new_segment_id() -> None:
    """The ids themselves are what scripts/build ranks within, so the boundary
    has to land on the right row and not merely somewhere."""
    dates = pd.Series(
        pd.to_datetime(["2019-01-04", "2019-01-11", "2021-01-08", "2021-01-15", "2021-01-22"])
    )
    assert S.segment_ids(dates).tolist() == [0, 0, 1, 1, 1]


def test_empty_inputs_do_not_raise() -> None:
    """A market can be filtered to nothing by a cohort or tier selection, and the
    build should produce no rows rather than a traceback."""
    empty_dates = pd.Series([], dtype="datetime64[ns]")
    empty_values = pd.Series([], dtype=float)
    assert S.segment_ids(empty_dates).tolist() == []
    assert len(M.expanding_percentile(empty_values)) == 0
    assert len(M.flow(empty_values, empty_dates)) == 0


# ----------------------------------------------- ETF creation/redemption flow

def test_flow_from_assets_is_immune_to_a_rounded_share_count() -> None:
    """Why flow is derived from ASSETS and not from the share count.

    The obvious formula is `diff(shares) * nav`, and it is wrong for a fund that has
    reverse-split often enough that its retroactively adjusted early share counts fall
    below the published column's resolution. Measured on the real ProShares file: SQQQ
    carries 140 rows whose share count rounds to exactly 0.0 thousands, 183 days where
    `diff(shares) * nav` computes $0 while over $1M genuinely moved, and a worst case
    16x wrong. TQQQ and UPRO, which only forward-split, show 0 such days.

    This fixture is that failure in miniature: a share count rounded to one decimal of
    thousands, where the true creation is visible in assets and invisible in shares.
    """
    nav = pd.Series([10.0, 10.0])
    # Rounded to the published resolution: both days read "0.0" thousands.
    shares = pd.Series([0.0, 0.0])
    # Assets say 40,000 dollars arrived.
    assets = pd.Series([10_000.0, 50_000.0])

    by_assets = M.etf_flow(assets, nav)
    by_shares = shares.diff() * 1000.0 * nav

    assert by_shares.iloc[1] == 0.0, "fixture no longer reproduces the rounding blindness"
    assert by_assets.iloc[1] == pytest.approx(40_000.0), (
        "flow must come from assets, which carry the creation the share column rounded away"
    )


def test_flow_separates_creation_from_market_move() -> None:
    """A pure NAV move with no creation is ZERO flow, not the change in assets.

    This is the whole reason the formula carries a NAV ratio: assets rise when the
    market rises, and calling that an inflow would make every rally look like buying.
    """
    nav = pd.Series([10.0, 12.0])
    assets = pd.Series([1_000.0, 1_200.0])  # +20% assets, entirely the +20% NAV
    out = M.etf_flow(assets, nav)
    assert pd.isna(out.iloc[0]), "the first observation has no prior day to difference"
    assert out.iloc[1] == pytest.approx(0.0, abs=1e-9)


def test_flow_is_signed_and_survives_a_drawdown() -> None:
    """The washout case: NAV collapses while money comes in."""
    nav = pd.Series([62.1887, 50.3126])
    assets = pd.Series([33_540_000_000.0, 27_970_000_000.0])
    out = M.etf_flow(assets, nav)
    # assets fell, but by less than the -19.1% NAV move, so the residual is an INFLOW
    assert out.iloc[1] > 0
    assert out.iloc[1] == pytest.approx(824_000_000.0, rel=0.05)


def test_rebalance_demand_is_exact_in_the_move_not_linearised() -> None:
    """(m^2 - m) * assets * move, derived rather than asserted.

    A leveraged fund holds m*A of exposure. After an index move r its target is
    m*A*(1+m*r) while its held exposure is worth m*A*(1+r); the difference it must
    trade is (m^2 - m)*A*r. Exact in r, not a first-order approximation.
    """
    assets = pd.Series([1_000.0])
    # +3x: coefficient 6. A 1% move forces 6% of assets.
    assert M.rebalance_demand(assets, 3.0, 0.01).iloc[0] == pytest.approx(60.0)
    # -3x: coefficient 12, NOT 6. An inverse fund rebalances harder than its bull twin.
    assert M.rebalance_demand(assets, -3.0, 0.01).iloc[0] == pytest.approx(120.0)
    # +2x: coefficient 2
    assert M.rebalance_demand(assets, 2.0, 0.01).iloc[0] == pytest.approx(20.0)
    # unlevered: no mechanical trading at all
    assert M.rebalance_demand(assets, 1.0, 0.01).iloc[0] == pytest.approx(0.0)


def test_a_bull_bear_pair_adds_rebalancing_demand_rather_than_offsetting() -> None:
    """The counterintuitive one, and the reason the panel must not net the pair.

    Both funds buy into a rally and sell into a decline, because the bear fund's short
    exposure grows as the index falls. Netting TQQQ against SQQQ would report roughly
    half the real mechanical demand.
    """
    assets = pd.Series([1_000.0])
    bull = M.rebalance_demand(assets, 3.0, 0.01).iloc[0]
    bear = M.rebalance_demand(assets, -3.0, 0.01).iloc[0]
    assert bull > 0 and bear > 0, "both sides trade in the same direction"
    assert bull + bear == pytest.approx(180.0)


def test_an_inverse_fund_rebalances_at_1x_while_its_bull_twin_does_not() -> None:
    """Why the board carries a -1x series and no +1x one.

    The asymmetry looks like an inconsistency on the board until the arithmetic is
    done, so it is pinned rather than left to the panel description. m^2-m is 0 at
    m=+1 -- an unleveraged fund that gains 1% on a 1% move is already correctly
    positioned and trades nothing -- but 2 at m=-1, because a SHORT fund's exposure
    grows as the index falls and it has to buy back. A -1x fund rebalances exactly
    as hard per dollar as a +2x.
    """
    assets = pd.Series([1_000.0])
    assert M.rebalance_demand(assets, 1.0, 0.01).iloc[0] == 0.0
    assert M.rebalance_demand(assets, -1.0, 0.01).iloc[0] == pytest.approx(20.0)
    assert (M.rebalance_demand(assets, -1.0, 0.01).iloc[0]
            == pytest.approx(M.rebalance_demand(assets, 2.0, 0.01).iloc[0]))


#: Only the magnitudes this repo actually maps. A 4x case was tried and
#: rebalance_demand refused it -- KNOWN_MULTIPLES fails loudly on a multiple nobody
#: has checked against a prospectus, which is the right answer and is pinned
#: elsewhere, so there is no point asserting the ratio on a fund that cannot exist.
@pytest.mark.parametrize(("magnitude", "ratio"), [(2.0, 3.0), (3.0, 2.0)])
def test_the_bear_bull_coefficient_ratio_is_not_a_fixed_factor(
    magnitude: float, ratio: float
) -> None:
    """(m^2+m)/(m^2-m) = (m+1)/(m-1), so the asymmetry SHRINKS as leverage rises.

    Pinned because the panel description used to say an inverse fund "rebalances
    twice as hard", generalising from the 3x pair. That is wrong on the 2x pair,
    which is three times -- and 2x is the only pair on the semiconductor board, so
    the claim was wrong exactly where a reader was most likely to check it.
    """
    assets = pd.Series([1_000.0])
    bull = M.rebalance_demand(assets, magnitude, 0.01).iloc[0]
    bear = M.rebalance_demand(assets, -magnitude, 0.01).iloc[0]
    assert bear / bull == pytest.approx(ratio)
    assert bear / bull == pytest.approx((magnitude + 1) / (magnitude - 1))
