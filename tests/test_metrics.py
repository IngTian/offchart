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

What these tests refuse to do: touch data/, import lib.store, lib.cache or
streamlit, or hit the network. Every fixture is either a literal from the corpus
(the 2023-05-02 NASDAQ re-basing, the 191691 aluminium code reuse, the 482.6%
legacy concentration print) or a seeded generator, because a test whose verdict
changes when the weekly ingest lands is not a test of the code.
"""
from __future__ import annotations

import inspect
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


#: name -> callable(values, dates) for every function that produces a ranking.
RANKERS = {
    "expanding_percentile": lambda v, d: M.expanding_percentile(v),
    "trailing_percentile": M.trailing_percentile,
    "cot_index": M.cot_index,
    "trailing_zscore": M.trailing_zscore,
}
WINDOWED = ["trailing_percentile", "cot_index", "trailing_zscore"]

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


@pytest.mark.parametrize("name", WINDOWED)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
@pytest.mark.parametrize("k", [200, 280, 350])
def test_windowed_rankers_causal_by_truncation(name: str, seed: int, k: int) -> None:
    """Causal once the window is saturated. Every cut is >= 158 on purpose, and
    the reason is a real limitation rather than test convenience -- see
    test_trailing_gate_is_not_prefix_stable."""
    _assert_prefix_stable(RANKERS[name], walk(seed), weekly(), k)


@pytest.mark.parametrize("name", sorted(NON_CAUSAL))
@pytest.mark.parametrize("k", [37, 200, 350])
def test_non_causal_implementations_fail_the_truncation_check(name: str, k: int) -> None:
    """The check has teeth. Without this it could be comparing a function to
    itself and passing forever."""
    with pytest.raises(AssertionError):
        _assert_prefix_stable(NON_CAUSAL[name], walk(0), weekly(), k)


@pytest.mark.parametrize("name", WINDOWED)
@pytest.mark.parametrize("k", [100, WINDOW_REPORTS])
def test_windowed_values_agree_wherever_both_are_published(name: str, k: int) -> None:
    """During warm-up a prefix and the full series disagree about WHICH points to
    publish, but never about the value of a published point -- so the look-ahead
    documented below is confined to the min-observation mask. No rank, index or
    z-score is contaminated by a future observation."""
    fn, v, d = RANKERS[name], walk(5), weekly()
    on_prefix, sliced = fn(v.iloc[:k], d.iloc[:k]), fn(v, d).iloc[:k]
    both = on_prefix.notna() & sliced.notna()
    assert both.sum() > 0
    np.testing.assert_array_equal(on_prefix[both].to_numpy(), sliced[both].to_numpy())


@pytest.mark.parametrize("k", [60, 100, WINDOW_REPORTS])
def test_trailing_gate_is_causal_during_warmup(k: int) -> None:
    """The publish/blank GATE must be causal too, not just the published values.

    This started life as an xfail documenting a real look-ahead bug: the min_obs
    floor was derived from `counts.mode()` over the whole series, so whether
    observation t published depended on how many reports arrived after it. On a
    clean 420-week walk a 157-row prefix blanked 1 point while the full series
    blanked 94 of those same 157 — identical values, different masks.

    lib.metrics._causal_floor now derives the floor from the nominal weekly
    cadence instead, which is data-independent and so causal by construction.
    The cut points here are all inside the warm-up on purpose, since that is the
    only region where the two derivations disagree.
    """
    _assert_prefix_stable(RANKERS["trailing_percentile"], walk(5), weekly(), k)


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
    # observations, so 50.0 -- not the 33.3 that counting the NaN gives.
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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_last_percentile_equals_final_expanding_value(seed: int) -> None:
    """The cross-market screen's shortcut must not be a different measure from
    the chart it sits next to."""
    v = walk(seed, 300)
    v.iloc[[7, 88, 201]] = np.nan  # interior holes, but not a trailing one
    assert M.last_percentile(v) == pytest.approx(M.expanding_percentile(v).iloc[-1])


def test_last_percentile_with_no_observations_is_nan() -> None:
    assert pd.isna(M.last_percentile(pd.Series([np.nan, np.nan])))


# --------------------------------------------------------------------------- #
# The percent-change gate -- semantic, never empirical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "safe"),
    [(k, True) for k in sorted(M.UNSIGNED_KINDS)] + [(k, False) for k in sorted(M.SIGNED_KINDS)],
)
def test_is_ratio_safe_classifies_every_declared_kind(kind: str, safe: bool) -> None:
    assert M.is_ratio_safe(kind) is safe


def test_is_ratio_safe_refuses_unknown_kind() -> None:
    """Failing closed is the point: an unclassified kind must not default to
    'percent change is fine'."""
    with pytest.raises(ValueError, match="unknown quantity kind"):
        M.is_ratio_safe("basis_points")


def test_is_ratio_safe_cannot_consult_the_data() -> None:
    """This replaced an empirical check that asked whether a series HAPPENED to
    cross zero -- which called 78 of 550 tff cohort nets safe, including S&P
    E-mini asset managers, positive for all 1,055 weeks on record and free to go
    negative next week. The structural guarantee the bug cannot return is that
    the function takes no data argument at all."""
    assert list(inspect.signature(M.is_ratio_safe).parameters) == ["kind"]
    assert M.is_ratio_safe("net") is False


# --------------------------------------------------------------------------- #
# Level measures
# --------------------------------------------------------------------------- #
def test_net_and_gross() -> None:
    long, short = pd.Series([100, 20]), pd.Series([40, 60])
    assert M.net(long, short).tolist() == [60, -40]  # signed
    assert M.gross(long, short).tolist() == [140, 80]  # non-negative


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
    """A null trader count is a real position over an UNKNOWN divisor -- 28.7% of
    2026 rows and 30 of the 94 markets on the latest report. It must be NaN, and
    must not be inf and must not emit a divide warning. Three dtypes because the
    column arrives as nullable Int32 from parquet and as float or object after a
    reindex."""
    positions = pd.Series([100.0, 100.0, 100.0])
    traders = pd.Series([4, 0, None], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = M.avg_position_per_trader(positions, traders)
    assert out.iloc[0] == 25.0
    assert out.iloc[1:].isna().all()
    assert np.isfinite(out.dropna().to_numpy(dtype=float)).all()


def test_clamp_percentage_drops_impossible_prints() -> None:
    """Code 148776 reports a top-8 long concentration of 482.6% on OI of 957.
    Plotting it faithfully renders a broken input; NaN is the honest render.
    Both bounds are inclusive -- 0% and 100% concentration are legal."""
    out = M.clamp_percentage(pd.Series([482.6, 100.0, -1.0, 0.0, 50.0, np.nan]))
    assert out.isna().tolist() == [True, False, True, False, False, True]
    assert out.dropna().tolist() == [100.0, 0.0, 50.0]


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


@pytest.mark.parametrize(
    ("days", "weeks"),
    [(7, 1.0), (14, 2.0), (6, 1.0), (8, 1.0), (3, 0.0), (3255, 465.0), (12341, 1763.0)],
)
def test_weeks_between_and_weeks_elapsed_agree_and_round(days: int, weeks: float) -> None:
    """Every CFTC gap is a multiple of 7 days to within one, so round(days/7) is
    exact. 3255 is the E-mini Russell hole, 12341 the 191691 aluminium hole."""
    base = pd.Timestamp("2000-01-07")
    dates = pd.Series([base, base + pd.Timedelta(days=days)])
    assert M.weeks_between(dates, 1).iloc[1] == weeks
    assert S.weeks_elapsed(dates).iloc[1] == weeks
    assert pd.isna(M.weeks_between(dates, 1).iloc[0])


# --------------------------------------------------------------------------- #
# segments: unit signatures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("units", "signature"),
    [
        ("(NASDAQ 100 INDEX X $100)", "100100"),
        ("(NASDAQ 100 INDEX X $20)", "10020"),
        ("(CONTRACTS OF 5,000 BUSHELS)", "5000"),
        ("(CONTRACTS OF 5000 BUSHELS)", "5000"),  # commas stripped, so these agree
        ("(THOUSAND BUSHELS)", ""),  # a real break vs 5,000-bushel contracts
        ("(25 Metric Tons)", "25"),
        ("(CONTRACTS OF 40,000 POUNDS)", "40000"),
        (None, ""),
        ("", ""),
        (float("nan"), ""),
    ],
)
def test_unit_signature(units, signature: str) -> None:
    assert S.unit_signature(units) == signature


def test_unit_signature_ignores_the_2022_name_shortening() -> None:
    """CFTC shortened names wholesale on 2022-02-08. If a signature tracked the
    words, 26-30% of codes would be cut into spurious segments on that date."""
    assert S.unit_signature("(E-MINI S&P 500 INDEX X $50)") == S.unit_signature(
        "(E-MINI S&P 500 STOCK INDEX X $50)"
    )


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
    assert S.last_segment_mask(dates, units).tolist() == [False, False, False, True, True]

    described = S.describe_segments(dates, units)
    assert described["reports"].tolist() == [3, 2]
    assert str(described["start"].iloc[1]) == "2023-05-02"
    assert described["unit_signature"].tolist() == ["100100", "10020"]


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
    assert S.last_segment_mask(dates, units).all()


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
    assert S.last_segment_mask(dates).index.tolist() == [11, 22, 33]


def test_last_segment_mask_selects_only_the_final_segment() -> None:
    dates = pd.Series(
        pd.to_datetime(["2019-01-04", "2019-01-11", "2021-01-08", "2021-01-15", "2021-01-22"])
    )
    mask = S.last_segment_mask(dates)
    assert mask.tolist() == [False, False, True, True, True]
    assert S.segment_ids(dates)[mask].nunique() == 1


def test_empty_inputs_do_not_raise() -> None:
    """A market can be filtered to nothing by a cohort or tier selection, and a
    panel should render an empty chart rather than a traceback."""
    empty_dates = pd.Series([], dtype="datetime64[ns]")
    empty_values = pd.Series([], dtype=float)
    assert S.segment_ids(empty_dates).tolist() == []
    assert S.last_segment_mask(empty_dates).tolist() == []
    assert S.weeks_elapsed(empty_dates).tolist() == []
    assert len(S.describe_segments(empty_dates)) == 0
    assert len(M.expanding_percentile(empty_values)) == 0
    assert pd.isna(M.last_percentile(empty_values))
