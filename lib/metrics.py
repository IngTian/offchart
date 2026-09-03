"""Derived measures. Deliberately boring, individually testable.

Four conventions are enforced here rather than left to discipline, because each
one was a real error before it was a rule.

1. NO LOOK-AHEAD, EVER. Every ranking function ranks observation t against
   observations at or before t. A percentile taken against the full sample -- the
   default for pandas' .rank(pct=True) -- tells you where a level sat relative to
   data that did not exist yet. tests/test_metrics.py proves causality by
   truncation: computing on a prefix must give bit-identical results to computing
   on the whole series and slicing.

2. SIGNED QUANTITIES ARE NEVER EXPRESSED AS A PERCENT CHANGE. A net position
   going -100,640 -> -96,727 is "+3,913 contracts, less short". Calling it
   "+3.89%" reads as growth while the position shrank toward zero. The gate for
   this is SEMANTIC, not empirical -- see is_ratio_safe.

3. A NORMALISER MUST BE STATED, NOT ASSUMED. Raw contract counts are not
   comparable across markets or across decades. Each function below says what it
   is invariant to and where it breaks.

4. CONCENTRATION SHARES ARE OF THE SIDE TOTAL, NOT OPEN INTEREST. See
   lib/cftc_spec.CONC_FIELDS for the two measurements that establish this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Quantity kinds -- the basis of the percent-change gate
# --------------------------------------------------------------------------- #
#: Quantities that can change sign. A ratio or percent change on any of these
#: inverts its own sense somewhere in its range, so it is banned outright.
#:
#: This replaces an earlier empirical check that asked whether a series HAPPENED
#: to cross zero in the observed sample. That check was actively dangerous: over
#: the tff universe it returned "safe" for 78 of 550 cohort-net series (14.2%),
#: including S&P 500 E-mini asset managers, whose net is positive for all 1,055
#: weeks on record and could still go negative next week. It also returned
#: "safe" for an all-NaN series. A gate keyed on the QUANTITY cannot be fooled
#: by a sample that has not yet done the thing you are guarding against.
SIGNED_KINDS = frozenset({"net", "flow", "change", "net_share", "purity"})

#: Quantities that are non-negative by construction, where a percent change is
#: meaningful.
#:
#: "price" is here because a price level is positive, so a percent change on it is
#: the number a reader actually wants. Note this is a claim about the QUANTITY, not
#: a promise about any particular series: a futures price can print negative (WTI
#: did in April 2020), and a spread between two prices is signed and belongs under
#: "net". Only an outright level goes here.
UNSIGNED_KINDS = frozenset(
    {
        "gross", "open_interest", "spread", "long", "short", "traders",
        "share", "gross_share", "price",
    }
)


def is_ratio_safe(kind: str) -> bool:
    """True if a percent change on a quantity of this kind is meaningful.

    Keyed on what the quantity IS, never on what the sample happens to show.
    Unknown kinds are refused -- failing closed is the point.
    """
    if kind in SIGNED_KINDS:
        return False
    if kind in UNSIGNED_KINDS:
        return True
    raise ValueError(
        f"unknown quantity kind {kind!r}; add it to SIGNED_KINDS or UNSIGNED_KINDS "
        "so the percent-change gate has an opinion about it"
    )


# --------------------------------------------------------------------------- #
# Level measures
# --------------------------------------------------------------------------- #
def net(long: pd.Series, short: pd.Series) -> pd.Series:
    """Directional lean. CHANGES SIGN -- report in contracts, never in percent.

    Not a stance. A cohort's net is a sum over firms running incompatible
    strategies: a basis trader long the cash index and short the future appears
    here as short while holding no market view at all.
    """
    return long - short


def gross(long: pd.Series, short: pd.Series) -> pd.Series:
    """Total book size: how much there is to unwind.

    Non-negative BY CONSTRUCTION, which is why 'gross' is in UNSIGNED_KINDS and a
    percent change on it is permitted. This is a fragility measure, not a
    direction measure -- it rises when a cohort adds on both sides, which is when
    a forced unwind has the most to sell.

    Caveat the classification cannot express: the premise fails on 46 rows of the
    real data. Pre-1999 legacy non-reportable long/short columns are occasionally
    NEGATIVE (29 rows in legacy_fut 1986-1997, 17 in legacy_futopt 1998, worst
    -228,000 contracts on code 005601 in 1986), presumably a publishing artifact
    of the era. UNSIGNED_KINDS is a statement about the QUANTITY, correctly, and
    those rows are a statement about the DATA. Anything ranking or ratio-ing gross
    over pre-1999 legacy history should filter them.
    """
    return long + short


def directional_oi(open_interest: pd.Series, spread_total: pd.Series) -> pd.Series:
    """Open interest excluding spread positions -- the denominator that matters.

    A spread position is long one expiry and short another, so it is in neither
    the long nor the short column and expresses no direction. Dividing a net by
    total open interest therefore understates the directional tilt by a factor
    of 1/(1 - spread_share), and the correction is not cosmetic: on the latest
    report the spread share is a median 3.2% of open interest but 44.5% in
    3-month SOFR, 47.2% in Fed Funds and 48.2% in the adjusted-rate S&P
    contract.

    Also the correct denominator for CR4/CR8, which are published as shares of
    the side total.
    """
    return open_interest - spread_total.fillna(0)


def share_of(part: pd.Series, whole: pd.Series, *, min_whole: float = 0.0) -> pd.Series:
    """`part` as a percentage of `whole`, guarding a vanishing denominator.

    min_whole defaults to 0 (guard division by zero only). Raising it drops
    small markets whose share swings wildly; note that on the current tff
    universe open interest is at or below 1,000 contracts in only 11 of 46,361
    market-weeks and in 0 of the markets on the latest report, so this guard is
    defensive rather than a fix for something observed.
    """
    w = whole.where(whole > max(min_whole, 0.0))
    return 100.0 * part / w.replace(0, np.nan)


def spread_share(spread_total: pd.Series, open_interest: pd.Series) -> pd.Series:
    """How much of the market is calendar structure rather than direction."""
    return share_of(spread_total.fillna(0), open_interest)


def directional_purity(net_: pd.Series, gross_: pd.Series) -> pd.Series:
    """net / gross in [-1, 1]. CHANGES SIGN.

    1 means every contract the cohort holds points the same way; near 0 means a
    big two-sided book with little net view. Distinguishes "flat because small"
    from "flat because hedged", which a net alone cannot.
    """
    return net_ / gross_.replace(0, np.nan)


def avg_position_per_trader(
    positions: pd.Series, traders: pd.Series
) -> pd.Series:
    """Contracts per reporting firm. NaN where the trader count is unavailable.

    A cohort net of 100,000 contracts across 5 firms is a different animal from
    the same net across 60: the first is a handful of decisions and the second
    is a consensus. This is the measure that says which.

    Two hazards, both measured, both handled by returning NaN rather than a
    number:

    - The per-cohort trader count is frequently NULL WHILE THE POSITION IS
      NONZERO -- 6,076 of 46,361 tff rows for leveraged-money longs, and in all
      6,076 the position is positive. So this is not a 0/0 case that NaN
      propagation handles for free; it is a real value divided by an unknown.
    - The null rate is not a historical artifact, it is growing and current:
      4.2% of 2015 rows, 12.6% of 2018, 22.0% of 2025, 28.7% of 2026, and 30 of
      the 94 markets on the latest report. Any cross-market table ranked on this
      measure is silently missing about a third of the board, so the UI must
      show the coverage rather than just dropping the rows.

    Non-reportable traders are never counted in any family, so the field does
    not exist at all -- that surfaces as an absent spec entry, not a null.
    """
    t = pd.to_numeric(traders, errors="coerce")
    return positions / t.where(t > 0)


# --------------------------------------------------------------------------- #
# Ranking -- all causal
# --------------------------------------------------------------------------- #
def expanding_percentile(values: pd.Series, *, min_history: int = 1) -> pd.Series:
    """Percentile of each observation against every observation up to it.

    Vectorised via argsort rather than the O(n^2) python loop this replaced
    (measured: 55.3s -> 0.22s on 290,000 points).

    Invariant to nothing except monotone rescaling of the whole history, which
    is exactly why it must be computed WITHIN a segment -- see lib/segments. Its
    weakness is the mirror of its strength: after 40 years the window is
    dominated by regimes that no longer exist. Use trailing_percentile when
    "extreme relative to how this market behaves NOW" is the question.

    The first observation is always 100.0 (it is the maximum of a one-element
    set), so `min_history` blanks the warm-up rather than shipping a meaningless
    100th percentile.
    """
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=values.index, name="pctile")

    ok = ~np.isnan(v)
    # rank of v[i] among v[:i+1]: count of non-NaN values <= v[i], by insertion
    # into the sorted prefix. Done with a running sorted array via searchsorted
    # on the argsort-ordered prefix would still be O(n log n) per step; instead
    # rank the whole array once and correct, which is exact for the "<=" count
    # because we only ever compare against earlier positions.
    order = np.argsort(np.where(ok, v, np.inf), kind="mergesort")
    rank_of_pos = np.empty(n, dtype=np.int64)
    rank_of_pos[order] = np.arange(n)

    # Fenwick tree over sorted ranks gives count of earlier values <= v[i].
    tree = np.zeros(n + 1, dtype=np.int64)

    def _add(i: int) -> None:
        i += 1
        while i <= n:
            tree[i] += 1
            i += i & (-i)

    def _sum(i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return int(s)

    seen = 0
    for i in range(n):
        if not ok[i]:
            continue
        r = int(rank_of_pos[i])
        _add(r)
        seen += 1
        if seen >= max(min_history, 1):
            out[i] = 100.0 * _sum(r) / seen
    return pd.Series(out, index=values.index, name="pctile")


def last_percentile(values: pd.Series) -> float:
    """Expanding percentile of the FINAL observation only.

    The cross-market screen needs one number per market, not a whole series, and
    computing the full expanding curve to read its last point is 1,300x more
    work than it needs to be (measured: 7.04 ms versus 0.0054 ms on a
    2,100-point series -- 9.9 s versus 0.008 s across 1,400 markets).
    """
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if v.size == 0:
        return float("nan")
    return 100.0 * float((v <= v[-1]).sum()) / v.size


#: CFTC publishes weekly, so a window of N days is expected to hold N/7 reports.
#: This is the causal way to say "is this window full enough" -- see _causal_floor.
DAYS_PER_REPORT = 7.0


def _causal_floor(window_days: int, min_obs_frac: float, hard_min: float) -> float:
    """Minimum observations before a trailing window is allowed to publish.

    DERIVED FROM THE NOMINAL CADENCE, NOT FROM THE SAMPLE. An earlier version
    took the MODE of the realised rolling counts, which is look-ahead: the mode
    is computed over the whole series, so whether observation t published
    depended on how many reports arrived AFTER t. That is a rule-2 violation
    hiding in a gate rather than in a statistic, which is the hardest kind to
    notice -- the published values were all causal, but the DECISION to publish
    them was not.

    Deriving the target from the cadence (weekly, so window_days/7 reports) is
    data-independent and therefore causal by construction. The cost is that
    markets with holes, and the pre-2002 legacy era where reports are
    semi-monthly, blank more often -- which is the honest outcome: a 3-year
    window that holds 40 reports genuinely is not a 3-year ranking.
    """
    expected = window_days / DAYS_PER_REPORT
    return max(hard_min, min_obs_frac * expected)


def _rank_last_in_window(w) -> float:
    """Percentile of the window's final value among the window's non-NaN values.

    Both halves matter. Comparing against NaN yields False, so a NaN final
    observation would otherwise score 0.0 -- and 0.0 reads as "the most extreme
    low in three years" when the truth is "no reading". Measured on the real
    corpus before this guard: 97 (market, cohort) series produced 10,720
    week-readings where the underlying measure was NaN and the percentile
    published exactly 0.0.

    Counting NaNs in the denominator is the second half: it silently shrinks
    every rank toward zero in proportion to how gappy the window is.
    """
    last = w[-1]
    if np.isnan(last):
        return np.nan
    vals = w[~np.isnan(w)]
    if vals.size == 0:
        return np.nan
    return 100.0 * float((vals <= last).sum()) / vals.size


def trailing_percentile(
    values: pd.Series,
    dates: pd.Series,
    *,
    window_days: int = 1095,
    min_obs_frac: float = 0.6,
) -> pd.Series:
    """Rank within a trailing time window ending at each observation.

    Strictly causal -- the window is (t - window_days, t] and the publish gate is
    derived from the cadence rather than from the realised sample. It asks a
    different question from the expanding version: extreme relative to the last
    three years, not relative to all history.
    """
    v = pd.to_numeric(values, errors="coerce")
    d = pd.to_datetime(pd.Series(dates))
    frame = pd.DataFrame({"v": v.to_numpy()}, index=pd.DatetimeIndex(d.to_numpy()))

    counts = frame["v"].rolling(f"{window_days}D").count()
    floor = _causal_floor(window_days, min_obs_frac, 2.0)

    ranks = frame["v"].rolling(f"{window_days}D").apply(_rank_last_in_window, raw=True)
    out = ranks.where(counts >= floor)
    out.index = values.index
    out.name = "trailing_pctile"
    return out


def cot_index(
    values: pd.Series,
    dates: pd.Series,
    *,
    window_days: int = 1095,
    min_obs_frac: float = 0.6,
) -> pd.Series:
    """The industry-standard COT index: 100*(v - min)/(max - min) over a window.

    Offered because it is what published COT commentary means by "positioning is
    at 90", so a board that cannot reproduce it cannot check that commentary.
    It is NOT the default here, and the reason is structural rather than a
    matter of taste:

    - It is a MIN-MAX, not a rank. One historical extreme pins the denominator
      for the whole window, so the index can sit at 50 for three years while the
      distribution underneath it shifts completely.
    - It is maximally sensitive to exactly the artifact this repo segments
      against: a single re-based observation becomes the min or the max and
      compresses everything else toward the middle.

    trailing_percentile answers the same question and degrades gracefully. Read
    the two together; where they disagree, the min-max is usually the one being
    driven by one old print.
    """
    v = pd.to_numeric(values, errors="coerce")
    d = pd.to_datetime(pd.Series(dates))
    frame = pd.DataFrame({"v": v.to_numpy()}, index=pd.DatetimeIndex(d.to_numpy()))
    roll = frame["v"].rolling(f"{window_days}D")

    counts = roll.count()
    floor = _causal_floor(window_days, min_obs_frac, 2.0)

    lo, hi = roll.min(), roll.max()
    span = (hi - lo).replace(0, np.nan)
    out = (100.0 * (frame["v"] - lo) / span).where(counts >= floor)
    out.index = values.index
    out.name = "cot_index"
    return out


def trailing_zscore(
    values: pd.Series,
    dates: pd.Series,
    *,
    window_days: int = 1095,
    min_obs_frac: float = 0.6,
) -> pd.Series:
    """Standardised level within a trailing window. Causal.

    Reported alongside a percentile because they fail differently: the z-score
    keeps resolution in the tails where a rank saturates at 0 or 100, and the
    rank survives the fat tails and regime shifts that make a z-score's sigma
    meaningless.
    """
    v = pd.to_numeric(values, errors="coerce")
    d = pd.to_datetime(pd.Series(dates))
    frame = pd.DataFrame({"v": v.to_numpy()}, index=pd.DatetimeIndex(d.to_numpy()))
    roll = frame["v"].rolling(f"{window_days}D")
    counts = roll.count()
    floor = _causal_floor(window_days, min_obs_frac, 3.0)
    z = (frame["v"] - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
    out = z.where(counts >= floor)
    out.index = values.index
    out.name = "trailing_z"
    return out


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
#: A span may be off a nominal week by at most this many days and still count.
#: CFTC's holes are integer weeks to within one day (the largest deviation across
#: ~46,000 tff gaps is 1 day, including a 3,255-day hole), so a tolerance of 1
#: admits every real weekly gap -- 6, 7 and 8 days -- and nothing else.
SPAN_TOLERANCE_DAYS = 1.0


def flow(values: pd.Series, dates: pd.Series, *, horizon_weeks: int = 1) -> pd.Series:
    """Change over exactly `horizon_weeks` weeks, NaN where the span is wrong.

    Uses the calendar rather than the row count. CFTC skips weeks -- the legacy
    gap distribution after 2002 is {3 days: 1, 4: 1, 6: 14, 7: 1,256, 8: 14} and
    holes of years exist -- so `.diff(1)` over a skipped week silently returns a
    two-week change wearing a one-week label.

    The span test is an absolute tolerance, NOT a rounded week count. Rounding
    accepts a 4-day gap as a week, since round(4/7) == 1, and those exist: the
    1997-12-19 -> 1997-12-23 and 2003-02-14 -> 2003-02-18 legacy pairs are
    re-issues four days apart. Measured, that rounding admitted 147 four-day
    comparisons in legacy_fut of which 141 disagreed with CFTC's own published
    weekly change -- i.e. the round() was manufacturing 96% of its own failures.

    Returns NaN rather than a mislabelled number, because a gap in a flow chart
    is honest and a wrong horizon is not.
    """
    v = pd.to_numeric(values, errors="coerce")
    d = pd.to_datetime(pd.Series(dates))
    span = d.diff(horizon_weeks).dt.days
    ok = (span - 7.0 * horizon_weeks).abs() <= SPAN_TOLERANCE_DAYS
    return v.diff(horizon_weeks).where(ok.to_numpy())


def weeks_between(dates: pd.Series, periods: int) -> pd.Series:
    """Whole weeks spanned by a `periods`-row shift, rounded.

    For LABELLING a span, not for validating one -- round(4/7) == 1 would call a
    four-day gap a week. Use `flow`, which applies SPAN_TOLERANCE_DAYS.
    """
    d = pd.to_datetime(pd.Series(dates))
    return (d.diff(periods).dt.days / 7.0).round()


def net_flow_balances(flows_by_cohort: pd.DataFrame) -> pd.Series:
    """Per-date sum of cohort net flows, which must be ~0.

    Every contract has two sides, so a week's net buying summed over all cohorts
    is zero up to the publisher's rounding. This is the second free invariant in
    the data and it is worth checking rather than assuming: it catches a dropped
    cohort, a bad field map and a misaligned join, all of which produce plausible
    individual series.

    Compare the result against lib.cftc_spec.FLOW_TOL, which is twice the level
    tolerance because a flow is a difference of two independently rounded levels.
    The tolerance is deliberately NOT applied here: this function returns the
    measurement and the caller decides what to do about a breach, which is what
    lets the integrity board show the distribution instead of just a verdict.
    (An earlier signature took a `tol` argument and ignored it -- worse than not
    offering one, since a caller passing tol=3 silently got no check at all.)
    """
    return flows_by_cohort.sum(axis=1)


def asof(
    value_dates: pd.Series,
    values: pd.Series,
    at_dates: pd.Series,
    *,
    max_staleness_days: int | None = None,
) -> pd.Series:
    """Sample `values` as of each date in `at_dates`: the last one at or BEFORE it.

    Lives here rather than in a panel because the direction is a correctness
    property, not a display choice. Positioning is reported as of a Tuesday and
    prices trade daily, so pairing them means picking a price for that Tuesday --
    and `direction="nearest"` or a forward fill would pick a price from LATER,
    putting a number that did not exist yet beside a position. That is look-ahead
    entering through a join instead of through a statistic, which is the variety
    that survives review because no ranking function is involved.

    Measured on the real data: strictly backward gives 0 of 846 NASDAQ report dates
    a price dated after the report, with a median lag of 0 days and a maximum of 4
    (a holiday Tuesday falling back to the previous Friday). A forward join leaks
    on 1 row per market -- small, and the wrong kind of small.

    `max_staleness_days` caps how old the carried-forward value may be. Without it,
    a backward join carries the last known value forward FOREVER, which turns a hole
    in the source into a flat line that reads as "the price stopped moving" -- and a
    flat line beside a series that moves every week invites exactly the wrong
    inference. Measured on the real data before this cap existed: platinum (PL=F) had
    54 report weeks staler than 30 days, and the fabricated plateau landed across the
    2008 crash, the single largest move in that contract's history. A dead symbol is
    the same failure at the live end -- the Bloomberg Commodity index stopped
    updating on 2026-07-17 and drew a flat line that grew by a week every week.

    Returns a Series indexed by `at_dates`, NaN where no earlier value exists or
    where the nearest earlier one is staler than the cap.
    """
    left = pd.DataFrame({"_at": pd.to_datetime(pd.Series(at_dates)).to_numpy()})
    right = pd.DataFrame(
        {
            "_on": pd.to_datetime(pd.Series(value_dates)).to_numpy(),
            "_v": pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
        }
    ).dropna(subset=["_on"])

    if left.empty or right.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex(left["_at"]), name="asof")

    merged = pd.merge_asof(
        left.sort_values("_at"),
        right.sort_values("_on"),
        left_on="_at",
        right_on="_on",
        direction="backward",  # never "nearest", never "forward" -- see above
    )
    out = merged["_v"]
    if max_staleness_days is not None:
        age = (merged["_at"] - merged["_on"]).dt.days
        out = out.where(age.notna() & (age <= max_staleness_days))
    return pd.Series(
        out.to_numpy(), index=pd.DatetimeIndex(merged["_at"]), name="asof"
    )


def clamp_percentage(values: pd.Series, *, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    """Drop out-of-range percentages to NaN rather than plotting them.

    The published concentration columns are not always valid percentages in the
    legacy datasets: 23 rows in legacy_fut and 18 in legacy_futopt exceed 100%,
    the worst being code 148776 reporting a top-8 long concentration of 482.6%
    on open interest of 957. disagg and tff have none. Plotting 482% would be a
    faithful rendering of a broken input; NaN is the honest one.
    """
    v = pd.to_numeric(values, errors="coerce")
    return v.where((v >= lo) & (v <= hi))
