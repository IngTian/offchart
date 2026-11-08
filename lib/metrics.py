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
   "+3.89%" reads as growth while the position shrank toward zero. So net, flow,
   change, net_share and purity are reported as differences; gross, open
   interest, long, short, trader counts, shares and prices may take a ratio.
   The rule is keyed on what the quantity IS, never on what the sample shows: an
   earlier empirical check asked whether a series HAPPENED to cross zero, and
   over the tff universe that returned "safe" for 78 of 550 cohort-net series
   (14.2%) -- including S&P 500 E-mini asset managers, whose net is positive for
   all 1,055 weeks on record and could still go negative next week -- as well as
   for an all-NaN series. Nothing in this repo divides a signed quantity today;
   board_flows' nav_change_pct is a ratio of NAV, which is a price level.

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

    Non-negative BY CONSTRUCTION, so a percent change on it is meaningful (rule 2
    in the module docstring). This is a fragility measure, not a direction
    measure -- it rises when a cohort adds on both sides, which is when a forced
    unwind has the most to sell.

    Caveat the construction cannot express: the premise fails on 46 rows of the
    real data. Pre-1999 legacy non-reportable long/short columns are occasionally
    NEGATIVE (29 rows in legacy_fut 1986-1997, 17 in legacy_futopt 1998, worst
    -228,000 contracts on code 005601 in 1986), presumably a publishing artifact
    of the era. Non-negativity is a claim about the QUANTITY and those rows are a
    fact about the DATA; they are left as published rather than clamped, because a
    clamp hides a bad input instead of showing it. Anything ranking or ratio-ing
    gross over pre-1999 legacy history should filter them.
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
    dominated by regimes that no longer exist, so "extreme against the whole
    record" is the question it answers and "extreme against how this market
    behaves NOW" is not. Trailing-window variants existed here and were deleted
    for having no caller; a board that needs one should narrow the time range
    instead, which Grafana does for free.

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


# --------------------------------------------------------------------------- #
# ETF creation/redemption flow and mechanical rebalancing
# --------------------------------------------------------------------------- #
def etf_flow(assets: pd.Series, nav: pd.Series) -> pd.Series:
    """Money created or redeemed, in dollars. Signed: positive is an inflow.

    DERIVED FROM ASSETS, NOT FROM THE SHARE COUNT, and that is the whole point of
    this function existing rather than being one line at the call site.

    The obvious formula is `diff(shares) * nav`. It is wrong for any fund that has
    reverse-split enough that its retroactively adjusted early share counts fall
    below the resolution of the published column, and it is wrong in the direction
    the chart plots. Measured on ProShares' own file: SQQQ carries 140 rows whose
    `Shares Outstanding (000)` rounds to exactly 0.0, 183 days where the share
    method computes a $0 flow while over $1M genuinely moved, and a worst case of
    16x (2010-08-30: +$103.9M by shares against +$6.5M by assets). On 21% of SQQQ
    rows `shares * nav * 1000` does not reconcile to the published AUM at all. TQQQ
    and UPRO, which only ever forward-split, show 0 such days -- so the bug hides on
    exactly the inverse funds a washout chart is about, and it worsens on the
    issuer's schedule: SQQQ reverse-split as recently as 2025-11-20.

    The assets column is unrounded, so the same identity taken from it is exact:

        flow_t = assets_t - assets_{t-1} * (nav_t / nav_{t-1})

    The NAV ratio is what separates creation from market move. Without it every
    rally reads as buying: assets rise 20% when NAV rises 20% and nothing was
    created. Reproduces TQQQ to the dollar on the 2026-11-05 washout -- NAV -19.10%
    and +$837.7M of inflow, which is the kind of day the measure exists for.

    First observation is NaN: there is no prior day to difference against, and a
    fund's inception-day assets are a creation of the whole fund rather than a flow.
    """
    a = pd.to_numeric(assets, errors="coerce")
    n = pd.to_numeric(nav, errors="coerce")
    grown = a.shift(1) * (n / n.shift(1))
    return a - grown


#: Multiples a leveraged ETF can carry here. Not in the published file -- it lives
#: only inside the marketing name ("UltraPro" is 3x, "Ultra" is 2x), so it is a hand
#: map in sources/leveraged_etf_flows.py and this set is the guard against a typo.
KNOWN_MULTIPLES = frozenset({-3.0, -2.0, -1.0, 1.0, 2.0, 3.0})


def rebalance_demand(assets: pd.Series, multiple: float, move: float = 0.01) -> pd.Series:
    """Notional a fund must mechanically trade for a `move` in its index. Non-negative.

    Derived, not folklore. A fund holding assets A at multiple m carries m*A of
    exposure. After an index move r, its mandate requires m*A*(1+m*r) while the
    exposure it holds is worth m*A*(1+r). The difference it has to trade is

        m*A*(1+m*r) - m*A*(1+r) = (m^2 - m) * A * r

    which is EXACT in r, not a linearisation. See Cheng & Madhavan (2009), "The
    Dynamics of Leveraged and Inverse Exchange-Traded Funds" -- it appears in no SEC
    filing, so the citation is the provenance.

    Two consequences worth stating because both are counterintuitive:

    - The coefficient is 6 for +3x and 12 for -3x. An inverse fund rebalances twice
      as hard as its bull twin, because (m^2 - m) is 12 at m=-3.
    - A bull/bear pair's demand ADDS rather than offsetting. Both buy into a rally:
      the bear fund's short exposure grows as the index falls, so it must buy to
      shrink it. Netting TQQQ against SQQQ reports about half the real demand.

    Unlevered is 0 by construction (m=1 gives m^2-m=0), which is the sanity check.

    This is mandate-implied DEMAND, never an observed trade. It is executed in swap
    and futures notional, and `assets` is an end-of-day figure, so row D's number is
    what a move on D+1 would force. Returned rather than stored per-row for a
    hypothetical move so the caller states the move it is asking about.
    """
    if float(multiple) not in KNOWN_MULTIPLES:
        raise ValueError(
            f"multiple {multiple!r} is not one this repo maps; add it to "
            "KNOWN_MULTIPLES once you have checked the fund's prospectus, because a "
            "wrong multiple scales this measure by its square"
        )
    a = pd.to_numeric(assets, errors="coerce")
    m = float(multiple)
    return (m * m - m) * a * abs(float(move))
