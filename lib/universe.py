"""Which markets the board shows by default, decided by a rule rather than a list.

CFTC reports on ~1,330 distinct market codes. Showing all of them buries the
signal -- ELECTRICITY AND SOURCES alone has 113 markets and EMISSIONS 40, most
with negligible open interest. Showing four (where this repo started) is the
other failure. So the default screen is chosen by a predicate over the ingested
history, and the predicate is here rather than in a hand-typed list because a
list of 200 names rots and a rule does not.

THE LIQUIDITY STATISTIC IS A TRAILING MAXIMUM, NOT THE LATEST REPORT.

This is the one non-obvious choice. Markets drop out of individual weeks and come
back: `124608 MICRO E-MINI DJIA` carried 63,517 contracts three weeks before the
latest report and is simply absent from it, as are `344607 7 YEAR ERIS SOFR SWAP`
(43,623) and `168DC1 DOGECOIN` (23,075). A latest-report predicate drops all of
them and re-admits them next week.

Measured by running the rule at each of the last 52 report dates and counting set
differences across the 51 transitions:

    liquidity statistic          adds/drops    changes per week
    trailing 26-week max OI          10 / 10        0.39
    latest report only               53 / 54        2.10      5.4x worse

So the trailing maximum cuts membership churn by ~5x at no cost in coverage.
"""
from __future__ import annotations

import pandas as pd

from lib import segments

#: Reports in the trailing liquidity window (~6 months).
LIQUIDITY_WINDOW = 26

#: Open interest thresholds, on the trailing-window maximum.
OI_CORE = 100_000
OI_WIDE = 25_000

#: Markets per subgroup admitted to CORE by liquidity alone. Without a cap, pure
#: open-interest ranking fills the top 30 with 8 natural-gas markets (6 of them
#: basis contracts) and admits no gold, no cotton, no bitcoin and no VIX.
CORE_PER_SUBGROUP = 2

#: A market this liquid gets in regardless of its subgroup's cap.
GLOBAL_FLOOR = 750_000

#: Reports needed before a trailing-3-year ranking means anything.
MIN_HISTORY = 140

# --------------------------------------------------------------------------- #
# Consolidated series and the supersession they imply
# --------------------------------------------------------------------------- #
# CFTC publishes exactly three market codes ending in '+': the Consolidated
# series for DJIA, S&P 500 and NASDAQ-100. They are the notional sum of the
# E-mini and Micro legs at 10:1, and that ratio is VERIFIED FROM THE DATA rather
# than asserted from a contract spec. On 2026-08-25:
#
#   S&P 500     13874+ = 2,074,931   13874A + 13874U/10 = 2,074,931.4  resid -0.4
#   NASDAQ-100  20974+ =   322,190   209742 + 209747/10 =   322,190.0  resid  0.0
#   DJIA        12460+ =    88,454   124603 + 124608/10 (Micro absent that week)
#
# and DJIA cross-checks on the five recent weeks where the Micro leg does report,
# with residuals between -0.4 and +0.5 contracts.
#
# So summing an E-mini and a Micro by CONTRACT COUNT double-counts exposure at
# the wrong weight -- which is what this repo did before, showing NDX E-mini and
# NDX Micro as two independent markets of comparable size when one is a tenth of
# the other per contract.
CONSOLIDATED = {
    "13874+": "S&P 500 Consolidated",
    "20974+": "NASDAQ-100 Consolidated",
    "12460+": "DJIA Consolidated",
}

#: leg code -> the Consolidated code that already contains it, at 10:1.
#: 124608 (MICRO E-MINI DJIA) is in this map deliberately: it is a verified
#: component of 12460+ and omitting it reintroduces exactly the double-count
#: this map exists to prevent.
SUPERSEDED_BY = {
    "13874A": "13874+",  # E-MINI S&P 500
    "13874U": "13874+",  # MICRO E-MINI S&P 500
    "209742": "20974+",  # E-MINI NASDAQ-100
    "209747": "20974+",  # MICRO E-MINI NASDAQ-100
    "124603": "12460+",  # E-MINI DJIA
    "124608": "12460+",  # MICRO E-MINI DJIA
}

# THE COST OF PREFERRING THE CONSOLIDATED SERIES, WHICH IS NOT FREE.
#
# The Consolidated codes are notionally correct and they are also the ones
# carrying the worst scale break in the dataset:
#
#     13874+ / 20974+   846 reports from 2010-06-15, unit break 2023-05-02,
#                       leaving 174 weeks in the current segment
#     13874A / 209742  1,055 reports from 2006-06-13, ZERO unit breaks
#
# So the notionally-correct series has 174 comparable weeks and excludes
# 2007-2009 entirely -- the most informative positioning episode in the sample --
# while the notionally-wrong single-leg series has twenty clean years. That is a
# real trade-off between notional correctness and history, not a bug to fix, and
# the board surfaces it as a toggle with this text rather than choosing silently.
CONSOLIDATED_TRADEOFF = (
    "Consolidated series are notionally correct (E-mini + Micro/10, verified "
    "against the published figures to under one contract) but short and seamed: "
    "846 reports from 2010-06-15 with a 5x contract re-basing on 2023-05-02, "
    "leaving 174 comparable weeks. The single-leg E-mini codes have 1,055 reports "
    "from 2006-06-13 with no unit break at all. Prefer Consolidated to size "
    "exposure; prefer the E-mini leg to rank today against history including "
    "2008."
)

#: Bitcoin Cash carries cftc_commodity_code 133, identical to CME Bitcoin, so
#: grouping crypto by commodity code silently folds BCH into BTC. Any
#: coin-equivalent aggregation therefore needs an explicit per-code map, not a
#: code-based fold. Multipliers below are read from the contract_units text and
#: are NOT verifiable against a published Consolidated identity the way the
#: equity-index 10:1 ratio is -- so anything derived from them is labelled as
#: text-derived wherever it is shown.
CRYPTO_UNITS_ARE_TEXT_DERIVED = (
    "Crypto coin-equivalent sizes are parsed from the free-text contract_units "
    "field. Unlike the equity-index 10:1 ratio, no published identity exists to "
    "check them against, and the field is already demonstrably loose -- Bitcoin "
    "Cash reports units of '(1 Bitcoin)'. Treat coin-notional as indicative."
)


def latest_report_date(df: pd.DataFrame, date_col: str = "report_date") -> pd.Timestamp | None:
    if df.empty:
        return None
    return pd.to_datetime(df[date_col]).max()


def market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per market code: liquidity, liveness, history, current label.

    Built from a single cohort's rows because the market-week columns (open
    interest, names, groups) are identical across cohorts, so scanning all of
    them is pure waste. `nonrept` is chosen because it is the only cohort present
    in every one of the seven report families -- filtering on any other cohort
    silently loses 62-92% of the market list.
    """
    if df.empty:
        return pd.DataFrame()

    one = df[df["cohort"] == "nonrept"]
    if one.empty:  # a source without that cohort: fall back to the first present
        one = df[df["cohort"] == df["cohort"].iloc[0]]

    one = one.copy()
    one["report_date"] = pd.to_datetime(one["report_date"])
    dates = sorted(one["report_date"].unique())
    window = set(dates[-LIQUIDITY_WINDOW:])
    live_window = set(dates[-8:])

    grouped = one.groupby("market_code", observed=True)
    rows = []
    for code, g in grouped:
        g = g.sort_values("report_date")
        in_win = g[g["report_date"].isin(window)]
        last = g.iloc[-1]
        # Reports in the CURRENT contract-unit segment, which is the only count a
        # ranking may use. Total reports over the file is the wrong gate: 30
        # legacy and 4 tff live markets have enough total history while their
        # current segment holds under MIN_HISTORY, so a board trusting the total
        # would rank a re-based market against a different contract.
        seg = segments.segment_ids(g["report_date"], g["contract_units"])
        n_segment = int((seg == seg.max()).sum()) if len(seg) else 0
        rows.append(
            {
                "market_code": code,
                # Latest NAME, not any historical one. 26-30% of codes have been
                # renamed and one code can carry up to 8 names, so taking
                # distinct names would fill a selector with dead labels.
                "market": last["market"],
                "market_full": last["market_full"],
                "commodity": last["commodity"],
                "group": last["group"],
                "subgroup": last["subgroup"],
                "contract_units": last["contract_units"],
                "oi_latest": float(last["open_interest"]),
                "oi_window_max": float(in_win["open_interest"].max()) if len(in_win) else 0.0,
                "last_report": last["report_date"],
                "first_report": g["report_date"].iloc[0],
                "n_reports": int(len(g)),
                "n_segment_reports": n_segment,
                "n_last8": int(g["report_date"].isin(live_window).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["superseded_by"] = out["market_code"].map(SUPERSEDED_BY)
    out["is_consolidated"] = out["market_code"].isin(CONSOLIDATED)
    return out.sort_values("oi_window_max", ascending=False).reset_index(drop=True)


def assign_tiers(summary: pd.DataFrame, *, allowlist: set[str] | None = None) -> pd.DataFrame:
    """Add a `tier` column: CORE, WIDE or THIN.

    CORE  -- on the default screen. Top `CORE_PER_SUBGROUP` by trailing-window
             open interest within each subgroup, subject to OI_CORE; plus
             anything above GLOBAL_FLOOR; plus the allowlist.
    WIDE  -- ingested and selectable, not on the default screen.
    THIN  -- below OI_WIDE, or too short a history to rank.

    Nothing is excluded from the DATA by this function. Tiering is a display
    decision only, so a market that falls out of CORE this week is still there to
    look at, and the base-rate board still sees the whole cross-section.
    """
    if summary.empty:
        return summary.assign(tier=[])

    allow = set(allowlist or set()) | set(CONSOLIDATED)
    s = summary.copy()

    # A superseded leg never reaches CORE -- its exposure is already counted in
    # the Consolidated series, and showing both invites adding them together.
    eligible = s["superseded_by"].isna() & (s["n_last8"] >= 1)

    subgroup_key = s["subgroup"].fillna("(ungrouped) " + s["group"].fillna("unknown"))
    rank_in_subgroup = (
        s.where(eligible)
        .groupby(subgroup_key, observed=True)["oi_window_max"]
        .rank(ascending=False, method="first")
    )

    is_core = eligible & (
        ((s["oi_window_max"] >= OI_CORE) & (rank_in_subgroup <= CORE_PER_SUBGROUP))
        | (s["oi_window_max"] >= GLOBAL_FLOOR)
        | s["market_code"].isin(allow)
    )
    is_wide = ~is_core & (s["oi_window_max"] >= OI_WIDE)

    s["tier"] = "THIN"
    s.loc[is_wide, "tier"] = "WIDE"
    s.loc[is_core, "tier"] = "CORE"
    s["rank_in_subgroup"] = rank_in_subgroup
    # can_rank is about the CURRENT segment, not the whole file. A market can have
    # twenty years of prints and still be unrankable today because it was re-based
    # last month, and that distinction is the entire point of lib/segments.
    s["can_rank"] = s.get("n_segment_reports", s["n_reports"]) >= MIN_HISTORY
    s["can_rank_full_history"] = s["n_reports"] >= MIN_HISTORY
    return s


def default_market(summary: pd.DataFrame, prefer: tuple[str, ...] = ()) -> str | None:
    """Pick a landing market: first available preference, else largest CORE."""
    if summary.empty:
        return None
    codes = set(summary["market_code"])
    for code in prefer:
        if code in codes:
            return code
    core = summary[summary.get("tier", "CORE") == "CORE"]
    pool = core if not core.empty else summary
    return str(pool.sort_values("oi_window_max", ascending=False)["market_code"].iloc[0])
