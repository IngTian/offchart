#!/usr/bin/env python3
"""Derive the wide board_* tables Grafana queries. Run: python -m scripts.build

TWO LAYERS, AND THE SPLIT IS THE POINT

The ARCHIVE mirrors each source faithfully: same rows, same columns, keyed as the
source declares. scripts/ingest writes it, through lib/db.upsert, and nothing in this
file touches it except to read.

The SERVING tables are wide -- one row per grid timestamp, one column per series --
because that is what a Grafana panel query wants and because both boards already put
every series on one grid: the price is sampled as-of each weekly CFTC report date, and
the survey board is monthly throughout.

WHY THE PERCENTILES ARE COMPUTED HERE, IN PANDAS, AND NOT IN SQL

This is the correctness core of the migration, so the reasoning is written down.

1. The segment boundaries cannot be expressed in SQLite at all.
   lib/segments.unit_signature pulls digits out of free-text contract units
   ('(NASDAQ 100 INDEX X $100)' -> '100') to detect a contract re-specification.
   SQLite has no REGEXP without a loadable extension, and the Grafana data source will
   not be loading one. Since the open-interest rank must be computed WITHIN a segment,
   the rank follows the segmentation into Python. That alone settles it.

2. The proof of causality lives with the pandas implementation. tests/test_metrics.py
   proves expanding_percentile is causal by truncation, WITH non-causal negative
   controls asserted to fail. Move the statistic into SQL and coverage of the repo's
   single most important property drops to zero.

3. A revision makes incremental materialisation wrong, not merely slow. CFTC revises
   already-published weeks, and a revision at week t changes every expanding rank from
   t forward. So this rebuilds everything every time; at that price no invalidation
   logic is needed and a stale percentile -- a plausible wrong number -- cannot exist.

4. A time filter cannot be pushed into a causal rank. The rank at t depends on all
   history up to t, so an in-SQL rank would scan a market's whole history on every
   panel load regardless of the visible window.

Each board table is built into a _new table and swapped inside one transaction, so
Grafana never reads a half-built table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from lib import cohort_groups, db, metrics, pricemap, segments, umich_spec  # noqa: E402
from lib.universe import assign_tiers, market_summary  # noqa: E402

#: Families charted on the positioning board. futopt is ingested but not served: a
#: combined-basis figure does not reconcile against a futures-only one, and putting
#: both in one dropdown invites comparing two numbers that are not comparable.
FAMILIES = ("cftc_tff_fut", "cftc_disagg_fut", "cftc_legacy_fut")

#: Markets given a serving row. CORE and WIDE are the liquidity tiers a person browses;
#: THIN is mostly dead electricity and basis contracts. This is the difference between
#: a serving table of a few million rows and one of thirty-nine million, and a THIN
#: market is still fully present in the ARCHIVE for anyone who queries it directly.
SERVED_TIERS = ("CORE", "WIDE")


def _epoch_ms(when) -> pd.Series:
    """Epoch MILLISECONDS, unit-independent.

    Milliseconds because Grafana's $__from and $__to are milliseconds, and the SQLite
    data source has no $__timeFilter macro, so panel SQL writes its own
    `ts BETWEEN $__from AND $__to` and both sides must agree on the unit.

    Normalised through as_unit('ns') rather than dividing by a hardcoded 1_000_000,
    because pandas resolution is not fixed: pandas 3 gives to_datetime on these ISO
    date strings a resolution of 'us', where older pandas gave 'ns'. A hardcoded
    divisor is therefore off by a factor of a thousand on one of them -- and the
    failure mode is a panel query that returns zero rows rather than one that errors,
    which is the kind that looks like missing data.
    """
    idx = pd.DatetimeIndex(when).as_unit("ns")
    return pd.Series(idx.astype("int64") // 1_000_000, index=range(len(idx)))


#: Markets forced into a served tier regardless of contract-count liquidity.
#:
#: 133741 is CME's full-size BITCOIN future (5 BTC). It misses OI_WIDE on a count of
#: CONTRACTS, while 133742 -- MICRO BITCOIN, a fiftieth of the notional -- clears it,
#: because a micro contract racks up fifty times the count for the same exposure. So
#: the board was serving the micro leg and calling it bitcoin: 2.5% of CME's bitcoin
#: open interest by notional. A liquidity filter denominated in contracts cannot see
#: that, and the fix belongs here rather than in OI_WIDE -- lowering the threshold to
#: admit this one market admits sixteen others and quietly reshapes the cross-section
#: the base-rate reasoning rests on. One code, auditable, with the reason written down.
FORCE_SERVED = {"133741"}


def _served_markets(dataset: str, frame: pd.DataFrame) -> dict[str, str]:
    """market_code -> the label a reader picks from, for every served market.

    Returns the label rather than a bare set because `board_positioning` is the only
    table a Grafana panel OR template variable may read (tests/test_grafana asserts
    both), so a name that is not carried here cannot reach the dropdown at all. The
    code stays inside the label: the dropdown fuzzy-matches on the label only, so
    "BITCOIN (133742)" is findable by `btc` AND by the code, where a bare name loses
    the latter.

    `market` is the short contract name, not `market_full`: measured over the served
    universe, `market` changes on 3 of 288 codes while `market_full` changes on 132 --
    the latter's churn is cosmetic exchange renaming, and a label that moves is a
    dropdown entry that moves under the reader.
    """
    summary = assign_tiers(market_summary(frame), allowlist=FORCE_SERVED)
    tiered = summary[summary["tier"].isin(SERVED_TIERS)]
    codes = tiered["market_code"].astype(str)
    names = tiered["market"].astype(str) if "market" in tiered else codes
    return {c: f"{n} ({c})" for c, n in zip(codes, names)}


def _closes_by_symbol(con) -> dict[str, pd.DataFrame]:
    """symbol -> its close series, read ONCE.

    Read once rather than per market because the naive version re-read the whole
    202,451-row prices table inside the per-market loop -- 700-odd times per family,
    invisible except as a build that takes minutes.
    """
    px = db.read_frame(con, "prices")
    if px.empty:
        return {}
    px = px[["symbol", "date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    return {str(sym): sub for sym, sub in px.groupby("symbol", observed=True)}


def _price_for(closes: dict[str, pd.DataFrame], symbol: str, dates: pd.Series) -> pd.Series:
    """Close as-of each report date, backward only, staleness-capped.

    metrics.asof, not a merge on nearest: nearest would put a price that did not exist
    yet beside a position, which is look-ahead smuggled in through a join.
    """
    one = closes.get(symbol)
    if one is None or one.empty:
        return pd.Series(index=dates.index, dtype=float)
    return metrics.asof(
        one["date"], one["close"], dates, max_staleness_days=10
    )


def _positioning_rows(dataset: str, frame: pd.DataFrame, closes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (market, cohort group, week), with every derived column."""
    frame = frame.copy()
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["market_code"] = frame["market_code"].astype(str)
    frame["cohort"] = frame["cohort"].astype(str)

    served = _served_markets(dataset, frame)
    cohorts = sorted(frame["cohort"].unique())
    groups = cohort_groups.groups_for(dataset, cohorts)
    print(f"    {len(served)} markets x {len(groups)} groups: {sorted(groups)}")

    out = []
    for code, sub in frame.groupby("market_code", observed=True):
        if code not in served:
            continue
        week = sub.groupby("report_date")
        oi = week["open_interest"].first().astype(float)
        units = week["contract_units"].first()
        seg = segments.segment_ids(pd.Series(oi.index), units)
        seg.index = oi.index

        ref = pricemap.for_code(code)
        price = (_price_for(closes, ref.symbol, pd.Series(list(oi.index)))
                 if ref else pd.Series(index=range(len(oi)), dtype=float))
        price.index = oi.index

        # Open interest is a LEVEL in contracts, so its rank must restart at a
        # re-specification; the share is a ratio and a re-denomination cancels.
        oi_rank = pd.Series(float("nan"), index=oi.index)
        for _, idx in oi.groupby(seg.to_numpy()).groups.items():
            oi_rank.loc[idx] = metrics.expanding_percentile(oi.loc[idx]).to_numpy()

        for group, members in groups.items():
            picked = sub[sub["cohort"].isin(members)]
            if picked.empty:
                continue
            by = picked.groupby("report_date")
            # The two sides separately, not only their difference. A net of zero is
            # two different animals -- a cohort holding nothing, and a cohort holding
            # a large book on both sides -- and only long and short can tell them
            # apart. metrics.net/gross/directional_purity carry the reasoning.
            longs = by["long"].sum().astype(float).reindex(oi.index)
            shorts = by["short"].sum().astype(float).reindex(oi.index)
            net = metrics.net(longs, shorts)
            gross = metrics.gross(longs, shorts)
            # net/gross in [-1, 1]: 1 means every contract points the same way, near
            # zero means a big two-sided book with little net view. This is the column
            # that distinguishes "flat because small" from "flat because hedged".
            purity = metrics.directional_purity(net, gross)
            share = metrics.share_of(net.reindex(oi.index), oi)
            rank = metrics.expanding_percentile(share)
            chg = metrics.flow(
                share.reset_index(drop=True), pd.Series(oi.index), horizon_weeks=2
            )
            out.append(pd.DataFrame({
                "dataset": dataset,
                "market_code": code,
                # Both labels are carried rather than derived in panel SQL because a
                # template variable may read only the serving tables, so a name that
                # is not here cannot reach a dropdown. `cohort_label` also keeps
                # lib/cohort_groups.LABEL the single source of truth -- the
                # alternative is a ten-branch CASE in committed dashboard JSON.
                "market_label": served[code],
                "cohort_group": group,
                "cohort_label": cohort_groups.label_for(group),
                "report_date": oi.index.strftime("%Y-%m-%d"),
                "ts": _epoch_ms(oi.index).to_numpy(),
                "segment_id": seg.to_numpy(),
                "net": net.reindex(oi.index).to_numpy(),
                "long": longs.to_numpy(),
                "short": shorts.to_numpy(),
                # long + short. NON-NEGATIVE by construction, which is why a percent
                # change on it is permitted where one on `net` is not -- a fragility
                # measure rather than a direction one: it rises when a cohort adds on
                # BOTH sides, which is when a forced unwind has the most to sell.
                #
                # One documented data caveat it cannot express: pre-1999 legacy
                # non-reportable long/short columns are occasionally NEGATIVE (46 rows,
                # worst -228,000 contracts in 1986), a publishing artefact of the era.
                # So gross can read below zero on that slice even though the quantity
                # cannot. Left as the publisher's number rather than clamped, because
                # a clamp would hide a bad input instead of showing it.
                "gross": gross.to_numpy(),
                "purity": purity.to_numpy(),
                "open_interest": oi.to_numpy(),
                "share": share.to_numpy(),
                "share_pctile": rank.to_numpy(),
                "oi_pctile": oi_rank.to_numpy(),
                "chg_2w": chg.to_numpy(),
                "price": price.to_numpy(),
                "n_pool": int(share.notna().sum()),
            }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _inflation_rows(con) -> pd.DataFrame:
    """The survey board, wide, with each rank taken over its own declared pool."""
    frame = db.read_frame(con, "umich_sca")
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame["survey_date"] >= umich_spec.MONTHLY_FROM]
    frame = frame.sort_values("survey_date").reset_index(drop=True)
    when = pd.to_datetime(frame["survey_date"])

    out = pd.DataFrame({
        "survey_date": frame["survey_date"].astype(str),
        "ts": _epoch_ms(when).to_numpy(),
    })
    for column in ("sentiment", "current_conditions", "expectations",
                   "infl_exp_1y", "infl_exp_5y10y"):
        out[column] = frame[column].astype(float).to_numpy()
        n = umich_spec.rankable(frame, column)
        out[f"{column}_pool"] = n
        if n < umich_spec.MIN_RANK_OBSERVATIONS:
            # NOT a number Grafana can draw. The web era is 26 months against a floor
            # of 36, so this column is NULL and the panel shows a gap -- the rule
            # lives in the data rather than in an `if` inside a dashboard.
            out[f"{column}_pctile"] = float("nan")
            continue
        pool = umich_spec.rank_pool(frame, column)
        ranked = metrics.expanding_percentile(pool.dropna()).reindex(frame.index)
        out[f"{column}_pctile"] = ranked.to_numpy()
    return out


def _swap_in(con, table: str, frame: pd.DataFrame, key: tuple[str, ...]) -> None:
    """Replace a serving table atomically.

    Built into <table>_new and renamed inside one transaction. Rebuilding in place
    would let Grafana read a half-written table and draw intermittently blank panels.

    The explicit BEGIN is load-bearing and not decoration. Python's sqlite3 opens a
    transaction implicitly before DML but NOT before DDL, so `with con:` around a DROP
    and a RENAME autocommits each statement as it runs and has nothing to roll back --
    a crash between the two left NO serving table, and every panel on the board read
    `no such table: board_positioning` until someone reran the build. SQLite does DDL
    transactionally without complaint; only the driver needed telling. Proved by
    tests/test_build.py, which fails the whole way back to an empty database without
    this line.
    """
    staging = f"{table}_new"
    con.execute(f'DROP TABLE IF EXISTS "{staging}"')
    db.upsert(con, staging, frame, key)
    con.execute("BEGIN IMMEDIATE")
    with con:
        con.execute(f'DROP TABLE IF EXISTS "{table}"')
        con.execute(f'ALTER TABLE "{staging}" RENAME TO "{table}"')


#: Every source feeding board_flows. Separate sources because each issuer's terms
#: differ -- ProShares and SSGA forbid mirroring, VanEck restricts commercial use --
#: and `license` is a per-source field the repo requires to be accurate. One board
#: because they answer one question.
FLOW_SOURCES = ("proshares_flows", "spy_flows", "gld_flows", "smh_flows")


def _flow_rows(con) -> pd.DataFrame:
    """One row per (fund, day) with flow, cumulative flow and rebalancing demand.

    Per TICKER rather than per index family, and nothing is summed across families:
    a dollar of NDX rebalancing demand and a dollar of SPX demand are not
    interchangeable, and adding them would assert the two indices move together.

    Everything derived is computed here in pandas for the same reason the
    percentiles are -- a panel may only read a serving table, and the cumulative sum
    in particular is an expanding window that no panel should be recomputing on
    every load.
    """
    frames = []
    for table in FLOW_SOURCES:
        one = db.read_frame(con, table)
        if one.empty:
            print(f"  ! {table} has no rows -- run `make pull` first")
            continue
        frames.append(one)
    if not frames:
        return pd.DataFrame()
    # Concat rather than one table per issuer: the funds share a grain (one fund-day),
    # a flow identity and a board, and only their LICENCES differ -- which is why the
    # sources stay separate upstream and meet here.
    raw = pd.concat(frames, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    out = []
    for ticker, sub in raw.groupby("ticker", observed=True):
        # Sorted before anything differences or accumulates. etf_flow and cumsum are
        # both position-based, so an unsorted frame would silently difference the
        # wrong pair of days -- the same hazard lib/segments documents for ranks.
        sub = sub.sort_values("trade_date").reset_index(drop=True)
        multiple = float(sub["multiple"].iloc[-1])
        flow = metrics.etf_flow(sub["assets"], sub["nav"])
        # Computed where the issuer does not publish it, which is both plain-ETF
        # sources. Not taken from VanEck's own '% Change' column even though it
        # exists: that column is TOTAL RETURN on SMH's annual December distribution
        # and disagrees with the NAV-derived change by up to 2.22 points on 14 rows.
        # Tested against `in sub`, not `sub.get(...) is not None`: on an absent
        # column pd.to_numeric(None, errors="coerce") returns np.float64(nan) rather
        # than None, so the obvious guard is always true and .fillna() then raises
        # AttributeError on a numpy scalar. Reachable whenever proshares_flows is
        # the absent table -- `ingest --source spy_flows` on a fresh store, or a
        # ProShares fetch that fails while the other three write rows -- and the
        # crash aborts main() before finalize_for_serving, handing Grafana a
        # WAL-mode file. See tests/test_build.test_a_flows_source_that_publishes_*.
        nav_change = (sub["nav"] / sub["nav"].shift(1) - 1.0) * 100.0
        if "nav_change_pct" in sub:
            published = pd.to_numeric(sub["nav_change_pct"], errors="coerce")
            nav_change = published.fillna(nav_change)
        out.append(pd.DataFrame({
            "ticker": ticker,
            "ticker_label": (
                f"{ticker} — {sub['fund_name'].iloc[-1]} "
                f"({multiple:+g}x {sub['index_family'].iloc[-1]})"
            ),
            "index_family": sub["index_family"].iloc[-1],
            "multiple": multiple,
            "trade_date": sub["trade_date"].dt.strftime("%Y-%m-%d"),
            "ts": _epoch_ms(pd.DatetimeIndex(sub["trade_date"])).to_numpy(),
            "nav": sub["nav"].to_numpy(),
            "nav_change_pct": nav_change.to_numpy(),
            "leverage_kind": "leveraged" if abs(multiple) != 1.0 else "unleveraged",
            "assets": sub["assets"].to_numpy(),
            "shares_outstanding": sub["shares_outstanding"].to_numpy(),
            "flow_usd": flow.to_numpy(),
            # Expanding sum from inception. Causal by construction, and the series
            # the washout question is actually about: did money keep arriving.
            "flow_cum_usd": flow.fillna(0.0).cumsum().to_numpy(),
            "rebalance_1pct_usd": metrics.rebalance_demand(
                sub["assets"], multiple, 0.01
            ).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def serving(con) -> None:
    closes = _closes_by_symbol(con)
    frames = []
    for dataset in FAMILIES:
        raw = db.read_frame(con, dataset)
        if raw.empty:
            print(f"  ! {dataset} has no rows -- run `make pull` (or `make backfill`) first")
            continue
        print(f"  . {dataset}")
        t0 = time.time()
        rows = _positioning_rows(dataset, raw, closes)
        print(f"    {len(rows):,} rows in {time.time() - t0:.1f}s")
        frames.append(rows)

    if frames:
        allrows = pd.concat(frames, ignore_index=True)
        _swap_in(con, "board_positioning", allrows,
                 ("dataset", "market_code", "cohort_group", "ts"))
        print(f"  = board_positioning {len(allrows):,} rows")

    infl = _inflation_rows(con)
    if not infl.empty:
        _swap_in(con, "board_inflation", infl, ("ts",))
        print(f"  = board_inflation {len(infl):,} rows")

    flows = _flow_rows(con)
    if not flows.empty:
        _swap_in(con, "board_flows", flows, ("ticker", "ts"))
        print(f"  = board_flows {len(flows):,} rows")



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="path to the sqlite file")
    args = ap.parse_args()

    con = db.connect(args.db)
    try:
        serving(con)
        con.commit()
        # Leave WAL mode. A WAL-mode file bind-mounted into the Grafana container
        # makes concurrent panel queries lose a lock race -- see db.finalize_for_serving.
        mode = db.finalize_for_serving(con)
        print(f"\njournal_mode now {mode} (readable by Grafana without a -shm race)")
    finally:
        con.close()

    path = Path(args.db) if args.db else db.DB_PATH
    if path.exists():
        print(f"\n{path} is {path.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
