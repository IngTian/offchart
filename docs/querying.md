# Querying the database yourself

The boards are a convenience; the database is the product. It is one ordinary SQLite
file at `data/offchart.sqlite` — open it with anything.

```
sqlite3 data/offchart.sqlite
```

```sql
-- who is most extreme this week, across every served market
SELECT market_label, cohort_label, round(share, 2) AS pct_oi, round(share_pctile) AS pctile
FROM board_positioning
WHERE dataset = 'cftc_tff_fut'
  AND ts = (SELECT max(ts) FROM board_positioning WHERE dataset = 'cftc_tff_fut')
  AND n_pool >= 200
ORDER BY share_pctile DESC
LIMIT 10;
```

## Two table layers, and the split matters

**Archive** — one table per source, faithful to the row, exactly what the upstream
published: `cftc_tff_fut`, `prices`, `umich_sca`, `proshares_flows`, … Read these for
research. Nothing is dropped, rounded or reinterpreted on the way in, which is why a
column the boards never use is still stored.

**Serving** — `board_positioning`, `board_inflation`, `board_flows`: wide, one row per
grid point, with every derived quantity already computed. Read these for charts.

**Do not recompute a percentile in SQL.** `PERCENT_RANK() OVER (ORDER BY share)` is
textbook look-ahead and reads as perfectly reasonable SQL; `tests/test_grafana.py` fails
the build if any dashboard query contains it, or `CUME_DIST`, or `NTILE`. The ranks in
the serving tables are already causal. See [internals](internals.md) for why they are
computed in pandas and cannot move here.

## Column conventions you cannot guess

These are the ones a reader will otherwise misread. The full set of data caveats is in
[data-notes](data-notes.md).

- **`ts` is epoch MILLISECONDS.** Grafana's `$__from` / `$__to` are milliseconds, so
  that is what is stored. The SQLite plugin reads a *time* column as epoch **seconds**,
  so panels project `ts/1000 AS time`. Querying by hand, use
  `datetime(ts/1000, 'unixepoch')`.
- **`chg_2w` is in CONTRACTS, not percent.** No signed quantity in this repo is ever
  expressed as a ratio — see `lib/metrics.py`'s convention 2 for the reason and the
  measurement behind it.
- **`share` is a percentage of open interest**, 0–100, not a fraction.
- **`purity` is `net / gross`**, in [−1, 1], and it CHANGES SIGN. Near 0 means a large
  two-sided book with little net view; ±1 means every contract points the same way.
- **`segment_id` is the comparability unit.** Ranks are computed within a segment, which
  is cut at contract-unit changes and at gaps over a quarter. Comparing a percentile
  across two `segment_id`s is comparing two different contracts.
- **`n_pool` is how many observations the percentile was ranked against.** A percentile
  with `n_pool = 12` is not a percentile. The boards filter on it; a hand-written query
  must too.
- **`*_pool` columns in `board_inflation`** name the regime each rank was taken within,
  for the same reason.
- **Two `board_flows` columns are COMPUTED, not published.** `nav_change_pct` is derived
  from NAV wherever the issuer publishes no usable change column — both plain-ETF
  sources — and SMH's `shares_outstanding` is derived as assets / NAV because VanEck
  publishes it nowhere. Flow itself never uses the share count.
- **`rebalance_1pct_usd` is mandate-implied, never an observed trade**, and is exactly 0
  for an unleveraged fund by arithmetic rather than by special case:
  `(multiple² − multiple) × assets × move` is zero at `multiple = 1`.

## The serving tables as they stand

Regenerate these with `make readme-table` and `PRAGMA table_info` rather than trusting
the shapes below, which were measured on 2026-11-08.

### `board_positioning` — 1,374,550 rows

```
dataset, market_code, market_label, cohort_group, cohort_label, report_date, ts,
segment_id, net, long, short, gross, purity, open_interest, share, share_pctile,
oi_pctile, chg_2w, price, n_pool
```

Three datasets (`cftc_tff_fut`, `cftc_disagg_fut`, `cftc_legacy_fut`), 578
(dataset, market_code) pairs over 289 distinct codes, 11 cohort groups. The
futures-and-options families and the supplemental are stored in the archive but not
served: see [data-notes](data-notes.md) on why they are not interchangeable.

### `board_flows` — 92,473 rows

```
ticker, ticker_label, index_family, multiple, trade_date, ts, nav, nav_change_pct,
leverage_kind, assets, shares_outstanding, flow_usd, flow_cum_usd, rebalance_1pct_usd
```

23 funds across 6 index families (BTC, GOLD, NDX, OIL, SOX, SPX), leveraged and
unleveraged in one table with `leverage_kind` to separate them.

### `board_inflation` — 586 rows

```
survey_date, ts, sentiment, current_conditions, expectations, infl_exp_1y,
infl_exp_5y10y   (each with _pool and _pctile)
```

## The price series are proxies

`prices` is keyed on a symbol, not on a CFTC contract, and **every entry is a proxy** —
nothing in the map is the reported contract itself. `lib/pricemap.py` names the gap for
each one in its `proxy` field; read it there rather than from a table here, because the
module is the single definition and a copy would drift.

The worst case is VIX: positioning is in VIX *futures* while the price is *spot* VIX, and
the two can move in opposite directions.

Prices are joined **as of** each report date — the last close at or before the Tuesday.
A `"nearest"` or forward join would put a price that did not exist yet beside a position,
which is look-ahead entering through a join rather than through a statistic.

## A note on reading the archive directly

The archive tables are keyed on their natural key with `PRIMARY KEY … WITHOUT ROWID`, so
a `SELECT` on the key prefix is already an index scan and no secondary indexes exist.
For CFTC that key leads with the report date, which means a date-range query is fast and
a `WHERE market_code = …` without a date is a full scan of that table. Add the date bound
if you have one.

Stored column types come from the first frame ever written to each table and are frozen
by `CREATE TABLE IF NOT EXISTS`, so a column's SQLite type is not always what today's
parser would produce. `make status` reports any declared-numeric column stored as text —
if it reports one, that column sorts lexically and should be rebuilt with
`make backfill` before you rank on it.
