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
- **`open_interest` is a MARKET figure, not a cohort one.** CFTC publishes one number
  per market-week and `scripts/build` repeats it verbatim on every cohort row, because
  it is the denominator rather than a property of the cohort. A cohort has no open
  interest of its own.
- **`gross` is `long + short`, and it is NOT the cohort's share of the market.** It is
  book size — how much there is to unwind. Summed across the cohorts of one
  market-week it comes to roughly **1.8× open interest**, because every contract is
  counted twice (once on its long side, once on its short) while spreads are excluded
  from both. The identity that does hold is
  `sum(long) == sum(short) == open_interest − sum(spread)`, a spread position being
  long one expiry and short another and so belonging to neither side. Measured on
  E-mini S&P 500's latest week: both sides 1,750,380, open interest 1,986,532, and the
  236,152 difference equals the archive's `sum(spread)` exactly.
- **`share` is a percentage of open interest**, 0–100, not a fraction — and the
  denominator is the FULL published open interest, spreads included. Note that this
  differs from the concentration columns below, which are shares of the side total
  (open interest *minus* spreads). Both conventions are defensible and they are not
  interchangeable: on E-mini S&P the same net reads 46.74% of open interest or 53.05%
  of the side total, and on 3-month SOFR, which is 44.5% spreads, the two differ by
  1.8×. `lib/metrics.directional_oi` computes the ex-spread denominator; `build` does
  not currently use it.
- **`purity` is `net / gross`**, in [−1, 1], and it CHANGES SIGN. Near 0 means a large
  two-sided book with little net view; ±1 means every contract points the same way.
- **`share_pctile` ranks the SHARE against its own history, not against other
  cohorts.** It is a causal expanding percentile over the same
  (dataset, market_code, cohort_group): week *t* is ranked only against weeks at or
  before *t*. Because `share` is signed, so is the reading — near 0 is unusually
  *short* for this cohort, near 100 unusually long, about 50 typical.
- **`oi_pctile` is a different animal: it ranks the MARKET's open interest**, so it is
  identical on every cohort row of a market-week, exactly as `open_interest` is.
- **The two are segmented DIFFERENTLY, and this is deliberate.** `oi_pctile` restarts
  at each `segment_id`; `share_pctile` runs straight through. A level in contracts is
  not comparable across a contract-size change, but a ratio is — the re-denomination
  cancels. NASDAQ-100 Consolidated's 2023-05-02 re-basing shows it:

  ```
  2023-04-25  seg=0  OI=   49,531  net=  8,230  share=16.62  share_pctile=52.4  oi_pctile= 32.6
  2023-05-02  seg=1  OI=  255,954  net= 43,252  share=16.90  share_pctile=53.2  oi_pctile=100.0
  ```

  Open interest jumps 5.17× and the net 5.25×, so `share` moves 0.3 points and its
  percentile 0.8. `oi_pctile` reads 100.0 for every week after the seam because in the
  new segment every value is a new maximum — that is correct, not a bug, and it is why
  you should not read `oi_pctile` across a seam as a trend.
- **`segment_id` is the comparability unit for LEVELS.** It is cut at contract-unit
  changes and at gaps over a quarter. Comparing an `oi_pctile` across two
  `segment_id`s is comparing two different contracts; comparing a `share_pctile` is
  fine, which is the whole reason the ratio is the headline measure.
- **A share percentile is comparable within one market and cohort, never across
  markets.** The net is bounded by the directional side (open interest *minus*
  spreads) while the denominator is the full open interest, so `|share|` cannot reach
  100% and its ceiling is market-specific: 88.1% on E-mini S&P 500 against 59.2% on
  3-month SOFR, which carries 41% spreads. Same percentile, different achievable
  range.
- **`n_pool` is the LENGTH OF THE HISTORY, not what this row was ranked against.**
  It is `share.notna().sum()` for the whole (market, cohort) — one scalar repeated on
  every row. Because the ranks are causal, week 5 really was ranked against 5
  observations while `n_pool` reads 848; the column answers "is there enough history
  here for a percentile to mean anything", which is what you want for a filter
  (`n_pool >= 200`) and not what you want for reading an early row. A percentile with
  `n_pool = 12` is not a percentile. The boards filter on it; a hand-written query
  must too.
- **`board_inflation`'s `*_pool` columns are COUNTS too, one per series**, because each
  series ranks within its own regime rather than over its whole history. Today:
  `sentiment_pool = 28`, `infl_exp_1y_pool = 536`, `infl_exp_5y10y_pool = 439` out of
  586 rows. The sentiment pool is 28 because that series ranks only within the
  post-July-2024 web era, and 28 months is below the 36-month floor — so
  `sentiment_pctile` is **null on all 586 rows**, deliberately. A blank rank there is
  the honest answer, not missing data; see
  [data-notes](data-notes.md) for the −6.6-point collection-mode break that forces it.
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
