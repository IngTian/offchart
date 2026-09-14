# offchart

**The data TradingView doesn't carry, in a database Grafana can read.**

Positioning, survey expectations and other published-but-badly-displayed series are
free and reproducible from primary sources. What's missing isn't the data, it's the
plumbing: a place they all land with the same shape, the same cadence handling, and the
same statistics applied honestly. That's all this repo is.

It owns **data and preprocessing**. It owns no charting code. Rendering is Grafana,
provisioned from two committed JSON files — because a hand-maintained chart layer turned
out to cost more than the data ever did, and Grafana already does crosshairs, zoom,
fullscreen and export better than a bespoke one will.

```
make venv        # once: .venv with the two dependencies, built by uv
make backfill    # once: full history into data/offchart.sqlite (~5 min, ~900 MB fetched)
make grafana     # start Grafana -> http://localhost:3000
make pull        # from then on, the only command you need
```

`make pull` is deliberately one command for a two-step job — fetch, then rebuild the
derived tables. Forgetting the second step leaves the charts showing last week's numbers
with no error anywhere, so the target that can't be half-run is the point.

---

## What's in it

4.6 million rows across ten sources, all fetched keyless from public endpoints.

| source | what it is | rows | markets | from |
|---|---|---:|---:|---|
| `cftc_tff_fut` / `_futopt` | CFTC financial futures — dealer, asset mgr, leveraged, other, non-rept | 232,755 / 233,015 | 147 | 2006-06-13 |
| `cftc_disagg_fut` / `_futopt` | CFTC commodities — producer, swap, managed money, other, non-rept | 923,955 / 954,390 | 655 / 685 | 2006-06-13 |
| `cftc_legacy_fut` / `_futopt` | CFTC everything — commercial, non-commercial, non-rept | 866,709 / 834,129 | 955 / 944 | **1986-01-15** |
| `cftc_supp_cit` | CFTC supplemental — **index traders (CIT)** in 13 ag markets | 54,648 | 13 | 2006-01-03 |
| `prices` | daily closes for the reported markets | 538,147 | 87 symbols | 1990-01-01 |
| `umich_sca` | UMich Surveys of Consumers — sentiment and inflation expectations | 681 | — | 1951-02-01 |
| `openrouter_pricing` | LLM token prices, **snapshot-only** | 5,617 | 456 models | 2026-09-01 |

Not a curated handful of markets — the whole CFTC cross-section, because the base-rate
question needs breadth and it turns out to be affordable.

## The two boards

**Positioning** — pick a report family, a market and a cohort group; read that group's
net position as a share of open interest against its own history, with the underlying
price above and open interest below. 1,344,182 precomputed rows covering 560 markets
across the three families.

**Inflation expectations (UMich)** — median expected price change over the next year and
the next 5–10 years, each ranked within its own regime, plus the sentiment index and its
two components.

They are **separate dashboards on purpose.** A monthly household survey and weekly
futures positioning share no market, no cadence and no denominator, and putting them on
one canvas invites reading a relationship that nothing here tests.

---

# User guide

## 1. Install

Needs [uv](https://docs.astral.sh/uv/) and Docker (for Grafana). Two Python
dependencies: pandas, and pytest for the suite. Everything else — SQLite, HTTP — is
standard library.

```
git clone https://github.com/IngTian/offchart.git
cd offchart

make venv        # .venv, built by uv. No activation needed afterwards.
```

Python itself is not a prerequisite: `make venv` runs `uv venv --python 3.12`, which
fetches a standalone interpreter when the system has none. Nothing here has a compiled
extension or a C library, so both dependencies are pure wheels from `requirements.txt`,
and there is no lock file for a dependency set of two.

Every target runs through `$(PYTHON)`, resolved in this order: an **activated**
environment (`$VIRTUAL_ENV`), then a `.venv/` directory in the
tree, then whatever `python3` is on `PATH`. Activation beats a directory on purpose — a
leftover `.venv/` is not a statement of intent, and having it win is how a suite ends up
passing against an environment you thought you'd left behind. `make which-python` prints
the choice and the reason for it, which is the first thing to check when the suite
behaves differently in two shells. Override any time with
`make test PYTHON=/path/to/python`.

## 2. Get the data

```
make backfill
```

Full history for all ten sources: about five minutes, roughly 900 MB fetched, landing in
`data/offchart.sqlite` (~1.7 GB — SQLite trades size for query speed, and the file is
disposable). It's incremental after this.

`make status` says what you have and how stale it is:

```
data/offchart.sqlite  1,664 MB

  source                        rows  latest          age  cadence
  cftc_tff_fut               232,755  2026-09-08       5d  Positions as of Tuesday, published Friday 1…
  umich_sca                      681  2026-08-01      43d  Monthly, 10:00 ET
  …
  board_positioning        1,344,182  2026-09-08       5d  serving

  committed snapshots (the history no API can return):
  openrouter_pricing       13 dates  2026-09-01 .. 2026-09-13
```

Freshness is reported because it's the failure this repo is most likely to suffer: a
chart drawn from a database that stopped updating three weeks ago looks exactly like a
chart drawn from a fresh one.

## 3. Start Grafana

```
make grafana        # then open http://localhost:3000
```

That runs `grafana/docker-compose.yml`, which is stateless by design — no volume for
Grafana's own database, because everything defining the boards is committed here and
reprovisioned on every start. Delete the container and nothing is lost.

The container installs the SQLite plugin itself via `GF_PLUGINS_PREINSTALL`, so the first
start takes ~20 seconds. Anonymous access is on and there is no login prompt; that's fine
for one reader on a laptop and should be turned off before the port is exposed anywhere.

Boards appear under the **Offchart** folder.

### Installing the plugin into a Grafana you already run

If you'd rather not use the container, three steps:

```
grafana-cli plugins install frser-sqlite-datasource     # or via Administration > Plugins
systemctl restart grafana-server                       # brew services restart grafana, etc.
```

Then point Grafana at this repo's committed provisioning. Either copy the two directories
into your Grafana provisioning path:

```
cp grafana/provisioning/datasources/sqlite.yaml   /etc/grafana/provisioning/datasources/
cp grafana/provisioning/dashboards/offchart.yaml  /etc/grafana/provisioning/dashboards/
```

…or add the datasource by hand: type **SQLite**, path = the absolute path to
`data/offchart.sqlite`, and import the two files in `grafana/dashboards/`.

**Two edits are required if you go native**, because the committed files use container
paths: `path:` in `sqlite.yaml` and `options.path` in `offchart.yaml` both say `/repo/…`
and must become real paths on your machine.

Notes on the plugin, since it is the one third-party piece here:

- `frser-sqlite-datasource` is a **signed community plugin**, one maintainer, catalog
  badge "Needs attention". It's pure Go via `modernc.org/sqlite`, so it needs no cgo and
  runs on the default Alpine Grafana image without `allow_loading_unsigned_plugins`.
- It reads a time column as **epoch seconds**, so every panel query divides: `ts/1000 AS
  time`. Passing the stored millisecond value renders 1953-11-19 instead of 2026-09-08 —
  a wrong chart that looks like a real one. A test asserts the division.
- It implements one macro, `$__unixEpochGroupSeconds`. No `$__timeFilter`, so panels
  write their own bound: `ts BETWEEN $__from AND $__to`, with `ts` stored in milliseconds
  because that's what Grafana's `$__from`/`$__to` are.
- If it ever stops being viable, the schema is ordinary SQL and moving these tables into
  Postgres is an afternoon.

## 4. Keep it current

```
make pull
```

CFTC publishes Friday 15:30 ET for positions as of Tuesday. The pull is incremental —
it asks for reports on or after `latest stored − 56 days`, about 1 MB, and the window
**overlaps** what you already have on purpose, because CFTC revises already-published
weeks and the upsert is keyed so a revision overwrites rather than duplicates.

Use `make backfill` after changing a field map or widening the market universe.

## 5. The rest of the commands

```
make status          what is in the database and how stale it is
make build           rebuild the derived tables only, no network
make which-python    which interpreter make will use, and why
make test            the suite: 357 tests, no network, ~1s
make verify          prove the CFTC field maps against the live API
make grafana-stop    stop Grafana
make grafana-logs    follow its logs
make fixtures        regenerate the committed test slice from your store
make compact         VACUUM
make reset-db        delete the database (needs CONFIRM=1; recovery is one backfill)
```

There is also a bare `make ingest`, which is the fetch half of `make pull` without the
rebuild. It is useful for debugging one source and it is not in `make help`, because
running it alone leaves Grafana showing the previous week's derived numbers — so it
prints a reminder to run `make build` rather than letting you discover that from a chart.

## 6. Querying it yourself

The boards are a convenience; the database is the product.

```sql
-- who is most extreme this week, across every served market
SELECT market_code, cohort_group, round(share, 2) AS pct_oi, round(share_pctile) AS pctile
FROM board_positioning
WHERE dataset = 'cftc_tff_fut'
  AND ts = (SELECT max(ts) FROM board_positioning WHERE dataset = 'cftc_tff_fut')
  AND n_pool >= 200
ORDER BY share_pctile DESC
LIMIT 10;
```

Two table layers, and the split matters:

- **archive** — one table per source, faithful to the row, exactly what the upstream
  published. `cftc_tff_fut`, `prices`, `umich_sca`, …
- **serving** — `board_positioning` and `board_inflation`, wide, one row per grid
  timestamp with every derived quantity already computed.

Read the archive for research. Read serving for charts. **Don't recompute a percentile in
SQL** — see below for why.

---

## Why the percentiles are computed in pandas and not in SQL

This is the correctness core, so the reasoning is written down rather than implied.

1. **The segment boundaries cannot be expressed in SQLite at all.** Detecting a contract
   re-specification means pulling digits out of free-text contract units
   (`(NASDAQ 100 INDEX X $100)` → `100`). SQLite has no REGEXP without a loadable
   extension, and the plugin won't load one. An open-interest rank must be computed
   *within* a segment, so the rank follows the segmentation into Python.
2. **The proof of causality lives with the pandas implementation.** `tests/test_metrics.py`
   proves `expanding_percentile` is causal by truncation — computing on a prefix must be
   bit-identical to computing on the whole series and slicing — *and* asserts that
   deliberately non-causal implementations fail the same check. Move the statistic into
   SQL and coverage of this repo's single most important property drops to zero.
3. **A revision makes incremental materialisation wrong, not merely slow.** A CFTC
   revision at week *t* changes every expanding rank from *t* forward. So `make build`
   rebuilds everything, every time, in about 30 seconds. At that price no invalidation
   logic is needed and a stale percentile — a plausible wrong number — cannot exist.
4. **A time filter cannot be pushed into a causal rank.** The rank at *t* depends on all
   history up to *t*, so an in-SQL rank would scan a market's whole history on every panel
   load regardless of the visible window.

`PERCENT_RANK() OVER (ORDER BY share)` is textbook look-ahead and reads as perfectly
reasonable SQL. `tests/test_grafana.py` fails the build if any dashboard query contains
it, or `CUME_DIST`, or `NTILE`, or reads an archive table.

---

## Licensing, and what is committed

Nine of the ten sources are US federal work or openly published APIs. One is not, and the
distinction is enforced by tests rather than by a comment.

**The database is never committed.** Two independent reasons: it contains `umich_sca`, and
it doesn't need to be. The University of Michigan grants permission-free *use* of its
public Surveys of Consumers tables and separately prohibits *redistribution* without
written consent (`faq.php` vs `agreement.php` — both quoted in `sources/umich.py`).
Charting is fine; mirroring is not. And nine sources are backfillable, so losing the file
costs one `make backfill`. Even the test fixture's UMich slice is **fabricated**, not
sampled, for exactly this reason.

**`data/snapshots/` is committed, and must stay that way.** `openrouter_pricing` serves
"today" with no archive, so a day nobody ran the job is a permanent hole. Its history
lives in git as one immutable CSV per snapshot date — 13 dates, 595 KB — written *before*
the database is touched, so the irreplaceable bytes never depend on a database write
succeeding, a migration completing, or this machine still existing tomorrow.

`tests/test_sources.py` asserts both directions by asking git's ignore rules directly:
a source we may not redistribute must have **no** committed artefact, and a source we
cannot re-fetch must have **one**, with no missing dates between its first and last.

The `snapshot` workflow runs daily and commits only `data/snapshots/`. It does *not* pull
CFTC — there's no committed state for CI to be incremental against, every CFTC source is
backfillable, and a runner that fetched 900 MB only to be destroyed would be theatre. Its
sibling job runs `scripts/verify_spec.py` against the live API, which is the monitor for
the failure mode with no symptom.

## Adding a source is one file

`sources/` is discovered with `pkgutil`, so there's no dispatch table to update:

1. Write `sources/yourthing.py` exposing either `build_sources()` or `SOURCE_KWARGS`, with
   a `TableSchema` declaring the columns you promise.
2. `python -m scripts.ingest --source yourthing --backfill`.

That's it. A module that raises on import is reported and **fails the suite** rather than
silently vanishing, because a source that disappears looks like a design decision.

Declare `backfillable=False` if the endpoint serves only "now" — that switches on the
snapshot machinery and the durability test. Declare `redistributable=False` if the terms
forbid mirroring, and supply `license`, `citation` and `caveats`; the dataclass refuses to
construct without them.

---

## Three corrections to what this repo previously claimed

Each was asserted confidently in an earlier version, with the right numbers next to the
wrong explanation — the worst failure mode available, because nothing looks broken. All
three were re-measured over 1,049,431 market-weeks, which melt to 4,099,601 tidy rows.
The identities are market-week quantities, so market-weeks is the denominator every rate
below is stated against; quoting the row count would inflate each by the cohort multiplier.

### 1. The open-interest residual is rounding noise, not an unpublished spread

The identity is `sum(long) == sum(short) == open_interest - sum(spread)` — a spread
position is long one expiry and short another, so it belongs to neither side. That part was
right. The leftover was explained as "the unpublished non-reportable spread". It isn't:

- it is frequently **negative** — 825 neg / 312 pos in `legacy_fut`, 1,192 / 428 in
  `tff_fut` (74% of its 1,620 nonzero market-weeks). A missing non-negative spread could
  only push it *positive*.
- it never leaves `-4..+3`, on markets whose open interest reaches **35,814,710**
  contracts. A real position bucket would scale with the market.
- `|corr(residual, open interest)| < 0.03` — none.
- it is **exactly zero on all 184,791 `disagg_fut` market-weeks**, the dataset where
  small-trader calendar spreads would be most visible.

It's integer rounding in the publisher, which rounds each column independently.

### 2. `sum(net) == 0` is not exact, and testing it exactly is the bug

Every contract has two sides, so cohort nets must sum to zero — but to within rounding. An
exact test fails on **2,298 of 520,245** futures-only market-weeks and on **51%** of
supplemental ones (6,970 of 13,662). The previous version showed a red "BROKEN — investigate before using
this week" banner at any nonzero value, which would fire constantly. Tolerance is 4
contracts; corpus max is 3.

### 3. CR4/CR8 are shares of the side total, not of open interest

The concentration columns divide by open interest **minus spreads**. Two independent
disproofs:

- 1,111 market-weeks report `conc_gross_8_long == 100.0` exactly, and **775 of them
  have `spread > 0`**. Against an open-interest denominator the long side caps at
  `100*(OI-spread)/OI`, which on those rows falls as low as **25.2%** — so 100.0 is
  arithmetically impossible there.
- `CR4/100 * OI` exceeds the entire long side in **14,868 of 1,035,745** market-weeks.
  Under the side-total reading: zero violations.

Reading it as a share of open interest overstates by `1/(1-spread_share)` — 1.8× in
3-month SOFR, which is 44.5% spreads.

## The traps that shape the code

**Markets are keyed on `cftc_contract_market_code`, never on name.** 26–30% of codes have
been renamed at least once and CFTC shortened names wholesale on 2022-02-08
(`U.S. TREASURY BONDS` → `UST BOND`). `(name, date)` isn't even unique — 15 collisions in
`legacy_fut` from truncated historical names, where a dedupe on name silently discards a
real market. `(code, date)` has zero duplicate groups in all seven datasets.

**A code's history is not one comparable series.** On 2023-05-02 CFTC re-based its
Consolidated equity indices from the big contract to the E-mini: `20974+` open interest
jumped 49,531 → 255,954 (×5.17) as `contract_units` went `(NASDAQ 100 INDEX X $100)` →
`($20)`. A unit change, not a positioning change — and it was an earlier version's default
market, so its headline percentile ranked $20-per-point observations against $100-per-point
ones. Code `191691` is worse: aluminium in 40,000-pound contracts 1986–1989, a 12,341-day
hole, then the same code in 25-metric-ton contracts from 2022. `lib/segments.py` cuts
history at unit changes and at gaps over a quarter, and percentiles are computed within a
segment. Unit breaks affect 96 of 955 `legacy_fut` codes, 56 of 655 `disagg_fut`, 13 of 147
`tff_fut`, none of the 13 supplemental.

**The three CFTC families are not interchangeable.** `legacy` is the only history before
2006 but its cohorts are coarse. `disagg` and `tff` partition the modern universe between
them — commodities and financials — and never overlap. "Commercial" is not "producer + swap
dealer", so splicing legacy onto disagg at 2006-06-13 produces a definitional break, not a
longer series.

**Consolidated = E-mini + Micro/10**, verified against the published figures to under one
contract (S&P 500: 2,074,931 vs 2,074,931.4). So never sum an E-mini and a Micro by
contract count. But it's a trade-off, not a free fix: the Consolidated codes start
2010-06-15 with 848 reports and carry the 2023-05-02 seam, while `13874A`/`209742` have
1,055 reports from 2006-06-13 and no break at all.

**Trader counts are often null while the position is nonzero.** Measured over the
3,050,170 reportable-cohort rows in all seven datasets: **18.0%** carry a null
`traders_long` beside a positive `long`. It is not a recent regression and it is not
getting worse — the rate rose from 14.6% in 2010 to about 20% by 2014 and has sat there
since (20.9% in 2025, 20.0% in 2026), and on the latest report it is 935 of 4,593 rows.
An earlier version of this file claimed a worsening trend from 4.2% to 28.7%; that does
not reproduce under any cohort definition and has been replaced with the measurement.

What matters is the storage rule, not the rate: a null count is stored as a null, never
as 0, because "nobody held the position" is a different and confident claim. `traders_*`
does not exist at all for the non-reportable cohort, whose rows are therefore excluded
from the figures above — including them adds a structural 100%-null block that flatters
nothing and explains nothing.

**The API misspells its own columns, inconsistently.** `noncomm_postions_spread_all`
(missing an `i`), `swap_positions_long_all` with one underscore but
`swap__positions_short_all` with two, MixedCase keys in the supplemental dataset whose
Socrata *metadata* is lowercase. Worse than a typo: in `disagg`, `prod_merc` and
`other_rept` have no `_all` variant, so the bare name is the all-maturity column and
`_1`/`_2` are crop-year buckets — reaching for `prod_merc_positions_long_1` silently
returns **old crop only**, type-checks, and breaks nothing except the arithmetic. Every
name lives in `lib/cftc_spec.py` and `make verify` proves them against the live API.

**Socrata omits null keys entirely.** Per-row key counts in `tff_fut` range 65–90 against a
90-key union, so a missing key means null, not schema drift. Three market codes end in `+`
(`12460+`, `13874+`, `20974+`), and a raw `+` in a query string decodes to a space — an
f-string `$where` silently matches nothing.

**Legacy isn't weekly before 2002-01-08**: 646 reports on mixed weekdays with 161 gaps of
11–18 days. Even after it, gaps of 3, 4, 6 and 8 days occur. So a one-row diff is not a
one-week change; `metrics.flow` computes the calendar span and returns NaN rather than
mislabelling the horizon.

**The UMich survey has a break its own chart doesn't show.** They moved from telephone to
web over April–June 2024 and measured the shift at −6.6 index points on sentiment, blending
the two instruments across three months so the level line has no visible step. Ranked
across that break, an ordinary web-era month looks near-record-low — measured on a
synthetic series with exactly that shift, the rank moves by more than 30 points. So
sentiment ranks only within the web era, and are **blank** while that era is shorter than 36
months. The inflation medians are *not* segmented there, because the effect UMich measured
is on the mean, which this repo doesn't ingest; their binding breaks are the 1982 wording
change and 1990-04. Every boundary in `lib/umich_spec.py` carries a citation URL, and a
test asserts it does.

## About the weaker sources, because they are weaker than the rest

`prices` is Yahoo's chart endpoint: no key, **not a documented public API**, and it can
change or start refusing without notice. That's a real step down from CFTC's Socrata
endpoint, which is a government publication with a stable schema. It's what's used because
the better keyless options don't work from here — FRED's `fredgraph.csv` times out and
Stooq returns a consent page instead of CSV. Two guards, both measured:

- **`range=max` makes the endpoint ignore `interval` and return monthly bars while still
  answering 200 OK.** `range=max&interval=1wk` on `^VIX` gives 440 points at
  `dataGranularity: "1mo"`, against 1,915 true weekly points for the same span requested
  with explicit period bounds. The fetch uses explicit bounds and **asserts** the
  granularity that came back.
- **Every series is a proxy**, never the exact contract CFTC reports on. `lib/pricemap.py`
  names the gap per entry. Worst case is VIX: positioning is in VIX *futures* while the
  price is *spot* VIX, and the two can move in opposite directions.

Prices are sampled **as of** each report date — the last close at or before the Tuesday,
via `metrics.asof`, which is `direction="backward"` and tested to be. `"nearest"` or a
forward fill would put a price that didn't exist yet beside a position: look-ahead entering
through a join rather than through a statistic, which is the variety that survives review.
Measured: 0 of 848 NASDAQ report dates get a price dated after the report; a forward join
leaks on 1 row per market.

`openrouter_pricing` routes for independent developers, not enterprise or first-party
traffic — good for direction and mix, never for level. And it measures one factor of a
product: inference revenue is tokens × price, so a falling line is not by itself bearish.

## What this cannot tell you

**Positioning is contemporaneous with price, not predictive.** Report the level; don't
infer a direction without a base-rate test. The board *shows* price beside positioning; it
does not test any relationship between them, and reading a turning point off two stacked
panels is eyeballing, not evidence — the eye is very good at finding leads and lags in
noise. What `prices` unlocks is a forward-return study conditioned on today's percentile,
with the overlap correction the weekly-observation/multi-week-horizon problem demands.
That's the next real piece of work, not something already done.

**Positions are as of Tuesday but published Friday 15:30 ET.** The price beside a position
is contemporaneous with the *position*, not with the moment you could first have seen it.

**A cohort's net is a sum over firms running incompatible strategies.** A basis trader long
the cash index and short the future appears here as short while holding no view at all. Net
is not a stance.

**Futures-only and futures-and-options-combined are different datasets**, not two views of
one. Their figures will not reconcile, and `supp_cit` is combined-basis only with no
futures-only counterpart.

**UMich inflation expectations are medians of "prices in general", not a CPI forecast.**
The mean is dragged by a fat right tail and is a different number; the question doesn't
mention any published index.

## Storage and durability

One SQLite file, one table per source, `STRICT` and `WITHOUT ROWID`, keyed on each source's
declared natural key.

`STRICT` is load-bearing rather than decoration. Open interest reaches 25,702,684 and
float32 is exact only to 2²⁴, so an earlier store spent forty lines deciding whether a
float column was integral in order to avoid narrowing a count. In a `STRICT` table that
reasoning becomes DDL: inserting 1.5 into an INTEGER column raises instead of silently
rounding. The check moves from "detect and hope" to "the engine refuses".

Writes are `ON CONFLICT DO UPDATE`, never `INSERT OR REPLACE` — REPLACE is a delete plus an
insert, so it would silently NULL every column an incoming partial frame didn't carry.
Change detection uses `IS NOT`, not `<>`, because a comparison against NULL yields NULL and
`<>` would report "unchanged" for all 81k null-bearing rows.

The build leaves the file in **rollback-journal** mode, not WAL. WAL is right for writing —
one writer, N readers — but with the file bind-mounted into the Grafana container on macOS,
three panels querying concurrently made one lose a lock race: `database is locked (5)
(SQLITE_BUSY)`, and Grafana drew an empty panel beside two full ones. So journal mode is a
phase, not a setting.

Serving tables are built into `<table>_new` and renamed inside one transaction, so Grafana
never reads a half-built table.

**Recovery.** `rm data/offchart.sqlite && make backfill` restores nine sources completely.
The tenth is why `data/snapshots/` exists.

## Tests

`make test` — 357 tests, no network, about a second, and no dependence on the 1.7 GB
database: the suite reads the 5 MB committed slice in `tests/fixtures/fixture.sqlite`,
regenerated by `make fixtures`. Every market code in that slice is there to make one
awkward case true — the 2023-05-02 re-basing, a market-week whose nets don't sum to zero, a
null trader count beside a nonzero position, two markets sharing one name, pre-2002
non-weekly cadence.

Three properties are what stop the suite rotting into decoration:

- **The causality check asserts that non-causal implementations fail it.** A full-sample
  `rank(pct=True)` and a centred rolling mean are run through the same truncation test and
  asserted to raise. It has earned this: it caught a live look-ahead bug in the
  min-observation *gate*, where the publish/blank threshold was derived from the whole
  sample's modal window occupancy — so whether observation *t* published depended on data
  arriving after it. The values were causal; the decision to show them wasn't.
- **`tests/test_grafana.py` lints the dashboards as the display layer's only guard.**
  Moving to Grafana cost this repo the ability to assert on a rendered figure, so what's
  left is that the JSON is authoritative — provisioned with `allowUiUpdates: false`, so
  Grafana refuses UI saves — and that these tests read it. Notably
  `test_every_series_declares_its_axis_side` is written as a *positive* assertion, because
  Grafana omits default-valued keys and `axisPlacement` defaults to `auto`, which means
  "first field left, everything else right". A lint looking for `"right"` passes on a
  dashboard that visibly renders a second y-axis.
- **The licence and durability rules are asked of git, not of a comment.** See above.

CI runs the suite on every push, plus `verify_spec` against the live API — the check that
catches a CFTC column which still exists but now means something else.

## Layout

```
lib/          db (SQLite store), snapshots (committed CSVs), metrics, segments,
              cftc_spec, cftc_api, universe, pricemap, cohort_groups, umich_spec, schema
sources/      one file per source, auto-discovered
scripts/      ingest, build (derived tables), status, verify_spec
grafana/      docker-compose.yml, provisioning/, dashboards/  — all committed
tests/        357 tests + fixture.sqlite
data/         offchart.sqlite (ignored) + snapshots/ (committed)
```

## Licence

Code: MIT, see [LICENSE](LICENSE).

**The data is not this repo's to license.** CFTC Commitments of Traders is US Government
work and in the public domain. Yahoo and OpenRouter data arrive under their own terms.
University of Michigan Surveys of Consumers data is used with permission for charting and
is **never redistributed here** — cite it as *"University of Michigan, Survey Research
Center, Surveys of Consumers."* If you publish anything derived from a source, check that
source's terms; `sources/*.py` records the ones this repo checked, verbatim, with URLs.
