# ◧ watchboard

Government and alternative data that TradingView either doesn't carry or doesn't
display usefully. Built because the numbers were always free — the failure was
**cadence**: two CFTC charts arrived via a social-media post on 2026-08-31 and
both reproduced exactly from a public JSON endpoint with no key.

The rule this repo encodes: **a number you can't reproduce yourself can raise a
question, but it shouldn't answer one.** Anything reproducible from a primary
source gets pulled here; anything genuinely proprietary (a bank's internal
vol-control-fund estimate, say) stays outside and stays discounted.

```
streamlit run app.py                      # the board
python -m scripts.ingest --backfill       # first run: full history, ~5 min, ~900 MB
python -m scripts.ingest                  # daily: incremental, ~1 MB
python -m scripts.verify_spec             # prove the field maps against the live API
python -m pytest -q                       # tests, no network
```

## What's in it

**All seven CFTC Commitments-of-Traders datasets**, full history, 4,087,353 rows,
115 MB committed. Not a curated handful of markets — the whole cross-section,
because the base-rate question needs breadth and it turns out to be affordable.

| source | coverage | cohorts | markets | rows | from |
|---|---|---|---|---|---|
| `cftc_tff_fut` / `_futopt` | financial futures | dealer, asset mgr, leveraged, other, non-rept | 146 | 231,805 / 232,065 | 2006-06-13 |
| `cftc_disagg_fut` / `_futopt` | physical commodities | producer, swap, managed money, other, non-rept | 652 | 921,145 / 951,510 | 2006-06-13 |
| `cftc_legacy_fut` / `_futopt` | everything | commercial, non-commercial, non-rept | 951 | 864,453 / 831,831 | **1986-01-15** |
| `cftc_supp_cit` | 13 ag markets | **index traders (CIT)**, comm ex-CIT, non-comm ex-CIT, non-rept | 13 | 54,544 | 2006-01-03 |
| `openrouter_pricing` | token prices | — | 419 models | snapshot-only | 2026-09-01 |

The three CFTC families are **not interchangeable**. `legacy` is the only history
before 2006 but its cohorts are coarse. `disagg` and `tff` partition the modern
universe between them — commodities and financials — and never overlap.
"Commercial" is not "producer + swap dealer", so splicing legacy onto disagg at
2006-06-13 produces a definitional break, not a longer series.

**One board.** Pick a market, tick which cohorts to group — dealers, asset
managers, hedge funds, other reportables, small traders — and read their combined
net position as a share of that market's open interest, against its own history.

Two readings reproduce the published figures to the digit, which is how you know
the pipeline is right rather than merely plausible:

| | reading | rank | 2-week |
|---|---|---|---|
| VIX · asset managers | −6.04% of OI | 4th pctile of 1,014 weeks | +1.05pp |
| NASDAQ-100 · asset mgr + hedge funds | +9.28% of OI | 40th of 846 | +20.58pp |

Export is the modebar's PNG at 3× (named after the selection, not `newplot.png`)
or **Copy image** for the clipboard. **Pull latest data** runs the ingest job on
demand — see the note under Architecture, because it is the one place this repo
bends its own rule.

Six further boards were built and then set aside as the wrong surface: a
cross-market screen, a per-market drill-down, week-over-week flows, crowding
(concentration and trader counts), forward base rates, and an integrity
reconciliation over all 4 million rows. They live in `.parked/`, which is
gitignored — untracked but still on disk, and still in git history at `42aa2dd`.

## Theme

Palette and type are read out of `ingtian.github.io`'s compiled CSS rather than
approximated: `#08090b` page, `#14171b` chart plane, `#dce1dc` ink, `#66c28c`
accent, Georgia for display, a system sans for body, mono for small labels. Both
of that site's modes are in `lib/theme.py` and switching is one constant.

Only dark is wired up, for a measured reason: the site's light accent `#c8a36a`
sits at **1.95:1** on its own cream paper. That is fine for the large serif
headings it was picked for and below the 3:1 a 2px data line needs. Light mode
would need its own darker step for the series colour before it could ship.

## Architecture — why it's split in two

GitHub Actions cannot serve a Streamlit app, and Streamlit cannot run unattended.
So they own different layers, and the split is better than either alone:

```
        ┌──────────────────────── ingest (unattended) ────────────────────────┐
        │  GitHub Actions, daily 20:15 UTC                                    │
        │    scripts/verify_spec.py   field maps vs the live API, fail early   │
        │    scripts/ingest.py  ->  sources/*.fetch(since=...)  ->  data/*.parquet │
        │    pytest, then commit data/ back to the repo                        │
        └────────────────────────────────┬────────────────────────────────────┘
                                         │  data/ is versioned, history accrues
        ┌────────────────────────────────▼──────────── display (interactive) ──┐
        │  app.py — Streamlit, reads data/ only, never fetches                 │
        │    panels/*.py   boards, auto-discovered                             │
        │    lib/*.py      metrics, segments, charts, store, universe          │
        └─────────────────────────────────────────────────────────────────────┘
```

The board **works offline** and cannot display a number that isn't in the
committed dataset. History accrues whether the laptop is awake or not — which
matters more than it looks, because one source has no backfill at all.

### The one exception: the pull button

The display layer is otherwise forbidden from fetching, so that a chart can never
show a number that isn't on disk. **Pull latest data** bends that, deliberately and
narrowly: it runs the *ingest job* — the same code path CI runs — which writes
parquet and then clears the read cache. It does not fetch into a chart. What is on
screen after it finishes is still exactly what is committed to disk.

It is incremental, asking the API for roughly the last eight weeks (~1 MB) rather
than the ~900 MB a full-history pull costs. Use `python -m scripts.ingest
--backfill` for that.

### Adding a source is one file

`sources/` and `panels/` are both discovered with `pkgutil`, so there is no
dispatch table to remember to update:

1. Write `sources/yourthing.py` exposing either `build_sources()` or
   `SOURCE_KWARGS`, with a `TableSchema` declaring the columns you promise.
2. Optionally write `panels/yourthing.py` exposing a `BOARD`.
3. `python -m scripts.ingest --source yourthing --backfill`.

That's it — two new files, zero edits elsewhere. A module that raises on import is
reported in the sidebar and fails the test suite rather than silently vanishing,
because a board that disappears looks like a design decision.

## Three corrections to what this repo previously claimed

Each of these was asserted confidently in an earlier version, with the right
numbers next to the wrong explanation — which is the worst failure mode available,
because nothing looks broken. All three were re-measured over the full
corpus: 1,046,369 market-weeks across the seven reports, which melt to 4,087,353
tidy rows. The identities are market-week quantities, so market-weeks is the
denominator every rate below is stated against — quoting the row count would
inflate each one by the cohort multiplier.

### 1. The open-interest residual is rounding noise, not an unpublished spread

The identity is `sum(long) == sum(short) == open_interest - sum(spread)` — a
spread position is long one expiry and short another, so it belongs to neither
side and gets its own column. That part was right. The leftover was explained as
"the unpublished non-reportable spread". It isn't:

- it is frequently **negative** — 822 neg / 312 pos in `legacy_fut`, 1,188 / 428
  in `tff_fut` (73% of its nonzero market-weeks), roughly even in the combined
  reports. A missing non-negative spread could only push it *positive*.
- it never leaves `-4..+4`, on markets whose open interest reaches **35,814,710**
  contracts. A real position bucket would scale with the market.
- `|corr(residual, open interest)| < 0.03` — none.
- it is **exactly zero on all 184,229 `disagg_fut` rows**, the dataset where
  small-trader calendar spreads would be most visible.

It's integer rounding in the publisher, which rounds each column independently.

### 2. `sum(net) == 0` is not exact, and testing it exactly is the bug

Every contract has two sides, so cohort nets must sum to zero — but to within
rounding, not exactly. An exact test fails on **2,292 of 518,741** futures-only
market-weeks and on **51%** of supplemental market-weeks. The previous version showed a
red "BROKEN — investigate before using this week" banner at any nonzero value,
which would fire constantly. Tolerance is now 4 contracts (corpus max is 3).

### 3. CR4/CR8 are shares of the side total, not of open interest

The concentration columns divide by open interest **minus spreads**. Two
independent disproofs:

- 158 market-weeks report `conc_gross_le_8_tdr_long == 100.0` exactly, and all 158
  have `spread > 0`. Against an open-interest denominator the long side caps at
  `100*(OI-spread)/OI`, which on those rows falls to **25.2%** — so 100.0 is
  arithmetically impossible.
- `CR4/100 * OI` exceeds the entire long side in **1,623 of 46,361** market-weeks.
  Under the side-total reading: zero violations.

Reading it as a share of open interest overstates by `1/(1-spread_share)` — 1.8x
in 3-month SOFR, which is 44.5% spreads.

## The traps that shape the code

**Markets are keyed on `cftc_contract_market_code`, never on name.** 26–30% of
codes have been renamed at least once and CFTC shortened names wholesale on
2022-02-08 (`U.S. TREASURY BONDS` → `UST BOND`, `E-MINI S&P 500 STOCK INDEX` →
`E-MINI S&P 500`). `(name, date)` isn't even unique — 15 collisions in
`legacy_fut` from truncated historical names, where a dedupe on name would
silently discard one of two real markets. `(code, date)` has zero duplicate groups
in all seven datasets.

**A code's history is not one comparable series.** On 2023-05-02 CFTC re-based its
Consolidated equity indices from the big contract to the E-mini: `20974+` open
interest jumped 49,531 → 255,954 (×5.17) as `contract_units` went
`(NASDAQ 100 INDEX X $100)` → `($20)`. That's a unit change, not a positioning
change — **and it was the previous version's default market**, so its headline
percentile was ranking $20-per-point observations against $100-per-point ones.
Code `191691` is worse: aluminium in 40,000-pound contracts 1986–1989, a
12,341-day hole, then the same code in 25-metric-ton contracts from 2022.
`lib/segments.py` cuts history at unit changes and at gaps over a quarter, and
percentiles are computed within a segment. Counted by that module's own rule over
the committed data, unit breaks affect 96 of 951 `legacy_fut` codes, 56 of 652
`disagg_fut`, 13 of 146 `tff_fut` and none of the 13 supplemental markets.

**Consolidated = E-mini + Micro/10**, verified against the published figures to
under one contract (S&P 500: 2,074,931 vs 2,074,931.4). So never sum an E-mini and
a Micro by contract count — the previous version showed NDX E-mini and NDX Micro as
two independent markets of comparable size when one is a tenth of the other per
contract. But this is a **trade-off, not a free fix**: the Consolidated codes start
2010-06-15 with 846 reports and carry the 2023-05-02 seam (174 comparable weeks),
while `13874A`/`209742` have 1,055 reports from 2006-06-13 and no break at all. So
the notionally-correct series excludes 2007–2009 and the notionally-wrong one has
twenty clean years. The board surfaces the choice instead of making it silently.

**Trader counts are often null while the position is nonzero, and it's getting
worse** — 4.2% of 2015 rows, 12.6% of 2018, 22.0% of 2025, **28.7% of 2026**, and
30 of the 94 markets on the latest report. So average-position-per-trader is
unavailable for about a third of the live cross-section, and the board shows that
coverage rather than quietly dropping the rows. `traders_*` doesn't exist at all
for the non-reportable cohort.

**The API misspells its own columns, inconsistently.** `noncomm_postions_spread_all`
(missing an `i`), `swap_positions_long_all` with one underscore but
`swap__positions_short_all` with two, MixedCase keys in the supplemental dataset
whose Socrata *metadata* is lowercase. Worse than a typo: in `disagg`,
`prod_merc` and `other_rept` have no `_all` variant, so the bare name is the
all-maturity column and `_1`/`_2` are crop-year buckets — reaching for
`prod_merc_positions_long_1` silently returns **old crop only**, type-checks, and
breaks nothing except the arithmetic. Every name lives in `lib/cftc_spec.py` and
`scripts/verify_spec.py` proves them against the live API.

**Socrata omits null keys entirely.** Per-row key counts in `tff_fut` range 65–90
against a 90-key union, so a missing key means null, not schema drift, and
`len(row)` says nothing about the schema. Three market codes end in `+`
(`12460+`, `13874+`, `20974+`), and a raw `+` in a query string decodes to a
space — an f-string `$where` silently matches nothing.

**Legacy isn't weekly before 2002-01-08**: 646 reports on mixed weekdays with 161
gaps of 11–18 days. Even after it, gaps of 3, 4, 6 and 8 days occur. So a one-row
diff is not a one-week change; `metrics.flow` computes the calendar span and
returns NaN rather than mislabelling the horizon.

## Storage: why one file per source, sorted by date

Parquet is a binary blob and git has no format-aware delta for it, so the cost of
committing data daily is decided entirely by how many *bytes move* when a week is
appended. Two choices do all the work, both measured with real git on the real
231,805-row `tff_fut` history over six sequential weekly commits:

| ordering | row group size | bytes/commit | % of file | MB/yr |
|---|---|---|---|---|
| `(market_code, report_date)` | default | 5,131,605 | 86.9% | 1,873 |
| `(report_date, market_code)` | default | 516,779 | 8.5% | 189 |
| `(report_date, market_code)` | 20,000 | 232,448 | 3.4% | 85 |
| **`(report_date, market_code)`** | **5,000** | **76,459** | **1.0%** | **28** |

Sorting by code first inserts a row into the middle of every market's run;
sorting by date first makes a new week a pure tail append. Row groups then bound
how much a tail append disturbs — with one row group per file, appending shifts
every column chunk's offset, so most of the file moves even though the data went
on the end. Together: **67× less churn** than the naive default, for a 22% larger
file, which dictionary-encoding the repeated market-week columns wins most of
back (−20%, 143 MB → 115 MB). `Source.__post_init__` asserts the sort key leads
with a date column, so this can't be silently undone.

Writes are byte-deterministic, so a day with no new report produces an identical
file and an empty diff — which is why this commits ~52 times a year rather than
365, and why nothing timestamped is written into `data/`. Verified rather than
assumed: two consecutive `scripts.ingest` runs report `file unchanged` for all
eight sources and leave all eight sha256 digests identical.

The determinism is only *within* a fixed write path. Changing anything that
affects the bytes — the pyarrow version, the sort key, `row_group_size`, the dtype
or dictionary-encoding choices in `lib/store._tighten` — rewrites every file
identically-in-content and costs one full-size no-op commit. That is a deliberate
act, not a routine bump, which is why pyarrow is pinned to the version that wrote
the committed files (`25.*`, visible in their parquet metadata as
`parquet-cpp-arrow version 25.0.1`).

Year-partitioning was considered and rejected: ~20×, but it needs ~170 files and
only works if every partition's dictionary is built from that partition alone,
since a global dictionary rewrites every cold file the moment a new market code
appears anywhere.

## What this board cannot tell you

**There is no price series here, so forward returns cannot be computed.** Every
measure is positioning-only. The base-rates board conditions forward *positioning*
on today's percentile — a real question, since it tests whether "extreme" is
mean-reverting or persistent — but it is not a claim about returns and doesn't
pretend to be. Extending to returns needs one thing: a futures settlement series
keyed to `cftc_contract_market_code`. That's the highest-value planned addition in
`sources/__init__.py`.

**Positioning is contemporaneous with price, not predictive.** Report the level; do
not infer a direction without a base-rate test.

**A cohort's net is a sum over firms running incompatible strategies.** A basis
trader long the cash index and short the future appears here as short while holding
no view at all. Net is not a stance.

**Futures-only and futures-and-options-combined are different datasets**, not two
views of one. Their figures will not reconcile, and `supp_cit` is combined-basis
only with no futures-only counterpart at all.

**`openrouter_pricing` is snapshot-only.** The endpoint returns today's prices with
no archive, so the series starts the day the job starts and a missed run is a
permanent hole. It also routes for independent developers, not enterprise or
first-party traffic — good for direction and mix, never for level. And it measures
one factor of a product: inference revenue is tokens × price, so a falling line is
not by itself bearish.

## Display rules, enforced in code

Each of these came from a real error, so none of them is a style preference.

1. **No dual axes.** Two series on one panel with two y-scales lets the author pick
   the scaling that makes a correlation look how they want, and the reader can't
   see a choice was made. `lib/charts.stacked()` builds independent subplots
   sharing only the x-axis and has no parameter that produces a secondary axis.
2. **Causal rankings only.** Every percentile ranks an observation against data at
   or before its own date. `tests/test_metrics.py` proves it by truncation:
   computing on a prefix must be bit-identical to computing on the whole series and
   slicing — and asserts a deliberately non-causal implementation *fails* the same
   check, because a causality test that can't fail isn't a test.
3. **Signed quantities keep their own units.** A net going −100,640 → −96,727 is
   "+3,913 contracts, less short", not "+3.89%". The gate is *semantic* — it asks
   what the quantity is, not whether this sample happens to have crossed zero. An
   earlier empirical version returned "safe" for 78 of 550 cohort-net series
   (14.2%), including the board's own default market, and for an all-NaN series.
4. **Open interest is always on screen beside positioning.** A cohort's share can
   rise because the cohort bought or because open interest fell, and the share
   alone can't distinguish them. `stacked()` raises if a figure plots a share of
   open interest without open interest.

Rules 1 and 3 are ultimately enforced by `tests/test_display_rules.py`, which
parses every board's AST for direct plotly imports, `secondary_y`,
`make_subplots`, `rank(pct=True)` and centred windows — because a panel can always
reach past a helper, so the test is the mechanism and the library is the
convenience. It also reads rule 1 back off the *rendered* figure JSON, which
catches the case an AST walk can't: post-processing a figure `stacked()` already
returned.

## Tests

`pytest -q` — ~350 tests, no network, about 2 seconds, and no dependence on the
115 MB `data/` directory: the suite reads the 596 KB committed slice in
`tests/fixtures/`, regenerated by `python -m tests.fixtures.build`. Every code in
that slice is there to make one awkward case true (the 2023-05-02 re-basing, a
market-week whose nets don't sum to zero, a null trader count beside a nonzero
position, two markets sharing one name, pre-2002 non-weekly cadence).

Two properties are worth knowing about, because they are what stops the suite
rotting into decoration:

- The causality check asserts that **non-causal implementations fail it**. A
  full-sample `rank(pct=True)` and a centred rolling mean are run through the same
  truncation test and asserted to raise. It has already earned this: it caught a
  live look-ahead bug in the min-observation *gate*, where the publish/blank
  threshold was derived from the whole sample's modal window occupancy, so whether
  observation *t* published depended on data arriving after it. The values were
  causal; the decision to show them wasn't.
- The board smoke tests assert **something was actually drawn**. `app.py` renders
  the title before it checks for data, so "no exception and the right title" is
  also exactly what a board that never ran looks like.

CI runs the suite on every push (`.github/workflows/tests.yml`), separately from
ingest, plus `scripts/verify_spec.py` against the live API — the check that catches
a CFTC column which still exists but now means something else.
