# What the data does and does not say

Read this before trusting a number off a board. The code handles each of these; the
point of writing them down is that a reader who doesn't know them will interpret a
correct chart incorrectly.

The measurements below were taken over 1,061,707 CFTC market-weeks, which melt to
4,148,705 tidy rows (measured 2026-11-08; both grow weekly). The identities are
market-week quantities, so market-weeks is the denominator every rate is stated
against — quoting the row count would inflate each by the cohort multiplier. Where a
rate below was measured against an earlier, smaller corpus, treat it as an order of
magnitude and re-measure rather than quoting it.

## Column conventions you cannot guess

Four facts about the serving tables that a reader will otherwise misread:

- **`ts` is epoch MILLISECONDS.** Grafana's `$__from` / `$__to` are milliseconds, so
  that is what is stored. The SQLite plugin reads a *time* column as epoch **seconds**,
  which is why every panel projects `ts/1000 AS time`.
- **`chg_2w` is in CONTRACTS, not percent.** A net position going −100,640 →
  −96,727 is "+3,913 contracts, less short". Calling it "+3.89%" reads as growth while
  the position shrank toward zero, so no signed quantity here is ever expressed as a
  ratio. See `lib/metrics.py`'s module docstring, convention 2.
- **Concentration shares are of the SIDE TOTAL, not of open interest.** See below.
- **Two flows columns are COMPUTED, not published.** `nav_change_pct` is derived from
  NAV wherever the issuer publishes no usable change column, which is both plain-ETF
  sources; and SMH's `shares_outstanding` is derived as assets / NAV, because VanEck
  publishes it nowhere. Flow itself never uses the share count.

## Positioning is contemporaneous with price, not predictive

Report the level; don't infer a direction without a base-rate test. The board *shows*
price beside positioning; it does not test any relationship between them, and reading a
turning point off two stacked panels is eyeballing, not evidence — the eye is very good
at finding leads and lags in noise.

What `prices` unlocks is a forward-return study conditioned on today's percentile, with
the overlap correction the weekly-observation / multi-week-horizon problem demands.
That is the next real piece of work, not something already done.

**Positions are as of Tuesday but published Friday 15:30 ET.** The price beside a
position is contemporaneous with the *position*, not with the moment you could first
have seen it.

**A cohort's net is a sum over firms running incompatible strategies.** A basis trader
long the cash index and short the future appears here as short while holding no view at
all. Net is not a stance — which is why the board also carries long, short, gross and
`purity` (net / gross): a net near zero can mean a cohort holding nothing or a cohort
holding a large book on both sides, and those are different animals.

**Futures-only and futures-and-options-combined are different datasets**, not two views
of one. Their figures will not reconcile, and `supp_cit` is combined-basis only with no
futures-only counterpart.

## Markets are keyed on code, never on name

26–30% of codes have been renamed at least once, and CFTC shortened names wholesale on
2022-02-08 (`U.S. TREASURY BONDS` → `UST BOND`). `(name, date)` is not even unique — 15
collisions in `legacy_fut` from truncated historical names, where a dedupe on name
silently discards a real market. `(code, date)` has zero duplicate groups in all seven
datasets.

The board's dropdown shows a *label* containing both the name and the code, because the
Grafana variable fuzzy-matches the label only — so searching either works.

## A code's history is not one comparable series

On 2023-05-02 CFTC re-based its Consolidated equity indices from the big contract to the
E-mini: `20974+` open interest jumped 49,531 → 255,954 (×5.17) as `contract_units` went
`(NASDAQ 100 INDEX X $100)` → `($20)`. A unit change, not a positioning change — and it
was an earlier version's default market, so its headline percentile ranked
$20-per-point observations against $100-per-point ones.

Code `191691` is worse: aluminium in 40,000-pound contracts 1986–1989, a 12,341-day
hole, then the same code in 25-metric-ton contracts from 2022.

`lib/segments.py` cuts history at unit changes and at gaps over a quarter, and
percentiles are computed within a segment. Unit breaks affect 96 of 955 `legacy_fut`
codes, 56 of 655 `disagg_fut`, 13 of 147 `tff_fut`, and none of the 13 supplemental.

## The three CFTC families are not interchangeable

`legacy` is the only history before 2006 but its cohorts are coarse. `disagg` and `tff`
partition the modern universe between them — commodities and financials — and never
overlap. "Commercial" is not "producer + swap dealer", so splicing legacy onto disagg at
2006-06-13 produces a definitional break, not a longer series.

**Consolidated = E-mini + Micro/10**, verified against the published figures to under
one contract (S&P 500: 2,074,931 vs 2,074,931.4). So never sum an E-mini and a Micro by
contract count. But it is a trade-off, not a free fix: the Consolidated codes start
2010-06-15 with 848 reports and carry the 2023-05-02 seam, while `13874A` / `209742`
have 1,055 reports from 2006-06-13 and no break at all. Both are tiered onto the default
screen so you can hold both — prefer Consolidated to size exposure, prefer the
single-leg code to rank today against history that includes 2008.

## Trader counts are often null while the position is nonzero

Measured over the 3,050,170 reportable-cohort rows in all seven datasets: **18.0%**
carry a null `traders_long` beside a positive `long`. It is not a recent regression and
it is not getting worse — the rate rose from 14.6% in 2010 to about 20% by 2014 and has
sat there since (20.9% in 2025, 20.0% in 2026), and on the latest report it is 935 of
4,593 rows.

What matters is the storage rule, not the rate: **a null count is stored as a null,
never as 0**, because "nobody held the position" is a different and confident claim.
`traders_*` does not exist at all for the non-reportable cohort, whose rows are
therefore excluded from the figures above — including them adds a structural 100%-null
block that flatters nothing and explains nothing.

## Legacy is not weekly before 2002-01-08

646 reports on mixed weekdays with 161 gaps of 11–18 days. Even after it, gaps of 3, 4,
6 and 8 days occur. So a one-row diff is not a one-week change; `metrics.flow` computes
the calendar span and returns NaN rather than mislabelling the horizon.

## The UMich survey has a break its own chart does not show

They moved from telephone to web collection, running both instruments side by side over
April–June 2024 and web-only from July 2024, and measured the shift at **−6.6 index
points** on sentiment. Blending across three months means the level line has no visible
step.

Ranked across that break, an ordinary web-era month looks near-record-low — measured on
a synthetic series with exactly that shift, the rank moves by more than 30 points. So
sentiment ranks only within the web era, and is **blank** while that era is shorter than
36 months.

The inflation medians are *not* segmented there, because the effect UMich measured is on
the mean, which this repo does not ingest; their binding breaks are the 1982 wording
change and 1990-04. Every boundary in `lib/umich_spec.py` carries a citation URL, and a
test asserts it does.

**UMich inflation expectations are medians of "prices in general", not a CPI forecast.**
The mean is dragged by a fat right tail and is a different number; the question does not
mention any published index.

## The weaker sources, because they are weaker than the rest

### `prices` is Yahoo's chart endpoint

No key, **not a documented public API**, and it can change or start refusing without
notice. That is a real step down from CFTC's Socrata endpoint, which is a government
publication with a stable schema. It is what is used because the better keyless options
do not work from here — FRED's `fredgraph.csv` times out and Stooq returns a consent
page instead of CSV. Two guards, both measured:

- **`range=max` makes the endpoint ignore `interval` and return monthly bars while
  still answering 200 OK.** `range=max&interval=1wk` on `^VIX` gives 440 points at
  `dataGranularity: "1mo"`, against 1,915 true weekly points for the same span
  requested with explicit period bounds. The fetch uses explicit bounds and **asserts**
  the granularity that came back.
- **Every series is a proxy**, never the exact contract CFTC reports on.
  `lib/pricemap.py` names the gap per entry. Worst case is VIX: positioning is in VIX
  *futures* while the price is *spot* VIX, and the two can move in opposite directions.

Prices are sampled **as of** each report date — the last close at or before the Tuesday,
via `metrics.asof`, which is `direction="backward"` and tested to be. `"nearest"` or a
forward fill would put a price that did not exist yet beside a position: look-ahead
entering through a join rather than through a statistic, which is the variety that
survives review. Measured: 0 of 848 NASDAQ report dates get a price dated after the
report; a forward join leaks on 1 row per market.

### The ETF flow sources are futures-based for three families

The ProShares bitcoin, gold and oil funds hold **derivatives, not the commodity**, so
their NAV tracks a rolling futures position and diverges from spot over time. Flow is
still flow, but the NAV panel is not a spot price. GLD holds bullion and is spot-backed,
so it is comparable to them as a flow and not as a price.

**Rebalancing demand is mandate-implied, never an observed trade.** It is executed in
swap and futures notional, and because assets are struck at the close, a given day's
figure is what a move on the *next* day would force. A bull/bear pair's demand **adds**
rather than offsetting — both buy into a rally, because the inverse fund's short
exposure grows as the index falls — so never net TQQQ against SQQQ.

The Nasdaq-100 has **no unleveraged long fund** on the board. QQQ is Invesco, which
serves only the latest day, so NDX is the one index with leveraged funds and no
ordinary-money denominator.

### There is no AI-inference data here any more

The reason generalises. Inference revenue is tokens × price, and only the first factor's
owner can measure it — no credible free token-volume series exists — so a price line
alone answers nothing. The price side was tried and removed; see the rejection note in
`sources/__init__.py`.

## How these were measured

Three of the claims above replaced earlier ones that were confidently wrong, with the
right numbers next to the wrong explanation — the worst failure mode available, because
nothing looks broken. The disproofs are kept because they are what make the current
reading more than an assertion.

### The open-interest residual is rounding noise, not an unpublished spread

The identity is `sum(long) == sum(short) == open_interest - sum(spread)` — a spread
position is long one expiry and short another, so it belongs to neither side. That part
was right. The leftover was explained as "the unpublished non-reportable spread". It is
not:

- it is frequently **negative** — 825 neg / 312 pos in `legacy_fut`, 1,192 / 428 in
  `tff_fut` (74% of its 1,620 nonzero market-weeks). A missing non-negative spread could
  only push it *positive*.
- it never leaves `-4..+3`, on markets whose open interest reaches **35,814,710**
  contracts. A real position bucket would scale with the market.
- `|corr(residual, open interest)| < 0.03` — none.
- it is **exactly zero on all 184,791 `disagg_fut` market-weeks**, the dataset where
  small-trader calendar spreads would be most visible.

It is integer rounding in the publisher, which rounds each column independently.

### `sum(net) == 0` is not exact, and testing it exactly is the bug

Every contract has two sides, so cohort nets must sum to zero — but to within rounding.
An exact test fails on **2,298 of 520,245** futures-only market-weeks and on **51%** of
supplemental ones (6,970 of 13,662). An earlier version showed a red "BROKEN —
investigate before using this week" banner at any nonzero value, which would fire
constantly. Tolerance is 4 contracts; corpus max is 3.

### CR4/CR8 are shares of the side total, not of open interest

The concentration columns divide by open interest **minus spreads**. Two independent
disproofs:

- 1,111 market-weeks report `conc_gross_8_long == 100.0` exactly, and **775 of them have
  `spread > 0`**. Against an open-interest denominator the long side caps at
  `100*(OI-spread)/OI`, which on those rows falls as low as **25.2%** — so 100.0 is
  arithmetically impossible there.
- `CR4/100 * OI` exceeds the entire long side in **14,868 of 1,035,745** market-weeks.
  Under the side-total reading: zero violations.

Reading it as a share of open interest overstates by `1/(1-spread_share)` — 1.8× in
3-month SOFR, which is 44.5% spreads.

No board renders the concentration columns today. They are stored faithfully and the
denominator is pinned by a test, against the day one does.
