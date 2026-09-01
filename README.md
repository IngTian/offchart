# ◧ watchboard

Alternative and government data that TradingView either doesn't carry or doesn't
display usefully. Built because the numbers were always free — the failure was
**cadence**: two CFTC charts arrived via a social-media post on 2026-08-31 and
both reproduced exactly from a public JSON endpoint with no key.

The rule this repo encodes: **a number you can't reproduce yourself can raise a
question, but it shouldn't answer one.** So anything reproducible from a primary
source gets pulled here; anything genuinely proprietary (a bank's internal
vol-control-fund estimate, say) stays outside and stays discounted.

## Architecture — why it's split in two

GitHub Actions cannot serve a Streamlit app, and Streamlit cannot run unattended.
So they own different layers, and the split is better than either alone:

```
        ┌────────────────────────── ingest (unattended) ──────────────────────────┐
        │  GitHub Actions, daily 20:15 UTC                                        │
        │    scripts/ingest.py  →  sources/*.fetch()  →  data/<id>.parquet         │
        │    commits data/ back to the repo                                       │
        └─────────────────────────────────┬───────────────────────────────────────┘
                                          │  data/ is versioned, history accrues
        ┌─────────────────────────────────▼──────────── display (interactive) ────┐
        │  app.py — Streamlit, reads data/ only, never fetches                    │
        │    local:  streamlit run app.py                                         │
        │    hosted: Streamlit Community Cloud, free, points at this repo          │
        └─────────────────────────────────────────────────────────────────────────┘
```

Consequences worth knowing:

- The board **works offline** and can't display a number that isn't in the
  committed dataset.
- History accrues **whether the laptop is awake or not**, which matters more than
  it looks — see the snapshot note below.
- Deploying to Streamlit Community Cloud gives a URL to open on Friday without
  keeping a terminal running. Same repo, same `app.py`, no code change.

## Sources

| id | what | cadence | history? |
|---|---|---|---|
| `cftc_tff` | CFTC Traders in Financial Futures — positioning by cohort for NDX (E-mini + Micro), VIX, SPX | Tuesday positions, published **Friday 15:30 ET** | ✅ full, from the API |
| `openrouter_pricing` | token prices, USD per token, 419 models | live | ❌ **snapshot only** |

### The snapshot distinction is the important one

`cftc_tff` serves 16 years of history on demand, so a missed run costs nothing.

`openrouter_pricing` returns **today's prices and nothing else**. There is no
backfill and no archive. So that dataset does not exist until this job starts
writing it, and every run that doesn't happen is a hole that can never be filled.
**Starting the job is what creates the series** — which is the argument for
starting it now rather than when it's needed.

Why it's worth having: the AI inference question is a claim about a *product*.

```
inference revenue  =  tokens  ×  price per token
```

If volume grows more slowly than price falls, revenue shrinks while "token
consumption is booming" stays literally true. This repo measures the second
factor. The first has no clean free endpoint yet — see the stub list in
`sources/__init__.py`.

## Use

```bash
pip install -r requirements.txt

python -m scripts.ingest --list          # what's registered
python -m scripts.ingest                 # pull everything
python -m scripts.ingest --source cftc_tff

streamlit run app.py
```

Ingest exits non-zero if any source fails, so a dead feed shows up as a red CI
run instead of a chart that quietly stops moving.

## Adding a source

Write a module in `sources/` exposing `SOURCE_KWARGS`, then add it to the tuple
in `sources/__init__.py`. Nothing else changes — ingest and the app both iterate
the registry.

`caveats` is a required-in-spirit field, not decoration: the app renders it next
to every chart built from the source. A limitation that lives only in a README
gets read without it.

**Verify the payload before writing the field names.** Two of the five CFTC
cohorts break the dataset's own naming pattern (`dealer` and `nonrept` take an
`_all` suffix; the other three don't), and `nonrept` has no spread field at all.
That was found by probing a live response, not by recalling it.

## Two invariants the board checks on every CFTC week

**1. Net positions sum to zero, exactly.** Every futures contract has two sides,
so a market-wide net of anything but zero would mean a contract with one side:

```
+74,231 +20,631 +4,130 −44,316 −54,676  =  0        (NDX, 2026-08-25)
```

**2. Longs, shorts and spreads reconcile to open interest — and it is *not*
`sum(long) == OI`.** A spread position is long one expiry and short another, so
the report counts it in neither column and gives it its own:

```
sum(long) = sum(short) = 288,110     spreads = 34,081     open interest = 322,190
288,110 + 34,081 = 322,191           residual = 1  (unpublished non-reportable spread)
```

The net identity survives spreads because a spread adds equally to both sides and
cancels. This is the reconciliation most write-ups get wrong.

## Display rules enforced in code

- **No dual axes.** Two series with two y-scales lets the author choose the
  scaling that makes a correlation look how they want. Stacked panels instead.
- **Expanding percentiles only.** Ranked against history up to that date, never
  the full series — a full-history percentile carries look-ahead bias.
- **Signed quantities keep their own units.** A net going −100,640 → −96,727 is
  *"+3,913 contracts, less short"*. Rendering that as *"+3.89%"* reads as growth
  while the position shrank toward zero. `metrics.pct_change_is_safe()` exists to
  be called before formatting.
- **Open interest is always on screen next to positioning.** A cohort's share can
  rise because contracts were created or because they changed hands, and the
  share alone cannot distinguish them.

## What positioning data does not do

Measured in the companion vault as **contemporaneous with price, not
predictive**. And a cohort's net is a sum over firms running incompatible
strategies — a basis trader long the cash index and short the future appears here
as *short* while holding no view whatsoever. **Net is not a stance.** Gross is
the more honest number for a fragility question: it says how much there is to
unwind.

Report the level. Don't infer a direction from it without a base-rate test.
