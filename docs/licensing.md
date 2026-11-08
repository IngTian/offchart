# Licensing, and what is deliberately absent

The short version: **the code is MIT, the data is not this repo's to license, and no
dataset from any source is stored here — now or in its history.**

The longer version matters because it is the reason two obvious series are missing.

## What "this repository contains no data" settles, and what it does not

Stated plainly, because it would be easy to over-claim. The framing settles
**redistribution** and **non-commercial use** completely.

It does **not** settle an upstream term that restricts *creating* a local database, or
using the data for a particular purpose — because those bind whoever runs `make pull`
rather than this repository. Where a source's terms work that way, the honest answer is
to find a different source for the same series, or to go without it.

**A source is not added here because its data would be useful.**

## The worked example: margin debt and short interest

Everyone means FINRA's monthly margin-debt series, back to 1997. It is not here.

FINRA grants use — content "shall be used ONLY for your own non-commercial personal or
professional use" — but separately prohibits the mechanism a fetcher needs:

- **(d)** "develop or create a database of data using the FINRA Website"
- **(e)** "any data mining, scraping or harvesting tools (including robots)"
- **(m)** use of any portion of the site "in the development of any software program"
- **(o)** any purpose without written permission

None of those are redistribution clauses, so "nothing is stored here" does not answer
them. Reading their page yourself is squarely inside the grant; publishing a tool that
automates (d), (e) and (m) is not. A permission request has been sent; if it is granted,
the grant will be quoted verbatim in the source's `license` field the way
`sources/umich.py` quotes UMich's.

**FRED is not a substitute.** Its nearest series is about 49% of the FINRA figure with a
drifting ratio, so it is a different quantity wearing a similar name.

**The Federal Reserve's Z.1 Financial Accounts were tried and rejected.** Its
broker-dealer customer receivables series carries undocumented reclassifications —
$146bn moved from "households" to "hedge funds" in 2012:Q4 with the total unchanged, the
household-bank-loan line jumps sixfold in 2024:Q4, and one 40-quarter stretch sits
pinned at exactly 28.390% — so a long chart of it reads as behaviour when it is
accounting. The source and its board were built and then withdrawn.

Not every gap has a substitute.

## Per-source terms

Of the 13 sources, five are marked `redistributable=False`. The rest are US federal work
or openly published APIs. The distinction is enforced by tests rather than by a comment.

| source | terms |
|---|---|
| `cftc_*` (7) | US Government work, public domain |
| `prices` | Yahoo's own terms; an undocumented endpoint |
| `umich_sca` | Use granted, redistribution prohibited without written consent |
| `proshares_flows` | Use granted, mirroring forbidden, non-commercial only |
| `spy_flows`, `gld_flows` | SSGA bars dissemination of site content |
| `smh_flows` | VanEck restricts **commercial** use; personal electronic storage is expressly permitted |

`umich_sca` — the University of Michigan grants permission-free *use* of its public
Surveys of Consumers tables and separately prohibits *redistribution* without written
consent (`faq.php` vs `agreement.php`, both quoted in `sources/umich.py`). Charting is
fine; mirroring is not. Even the test fixture's UMich slice is **fabricated**, not
sampled, for exactly this reason.

Each source's `license` field carries the upstream's own words, verbatim, with a URL.
`citation` carries the attribution it requires — and the dataclass refuses to construct
a `redistributable=False` source without both.

## How the rule is enforced

`tests/test_sources.py` asks **git** rather than trusting `.gitignore` to say what it
means:

- the store is unignorable at *every* path a `--db` run can write to — ten of them,
  enumerated, including `data/sub/offchart.sqlite` which `data/*.sqlite` does not reach
- nothing under `data/` is tracked at all
- no committed file is named for a source we may not redistribute

And every source serves its own history back, so losing the file costs one
`make backfill`. That is the whole recovery story: nothing here is kept only by us.

## The rule that used to pull the other way

A since-deleted source, `openrouter_pricing`, served only "today" with no archive, so it
*had* to be committed to exist — one CSV per date, written before the database was
touched, with a test asserting no gaps.

It is gone, and the honest reason is worth recording: it rendered on no panel, and
`/api/v1/models` reports one host's price per model out of a spread measured at a median
20.7× (max 150×) across hosts, so the series tracked which host was cheapest that
morning rather than any price.

The daily job that maintained it also **destroyed six days of it** — the commit step sat
*after* the test step, so one dropped cron created a gap, the gap-detection test went
red, and the commit was skipped from then on. If anything here ever commits data again,
the commit goes first.

Every source now serves history, which means no test can be satisfied by writing data
into the repo. The only artefact question left is whether committing is *allowed* —
never whether it is required.

## What CI does

One live call to `scripts/verify_spec.py` to catch CFTC field-map drift, which is the
monitor for the failure mode with no symptom: a column that still exists but now means
something else, producing a plausible chart of the wrong number.

It commits nothing. CI does **not** pull CFTC — every source is re-fetchable, and a
runner that fetched 900 MB only to be destroyed would be theatre.

## Attribution

Code: MIT, see [LICENSE](../LICENSE).

CFTC Commitments of Traders is US Government work and in the public domain. Yahoo data
arrives under its own terms. University of Michigan Surveys of Consumers data is used
with permission for charting and is **never redistributed here** — cite it as
*"University of Michigan, Survey Research Center, Surveys of Consumers."* ProShares,
State Street and VanEck figures carry their own required attribution, recorded in each
source's `citation`.

If you publish anything derived from a source, check that source's terms; `sources/*.py`
records the ones this repo checked, verbatim, with URLs.
