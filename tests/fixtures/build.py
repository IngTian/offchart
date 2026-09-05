"""Regenerate the committed test slice from data/. Run: python -m tests.fixtures.build

The suite must never touch the network and must finish in seconds, so it runs
against a small committed slice of the real parquet files rather than against
data/ (115 MB) or a mock. A mock would be worse than useless here: every bug this
suite defends against is a property of the REAL published figures -- rounding
that breaks exact equality, nulls that co-occur with nonzero positions, a
contract re-basing mid-history, two markets sharing one name. A hand-written
frame reproduces only the cases its author already thought of.

WHY THESE CODES. Each one is here to make a specific awkward case true, named
below next to it. Counts are measured on the slice this script writes.

  cftc_tff_fut
    20974+   full history, 846 reports from 2010-06-15. The 2023-05-02 contract
             re-basing: open interest 49,531 -> 255,954 (x5.17) while
             contract_units goes (NASDAQ 100 INDEX X $100) -> ($20), i.e. a unit
             break, not a positioning move. Also 458 market-weeks where
             sum(net) != 0 and 251 where the open-interest residual is NEGATIVE
             -- the two facts that make exact-equality assertions wrong.
    133741   BITCOIN, full history. 281 rows with a NULL traders_long beside a
             NONZERO long in a REPORTABLE cohort, where the column exists and is
             simply not published. The nonrept cohort's nulls are structural
             (no traders_* column at all in any family) and prove nothing.
    209742   last 65 reports only. A SUPERSEDED_BY leg (E-mini NDX, already
             inside 20974+) still carrying ~378k contracts, so the tier test can
             show supersession is what keeps it out of CORE rather than size.
             Only the trailing liquidity window is needed for that, so this one
             is trimmed hardest.
  cftc_legacy_fut
    088691   GOLD, full history, 1,931 reports from 1986-01-15. Legacy is NOT
             weekly before 2002-01-08: gaps of 3,6,7,8 and 13-18 days across all
             five weekdays, and even after it gaps of 3, 6, 8 and 11 days occur.
             Anything that treats one row back as one week is wrong here. Legacy
             also supplies the comm cohort, whose spread column does not exist.
    209741   1996-2000 prefix. With 209742 it gives the name collision: on
    209742   1999-06-22 both report market_and_exchange_names exactly
             "NASDAQ-100 STOCK INDEX - INTERNATIONAL MONETARY MARKET" while
             holding 22,998 and 537 contracts. A name-keyed dedupe discards a
             real market. 209742 additionally carries FOUR distinct names inside
             those 18 months, so a name-keyed series splits into four.
  cftc_disagg_fut
    088691   2016 onwards. The same code in a second family under a different
             cohort taxonomy -- the shape that makes a family-blind read
             silently wrong (see sources/cftc.py). disagg is also the family
             whose residual is exactly zero on every row, so it is the control
             for the residual test, and it supplies prod_merc, the other cohort
             with no spread column.
  cftc_supp_cit
    001602   2016 onwards. Rounds worst of the four: sum(net) != 0 on 295 of 556
             market-weeks, with the residual negative on 121 and positive on 138
             -- the symmetry that rules out an unpublished non-negative spread.
             Publishes no concentration columns, and three of its four cohorts
             have no spread column.

WHAT THE DATE WINDOWS COST. Full history is kept where the case needs it, and
trimming is by contiguous window only -- never by sampling and never with an
interior hole, because lib/segments reads a hole as a contract re-specification
and a sampled slice would fabricate breaks that are not in the data. Truncating
everything to a few hundred weeks, which is what a naive slice does, would
destroy the cases above: the re-basing needs both sides of 2023-05-02, the
cadence case needs 1986-1993, universe.MIN_HISTORY is 140 reports and the
trailing windows in lib/metrics are 1,095 days, so a 300-week slice makes every
ranking test vacuous while saving ~200 KB.

The one artefact the windows introduce: a prefix window makes a market look
delisted at the cut (legacy 209741/209742 appear to stop in 2000; they do not)
and a suffix window makes one look young. Neither is a property of the data.
Tests that care about liveness or tiering use cftc_tff_fut, where the codes run
to the real last report.

openrouter_pricing is copied whole -- it is snapshot-only and currently holds one
snapshot date, so there is nothing to trim.

The slice is written through lib.store.upsert with the real Source key and sort
order, so it has production dtypes (Int32 that keeps its nulls, dictionary-
encoded strings, plain-string dates) and production row-group layout. A fixture
written by a different code path would not exercise the dtypes the app sees.
Total on disk is ~0.5 MB.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sources  # noqa: E402
from lib import store  # noqa: E402

#: (first report to keep, last report to keep), inclusive; None means unbounded.
FULL: tuple[str | None, str | None] = (None, None)

#: source_id -> {market_code: window}. `None` instead of a dict means "every row",
#: for a source that is already tiny. See the module docstring for what each code
#: is here to make true.
SELECTION: dict[str, dict[str, tuple[str | None, str | None]] | None] = {
    "cftc_tff_fut": {
        "20974+": FULL,  # 2023-05-02 unit break, net != 0, negative residual
        "133741": FULL,  # null traders_long beside a nonzero long
        "209742": ("2025-06-01", None),  # superseded leg, liquidity window only
    },
    "cftc_legacy_fut": {
        "088691": FULL,  # pre-2002 non-weekly cadence, 40-year history
        "209741": ("1996-01-01", "2000-12-31"),  # name collision, 1999-06-22
        "209742": ("1996-01-01", "2000-12-31"),  # ... and four names in 18 months
    },
    "cftc_disagg_fut": {"088691": ("2016-01-01", None)},  # residual exactly zero
    "cftc_supp_cit": {"001602": ("2016-01-01", None)},  # worst rounding, no conc
    "openrouter_pricing": None,
}


def _slice(source_id: str, windows: dict[str, tuple[str | None, str | None]] | None) -> pd.DataFrame:
    df = pd.read_parquet(REPO_ROOT / "data" / f"{source_id}.parquet")
    if df.empty:
        raise SystemExit(f"{source_id}: data/ is empty -- run scripts/ingest first")
    if windows is None:
        out = df.copy()
    else:
        # .astype(str) because market_code arrives dictionary-encoded, and a
        # comparison that leaves it categorical carries the full 951-code
        # category set into the output and inflates the fixture's dictionary
        # pages with codes it does not contain.
        code = df["market_code"].astype(str)
        date = df["report_date"].astype(str)
        keep = pd.Series(False, index=df.index)
        for c, (lo, hi) in windows.items():
            m = code == c
            if not m.any():
                raise SystemExit(f"{source_id}: code {c!r} absent from data/")
            if lo:
                m &= date >= lo
            if hi:
                m &= date <= hi
            if not m.any():
                raise SystemExit(f"{source_id}: code {c!r} has no rows in {lo}..{hi}")
            keep |= m
        out = df[keep].copy()
    # Drop the stale category sets so store._tighten rebuilds them from the
    # slice alone.
    cats = [c for c in out.columns if isinstance(out[c].dtype, pd.CategoricalDtype)]
    return out.astype({c: "object" for c in cats})


def _umich_fabricated() -> pd.DataFrame:
    """umich_sca is FABRICATED, not sliced, and that is the point.

    Every other fixture here is a window cut out of data/. This one cannot be: the
    University of Michigan permits use of its public tables but not redistribution,
    which is why data/umich_sca.parquet is gitignored (see sources/umich). Copying
    even a few hundred of their values into a committed fixture would be exactly the
    redistribution the gitignore exists to avoid.

    Fabricating instead costs nothing and buys something: the shape can be built to
    contain every edge the panel has to survive, rather than whatever the real file
    happens to hold this month.

      * a sparse pre-1978 era (quarterly), so gap rendering is exercised
      * monthly from 1978, so the cadence break is real
      * infl_exp_5y10y blank before 1990-04, as upstream
      * the three 2024 blend months present, so the panel must exclude them
      * a web era of 26 months -- shorter than MIN_RANK_OBSERVATIONS -- so the
        suppressed-rank path is the one under test, which is the live case
      * inflation pools long enough to clear the threshold, so the bubble path is
        under test too

    Deterministic arithmetic, no RNG: a fixture that changes between rebuilds turns
    an unrelated failure into a hunt.
    """
    from lib import umich_spec

    quarterly = pd.date_range("1970-02-01", "1977-11-01", freq="QS-FEB")
    monthly = pd.date_range("1978-01-01", "2026-08-01", freq="MS")
    months = quarterly.union(monthly)

    web_start = pd.Timestamp(umich_spec.WEB_ERA_START)
    blend = {pd.Timestamp(m) for m in umich_spec.BLEND_MONTHS}

    rows = []
    for i, when in enumerate(months):
        # A slow wave plus a shorter one: no RNG, but not a straight line either, so
        # a percentile computed over it is not degenerate.
        wave = 12.0 * math.sin(i / 19.0) + 5.0 * math.cos(i / 4.0)
        # The mode shift UMich measured, applied from the web era onward so the
        # fixture reproduces the trap the panel exists to avoid.
        shift = -6.6 if when >= web_start else 0.0
        if when in blend:
            shift = -3.3  # part way through the four-month blend
        rows.append(
            {
                "survey_date": when.strftime("%Y-%m-01"),
                "year": int(when.year),
                "sentiment": round(85.0 + wave + shift, 1),
                "current_conditions": round(90.0 + wave * 1.1 + shift * 2.0, 1),
                "expectations": round(80.0 + wave * 0.9 + shift * 0.3, 1),
                "infl_exp_1y": (
                    None if when < pd.Timestamp("1978-01-01")
                    else round(3.2 + 1.4 * math.sin(i / 11.0), 1)
                ),
                "infl_exp_5y10y": (
                    None if when < pd.Timestamp("1990-04-01")
                    else round(2.9 + 0.8 * math.sin(i / 23.0), 1)
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build() -> list[dict]:
    reports = []
    for source_id, windows in SELECTION.items():
        src = sources.get(source_id)
        frame = _slice(source_id, windows)
        target = FIXTURE_DIR / f"{source_id}.parquet"
        target.unlink(missing_ok=True)  # full rewrite: a dropped code must vanish
        reports.append(store.upsert(source_id, frame, src.key, sort_key=src.sort_key))

    src = sources.get("umich_sca")
    (FIXTURE_DIR / "umich_sca.parquet").unlink(missing_ok=True)
    reports.append(
        store.upsert("umich_sca", _umich_fabricated(), src.key, sort_key=src.sort_key)
    )
    return reports


def main() -> None:
    # store writes to store.DATA_DIR; point it here rather than reimplementing
    # the write, so the fixture and the real files come off one code path.
    original = store.DATA_DIR
    store.DATA_DIR = FIXTURE_DIR
    try:
        reports = build()
    finally:
        store.DATA_DIR = original

    total = 0
    for r in reports:
        total += r["bytes"]
        print(f"  {r['source']:<22} {r['rows_total']:>7,} rows  {r['bytes'] / 1024:>7.1f} KB")
    print(f"  {'TOTAL':<22} {'':>7}       {total / 1024:>7.1f} KB")


if __name__ == "__main__":
    main()
