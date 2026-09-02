#!/usr/bin/env python3
"""Prove lib/cftc_spec.py against the live API. Run this after touching a field name.

    python -m scripts.verify_spec              # all seven datasets
    python -m scripts.verify_spec --source cftc_tff_fut

Two independent checks per dataset:

1. EXISTENCE. Ask Socrata for exactly the columns the spec names. A column that
   does not exist comes back as HTTP 400 with the offending name in the body,
   so this fails at the network boundary and names the culprit -- no parsing, no
   guessing. This is also the drift detector the ingest job relies on.

2. ACCOUNTING. Pull a real sample and check, per market-week:
       sum(long) == sum(short)                          within NET_ZERO_TOL
       sum(net)  == 0                                   within NET_ZERO_TOL
       OI - sum(long) - sum(spread) == 0                within OI_RESIDUAL_TOL
   A wrong field name that happens to exist (say the old-crop bucket instead of
   the all-maturity column) passes check 1 and fails check 2. That is the whole
   point of running both.

Exit code is non-zero if any dataset fails either check.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import cftc_api, cftc_spec  # noqa: E402


def check_existence(spec: cftc_spec.ReportSpec) -> tuple[bool, str]:
    """Ask for every spec'd column. HTTP 400 means a name is wrong."""
    try:
        rows = cftc_api.fetch_rows(spec, limit=5)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        return False, f"HTTP {exc.code} -- a column in the spec does not exist: {body}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if not rows:
        return False, "query succeeded but returned no rows"
    return True, f"all {len(spec.select_fields())} spec'd columns accepted"


def check_accounting(spec: cftc_spec.ReportSpec, n: int) -> tuple[bool, str]:
    """Pull recent rows and test the identities the field map has to satisfy."""
    rows = cftc_api.fetch_rows(spec, limit=n, order="report_date_as_yyyy_mm_dd DESC")
    if not rows:
        return False, "no rows"

    worst_ls = worst_net = worst_resid = 0.0
    n_checked = 0
    for raw in rows:
        row = cftc_api.normalise_keys(raw)
        oi = cftc_api.num(row, spec.meta_field("open_interest_all"))
        if oi is None:
            continue
        tot_long = tot_short = tot_spread = 0.0
        ok = True
        for c in spec.cohorts:
            lo = cftc_api.num(row, c.long)
            sh = cftc_api.num(row, c.short)
            if lo is None or sh is None:
                ok = False
                break
            tot_long += lo
            tot_short += sh
            if c.spread:
                tot_spread += cftc_api.num(row, c.spread) or 0.0
        if not ok:
            continue
        n_checked += 1
        worst_ls = max(worst_ls, abs(tot_long - tot_short))
        worst_net = max(worst_net, abs((tot_long - tot_short)))
        worst_resid = max(worst_resid, abs(oi - tot_long - tot_spread))

    if not n_checked:
        return False, "no rows had a complete cohort set -- field map is wrong"

    fails = []
    if worst_ls > cftc_spec.NET_ZERO_TOL:
        fails.append(f"|sum(long)-sum(short)| max {worst_ls:,.0f} > {cftc_spec.NET_ZERO_TOL}")
    if worst_resid > cftc_spec.OI_RESIDUAL_TOL:
        fails.append(f"|OI-long-spread| max {worst_resid:,.0f} > {cftc_spec.OI_RESIDUAL_TOL}")
    msg = (
        f"{n_checked:,} market-weeks: max |long-short| = {worst_ls:,.0f}, "
        f"max |OI-long-spread| = {worst_resid:,.0f}"
    )
    return (not fails), msg + ("  FAIL: " + "; ".join(fails) if fails else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="all")
    ap.add_argument("--rows", type=int, default=3000, help="sample size per dataset")
    args = ap.parse_args()

    specs = (
        list(cftc_spec.SPECS)
        if args.source == "all"
        else [cftc_spec.spec_for(args.source)]
    )

    failed = 0
    for spec in specs:
        print(f"\n=== {spec.id}  ({spec.dataset}, {spec.family}/{spec.scope})")
        ok, msg = check_existence(spec)
        print(f"  {'ok ' if ok else 'FAIL'} existence   {msg}")
        if not ok:
            failed += 1
            continue
        ok, msg = check_accounting(spec, args.rows)
        print(f"  {'ok ' if ok else 'FAIL'} accounting  {msg}")
        if not ok:
            failed += 1

    print(f"\n{len(specs) - failed}/{len(specs)} datasets verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
