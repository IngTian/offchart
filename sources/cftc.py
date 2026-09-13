"""All seven CFTC Commitments-of-Traders datasets, from one parser.

Each dataset becomes its own Source and its own table. That is a deliberate
choice over one combined table with a `family` column, and the reason is a bug
that combining invites:

    the same cftc_contract_market_code exists in several families

20974+ (NASDAQ-100 Consolidated) appears in legacy_fut, legacy_futopt, tff_fut
and tff_futopt. A read that filters on code and date but forgets to filter on
family gets 16 rows for one week across 4 families, and the obvious
`groupby(report_date).sum()` on top of it silently reports a NASDAQ net of
+80,950 (25.1% of open interest) where the truth for the futures-only TFF report
is +20,631 (6.4%) -- wrong by 3.9x, with every individual number in the frame
correct. Separate files make that mistake unrepresentable rather than merely
discouraged.

The tidy output is one row per (report_date, market_code, cohort), which is the
shape every panel wants and the shape that keeps the git delta small.
"""
from __future__ import annotations

import pandas as pd

from lib import cftc_api, cftc_spec
from lib.schema import TableSchema
from sources.base import Source

#: Emitted for every row. Order matters only for readability.
TIDY_COLUMNS = (
    "report_date",
    "market_code",
    "cohort",
    "cohort_label",
    "market",
    "market_full",
    "commodity",
    "group",
    "subgroup",
    "contract_units",
    "long",
    "short",
    "spread",
    "traders_long",
    "traders_short",
    "open_interest",
    "oi_change_published",
    "traders_total",
    "change_long_published",
    "change_short_published",
    "conc_gross_4_long",
    "conc_gross_4_short",
    "conc_gross_8_long",
    "conc_gross_8_short",
    "conc_net_4_long",
    "conc_net_4_short",
    "conc_net_8_long",
    "conc_net_8_short",
)

_CONC_MAP = {
    "conc_gross_4_long": "conc_gross_le_4_tdr_long",
    "conc_gross_4_short": "conc_gross_le_4_tdr_short",
    "conc_gross_8_long": "conc_gross_le_8_tdr_long",
    "conc_gross_8_short": "conc_gross_le_8_tdr_short",
    "conc_net_4_long": "conc_net_le_4_tdr_long_all",
    "conc_net_4_short": "conc_net_le_4_tdr_short_all",
    "conc_net_8_long": "conc_net_le_8_tdr_long_all",
    "conc_net_8_short": "conc_net_le_8_tdr_short_all",
}


def parse(rows: list[dict], spec: cftc_spec.ReportSpec) -> pd.DataFrame:
    """Melt raw Socrata rows into the tidy long format. Pure -- no I/O.

    Being pure is what lets tests exercise the field maps against a committed
    fixture without touching the network.
    """
    records: list[dict] = []
    for raw in rows:
        row = cftc_api.normalise_keys(raw)

        oi = cftc_api.num(row, spec.meta_field("open_interest_all"))
        date = row.get("report_date_as_yyyy_mm_dd")
        code = cftc_api.text(row, "cftc_contract_market_code")
        if oi is None or not date or not code:
            continue

        # Concentration is shared across cohorts within a market-week. Values
        # outside [0, 100] are dropped: legacy reports 482.6% top-8 long
        # concentration on one 957-contract market-week, and plotting that
        # faithfully renders a broken input.
        conc: dict[str, float | None] = {}
        for out_name, field in _CONC_MAP.items():
            v = cftc_api.num(row, field) if spec.has_conc else None
            conc[out_name] = v if (v is not None and 0.0 <= v <= cftc_spec.CONC_VALID_MAX) else None

        base = {
            "report_date": str(date)[:10],
            "market_code": code.strip(),
            "market_full": cftc_api.text(row, "market_and_exchange_names"),
            "market": cftc_api.text(row, "contract_market_name")
            or cftc_api.text(row, "market_and_exchange_names"),
            "commodity": cftc_api.text(row, "commodity_name"),
            "group": cftc_api.text(row, "commodity_group_name"),
            "subgroup": cftc_api.text(row, "commodity_subgroup_name"),
            "contract_units": cftc_api.text(row, "contract_units"),
            "open_interest": oi,
            "oi_change_published": cftc_api.num(
                row, spec.meta_field("change_in_open_interest_all")
            ),
            "traders_total": cftc_api.num(row, "traders_tot_all"),
            **conc,
        }

        for c in spec.cohorts:
            lo = cftc_api.num(row, c.long)
            sh = cftc_api.num(row, c.short)
            if lo is None or sh is None:
                # Position columns are never null in any of the seven datasets
                # (0 of 30,100 sampled rows), so this means the field map is
                # wrong, not that the data is sparse. Skipping keeps the
                # accounting check honest -- it will fail and say so.
                continue
            records.append(
                {
                    **base,
                    "cohort": c.id,
                    "cohort_label": c.label,
                    "long": lo,
                    "short": sh,
                    "spread": cftc_api.num(row, c.spread),
                    "traders_long": cftc_api.num(row, c.traders_long),
                    "traders_short": cftc_api.num(row, c.traders_short),
                    "change_long_published": cftc_api.num(row, c.change_long),
                    "change_short_published": cftc_api.num(row, c.change_short),
                }
            )

    if not records:
        return pd.DataFrame(columns=list(TIDY_COLUMNS))

    df = pd.DataFrame.from_records(records)
    for col in TIDY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[list(TIDY_COLUMNS)]


def _make_fetch(spec: cftc_spec.ReportSpec):
    def fetch(since: str | None = None) -> pd.DataFrame:
        rows = cftc_api.fetch_rows(spec, since=since)
        print(f"    pulled {len(rows):,} raw market-weeks" + (f" since {since}" if since else " (full history)"))
        return parse(rows, spec)

    fetch.__name__ = f"fetch_{spec.id}"
    return fetch


#: Caveats that apply to every CFTC source. Rendered next to every chart.
COMMON_CAVEATS: tuple[str, ...] = (
    "Positioning here is CONTEMPORANEOUS with price, not predictive. Report the "
    "level; do not infer a direction without a base-rate test. The forward-outcome "
    "board exists so that test can be run rather than assumed.",
    "A cohort's net is a sum over firms running incompatible strategies. A basis "
    "trader long the cash index and short the future appears here as short while "
    "holding no view at all. Net is not a stance.",
    "Percentiles are causal -- each observation is ranked only against data at or "
    "before its own date. A full-sample percentile would carry look-ahead bias.",
    "sum(long) == sum(short) == open interest MINUS spreads. A spread position is "
    "long one expiry and short another, so it belongs to neither side and gets its "
    "own column.",
    # Do not restate the residual argument here -- render
    # cftc_spec.RESIDUAL_EXPLANATION, which is the single source of that wording.
    # An earlier copy of it drifted and contradicted the canonical text on the
    # same screen, which is exactly the failure a duplicated caption invites.
    "The identities hold to within a few contracts, not exactly, because CFTC "
    "rounds each published column independently. The leftover is rounding noise, "
    "not a hidden position bucket -- see the reconciliation on the Integrity "
    f"board, which states the evidence. Tolerances: "
    f"{cftc_spec.NET_ZERO_TOL:.0f} contracts on the sum of cohort nets, "
    f"{cftc_spec.OI_RESIDUAL_TOL:.0f} on the open-interest "
    "reconciliation.",
    "Markets are keyed on cftc_contract_market_code, never on name: 26-30% of codes "
    "have been renamed at least once, and CFTC shortened names wholesale on "
    "2022-02-08. A name-keyed series silently splits in two there.",
    "History is split into segments at contract re-specifications and at gaps "
    "longer than a quarter. Levels are only comparable within a segment -- CFTC "
    "re-based its Consolidated equity indices on 2023-05-02 and open interest "
    "jumped 5x with no change in positioning.",
    "Per-cohort trader counts are often null while the position is nonzero, and the "
    "null rate is growing (4% of 2015 rows, 29% of 2026). Measures that divide by "
    "trader count are blank for about a third of the current cross-section.",
)


def _schema_for(spec: cftc_spec.ReportSpec) -> TableSchema:
    return TableSchema(
        required=("report_date", "market_code", "cohort", "long", "short", "open_interest"),
        numeric=(
            "long",
            "short",
            "spread",
            "open_interest",
            "traders_long",
            "traders_short",
            "traders_total",
            "oi_change_published",
            "change_long_published",
            "change_short_published",
            *_CONC_MAP.keys(),
        ),
        optional=tuple(
            c
            for c in TIDY_COLUMNS
            if c
            not in {"report_date", "market_code", "cohort", "long", "short", "open_interest"}
        ),
    )


def _cadence(spec: cftc_spec.ReportSpec) -> str:
    base = "Positions as of Tuesday, published Friday 15:30 ET"
    if spec.family == "legacy":
        return base + " -- but NOT weekly before 2002-01-08 (646 earlier reports, mixed weekdays, gaps of 11-18 days)"
    return base


def build_sources() -> list[Source]:
    out: list[Source] = []
    for spec in cftc_spec.SPECS:
        caveats = list(COMMON_CAVEATS)
        caveats.insert(0, f"Coverage: {spec.coverage}")
        if spec.is_combined:
            caveats.insert(
                1,
                "This is the COMBINED futures-and-options report. Its figures will "
                "not reconcile against the futures-only report for the same market; "
                "the two are different datasets, not two views of one.",
            )
        if not spec.has_conc:
            caveats.append("This report publishes no concentration columns.")

        out.append(
            Source(
                id=spec.id,
                label=f"CFTC {spec.label}",
                fetch=_make_fetch(spec),
                # (date, code, cohort) is a true primary key: (code, report_date)
                # has zero duplicate groups in all seven datasets, while
                # (name, report_date) has 15 duplicates in legacy_fut alone from
                # truncated historical names, where a dedupe would silently
                # discard one of two real markets.
                key=("report_date", "market_code", "cohort"),
                sort_key=("report_date", "market_code", "cohort"),
                schema=_schema_for(spec),
                cadence=_cadence(spec),
                backfillable=True,
                incremental=True,
                provenance=(
                    f"publicreporting.cftc.gov/resource/{spec.dataset}.json "
                    f"(Socrata, no key). {spec.label}. Columns are requested by name "
                    "with $select, so an upstream rename fails the run with HTTP 400 "
                    "rather than silently nulling a chart."
                ),
                caveats=tuple(caveats),
                group="CFTC positioning",
            )
        )
    return out
