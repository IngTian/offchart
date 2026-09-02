"""Field maps for the seven CFTC Commitments-of-Traders datasets.

Every field name in this file was read off a live payload and then checked by
`scripts/verify_spec.py`, which asserts (a) the name exists in the payload and
(b) the accounting invariants hold once you use it. None of it was recalled.
That matters more here than anywhere else in the repo, because a wrong field
name does not raise -- it silently produces a plausible chart of the wrong
number.

The API misspells its own columns, inconsistently, and only in some families:

    legacy   noncomm_postions_spread_all      "postions", missing the 'i'
    disagg   swap_positions_long_all          one underscore
             swap__positions_short_all        TWO underscores
             swap__positions_spread_all       TWO underscores
    tff      dealer/nonrept use *_long_all,   asset_mgr/lev_money/other_rept
             *_short_all                      use *_long / *_short, no suffix
    supp     Comm_Positions_Short_All_NoCIT   MixedCase in the payload while
                                              the Socrata *metadata* for the
                                              same column is lowercase

Two more traps that cost real debugging time:

1. In disagg, `prod_merc` and `other_rept` have NO `_all` variant. The bare
   base name IS the all-maturity column and `_1` / `_2` are the old-crop and
   other-crop buckets. So `prod_merc_positions_long_all` raises KeyError (loud,
   fine) but `prod_merc_positions_long_1` returns the OLD CROP YEAR silently.
   Confirmed the bare name is all-maturities by reconciliation: with the bare
   names disagg_fut balances on 184,229 of 184,229 rows with residual exactly 0.

2. Socrata OMITS a key entirely when its value is null. A row is not missing a
   field because the schema changed; it is missing because that value is null.
   Per-row key counts in tff_fut range 65..90 against a 90-key union. NEVER
   infer the schema from len(row).

WHAT THE COHORTS ARE. Three incompatible taxonomies over the same futures:

    legacy   1986+  commercial / non-commercial / non-reportable
                    Coarse, but it is the only 40-year history.
    disagg   2006+  producer-merchant / swap dealer / managed money /
                    other reportable / non-reportable      COMMODITIES ONLY
    tff      2006+  dealer / asset manager / leveraged funds /
                    other reportable / non-reportable      FINANCIALS ONLY

disagg and tff partition the modern universe between them and never overlap;
legacy covers both and reaches back furthest. They are NOT interchangeable --
"commercial" is not "producer + swap", and splicing legacy onto disagg at
2006-06-13 creates a definitional break, not a longer series.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cohort:
    """One trader classification within one report family.

    `spread` is None where the report publishes no spread column for this
    cohort -- that is a real property of the report, not missing data.
    Commercials in legacy and producer-merchants in disagg genuinely have no
    spread column, and non-reportables never do in any family.

    `traders_*` are None where the column does not exist at all. Note this is
    a DIFFERENT condition from the column existing but being null, which is
    common and growing: per-cohort trader counts are null on ~29% of 2026 tff
    rows and null co-occurs with a NONZERO position, so a trader count is not
    a reliable denominator. See lib/metrics.avg_position_per_trader.
    """

    id: str
    label: str
    long: str
    short: str
    spread: str | None = None
    traders_long: str | None = None
    traders_short: str | None = None
    traders_spread: str | None = None
    change_long: str | None = None
    change_short: str | None = None

    def position_fields(self) -> tuple[str, ...]:
        return tuple(f for f in (self.long, self.short, self.spread) if f)

    def aux_fields(self) -> tuple[str, ...]:
        return tuple(
            f
            for f in (
                self.traders_long,
                self.traders_short,
                self.traders_spread,
                self.change_long,
                self.change_short,
            )
            if f
        )


# --------------------------------------------------------------------------- #
# Cohort maps, one per family
# --------------------------------------------------------------------------- #
LEGACY_COHORTS = (
    Cohort(
        id="noncomm",
        label="Non-commercial (speculative)",
        long="noncomm_positions_long_all",
        short="noncomm_positions_short_all",
        spread="noncomm_postions_spread_all",  # sic: "postions"
        traders_long="traders_noncomm_long_all",
        traders_short="traders_noncomm_short_all",
        traders_spread="traders_noncomm_spread_all",
        change_long="change_in_noncomm_long_all",
        change_short="change_in_noncomm_short_all",
    ),
    Cohort(
        id="comm",
        label="Commercial (hedger)",
        long="comm_positions_long_all",
        short="comm_positions_short_all",
        spread=None,  # legacy publishes no commercial spread column
        traders_long="traders_comm_long_all",
        traders_short="traders_comm_short_all",
        change_long="change_in_comm_long_all",
        change_short="change_in_comm_short_all",
    ),
    Cohort(
        id="nonrept",
        label="Non-reportable (small traders)",
        long="nonrept_positions_long_all",
        short="nonrept_positions_short_all",
        spread=None,
        change_long="change_in_nonrept_long_all",
        change_short="change_in_nonrept_short_all",
    ),
)

DISAGG_COHORTS = (
    Cohort(
        id="prod_merc",
        label="Producer / merchant / processor / user",
        long="prod_merc_positions_long",  # bare name IS all-maturities
        short="prod_merc_positions_short",
        spread=None,
        traders_long="traders_prod_merc_long_all",
        traders_short="traders_prod_merc_short_all",
        change_long="change_in_prod_merc_long",
        change_short="change_in_prod_merc_short",
    ),
    Cohort(
        id="swap",
        label="Swap dealers",
        long="swap_positions_long_all",  # one underscore
        short="swap__positions_short_all",  # two underscores
        spread="swap__positions_spread_all",  # two underscores
        traders_long="traders_swap_long_all",
        traders_short="traders_swap_short_all",
        traders_spread="traders_swap_spread_all",
        change_long="change_in_swap_long_all",
        change_short="change_in_swap_short_all",
    ),
    Cohort(
        id="m_money",
        label="Managed money",
        long="m_money_positions_long_all",
        short="m_money_positions_short_all",
        spread="m_money_positions_spread",
        traders_long="traders_m_money_long_all",
        traders_short="traders_m_money_short_all",
        traders_spread="traders_m_money_spread_all",
        change_long="change_in_m_money_long_all",
        change_short="change_in_m_money_short_all",
    ),
    Cohort(
        id="other_rept",
        label="Other reportables",
        long="other_rept_positions_long",  # bare name IS all-maturities
        short="other_rept_positions_short",
        spread="other_rept_positions_spread",
        traders_long="traders_other_rept_long_all",
        traders_short="traders_other_rept_short",
        traders_spread="traders_other_rept_spread",
        change_long="change_in_other_rept_long",
        change_short="change_in_other_rept_short",
    ),
    Cohort(
        id="nonrept",
        label="Non-reportable (small traders)",
        long="nonrept_positions_long_all",
        short="nonrept_positions_short_all",
        spread=None,
        change_long="change_in_nonrept_long_all",
        change_short="change_in_nonrept_short_all",
    ),
)

TFF_COHORTS = (
    Cohort(
        id="dealer",
        label="Dealer / intermediary",
        long="dealer_positions_long_all",
        short="dealer_positions_short_all",
        spread="dealer_positions_spread_all",
        traders_long="traders_dealer_long_all",
        traders_short="traders_dealer_short_all",
        traders_spread="traders_dealer_spread_all",
        change_long="change_in_dealer_long_all",
        change_short="change_in_dealer_short_all",
    ),
    Cohort(
        id="asset_mgr",
        label="Asset managers",
        long="asset_mgr_positions_long",  # no _all suffix
        short="asset_mgr_positions_short",
        spread="asset_mgr_positions_spread",
        traders_long="traders_asset_mgr_long_all",
        traders_short="traders_asset_mgr_short_all",
        traders_spread="traders_asset_mgr_spread",
        change_long="change_in_asset_mgr_long",
        change_short="change_in_asset_mgr_short",
    ),
    Cohort(
        id="lev_money",
        label="Leveraged funds",
        long="lev_money_positions_long",  # no _all suffix
        short="lev_money_positions_short",
        spread="lev_money_positions_spread",
        traders_long="traders_lev_money_long_all",
        traders_short="traders_lev_money_short_all",
        traders_spread="traders_lev_money_spread",
        change_long="change_in_lev_money_long",
        change_short="change_in_lev_money_short",
    ),
    Cohort(
        id="other_rept",
        label="Other reportables",
        long="other_rept_positions_long",
        short="other_rept_positions_short",
        spread="other_rept_positions_spread",
        traders_long="traders_other_rept_long_all",
        traders_short="traders_other_rept_short",
        traders_spread="traders_other_rept_spread",
        change_long="change_in_other_rept_long",
        change_short="change_in_other_rept_short",
    ),
    Cohort(
        id="nonrept",
        label="Non-reportable (small traders)",
        long="nonrept_positions_long_all",
        short="nonrept_positions_short_all",
        spread=None,
        change_long="change_in_nonrept_long_all",
        change_short="change_in_nonrept_short_all",
    ),
)

# Supplemental splits the legacy commercial/non-commercial cohorts by removing
# index traders, then reports index traders (CIT) separately. So its cohorts
# are legacy-minus-CIT plus CIT, and they are NOT comparable to plain legacy.
SUPP_COHORTS = (
    Cohort(
        id="cit",
        label="Index traders (CIT)",
        long="cit_positions_long_all",
        short="cit_positions_short_all",
        spread=None,
        traders_long="traders_cit_long_all",
        traders_short="traders_cit_short_all",
        change_long="change_cit_long_all",
        change_short="change_cit_short_all",
    ),
    Cohort(
        id="noncomm_nocit",
        label="Non-commercial, ex-index",
        long="ncomm_postions_long_all_nocit",
        short="ncomm_postions_short_all_nocit",
        spread="ncomm_postions_spread_all_nocit",
        traders_long="traders_noncomm_long_all_nocit",
        traders_short="traders_noncomm_short_all_nocit",
        traders_spread="traders_noncomm_spread_all_nocit",
        change_long="change_noncomm_long_all_nocit",
        change_short="change_noncomm_short_all_nocit",
    ),
    Cohort(
        id="comm_nocit",
        label="Commercial, ex-index",
        long="comm_positions_long_all_nocit",
        short="comm_positions_short_all_nocit",
        spread=None,
        traders_long="traders_comm_long_all_nocit",
        traders_short="traders_comm_short_all_nocit",
        change_long="change_comm_long_all_nocit",
        change_short="change_comm_short_all_nocit",
    ),
    Cohort(
        id="nonrept",
        label="Non-reportable (small traders)",
        long="nonrept_positions_long_all",
        short="nonrept_positions_short_all",
        spread=None,
        change_long="change_nonrept_long_all",
        change_short="change_nonrept_short_all",
    ),
)


# --------------------------------------------------------------------------- #
# Concentration and meta columns
# --------------------------------------------------------------------------- #
# CR4 / CR8. The DENOMINATOR IS THE SIDE TOTAL (open interest minus spreads),
# NOT open interest. Two independent disproofs of the open-interest reading,
# both measured over the full 46,361-row tff_fut history:
#
#   - 158 rows report conc_gross_le_8_tdr_long == 100.0 exactly, and all 158
#     have spread > 0. If the denominator were open interest, the long side can
#     hold at most 100*(OI - spread)/OI, which on those rows falls to 25.2%.
#     A reading of exactly 100.0 is then arithmetically impossible.
#   - CR4/100 * OI exceeds the entire long side in 1,623 of 46,361 market-weeks
#     (3.5%) -- the top four longs holding more than all longs combined. Under
#     the side-total reading there are ZERO violations.
#
# Getting this wrong overstates by 1/(1 - spread_share): harmless in Micro SPX
# (spread 1.0% of OI) and a factor of 1.8 in SOFR-3M (44.5%).
CONC_FIELDS = (
    "conc_gross_le_4_tdr_long",
    "conc_gross_le_4_tdr_short",
    "conc_gross_le_8_tdr_long",
    "conc_gross_le_8_tdr_short",
    "conc_net_le_4_tdr_long_all",
    "conc_net_le_4_tdr_short_all",
    "conc_net_le_8_tdr_long_all",
    "conc_net_le_8_tdr_short_all",
)

# Concentration is not always a valid percentage in the RAW legacy payload: 23
# rows in legacy_fut and 18 in legacy_futopt report above 100%, the worst being
# code 148776 (STOCK INDEX NYSE CMP NEW) on 1998-12-22 reporting
# conc_gross_le_8_tdr_long = 482.6 on open interest of 957. disagg and tff have
# none.
#
# sources/cftc.parse() drops anything outside [0, 100] to NaN, so the COMMITTED
# data contains no such value -- max conc_* is exactly 100.00 in all seven files
# and no cell exceeds it across 4,087,353 rows. metrics.clamp_percentage is
# therefore currently inert on stored data, and is defence for a future ingest
# rather than a guard that fires today. Worth knowing before someone "simplifies"
# either layer away: it is the parse-time clamp doing the work.
CONC_VALID_MAX = 100.0

META_FIELDS = (
    "report_date_as_yyyy_mm_dd",
    "cftc_contract_market_code",
    "market_and_exchange_names",
    "contract_market_name",
    "contract_units",
    "commodity_name",
    "commodity_group_name",
    "commodity_subgroup_name",
    "open_interest_all",
    "change_in_open_interest_all",
    "traders_tot_all",
)

# supplemental uses different names for two of the meta columns and has no
# futonly_or_combined / cftc_subgroup_code at all.
SUPP_META_OVERRIDES = {
    "change_in_open_interest_all": "change_open_interest_all",
}


@dataclass(frozen=True)
class ReportSpec:
    """One CFTC dataset: a Socrata id plus the cohort vocabulary it uses."""

    id: str  # our source id, e.g. "cftc_tff_fut"
    dataset: str  # Socrata four-by-four, e.g. "gpe5-46if"
    family: str  # legacy | disagg | tff | supp
    scope: str  # fut | futopt | combined
    label: str
    cohorts: tuple[Cohort, ...]
    coverage: str  # what slice of the universe this family reports on
    has_conc: bool = True
    meta_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def is_combined(self) -> bool:
        """True where the figures include options delta-equivalents.

        Futures-only and futures-and-options-combined are DIFFERENT datasets
        and their numbers do not reconcile against each other. Never present
        one as a continuation of the other.
        """
        return self.scope in ("futopt", "combined")

    def meta_field(self, canonical: str) -> str:
        return self.meta_overrides.get(canonical, canonical)

    def select_fields(self) -> tuple[str, ...]:
        """Every column we ask Socrata for, deduped and ordered.

        Projecting with $select rather than taking all 133-194 columns is worth
        roughly 10x on wire bytes and 9x on peak memory (measured: legacy_fut
        full history is 1,343 MB of JSON and 3,764 MB of json.loads peak
        unprojected, versus 130 MB / 421 MB projected). It is also the cheapest
        schema-drift detector available: Socrata answers a $select naming a
        column that no longer exists with HTTP 400 and the column name in the
        body, at the network boundary, before any parsing happens.
        """
        out: list[str] = [self.meta_field(m) for m in META_FIELDS]
        for c in self.cohorts:
            out.extend(c.position_fields())
            out.extend(c.aux_fields())
        if self.has_conc:
            out.extend(CONC_FIELDS)
        seen: set[str] = set()
        ordered: list[str] = []
        for f in out:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
        return tuple(ordered)


SPECS: tuple[ReportSpec, ...] = (
    ReportSpec(
        id="cftc_tff_fut",
        dataset="gpe5-46if",
        family="tff",
        scope="fut",
        label="Traders in Financial Futures, futures only",
        cohorts=TFF_COHORTS,
        coverage="Financial futures only: equity indices, rates, FX, digital assets.",
    ),
    ReportSpec(
        id="cftc_tff_futopt",
        dataset="yw9f-hn96",
        family="tff",
        scope="futopt",
        label="Traders in Financial Futures, futures + options",
        cohorts=TFF_COHORTS,
        coverage="Financial futures only, including options on a delta basis.",
    ),
    ReportSpec(
        id="cftc_disagg_fut",
        dataset="72hh-3qpy",
        family="disagg",
        scope="fut",
        label="Disaggregated COT, futures only",
        cohorts=DISAGG_COHORTS,
        coverage="Physical commodities only: agriculture and natural resources.",
    ),
    ReportSpec(
        id="cftc_disagg_futopt",
        dataset="kh3c-gbw2",
        family="disagg",
        scope="futopt",
        label="Disaggregated COT, futures + options",
        cohorts=DISAGG_COHORTS,
        coverage="Physical commodities only, including options on a delta basis.",
    ),
    ReportSpec(
        id="cftc_legacy_fut",
        dataset="6dca-aqww",
        family="legacy",
        scope="fut",
        label="Legacy COT, futures only",
        cohorts=LEGACY_COHORTS,
        coverage="Every market, financial and physical. The only history before 2006.",
    ),
    ReportSpec(
        id="cftc_legacy_futopt",
        dataset="jun7-fc8e",
        family="legacy",
        scope="futopt",
        label="Legacy COT, futures + options",
        cohorts=LEGACY_COHORTS,
        coverage="Every market, including options on a delta basis. Starts 1995.",
    ),
    ReportSpec(
        id="cftc_supp_cit",
        dataset="4zgm-a668",
        family="supp",
        scope="combined",
        label="Supplemental COT with index traders",
        cohorts=SUPP_COHORTS,
        coverage=(
            "13 agricultural markets only, and COMBINED BASIS ONLY -- there is "
            "no futures-only counterpart, so these figures will not reconcile "
            "against any *_fut dataset."
        ),
        has_conc=False,  # supplemental publishes no concentration columns
        meta_overrides=SUPP_META_OVERRIDES,
    ),
)

BY_ID: dict[str, ReportSpec] = {s.id: s for s in SPECS}


# --------------------------------------------------------------------------- #
# Tolerances, all measured over the full 1,046,469-row corpus
# --------------------------------------------------------------------------- #
# CFTC rounds published cohort figures independently, so the accounting
# identities hold to within a few contracts rather than exactly. Both of these
# were originally written as exact checks in this repo and both were wrong:
#
#   sum(net) == 0                 fails on 2,292 of 518,741 futures-only
#                                 market-weeks, and on 52.6% of supplemental
#   OI - sum(long) - sum(spread)  reaches +/-4 over full history
#
# A tolerance of 3 on net fires on 0 of 1,046,369 triples (corpus max is 3).
# A tolerance of 4 on the residual is the corpus max. Both are set one wider
# than the observed maximum and are reported as WARNINGS, not failures: a real
# break should be visible on the integrity board, not silently abort ingest.
NET_ZERO_TOL = 4.0
OI_RESIDUAL_TOL = 5.0

#: Tolerance on a FLOW (a week-over-week change), which is a difference of two
#: independently rounded levels and so carries twice the level rounding.
FLOW_TOL = 2.0 * NET_ZERO_TOL

#: Both tolerances above are set one contract wider than the observed corpus
#: maximum (3 for net, 4 for the residual). That makes them forward-looking
#: tripwires rather than tests that "passed": nothing in the current data can
#: breach them by construction, so a zero count is a statement about calibration,
#: not a finding. Say so wherever the count is displayed.

# The published change_* columns reconcile against our own diff to +/-2, for
# the same rounding reason -- except across the July 2008 trader
# reclassification and a handful of genuine revisions, where they differ by up
# to 323,944 contracts because the published change compares a new
# classification against an old-classification prior level.
CHANGE_FIELD_TOL = 2.0

# THE RESIDUAL IS ROUNDING NOISE, NOT AN UNPUBLISHED SPREAD.
#
# This repo previously rendered "residual = unpublished non-reportable spread"
# on screen. That explanation is wrong and the evidence is decisive:
#
#   - it is NEGATIVE in 822 of 1,134 nonzero legacy_fut rows and 1,188 of 1,616
#     in tff_fut. A non-negative missing spread can only make
#     OI - long - spread MORE positive, never negative.
#   - it is confined to {-4..+4} while open interest spans 724 to 25,702,684
#     contracts. A real position bucket would scale with the market.
#   - corr(residual, open interest) is between -0.02 and +0.03, i.e. none.
#   - it is EXACTLY ZERO on all 184,229 disagg_fut rows -- the commodity
#     dataset where small-trader calendar spreads would be most visible.
#
# So it is integer rounding in the publisher, and the honest caption says so.
RESIDUAL_EXPLANATION = (
    "Integer rounding in the published figures. CFTC rounds each cohort column "
    "independently, so the columns reconcile to within a few contracts rather "
    "than exactly. It is NOT an unpublished non-reportable spread, and four "
    "measurements say so: the residual is frequently NEGATIVE -- 74% of nonzero "
    "rows in the TFF futures-only report and about half in the combined reports "
    "-- while a missing non-negative spread could only ever push it positive; it "
    "never leaves the range -4..+4 even on markets carrying 25 million contracts "
    "of open interest, where a real position bucket would scale with the market; "
    "its correlation with open interest is under 0.03 in absolute value; and it "
    "is exactly zero on all 184,229 rows of the disaggregated futures-only "
    "report, the dataset where small-trader calendar spreads would be most "
    "visible."
)


def spec_for(source_id: str) -> ReportSpec:
    if source_id not in BY_ID:
        raise KeyError(f"unknown CFTC spec {source_id!r}; known: {sorted(BY_ID)}")
    return BY_ID[source_id]
