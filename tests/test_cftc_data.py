"""The accounting layer: does the ingested CFTC data mean what the app says it means.

This module tests the things that do not raise when they are wrong. A field map
that points at the old-crop bucket, a residual explained as a hidden position
bucket, a dedupe keyed on a name that two markets share -- each one leaves the
pipeline running and the charts plausible. So each assertion here is either an
accounting identity on real published figures or a regression pin on a
correction this repo has already had to make once.

Three corrections are pinned deliberately, and every one of them started life as
an exact-equality assertion or a confident caption:

  sum(net) == 0                    is FALSE on 458 tff and 295 supplemental
                                   market-weeks in the fixture. Tolerance, not
                                   equality -- see cftc_spec.NET_ZERO_TOL.
  the open-interest residual       is NEGATIVE on 251 tff market-weeks, which is
                                   arithmetically impossible if it were an
                                   unpublished non-negative spread. It is rounding.
  (name, date) identifies a market is FALSE: on 1999-06-22 codes 209741 and
                                   209742 publish the identical name while
                                   holding 22,998 and 537 contracts.

What this module does NOT test: anything about display (tests/test_display_rules)
or about the causal-ranking maths (tests/test_metrics). It also never fetches --
every frame comes from the committed slice described in tests/fixtures/build.py.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import sources
from lib import cftc_spec, metrics, segments, store, universe
from lib.cftc_spec import NET_ZERO_TOL, OI_RESIDUAL_TOL
from lib.schema import TableSchema
from sources import cftc as cftc_source
from sources.base import Source

PRIMARY_KEY = ("report_date", "market_code", "cohort")

# Codes the assertions below name explicitly. Restated here rather than imported
# from conftest so this module has no dependency on the fixture plumbing; what
# each one is in the slice for is documented in tests/fixtures/build.py.
NASDAQ_BIG = "209741"  # legacy: shares one name with 209742 on 1999-06-22
NASDAQ_EMINI = "209742"  # a SUPERSEDED_BY leg of 20974+
GOLD = "088691"  # legacy from 1986-01-15: pre-2002 non-weekly cadence
UNIT_BREAK_DATE = "2023-05-02"  # 20974+ re-based from $100 to $20 per point


# --------------------------------------------------------------------------- #
# The primary key
# --------------------------------------------------------------------------- #
def test_market_code_date_cohort_is_the_primary_key(cftc_frame):
    """Not (name, date): see test_names_are_not_a_safe_key below."""
    source_id, df = cftc_frame
    assert sources.get(source_id).key == PRIMARY_KEY, "the declared dedupe key moved"
    dups = df[df.duplicated(subset=list(PRIMARY_KEY), keep=False)]
    assert dups.empty, f"{source_id}: {len(dups)} rows share a primary key"


def test_every_market_week_carries_the_full_cohort_set(cftc_frame, market_weeks):
    """A missing cohort row would make every sum in the app quietly incomplete,
    and the net-zero test below would then pass for the wrong reason."""
    source_id, df = cftc_frame
    expected = {c.id for c in cftc_spec.spec_for(source_id).cohorts}
    assert set(df["cohort"].astype(str)) == expected
    assert (market_weeks(df)["cohorts"] == len(expected)).all()


# --------------------------------------------------------------------------- #
# The accounting identities -- to a tolerance, never to equality
# --------------------------------------------------------------------------- #
def test_net_sums_to_zero_within_tolerance(cftc_frame):
    """Every long is somebody's short, so the cohort nets must cancel.

    Computed through metrics.net on the nullable Int32 columns rather than on
    floats, because that is the path a panel takes and a silent NA there would
    make the sum look balanced by dropping a cohort.
    """
    source_id, df = cftc_frame
    per_row = metrics.net(df["long"], df["short"])
    assert per_row.notna().all(), f"{source_id}: net is null somewhere -- a position column is null"
    worst = per_row.groupby([df["market_code"], df["report_date"]], observed=True).sum().abs().max()
    assert worst <= NET_ZERO_TOL, f"{source_id}: worst |sum(net)| is {worst}"


def test_long_side_equals_short_side_within_tolerance(cftc_frame, market_weeks):
    """The same identity read off the sides instead of the nets. Algebraically the
    same number; a different code path, and the one a cross-market screen uses."""
    source_id, df = cftc_frame
    mw = market_weeks(df)
    gap = (mw["long_sum"] - mw["short_sum"]).abs()
    assert gap.max() <= NET_ZERO_TOL, f"{source_id}: sides differ by up to {gap.max()}"


def test_net_zero_is_not_exact_so_an_equality_assertion_would_be_wrong(cftc_frames, market_weeks):
    """REGRESSION. This repo asserted sum(net) == 0 exactly and it fired on real data.

    CFTC rounds each published column independently, so the identity holds to a
    few contracts. The fixture keeps live counterexamples on purpose (tff and
    supplemental), and this test fails if a future slice loses them -- because
    then the tolerance above would be untested and could quietly be tightened
    back to equality.
    """
    violations = {}
    for sid, df in cftc_frames.items():
        mw = market_weeks(df)
        violations[sid] = int((mw["long_sum"] - mw["short_sum"]).ne(0).sum())
    assert sum(violations.values()) > 0, f"fixture no longer exercises the tolerance: {violations}"
    # Named explicitly because it is the worst-rounding family: 52.6% of
    # supplemental market-weeks fail exact equality across the full corpus.
    assert violations["cftc_supp_cit"] > 0, "supplemental rounds worst; it must stay in the slice"


def test_open_interest_reconciles_against_the_sides_and_spreads(cftc_frame, market_weeks):
    """open_interest == sum(long) + sum(spread), to OI_RESIDUAL_TOL.

    A spread position is long one expiry and short another, so it belongs to
    neither side and is counted once in open interest. spread is NULL for the
    cohorts whose report has no spread column (legacy comm, disagg prod_merc,
    nonrept everywhere) and a null there means "no such column", which
    contributes zero contracts.
    """
    source_id, df = cftc_frame
    mw = market_weeks(df)
    resid = mw["open_interest"] - mw["long_sum"] - mw["spread_sum"].fillna(0)
    worst = resid.abs().max()
    assert worst <= OI_RESIDUAL_TOL, f"{source_id}: residual reaches {worst}"


def test_the_residual_goes_negative_which_disproves_the_hidden_spread_reading(cftc_frames, market_weeks):
    """REGRESSION, and the most important test in this file.

    An earlier version of this repo rendered "residual = unpublished
    non-reportable spread" next to the number. That is wrong, and the decisive
    evidence is the SIGN: a missing non-negative position bucket can only make
    open_interest - long - spread MORE positive. It is negative on 251 tff
    market-weeks and 121 supplemental ones, so no such bucket exists and the
    leftover is integer rounding in the publisher.

    Three supporting properties are asserted with it, because the sign alone
    invites the reply "then it is a signed adjustment": the residual is bounded by
    a few contracts while open interest across the slice spans 1,484x, it is
    uncorrelated with open interest, and it is exactly zero on every
    disaggregated row -- the family where small-trader calendar spreads would be
    most visible.
    """
    pooled = []
    for sid, df in cftc_frames.items():
        mw = market_weeks(df)
        resid = (mw["open_interest"] - mw["long_sum"] - mw["spread_sum"].fillna(0)).astype("float64")
        assert resid.abs().max() <= OI_RESIDUAL_TOL, f"{sid}: residual reaches {resid.abs().max()}"
        if sid == "cftc_disagg_fut":
            assert (resid == 0).all(), "disagg is the control: its residual is exactly zero"
        pooled.append(pd.DataFrame({"oi": mw["open_interest"].astype("float64"), "resid": resid}))

    p = pd.concat(pooled, ignore_index=True)
    negative, positive = int((p["resid"] < 0).sum()), int((p["resid"] > 0).sum())
    assert negative > 0, "the correction is no longer exercised by the fixture"
    # Roughly symmetric, which is what rounding looks like and what a missing
    # non-negative bucket cannot look like. Loose bounds: a sign test, not a fit.
    assert 0.3 < negative / (negative + positive) < 0.7, f"lopsided: {negative} neg / {positive} pos"
    # Does not scale with the market, and is uncorrelated with it.
    assert p["oi"].max() / p["oi"].min() > 100
    assert abs(p["oi"].corr(p["resid"])) < 0.1

    # The caption the app renders must keep saying so, verbatim.
    assert "NOT an unpublished non-reportable spread" in cftc_spec.RESIDUAL_EXPLANATION


def test_market_week_columns_are_constant_across_cohorts(cftc_frame):
    """open_interest, traders_total and the concentration ratios are market-week
    level and repeated on every cohort row, so aggregating them with .sum()
    multiplies them by the cohort count -- 3x in legacy, 5x in tff -- and the
    result is a plausible number nobody questions. Pinned so .first() stays
    justified rather than folkloric."""
    source_id, df = cftc_frame
    level_cols = ["open_interest", "traders_total", "oi_change_published", *cftc_source._CONC_MAP]
    g = df.groupby(["market_code", "report_date"], observed=True)
    for col in level_cols:
        assert g[col].nunique(dropna=True).max() <= 1, f"{source_id}: {col} varies within a market-week"


# --------------------------------------------------------------------------- #
# Keys, names and breaks in the data itself
# --------------------------------------------------------------------------- #
def test_names_are_not_a_safe_key(legacy_fut):
    """REGRESSION. Two disproofs, both from the fixture.

    (1) (market_full, report_date) is not unique: on 1999-06-22 codes 209741 and
        209742 both publish "NASDAQ-100 STOCK INDEX - INTERNATIONAL MONETARY
        MARKET" while holding 22,998 and 537 contracts. A name-keyed dedupe does
        not merge duplicates there, it deletes a real market.
    (2) One code carries many names over time -- 209742 has four inside 18
        months -- so a name-keyed series also splits one market into several.
    """
    one = legacy_fut[legacy_fut["cohort"] == "nonrept"]
    collisions = one.groupby(["market_full", "report_date"], observed=True)["market_code"].nunique()
    clash = collisions[collisions > 1]
    assert not clash.empty, "fixture lost the name collision (legacy 209741/209742, 1999-06-22)"

    name, date = clash.index[0]
    rows = one[(one["market_full"] == name) & (one["report_date"] == date)]
    assert set(rows["market_code"].astype(str)) == {NASDAQ_BIG, NASDAQ_EMINI}
    assert rows["open_interest"].nunique() == 2, "the colliding rows are two different markets"

    names_per_code = one.groupby("market_code", observed=True)["market_full"].nunique()
    assert names_per_code.max() >= 2, "fixture lost the renamed-code case"


def test_the_2023_rebasing_is_a_unit_break_not_a_positioning_move(nasdaq_consolidated):
    """20974+ open interest goes 49,531 -> 255,954 on 2023-05-02 because the
    contract was re-based from $100 to $20 per index point. Positioning barely
    moves: leveraged funds' net stays near -14% of open interest across the seam.
    lib.segments must therefore cut the history there, and this test pins both
    the fixture case and the cut."""
    lev = nasdaq_consolidated[nasdaq_consolidated["cohort"] == "lev_money"].reset_index(drop=True)
    pre = lev[lev["report_date"] < UNIT_BREAK_DATE].iloc[-1]
    post = lev[lev["report_date"] == UNIT_BREAK_DATE].iloc[0]

    assert (int(pre["open_interest"]), int(post["open_interest"])) == (49_531, 255_954)
    assert int(post["open_interest"]) / int(pre["open_interest"]) > 5
    assert segments.unit_signature(pre["contract_units"]) != segments.unit_signature(post["contract_units"])

    share = [100 * (int(r["long"]) - int(r["short"])) / int(r["open_interest"]) for r in (pre, post)]
    assert abs(share[1] - share[0]) < 2, f"composition moved too: {share}"

    seg = segments.segment_ids(lev["report_date"], lev["contract_units"])
    assert seg.nunique() == 2
    assert lev.loc[seg.ne(seg.shift()).fillna(True), "report_date"].tolist()[1] == UNIT_BREAK_DATE


def test_legacy_history_is_not_weekly(gold_legacy):
    """Legacy runs on mixed weekdays before 2002-01-08 with gaps of 11-18 days,
    and even after it skips weeks. 088691 keeps that in the fixture, so anything
    that reads a one-row diff as a one-week change is testably wrong here."""
    dates = pd.to_datetime(sorted(gold_legacy["report_date"].unique()))
    gaps = pd.Series(np.diff(dates.values).astype("timedelta64[D]").astype(int))
    assert (gaps != 7).sum() > 100, "fixture lost the irregular cadence"
    assert gaps.max() >= 13
    early = dates[dates < segments.LEGACY_WEEKLY_FROM]
    assert len(set(early.dayofweek)) > 1, "pre-2002 reports land on one weekday only"


def test_null_trader_counts_coexist_with_nonzero_positions(tff_fut):
    """Fact D, pinned. traders_long is null on 7.9% of asset-manager rows and
    11.1% of dealer rows in this slice WHILE the position is nonzero, so a table
    ranked on a per-trader measure silently omits them. The coverage fraction is
    computed and reported in the failure message for the same reason a panel has
    to show it: the omission is invisible in the ranking itself."""
    reportable = tff_fut[tff_fut["cohort"] != "nonrept"]
    held = reportable[reportable["long"] > 0]
    blind = held["traders_long"].isna()
    coverage = f"{int((~blind).sum())} of {len(held)} rows have a trader count"
    assert blind.sum() > 0, f"fixture lost the null-trader case (BITCOIN 133741): {coverage}"
    assert blind.mean() < 0.5, f"implausibly sparse -- check the field map: {coverage}"
    assert metrics.avg_position_per_trader(
        held.loc[blind, "long"], held.loc[blind, "traders_long"]
    ).isna().all(), "a null trader count must give NaN, never a division by zero"

    # nonrept has no traders_* column at all in any family -- a structural
    # absence, not sparse data, and it must not be mistaken for the above.
    assert tff_fut[tff_fut["cohort"] == "nonrept"]["traders_long"].isna().all()
    assert all(c.traders_long is None for s in cftc_spec.SPECS for c in s.cohorts if c.id == "nonrept")


def test_spread_is_null_exactly_where_the_report_has_no_spread_column(cftc_frame):
    """Null spread is a property of the report, not missing data, so it must be
    null for ALL rows of those cohorts and never null for the others."""
    source_id, df = cftc_frame
    for cohort in cftc_spec.spec_for(source_id).cohorts:
        rows = df[df["cohort"] == cohort.id]["spread"]
        if cohort.spread is None:
            assert rows.isna().all(), f"{source_id}/{cohort.id}: spread present but spec says none"
        else:
            assert rows.notna().all(), f"{source_id}/{cohort.id}: spread column has holes"


# --------------------------------------------------------------------------- #
# lib/cftc_spec -- the field maps, which fail silently when wrong
# --------------------------------------------------------------------------- #
def test_select_fields_are_unique_and_complete():
    for spec in cftc_spec.SPECS:
        fields = spec.select_fields()
        assert len(fields) == len(set(fields)), f"{spec.id}: duplicate columns in $select"
        for cohort in spec.cohorts:
            assert cohort.long and cohort.short, f"{spec.id}/{cohort.id}: missing a side"
            assert set(cohort.position_fields()) <= set(fields)
        assert spec.meta_field("change_in_open_interest_all") in fields
        has_conc = any(f in fields for f in cftc_spec.CONC_FIELDS)
        assert has_conc is spec.has_conc, f"{spec.id}: has_conc disagrees with $select"


def test_the_misspelled_upstream_names_are_intentional():
    """DO NOT "FIX" THESE. They are the names Socrata actually serves, verified
    against live payloads. Correcting the spelling makes $select return HTTP 400
    at best and, if the corrected name happens to exist, a chart of the wrong
    column at worst. This test exists so a well-meaning spell-check fails loudly."""
    legacy = cftc_spec.BY_ID["cftc_legacy_fut"].select_fields()
    assert "noncomm_postions_spread_all" in legacy  # "postions", missing the i
    assert "noncomm_positions_spread_all" not in legacy

    disagg = cftc_spec.BY_ID["cftc_disagg_fut"].select_fields()
    assert "swap_positions_long_all" in disagg  # one underscore
    assert "swap__positions_short_all" in disagg  # two underscores
    assert "swap__positions_spread_all" in disagg  # two underscores

    supp = cftc_spec.BY_ID["cftc_supp_cit"]
    assert "ncomm_postions_long_all_nocit" in supp.select_fields()  # "postions" again
    # disagg prod_merc/other_rept have no _all variant: the bare name IS
    # all-maturities and *_1 is the old crop year, which would be silently wrong.
    assert "prod_merc_positions_long" in disagg
    assert not any(f.endswith(("_1", "_2")) for f in disagg)


def test_scope_and_coverage_are_declared_consistently():
    for spec in cftc_spec.SPECS:
        assert spec.is_combined == (spec.scope in ("futopt", "combined"))
        assert spec.coverage
    # The supplemental report has no futures-only counterpart, so nothing may
    # present it as a continuation of a *_fut series.
    assert cftc_spec.BY_ID["cftc_supp_cit"].is_combined
    assert cftc_spec.BY_ID["cftc_supp_cit"].meta_field("change_in_open_interest_all") == (
        "change_open_interest_all"
    )
    with pytest.raises(KeyError):
        cftc_spec.spec_for("cftc_tff")  # the pre-split source id must not resolve


# --------------------------------------------------------------------------- #
# sources/cftc.parse -- hand-written raw rows, including the awkward shapes
# --------------------------------------------------------------------------- #
#: One complete tff row, keys written out literally rather than read from the
#: spec, so a spec rename fails here instead of agreeing with itself. Values are
#: strings because Socrata serves every number as a string. It balances:
#: spreads 5,000; both sides 295,000; open interest 300,000.
RAW_TFF = {
    "report_date_as_yyyy_mm_dd": "2026-08-25T00:00:00.000",
    "cftc_contract_market_code": " 20974+ ",  # padded: the '+' codes get trimmed
    "market_and_exchange_names": "NASDAQ-100 Consolidated  - CHICAGO MERCANTILE EXCHANGE",
    "contract_market_name": "NASDAQ-100 Consolidated",
    "contract_units": "(NASDAQ 100 INDEX X $20)",
    "commodity_name": "NASDAQ 100 STOCK INDEX",
    "commodity_group_name": "FINANCIAL INSTRUMENTS",
    "commodity_subgroup_name": "STOCK INDICES",
    "open_interest_all": "300000",
    "change_in_open_interest_all": "-1500",
    "traders_tot_all": "300",
    "dealer_positions_long_all": "10000",
    "dealer_positions_short_all": "50000",
    "dealer_positions_spread_all": "1000",
    "traders_dealer_long_all": "5",
    "asset_mgr_positions_long": "120000",
    "asset_mgr_positions_short": "20000",
    "asset_mgr_positions_spread": "500",
    "lev_money_positions_long": "60000",
    "lev_money_positions_short": "130000",
    "lev_money_positions_spread": "2000",
    "other_rept_positions_long": "30000",
    "other_rept_positions_short": "25000",
    "other_rept_positions_spread": "1500",
    "nonrept_positions_long_all": "75000",
    "nonrept_positions_short_all": "70000",
    "conc_gross_le_4_tdr_long": "22.5",
    "conc_gross_le_8_tdr_long": "35.1",
}

#: One supplemental row in the casing the payload actually uses: MixedCase on 16
#: of 60 columns, and the split runs per side within a cohort. Selecting by the
#: lowercase metadata name works; READING the response by it does not, which is
#: why cftc_api.normalise_keys exists. Balances at 397,000 a side.
RAW_SUPP_MIXEDCASE = {
    "report_date_as_yyyy_mm_dd": "2026-08-25",
    "cftc_contract_market_code": "001602",
    "market_and_exchange_names": "WHEAT-SRW - CHICAGO BOARD OF TRADE",
    "contract_market_name": "WHEAT-SRW",
    "contract_units": "(CONTRACTS OF 5,000 BUSHELS)",
    "open_interest_all": "400000",
    "change_open_interest_all": "2500",
    "CIT_Positions_Long_All": "150000",
    "CIT_Positions_Short_All": "5000",
    "ncomm_postions_long_all_nocit": "90000",
    "ncomm_postions_short_all_nocit": "60000",
    "ncomm_postions_spread_all_nocit": "3000",
    "Comm_Positions_Long_All_NoCIT": "120000",
    "Comm_Positions_Short_All_NoCIT": "300000",
    "NonRept_Positions_Long_All": "37000",
    "NonRept_Positions_Short_All": "32000",
    "conc_gross_le_4_tdr_long": "19.0",  # present in the row, absent from the report
}

TFF = cftc_spec.BY_ID["cftc_tff_fut"]
SUPP = cftc_spec.BY_ID["cftc_supp_cit"]


def _one(rows, spec):
    return cftc_source.parse(rows, spec)


def test_parse_produces_the_tidy_shape():
    out = _one([RAW_TFF], TFF)
    assert list(out.columns) == list(cftc_source.TIDY_COLUMNS)
    assert list(out["cohort"]) == [c.id for c in TFF.cohorts]
    assert out["report_date"].unique().tolist() == ["2026-08-25"]  # timestamp truncated
    assert out["market_code"].unique().tolist() == ["20974+"]  # trimmed, '+' intact
    # Internal whitespace is collapsed: the same market is spelled with one space
    # in one dataset and two in another, and it must match itself.
    assert "  " not in out["market_full"].iloc[0]

    row = out[out["cohort"] == "lev_money"].iloc[0]
    assert (row["long"], row["short"], row["spread"]) == (60000.0, 130000.0, 2000.0)
    assert metrics.net(out["long"], out["short"]).sum() == 0
    assert out["open_interest"].iloc[0] - out["long"].sum() - out["spread"].sum() == 0


def test_parse_turns_an_absent_key_into_nan_and_never_into_zero():
    """Socrata OMITS a key whose value is null, so an absent key means null, not
    zero and not schema drift. Zero would be a lie with consequences:
    avg_position_per_trader would divide by it and traders_total == 0 would read
    as "no firms hold this" on a market with 300,000 contracts open."""
    stripped = {k: v for k, v in RAW_TFF.items() if k not in
                {"traders_dealer_long_all", "traders_tot_all", "conc_gross_le_4_tdr_long",
                 "asset_mgr_positions_spread"}}
    out = _one([stripped], TFF)
    dealer = out[out["cohort"] == "dealer"].iloc[0]
    asset_mgr = out[out["cohort"] == "asset_mgr"].iloc[0]

    absent = {
        "traders_long": dealer["traders_long"],
        "traders_total": dealer["traders_total"],
        "conc_gross_4_long": dealer["conc_gross_4_long"],
        "spread": asset_mgr["spread"],
    }
    for name, value in absent.items():
        # isna() IS the check. The failure mode being pinned is a 0 turning up
        # here, and 0 is not NaN, so this catches it and names the column.
        assert pd.isna(value), f"absent {name} parsed as {value!r}; absent must mean null"

    # Present-but-null is the same condition and must land the same way.
    assert pd.isna(_one([{**RAW_TFF, "traders_tot_all": None}], TFF)["traders_total"].iloc[0])
    # A cohort whose own long or short is absent is dropped rather than zeroed:
    # position columns are never null upstream, so that means the field map is
    # wrong, and dropping the row makes the accounting test fail and say so.
    assert len(_one([{k: v for k, v in RAW_TFF.items() if k != "lev_money_positions_long"}], TFF)) == 4


def test_parse_reads_mixedcase_keys_and_honours_has_conc():
    out = _one([RAW_SUPP_MIXEDCASE], SUPP)
    assert list(out["cohort"]) == [c.id for c in SUPP.cohorts]
    cit = out[out["cohort"] == "cit"].iloc[0]
    comm = out[out["cohort"] == "comm_nocit"].iloc[0]
    assert (cit["long"], cit["short"]) == (150000.0, 5000.0)
    assert (comm["long"], comm["short"]) == (120000.0, 300000.0)  # MixedCase, both sides
    assert pd.isna(comm["spread"])  # commercial ex-index has no spread column
    assert metrics.net(out["long"], out["short"]).sum() == 0
    assert out["open_interest"].iloc[0] - out["long"].sum() - out["spread"].sum() == 0
    assert out["oi_change_published"].iloc[0] == 2500.0  # via the meta override
    # supplemental publishes no concentration columns, so a conc value that turns
    # up in the payload is not carried through.
    assert out[list(cftc_source._CONC_MAP)].isna().all().all()


def test_parse_rejects_impossible_and_unusable_rows():
    # Concentration outside [0, 100]: legacy reports 482.6% top-8 long on a
    # 957-contract market-week. Plotting that faithfully renders a broken input.
    hot = _one([{**RAW_TFF, "conc_gross_le_8_tdr_long": "482.6"}], TFF)
    assert hot["conc_gross_8_long"].isna().all()
    # No open interest, no market code, no date -> nothing to reconcile against.
    for bad in ("open_interest_all", "cftc_contract_market_code", "report_date_as_yyyy_mm_dd"):
        out = _one([{k: v for k, v in RAW_TFF.items() if k != bad}], TFF)
        assert out.empty and list(out.columns) == list(cftc_source.TIDY_COLUMNS)
    assert _one([], TFF).empty


# --------------------------------------------------------------------------- #
# lib/store and the Source contract
# --------------------------------------------------------------------------- #
def _frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        list(rows),
        columns=["report_date", "market_code", "cohort", "long", "traders_long", "open_interest"],
    )


def _dummy_source(**over) -> Source:
    kwargs = dict(
        id="dummy", label="dummy", fetch=lambda since=None: pd.DataFrame(),
        key=PRIMARY_KEY, sort_key=PRIMARY_KEY,
        schema=TableSchema(required=("report_date",)), cadence="never", backfillable=True,
    )
    return Source(**{**kwargs, **over})


def test_upsert_dedupes_on_the_key_with_last_wins(tmp_data_dir):
    """CFTC revises already-published weeks, so ingest re-fetches an overlapping
    window every run. Last-wins is what makes that pick up the revision instead
    of storing the week twice."""
    incoming = _frame(
        ("2026-08-18", "20974+", "dealer", 10, 4, 300000),
        ("2026-08-25", "20974+", "dealer", 20, 5, 300000),
        ("2026-08-25", "20974+", "dealer", 21, 5, 300000),  # same key, later row wins
    )
    first = store.upsert("t", incoming, PRIMARY_KEY, sort_key=PRIMARY_KEY)
    assert first["rows_total"] == 2
    assert store.read("t")["long"].tolist() == [10, 21]

    revision = _frame(("2026-08-25", "20974+", "dealer", 99, 5, 300000))
    second = store.upsert("t", revision, PRIMARY_KEY, sort_key=PRIMARY_KEY)
    assert (second["rows_new"], second["rows_revised"], second["rows_total"]) == (0, 1, 2)
    assert store.read("t")["long"].tolist() == [10, 99]


def test_sort_key_must_lead_with_the_date_column():
    """Sorting code-first makes a weekly append rewrite the middle of every
    market's run: ~2.54 MB of .git growth per commit against ~5.3 KB date-first,
    a factor of ~480 (measured, see lib/store). Asserted rather than trusted
    because the tuple is easy to reorder and nothing else would complain."""
    with pytest.raises(ValueError, match="lead with the date column"):
        _dummy_source(sort_key=("market_code", "report_date", "cohort"))
    with pytest.raises(ValueError):
        _dummy_source(key=())
    assert _dummy_source().sort_key[0] == "report_date"
    for src in sources.all_sources():
        assert src.sort_key[0].endswith("date"), f"{src.id} sorts on {src.sort_key[0]} first"


def test_tighten_never_turns_a_null_trader_count_into_zero(tmp_data_dir):
    """A null per-cohort trader count co-occurs with a NONZERO position, so 0
    would turn "not published" into "no firms hold this" and manufacture a
    division by zero downstream. Also pinned: counts narrow to nullable Int32 and
    not to float32, which only represents integers exactly to 2**24 while open
    interest reaches 25,702,684."""
    frame = _frame(
        ("2026-08-18", "20974+", "dealer", 10, 4, 25_702_684),
        ("2026-08-25", "20974+", "dealer", 20, None, 25_702_684),
    )
    tight = store._tighten(frame)
    assert tight["traders_long"].tolist()[0] == 4
    assert pd.isna(tight["traders_long"].iloc[1])
    assert str(tight["open_interest"].dtype) == "Int32"
    assert int(tight["open_interest"].iloc[0]) == 25_702_684

    # ... and it survives the parquet round trip, which is where it matters.
    store.upsert("t", frame, PRIMARY_KEY, sort_key=PRIMARY_KEY)
    back = store.read("t")
    assert back["traders_long"].isna().tolist() == [False, True]
    # Date columns stay plain strings: pd.to_datetime on an unordered Categorical
    # produces something whose .max() raises, and every caller would have to know.
    assert not isinstance(back["report_date"].dtype, pd.CategoricalDtype)


def test_writing_the_same_frame_twice_produces_identical_bytes(tmp_data_dir):
    """This is what makes "skip the commit when nothing changed" work. Without it
    the ingest job commits ~115 MB of identical-in-content parquet every day.
    Byte-determinism is a pyarrow property, which is why requirements.txt pins it
    exactly rather than with a floor."""
    frame = _frame(
        ("2026-08-18", "20974+", "dealer", 10, 4, 300000),
        ("2026-08-25", "20974+", "dealer", 20, None, 300000),
    )
    first = store.upsert("a", frame, PRIMARY_KEY, sort_key=PRIMARY_KEY)
    again = store.upsert("a", frame, PRIMARY_KEY, sort_key=PRIMARY_KEY)
    other = store.upsert("b", frame, PRIMARY_KEY, sort_key=PRIMARY_KEY)

    assert first["changed"] is True
    assert again["changed"] is False, "a no-op rewrite must be byte-identical"

    def digest(source_id: str) -> str:
        return hashlib.sha256(store.path_for(source_id).read_bytes()).hexdigest()

    assert digest("a") == digest("b")
    assert first["bytes"] == other["bytes"]


# --------------------------------------------------------------------------- #
# lib/universe -- the supersession map that stops a double count
# --------------------------------------------------------------------------- #
def test_superseded_by_covers_all_six_legs():
    """REGRESSION. 124608 (MICRO E-MINI DJIA) was missing from this map, and its
    absence double-counts DJIA exposure: the Micro leg is already inside 12460+
    at 10:1, so showing both invites adding them. Consolidated == E-mini +
    Micro/10 is verified against the published figures to under one contract, so
    every Consolidated code must have exactly its two legs listed."""
    assert set(universe.SUPERSEDED_BY) == {"13874A", "13874U", "209742", "209747", "124603", "124608"}
    assert universe.SUPERSEDED_BY["124608"] == "12460+"
    assert set(universe.SUPERSEDED_BY.values()) == set(universe.CONSOLIDATED)
    per_parent = pd.Series(list(universe.SUPERSEDED_BY.values())).value_counts()
    assert (per_parent == 2).all(), f"a Consolidated code is missing a leg: {per_parent.to_dict()}"


def test_a_superseded_leg_never_reaches_core(tff_fut):
    """Tiering is a display decision, but this part of it is an arithmetic one:
    the E-mini NDX leg is inside 20974+, so putting both on the default screen is
    an invitation to add them. The second half of the test isolates supersession
    as the cause -- with the flag cleared the same row is liquid enough for
    CORE, so it was excluded for the right reason and not by accident of size."""
    summary = universe.market_summary(tff_fut)
    tiered = universe.assign_tiers(summary)
    assert not (tiered["tier"].eq("CORE") & tiered["superseded_by"].notna()).any()

    leg = tiered[tiered["market_code"] == NASDAQ_EMINI].iloc[0]
    assert leg["superseded_by"] == "20974+" and leg["tier"] != "CORE"
    assert leg["oi_window_max"] >= universe.OI_CORE  # liquid enough, still excluded

    unsuperseded = summary.copy()
    unsuperseded.loc[unsuperseded["market_code"] == NASDAQ_EMINI, "superseded_by"] = None
    promoted = universe.assign_tiers(unsuperseded)
    assert promoted.loc[promoted["market_code"] == NASDAQ_EMINI, "tier"].iloc[0] == "CORE"

    # The Consolidated series itself is always on the default screen, and the
    # cost of preferring it -- 174 comparable weeks against 1,055 reports on the
    # single-leg code -- is text the board has to render rather than decide.
    assert tiered.loc[tiered["market_code"] == "20974+", "tier"].iloc[0] == "CORE"
    assert "2023-05-02" in universe.CONSOLIDATED_TRADEOFF


def test_market_summary_labels_a_code_with_its_latest_name(legacy_fut):
    """26-30% of codes have been renamed, one carries up to eight names, and the
    fixture's 209742 carries four. The selector must show the current one."""
    summary = universe.market_summary(legacy_fut)
    row = summary[summary["market_code"] == NASDAQ_EMINI].iloc[0]
    latest = legacy_fut[legacy_fut["market_code"].astype(str) == NASDAQ_EMINI].sort_values("report_date")
    assert row["market_full"] == latest["market_full"].iloc[-1]
    assert row["n_reports"] == latest["report_date"].nunique()
    gold = summary[summary["market_code"] == GOLD].iloc[0]
    assert str(gold["first_report"].date()) == "1986-01-15"
