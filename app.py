"""Watchboard -- alternative and government data that TradingView either does not
carry or does not display usefully.

    streamlit run app.py

Reads only from data/*.parquet. It never fetches, so it works offline and it is
impossible for a chart to show a number that is not in the committed dataset.
Ingest is a separate job (scripts/ingest.py, run by CI).

Two display rules are enforced in code rather than left to discipline:

  1. NO DUAL AXES. Two series on one panel with two y-scales lets the author pick
     the scaling that makes the correlation look how they want. Stacked subplots
     with independent panels instead.
  2. SIGNED QUANTITIES IN THEIR OWN UNITS. Net position is offered in contracts
     and as a share of open interest, and its CHANGE is never offered as a
     percent -- because -100,640 -> -96,727 is "+3,913 contracts, less short",
     while "+3.89%" reads as growth.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import metrics, store
from sources import REGISTRY, all_sources
from sources.cftc_tff import COHORTS

st.set_page_config(page_title="Watchboard", page_icon="◧", layout="wide")

ACCENT = "#2E6F9E"
MUTED = "#9AA5B1"
WARN = "#B4623A"


def caveat_block(source_id: str) -> None:
    src = REGISTRY[source_id]
    with st.expander("What these numbers can and cannot tell you", expanded=False):
        st.caption(f"**Source.** {src.provenance}")
        st.caption(f"**Cadence.** {src.cadence}")
        if not src.backfillable:
            st.caption(
                ":warning: **Snapshot-only.** No history is retrievable. The series "
                "begins when the job began; missed runs are permanent gaps."
            )
        for c in src.caveats:
            st.caption(f"- {c}")


def freshness(dates: pd.Series, label: str) -> None:
    latest = pd.to_datetime(dates).max()
    age = (pd.Timestamp.today().normalize() - latest.normalize()).days
    msg = f"Latest {label}: **{latest.date()}** ({age} days ago)"
    (st.warning if age > 10 else st.caption)(msg)


# --------------------------------------------------------------------------- #
# CFTC positioning
# --------------------------------------------------------------------------- #
def panel_cftc() -> None:
    df = store.read("cftc_tff")
    if df.empty:
        st.info("No data yet. Run `python -m scripts.ingest --source cftc_tff`.")
        return

    df["report_date"] = pd.to_datetime(df["report_date"])
    markets = sorted(df["market"].unique())

    c1, c2, c3 = st.columns([2, 2, 1.4])
    market = c1.selectbox("Market", markets, index=markets.index("NDX (E-mini, $20/pt)") if "NDX (E-mini, $20/pt)" in markets else 0)
    labels = {k: v[1] for k, v in COHORTS.items()}
    picked = c2.multiselect(
        "Cohorts (summed, the way the circulating charts build them)",
        options=list(labels),
        default=["asset_mgr", "lev_money"],
        format_func=lambda k: labels[k],
    )
    measure = c3.radio("Measure", ["net", "gross"], horizontal=True,
                       help="net = which way they lean (can change sign). "
                            "gross = how big the book is, i.e. how much there is "
                            "to unwind. gross is a fragility measure.")

    if not picked:
        st.info("Pick at least one cohort.")
        return

    sub = df[(df["market"] == market) & (df["cohort"].isin(picked))]
    oi = sub.groupby("report_date")["open_interest"].first()
    agg = sub.groupby("report_date")[["long", "short"]].sum()

    series = (metrics.net if measure == "net" else metrics.gross)(agg["long"], agg["short"])
    series.name = measure
    share = metrics.share_of(series, oi)
    pctile = metrics.expanding_percentile(share)

    unit = st.radio("Unit", ["% of open interest", "contracts"], horizontal=True)
    plotted = share if unit.startswith("%") else series

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        row_heights=[0.42, 0.3, 0.28],
        subplot_titles=(
            f"{labels_join(picked, labels)} — {measure}, {unit}",
            "Expanding percentile of that level (no look-ahead)",
            "Open interest (contracts) — the column that says whether contracts were CREATED",
        ),
    )
    fig.add_trace(go.Scatter(x=plotted.index, y=plotted, name=measure,
                             line=dict(color=ACCENT, width=1.6)), row=1, col=1)
    if measure == "net" and unit.startswith("%"):
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pctile.index, y=pctile, name="percentile",
                             line=dict(color=WARN, width=1.4)), row=2, col=1)
    for level in (10, 90):
        fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)
    fig.add_trace(go.Scatter(x=oi.index, y=oi, name="open interest",
                             line=dict(color=MUTED, width=1.3)), row=3, col=1)

    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text="percentile", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="contracts", row=3, col=1)
    fig.update_layout(height=760, hovermode="x unified", showlegend=False,
                      margin=dict(l=70, r=30, t=60, b=40))
    st.plotly_chart(fig, width="stretch")

    freshness(sub["report_date"], "report (Tuesday positions, published Friday)")

    # ---- current reading -------------------------------------------------- #
    latest = plotted.index.max()
    st.subheader(f"Reading for {latest.date()}")
    prev2 = plotted.index[-3] if len(plotted) >= 3 else None
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{measure} ({unit})", f"{plotted.loc[latest]:,.2f}")
    m2.metric("expanding percentile", f"{pctile.loc[latest]:.1f}th",
              help=f"ranked against {pctile.notna().sum():,} prior weekly reports")
    if prev2 is not None:
        delta = plotted.loc[latest] - plotted.loc[prev2]
        m3.metric("2-week change",
                  f"{delta:+,.2f} {'pp' if unit.startswith('%') else 'contracts'}",
                  help="Stated as an absolute change on purpose: a percent change "
                       "on a sign-changing quantity inverts its sense.")

    # ---- every cohort, and the identity check ----------------------------- #
    st.subheader(f"All cohorts on {latest.date()} — and the two invariants")
    week = df[(df["market"] == market) & (df["report_date"] == latest)].copy()
    week["net"] = week["long"] - week["short"]
    week["gross"] = week["long"] + week["short"]
    table = (week[["cohort_label", "long", "short", "spread", "net", "gross"]]
             .rename(columns={"cohort_label": "cohort"})
             .sort_values("net", ascending=False)
             .reset_index(drop=True))
    st.dataframe(table.style.format({c: "{:,.0f}" for c in
                 ["long", "short", "spread", "net", "gross"]}, na_rep="—"),
                 width="stretch", hide_index=True)

    total_oi = float(week["open_interest"].iloc[0])
    sum_net = float(week["net"].sum())
    sum_long, sum_short = float(week["long"].sum()), float(week["short"].sum())
    sum_spread = float(week["spread"].sum(skipna=True))
    residual = total_oi - sum_long - sum_spread

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Invariant 1 — every net position sums to zero**")
        st.code(f"sum(net) = {sum_net:,.0f}", language="text")
        (st.success if abs(sum_net) < 1 else st.error)(
            "Holds exactly. Every contract has two sides, so a market-wide net "
            "of anything other than zero would mean a contract with one side."
            if abs(sum_net) < 1 else f"BROKEN by {sum_net:,.0f} contracts — investigate before using this week."
        )
    with i2:
        st.markdown("**Invariant 2 — longs, shorts and spreads reconcile to open interest**")
        st.code(
            f"sum(long)   = {sum_long:>12,.0f}\n"
            f"sum(short)  = {sum_short:>12,.0f}\n"
            f"sum(spread) = {sum_spread:>12,.0f}\n"
            f"open int.   = {total_oi:>12,.0f}\n"
            f"residual    = {residual:>12,.0f}   (unpublished non-reportable spread)",
            language="text",
        )
        st.caption(
            "Note it is **not** `sum(long) == open interest`. A spread position is "
            "long one expiry and short another, so it sits in neither column and "
            "gets its own. The net identity is unaffected because a spread adds "
            "equally to both sides and cancels."
        )

    caveat_block("cftc_tff")


def labels_join(picked: list[str], labels: dict[str, str]) -> str:
    return " + ".join(labels[k] for k in picked)


# --------------------------------------------------------------------------- #
# Token prices
# --------------------------------------------------------------------------- #
def panel_token_prices() -> None:
    df = store.read("openrouter_pricing")
    if df.empty:
        st.info("No data yet. Run `python -m scripts.ingest --source openrouter_pricing`.")
        return

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    n_snapshots = df["snapshot_date"].nunique()

    if n_snapshots < 2:
        st.warning(
            f"**{n_snapshots} snapshot so far.** This endpoint serves only today's "
            "prices — there is no backfill, so the time series starts now and "
            "grows one run at a time. Today's cross-section is below; the trend "
            "panel becomes meaningful after a few weeks of ingests."
        )

    latest = df["snapshot_date"].max()
    today = df[df["snapshot_date"] == latest].copy()

    st.subheader(f"Frontier price, {latest.date()} — cheapest model per vendor")
    vendors = st.multiselect(
        "Vendors", sorted(today["vendor"].dropna().unique()),
        default=[v for v in ("anthropic", "openai", "google", "meta-llama", "deepseek")
                 if v in set(today["vendor"])],
    )
    view = today[today["vendor"].isin(vendors)] if vendors else today
    priced = view[view["usd_per_mtok_completion"].notna() & (view["usd_per_mtok_completion"] > 0)]

    if priced.empty:
        st.info("No priced models for that selection.")
        return

    cheapest = (priced.sort_values("usd_per_mtok_completion")
                      .groupby("vendor", as_index=False).first())
    fig = go.Figure(go.Bar(
        x=cheapest["usd_per_mtok_completion"], y=cheapest["vendor"], orientation="h",
        marker_color=ACCENT, text=cheapest["model_id"], textposition="outside",
        hovertemplate="%{y}<br>%{text}<br>$%{x:.3f} / M output tokens<extra></extra>",
    ))
    fig.update_layout(height=60 + 42 * len(cheapest), xaxis_title="USD per million output tokens",
                      margin=dict(l=110, r=180, t=20, b=45))
    st.plotly_chart(fig, width="stretch")

    if n_snapshots >= 2:
        st.subheader("Price over time")
        tracked = st.multiselect(
            "Models", sorted(df["model_id"].dropna().unique()),
            default=list(cheapest["model_id"].head(4)),
        )
        hist = df[df["model_id"].isin(tracked)]
        fig2 = go.Figure()
        for mid, grp in hist.groupby("model_id"):
            grp = grp.sort_values("snapshot_date")
            fig2.add_trace(go.Scatter(x=grp["snapshot_date"], y=grp["usd_per_mtok_completion"],
                                      mode="lines+markers", name=str(mid)))
        fig2.update_layout(height=420, yaxis_title="USD per million output tokens",
                           hovermode="x unified", margin=dict(l=70, r=30, t=20, b=40))
        st.plotly_chart(fig2, width="stretch")

    st.info(
        "**What this is half of.** Inference revenue = tokens × price per token. "
        "This panel is the second factor only. If volume grows more slowly than "
        "price falls, revenue shrinks while 'token consumption is booming' stays "
        "literally true — so a falling line here is not by itself bearish or bullish."
    )
    freshness(df["snapshot_date"], "snapshot")
    caveat_block("openrouter_pricing")


# --------------------------------------------------------------------------- #
PANELS = {
    "CFTC positioning": panel_cftc,
    "Token prices": panel_token_prices,
}

st.sidebar.title("◧ Watchboard")
st.sidebar.caption(
    "Alternative and government data that TradingView does not carry or does not "
    "display usefully."
)
choice = st.sidebar.radio("Board", list(PANELS))
st.sidebar.divider()
st.sidebar.caption("**Sources ingested**")
for s in all_sources():
    stored = store.read(s.id)
    rows = f"{len(stored):,} rows" if not stored.empty else "empty"
    st.sidebar.caption(f"`{s.id}` — {rows}" + ("" if s.backfillable else "  ·  snapshot-only"))

st.title(choice)
PANELS[choice]()
