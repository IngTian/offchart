"""Token prices -- the price factor of "inference revenue = tokens x price".

ANSWERS: where the posted per-token price sits across vendors today, and -- once
two or more snapshots exist -- which way a named model's price moves and how fast.

REFUSES: whether inference revenue is growing (revenue is a product of two
factors and this source measures one, so a falling line is neither bearish nor
bullish on its own), and LEVEL at all. OpenRouter routes for independent
developers and small shops and carries almost none of the enterprise or
first-party OpenAI / Anthropic / Google traffic, so these are retail list prices
on a venue that is not the market. Direction and mix survive that bias; level
does not.

The endpoint serves today's prices with no date parameter and no archive, so the
dataset does not exist before the job started writing it and a missed run is a
permanent hole -- hence an explicit empty state rather than a one-point chart
that reads as a flat price. Of the four display rules only rule 1 bites here (a
price is unsigned, nothing is a share of open interest, and one snapshot cannot
support a causal percentile); the previous version of this panel built its own
plotly figures.
"""
from __future__ import annotations

import sys

import pandas as pd
import streamlit as st

from lib import cache, charts, metrics
from panels import Board

_SOURCE = "openrouter_pricing"

#: Which leg of the price to read. The two move for different reasons -- output
#: tracks decode cost and margin, input tracks prefill and dominates a
#: long-context or RAG bill -- so the choice is the reader's, not a default
#: buried in the code.
_LEGS = {
    "output (completion)": "usd_per_mtok_completion",
    "input (prompt)": "usd_per_mtok_prompt",
}

_UNIT = "USD per million tokens"

#: Pre-selected vendors: the ones a capex or positioning thesis would reference.
#: Intersected with what the snapshot actually contains.
_DEFAULT_VENDORS = ("anthropic", "openai", "google", "deepseek", "meta-llama", "qwen")

#: Bar cap when no vendor filter is set. 55 vendors is a 1,200-pixel chart.
_MAX_BARS = 24

_LEVEL_CAVEAT = (
    "**Direction and mix, never level.** OpenRouter is an aggregator for indie "
    "developers and small shops. It carries almost none of the enterprise or "
    "first-party OpenAI / Anthropic / Google traffic, which is where the volume "
    "is. Every number here is a posted retail price on a venue that is not the "
    "market: useful for which way prices move and which vendors sit where, "
    "useless as an estimate of what inference costs at scale."
)

_HALF_THE_QUESTION = (
    "**This is one factor of two.** Inference revenue = tokens x price per token, "
    "and this source measures price only. If volume grows more slowly than price "
    "falls, revenue shrinks while \"token consumption is booming\" stays literally "
    "true. A falling line here is not by itself bearish, and a flat one is not "
    "bullish."
)


def _app_helpers():
    """app.py's shared renderers, obtained without re-executing app.py.

    Streamlit execs the entry script as `__main__` and does not alias it in
    sys.modules under its filename, so `from app import caveat_block` inside a
    board imports a SECOND copy of app.py and runs its module body again -- a
    second sidebar radio with identical parameters, which raises
    StreamlitDuplicateElementId before either helper is bound. So reuse the
    module already running when it defines them, and keep the documented import
    as the fallback for when app.py is made import-safe.
    """
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "caveat_block"):
        return main.caveat_block, main.freshness
    from app import caveat_block, freshness  # noqa: PLC0415

    return caveat_block, freshness


def _load() -> pd.DataFrame:
    df = cache.read(_SOURCE)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    # Stored as category to keep the parquet small. Widening once here beats
    # threading observed=True through every groupby and pivot below, and stops a
    # filtered frame from dragging all 419 model categories along with it.
    for c in ("model_id", "model_name", "vendor"):
        df[c] = df[c].astype("string")
    return df


def _priced(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows with a real posted price, and the two ways a row fails to have one.

    -1 per token is OpenRouter's sentinel for "decided at route time" (the
    openrouter/auto family), landing here as -1,000,000 per million. 0 is a free
    or promotional tier, and dropping those is not cosmetic: several vendors
    publish a `:free` variant, so a cheapest-per-vendor read that keeps zeros
    reports $0.00 for them and becomes a list of promos.
    """
    v = df[col]
    return df[v.notna() & (v > 0)]


def _per_vendor(priced: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cheapest, median and dearest priced model per vendor.

    The cheapest model is a vendor's FLOOR, not its frontier price: it moves when
    a vendor ships a nano model, a different event from a flagship price cut. So
    the dearest model travels with it everywhere below.
    """
    s = priced.sort_values(col)
    out = s.groupby("vendor", as_index=False).agg(
        cheapest=(col, "first"),
        cheapest_model=("model_id", "first"),
        dearest=(col, "last"),
        dearest_model=("model_id", "last"),
        median=(col, "median"),
        n_priced=(col, "size"),
    )
    return out.sort_values("cheapest").reset_index(drop=True)


def _bar_label(vendor: str, model_id: str) -> str:
    """Vendor plus the model the number came from. charts.ranked_bars has no
    per-bar text argument, and "$0.10" means nothing until you know it came from
    a 4B model, so the identity goes in the label, not into a tooltip.
    """
    short = str(model_id).split("/", 1)[-1]
    return f"{vendor}  ·  {short if len(short) <= 30 else short[:29] + '…'}"


def _cross_section(per_vendor: pd.DataFrame, leg: str) -> None:
    shown = per_vendor.head(_MAX_BARS)
    fig = charts.ranked_bars(
        [_bar_label(r.vendor, r.cheapest_model) for r in shown.itertuples()],
        [float(r.cheapest) for r in shown.itertuples()],
        unit=_UNIT,
        # A price cannot go negative, so the unsigned class is the right one;
        # lib.metrics has no 'price' kind and a panel does not get to add one.
        kind="gross",
        hover=[
            f"dearest priced model {r.dearest_model} at {r.dearest:,.2f}"
            f" · median {r.median:,.2f} · {r.n_priced} priced models"
            for r in shown.itertuples()
        ],
    )
    st.plotly_chart(fig, width="stretch")

    if len(shown) < len(per_vendor):
        st.caption(
            f"Cheapest {len(shown)} of {len(per_vendor)} vendors with a priced "
            "model. Narrow the vendor filter to see the rest."
        )
    st.caption(
        f"Each bar is the cheapest {leg} price a vendor posts -- its floor, not "
        "its frontier. Hover for that vendor's dearest model and median, which is "
        "what tells you whether a low bar is a price cut or a small model."
    )


def _history(df: pd.DataFrame, col: str, leg: str, seeds: list[str]) -> None:
    dates = df["snapshot_date"].drop_duplicates().sort_values()
    wide = df.pivot_table(index="snapshot_date", columns="model_id", values=col)
    wide = wide.reindex(dates)  # a snapshot a model is absent from becomes NaN
    wide = wide.where(wide > 0)  # sentinels and free tiers must not enter a trend

    # charts.Series draws lines only, so a model priced on one snapshot is an
    # invisible trace. Offer only what can actually draw.
    drawable = [m for m in wide.columns if int(wide[m].notna().sum()) >= 2]
    if not drawable:
        st.info(
            "No model has a priced observation on two different snapshots yet, so "
            "there is nothing to draw a line through."
        )
        return

    tracked = st.multiselect(
        "Models",
        sorted(drawable),
        default=[m for m in seeds if m in drawable][:4] or drawable[:4],
        key="tokens_models",
    )
    if not tracked:
        st.caption("Pick at least one model.")
        return

    # NaNs stay in the plotted series on purpose. charts.stacked() sets
    # connectgaps=False, but that only shows a break if the missing snapshot is
    # present as NaN -- dropna() here would draw a straight line across a missed
    # ingest and make a hole look like a smooth decline.
    fig = charts.stacked(
        [
            charts.Panel(
                series=[
                    charts.Series(wide[m], label=str(m), kind="gross")
                    for m in tracked
                ],
                unit=_UNIT,
                title=f"Posted {leg} price per model",
            )
        ],
        height_per_panel=340,
    )
    st.plotly_chart(fig, width="stretch")

    # First-to-last change. A price is unsigned, so a ratio IS meaningful -- the
    # case rule 3 permits -- but the gate is asked rather than assumed so a
    # signed quantity added to this board later cannot inherit the percent.
    ratio_ok = metrics.is_ratio_safe("gross")
    lines = []
    for m in tracked:
        obs = wide[m].dropna()  # endpoints of what was OBSERVED, not of the axis
        first, last = float(obs.iloc[0]), float(obs.iloc[-1])
        move = charts.fmt_change(last - first, "gross", _UNIT)
        if ratio_ok and first > 0:
            move += f" ({100.0 * (last / first - 1):+.1f}%)"
        holes = int(wide[m].loc[obs.index[0] : obs.index[-1]].isna().sum())
        gap = f", {holes} snapshot(s) missing in between" if holes else ""
        lines.append(
            f"`{m}` {obs.index[0].date()} -> {obs.index[-1].date()}: {move}{gap}"
        )
    st.caption(
        "Change over the observed window, and it is a window, not a trend -- two "
        "snapshots are a first difference. Gaps are drawn as gaps: a missed "
        "ingest is missing data, and a model that leaves the endpoint was "
        "delisted, not repriced. " + "  ·  ".join(lines)
    )


def _no_history_yet(n_snapshots: int, latest: pd.Timestamp) -> None:
    st.info(
        f"**{n_snapshots} snapshot, {latest.date()}. Not enough for a series, and "
        "no backfill exists.** The endpoint serves today's prices and nothing "
        "else: no date parameter, no archive. This series cannot be built "
        "retroactively -- it accumulates one ingest at a time from the day the job "
        "started, and a run that does not happen is a hole that stays a hole. Two "
        "snapshots give a first difference; a decline rate worth quoting needs "
        "months, because model turnover moves a vendor's cheapest price far more "
        "than any single price cut does."
    )


def _render() -> None:
    caveat_block, freshness = _app_helpers()

    df = _load()
    n_snapshots = int(df["snapshot_date"].nunique())
    latest = df["snapshot_date"].max()
    today = df[df["snapshot_date"] == latest]

    st.caption(_LEVEL_CAVEAT)

    # Explicit widget keys: an id is derived from label plus options, so an
    # unkeyed radio here collides with any other board's unkeyed radio.
    leg = st.radio("Price leg", list(_LEGS), horizontal=True, key="tokens_leg")
    col = _LEGS[leg]

    st.subheader(f"Cheapest model per vendor, {latest.date()}")
    priced_today = _priced(today, col)
    st.caption(
        f"{len(priced_today)} of {len(today)} models carry a positive {leg} price. "
        f"Excluded: {int((today[col] < 0).sum())} dynamic-router rows priced at -1 "
        "per token (OpenRouter's sentinel for a price decided at route time), "
        f"{int((today[col] == 0).sum())} free or promotional tiers at 0, "
        f"{int(today[col].isna().sum())} with no price field. The zeros matter -- "
        "several vendors publish a `:free` variant, so keeping them would report "
        "$0.00 as those vendors' cheapest model."
    )

    vendor_options = sorted(priced_today["vendor"].dropna().unique())
    vendors = st.multiselect(
        "Vendors",
        vendor_options,
        default=[v for v in _DEFAULT_VENDORS if v in set(vendor_options)],
        help=f"Empty shows every vendor, cheapest first, capped at {_MAX_BARS} bars.",
        key="tokens_vendors",
    )
    view = priced_today[priced_today["vendor"].isin(vendors)] if vendors else priced_today

    # No early return on an empty selection: the caveats below are part of the
    # reading, and a board that drops them on one branch has taught the reader
    # they are optional.
    if view.empty:
        st.info("No priced models for that selection.")
    else:
        per_vendor = _per_vendor(view, col)
        _cross_section(per_vendor, leg)

        with st.expander("Per-vendor range (cheapest / median / dearest)"):
            st.dataframe(
                per_vendor[
                    ["vendor", "n_priced", "cheapest", "cheapest_model",
                     "median", "dearest", "dearest_model"]
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Model IDs are recycled and re-versioned, so a vendor's cheapest "
                "model on two dates is often not the same product. Compare a named "
                "model across dates below; compare vendors only within a date."
            )

        st.subheader("Price over time")
        if n_snapshots < 2:
            _no_history_yet(n_snapshots, latest)
        else:
            _history(df, col, leg, list(per_vendor["cheapest_model"].astype(str)))

    st.caption(_HALF_THE_QUESTION)
    # Daily job, so 3 days late is already a hole rather than a slow week.
    freshness(df["snapshot_date"], "snapshot", stale_days=3)
    caveat_block(_SOURCE)


BOARD = Board(
    id="tokens",
    title="Token prices",
    render=_render,
    sources=(_SOURCE,),
    order=90,
    blurb=(
        "Posted per-token prices on OpenRouter -- the price factor of "
        "tokens x price. Snapshot-only: no history exists before the job began, "
        "and a missed run is a permanent gap."
    ),
    group="Token economics",
)
