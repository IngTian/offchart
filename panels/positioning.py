"""One panel: pick a market, pick which cohorts to group, read the positioning.

The measure is a cohort group's net position as a share of that market's open
interest. Two readings reproduce the published figures to the digit, which is the
check that the pipeline is right:

    VIX          asset managers          -6.04% of OI,  4th pctile, n=1,014
    NASDAQ-100   asset mgr + leveraged   +9.28% of OI, 40th pctile, n=846

WHY THE CONTROLS DO NOT MAKE IT FEEL LIKE A SLIDE AGAIN

Streamlit reruns the whole script on every widget change, which is what made the
earlier multi-control boards feel like a deck advancing. The controls here live in
an `st.fragment`, so changing a cohort re-runs THIS FUNCTION ONLY -- the masthead,
the CSS and the sibling sections are untouched. Zoom and hover stay client-side in
plotly and never reach the server at all.

WHY SEGMENTING IS NOT USED HERE, since the rest of the repo leans on it

Net as a share of open interest is dimensionless. CFTC re-specified the Consolidated
equity contracts on 2023-05-02 and NASDAQ-100 open interest jumped 49,531 ->
255,954 with no change in positioning; that break multiplies numerator and
denominator by the same factor, so the ratio is untouched. Measured, computing the
percentile within the post-break segment alone moves NASDAQ 40.2 -> 40.8 and VIX
4.0 -> 4.6, and the whole difference is the smaller sample (846 -> 174 reports).
The break is still DRAWN, because a reader looking at the open-interest strip needs
to know why it steps.
"""
from __future__ import annotations

import contextlib
import io

import pandas as pd
import streamlit as st

from lib import cache, cftc_spec, charts, metrics, overlay, pricemap, segments, theme
from lib.ui import caveat_block
from panels import Board

#: The report families offered, in the order a person would reach for them. Each is
#: a separate CFTC publication with its OWN cohort vocabulary, which is why the
#: checkbox row is derived from the chosen family rather than hardcoded.
#:
#: Not offered: the futures-and-options-combined variants and the supplemental
#: report. They are ingested and available, but combined-basis figures do not
#: reconcile against futures-only ones, and putting them in the same picker invites
#: comparing two numbers that are not comparable. Add them when there is a reason.
FAMILIES = (
    ("cftc_tff_fut", "Financial futures — equity indices, rates, FX, crypto"),
    ("cftc_disagg_fut", "Commodities — energy, metals, grains, softs, livestock"),
    ("cftc_legacy_fut", "Legacy, every market — the only history before 2006"),
)
DEFAULT_FAMILY = "cftc_tff_fut"
PRICE_SOURCE = "prices"

#: Series that have compounded enough over their history that a linear axis is
#: useless -- an equity index or a coin spends its first decade indistinguishable
#: from zero, which hides the early positioning it is meant to sit beside. Rates,
#: FX and the vol index stay linear, where the level itself is the information.
_LOG_PRICE = frozenset({"^GSPC", "^NDX", "^DJI", "^RUT", "^SP400",
                        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"})

#: Short checkbox labels for the cohort ids that appear across the three families.
#: The spec's own labels are accurate and too long for a checkbox row
#: ("Producer / merchant / processor / user"), so these are the reading names; any
#: id not listed falls back to the spec label, which means a new cohort shows up
#: with its real name rather than vanishing.
_COHORT_LABEL = {
    # tff, sell side through to retail
    "dealer": "Dealers",
    "asset_mgr": "Asset managers",
    "lev_money": "Hedge funds",
    # disaggregated
    "prod_merc": "Producers",
    "swap": "Swap dealers",
    "m_money": "Managed money",
    # legacy
    "noncomm": "Non-commercial",
    "comm": "Commercial",
    # shared
    "other_rept": "Other reportables",
    "nonrept": "Small traders",
}

#: Which cohorts start ticked, per family. In each case the speculative money: the
#: cohorts whose position is a view rather than a hedge against physical or a
#: market-making book.
_DEFAULT_COHORTS = {
    "cftc_tff_fut": ("asset_mgr", "lev_money"),
    "cftc_disagg_fut": ("m_money",),
    "cftc_legacy_fut": ("noncomm",),
}

#: Markets offered in the picker, narrowest first. See lib/universe for the rule.
#:
#: DEFAULT IS "everything", which is what a data terminal does: you type a ticker and
#: it is either in the file or it is not. The tiers stay available because they are
#: genuinely useful for browsing -- CORE is the couple-of-dozen markets worth
#: scanning, and the liquidity rule behind it is what a cross-market screen needs --
#: but making them the default meant a market being absent from the picker looked
#: like a missing dataset rather than a filter.
#:
#: The cost is a long list: 140 markets in financials, 652 in commodities, 945 in
#: legacy, most of them near-dead electricity and basis contracts. The selectbox is
#: searchable, so the cost lands on browsing rather than on looking something up.
TIERS = ("CORE", "CORE + WIDE", "everything")
DEFAULT_TIER = "everything"


@st.cache_data(show_spinner=False, max_entries=64)
def _market_frame(source: str, code: str) -> pd.DataFrame:
    """Every cohort row for one market, indexed by report date. Cached per market."""
    df = cache.read(source)
    sub = df[df["market_code"].astype(str) == code].copy()
    if sub.empty:
        return sub
    sub["report_date"] = pd.to_datetime(sub["report_date"])
    return sub


def _measure(sub: pd.DataFrame, cohorts: tuple[str, ...]) -> pd.DataFrame:
    """Net share of open interest for a cohort GROUP, plus its causal rank."""
    picked = sub[sub["cohort"].isin(cohorts)]
    by_date = picked.groupby("report_date")
    net = (by_date["long"].sum() - by_date["short"].sum()).astype(float)

    # .first(), not .sum(): open interest is a market-week quantity repeated on
    # every cohort row, so summing it multiplies by the number of cohorts.
    week = sub.groupby("report_date")
    oi = week["open_interest"].first().astype(float)

    out = pd.DataFrame({"net": net, "open_interest": oi}).dropna(subset=["open_interest"])
    out["share"] = metrics.share_of(out["net"], out["open_interest"])
    out["pctile"] = metrics.expanding_percentile(out["share"])
    out["chg_2w"] = metrics.flow(
        out["share"].reset_index(drop=True), pd.Series(out.index), horizon_weeks=2
    ).to_numpy()
    return out


def _oi_pctile(out: pd.DataFrame, units: pd.Series) -> pd.Series:
    """Causal rank of open interest, WITHIN its contract-unit segment.

    The share's rank can span a contract re-specification safely, because
    net/open-interest is dimensionless and a re-denomination cancels top and bottom.
    Open interest in CONTRACTS is not: when CFTC re-based the Consolidated equity
    indices on 2023-05-02, NASDAQ-100 open interest stepped 49,531 -> 255,954 with no
    change in positioning, so an expanding rank over the whole history says "98th
    percentile" about a number that is mostly a units change. That reading was on
    screen one pixel-column to the right of the break line proving it.

    So this one is segmented and the share's is not -- an asymmetry that looks
    inconsistent until you notice one measure is a ratio and the other is a level.
    Ranking restarts at each break, which means it is honestly blank-ish early in a
    new segment rather than confidently wrong.
    """
    dates = pd.Series(out.index)
    seg = segments.segment_ids(dates, units.reindex(out.index))
    oi = out["open_interest"]
    ranked = pd.Series(float("nan"), index=out.index)
    for _, idx in oi.groupby(seg.to_numpy()).groups.items():
        ranked.loc[idx] = metrics.expanding_percentile(oi.loc[idx]).to_numpy()
    return ranked


@st.cache_data(show_spinner=False, max_entries=32)
def _price_at(symbol: str, dates: tuple) -> pd.Series:
    """The mapped price series, sampled AS OF each CFTC report date.

    As-of, backward, and that direction is the whole point. A report date is a
    Tuesday; markets close daily; so the price beside a position must be the last
    close at or before that Tuesday. `direction="nearest"` or a forward fill would
    quietly put a price that did not exist yet next to a position -- look-ahead
    smuggled in through a join rather than through a statistic, which is the kind
    that survives review.

    Returns an empty Series when the symbol is not in the store, so a missing price
    source degrades to "no price panel" rather than taking the board down.
    """
    px = cache.read("prices")
    if px.empty or "symbol" not in px.columns:
        return pd.Series(dtype=float)
    one = px[px["symbol"] == symbol]
    if one.empty:
        return pd.Series(dtype=float)

    # 10 days: a weekly report should always have a close within about a week, so
    # anything older means the price series has a hole or has died. Beyond the cap
    # the panel gets NaN and simply draws a gap, which is honest, instead of a
    # carried-forward flat line that reads as a price standing still.
    return metrics.asof(
        one["date"], one["close"], pd.Series(list(dates)), max_staleness_days=10
    )


def _hero(frame: pd.DataFrame, who: str) -> None:
    last = frame.iloc[-1]
    share, pctile, chg = last["share"], last["pctile"], last["chg_2w"]
    side = "net long" if share > 0 else "net short" if share < 0 else "flat"
    n = int(frame["share"].notna().sum())
    delta = f"{chg:+.2f}pp over 2 weeks" if pd.notna(chg) else "no week two back"
    # The ordinal comes from lib.charts, which is also what the bubble on the chart
    # uses. This line used to build its own by pasting a hardcoded "th" onto
    # `{pctile:.0f}`, which printed "53th" next to a bubble reading "53rd" -- and
    # rounded, so 99.75 read "100th", a record the series had not set.
    rank_n, rank_suffix = charts.ordinal_parts(pctile)
    where = (
        # A blank rank is not a middling one. Saying "unremarkable" about a number
        # that does not exist yet is the same error as printing "nanth".
        "no rank yet -- too little history" if pd.isna(pctile)
        else "lower than all but a handful of weeks on record" if pctile <= 10
        else "higher than all but a handful of weeks on record" if pctile >= 90
        else "unremarkable against its own history"
    )
    st.markdown(
        f"""<div class="hero">
              <div class="hero-figure">{share:+.2f}<span class="hero-unit">% of OI</span></div>
              <div class="hero-side">{who} &mdash; <strong>{side}</strong></div>
            </div>
            <div class="hero-row">
              <div><span class="k">{rank_n}<span class="k-unit">{rank_suffix}</span></span>
                   <span class="v">percentile of its own {n:,} weeks &mdash; {where}</span></div>
              <div><span class="k-sm">{delta}</span></div>
            </div>""",
        unsafe_allow_html=True,
    )


def _copy_button(slug: str) -> None:
    """Copy the chart to the clipboard as a PNG.

    Done with a script injected into the parent document rather than a Streamlit
    component, because the plot lives in the parent and an iframe cannot reach
    another document's plotly instance. Same origin, so this is allowed.

    The click has to happen in the document that writes to the clipboard -- browsers
    require a user gesture -- which is why the button is created in the parent DOM
    instead of being an st.button that posts a message.
    """
    st.components.v1.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent.document;
          const id = "copy-{slug}";

          // POLL, do not run once. This script executes as soon as its iframe
          // loads, which can be before Streamlit has mounted the plotly div -- and
          // a single attempt then finds nothing and gives up silently. The chart is
          // also replaced wholesale whenever the fragment reruns, taking the button
          // with it, so the attach has to be re-tried rather than assumed.
          let tries = 0;
          const timer = setInterval(attach, 250);
          attach();

          function attach() {{
            if (++tries > 80) {{ clearInterval(timer); return; }}   // ~20s
            const plot = doc.querySelector('.js-plotly-plot');
            if (!plot || !window.parent.Plotly) return;
            const bar = plot.closest('[data-testid="stPlotlyChart"]');
            if (!bar) return;
            if (bar.querySelector('.wb-copy')) {{ clearInterval(timer); return; }}
            bar.appendChild(build(plot));
            clearInterval(timer);
          }}

          function build(plot) {{
          const btn = doc.createElement('button');
          btn.id = id;
          btn.className = 'wb-copy';
          btn.textContent = 'Copy image';
          btn.onclick = async function () {{
            try {{
              const W = window.parent;
              const url = await W.Plotly.toImage(plot, {{ format: 'png', scale: 3 }});
              // Every realm-sensitive call has to be the PARENT's. This closure is
              // defined inside a sandboxed component iframe, so a bare
              // navigator.clipboard here is the iframe's -- which has no
              // clipboard-write permission and fails silently. The blob and the
              // ClipboardItem must also come from the parent realm or Chrome
              // rejects the item as being from the wrong context.
              const blob = await (await W.fetch(url)).blob();
              await W.navigator.clipboard.write([
                new W.ClipboardItem({{ 'image/png': blob }})
              ]);
              btn.textContent = 'Copied';
            }} catch (e) {{
              // Clipboard image write is not universally permitted. Say so rather
              // than failing silently, and leave the PNG download as the path
              // that always works.
              btn.textContent = 'Use the download icon';
            }}
            setTimeout(() => {{ btn.textContent = 'Copy image'; }}, 2200);
          }};
          return btn;
          }}
        }})();
        </script>
        """,
        height=0,
    )


def _pull_button(source: str) -> None:
    """Fetch the newest reports for the SELECTED family, plus prices, on demand.

    NOTE THIS BREAKS A RULE THIS REPO OTHERWISE KEEPS. The display layer is
    supposed never to fetch, so that a chart cannot show a number that is not in
    the committed dataset. This button is the one exception and it is deliberate:
    it runs the ingest JOB (the same code path CI runs), writes parquet, and then
    clears the read cache -- rather than fetching into a chart. What is on screen
    after it finishes is still exactly what is on disk.

    Incremental, so it asks the API for roughly the last eight weeks rather than
    the ~900 MB a full history pull costs. Use `python -m scripts.ingest --backfill`
    for the full thing.
    """
    from scripts import ingest as ingest_job
    from sources import get as get_source

    if not st.button("Pull latest data", type="secondary"):
        return

    log = io.StringIO()
    done: list[str] = []
    with st.spinner("Fetching new reports and closes..."):
        # Positioning and prices, in that order: a failed price pull must not stop
        # the CFTC data from landing, since the board works without a price panel
        # and does not work without positions.
        for sid in (source, PRICE_SOURCE):
            try:
                with contextlib.redirect_stdout(log):
                    report = ingest_job.run_one(get_source(sid), backfill=False)
            except Exception as exc:  # noqa: BLE001 -- surfaced, not swallowed
                done.append(f"{sid} failed ({type(exc).__name__})")
                continue
            if report is None:
                done.append(f"{sid} failed, nothing written")
            elif report["changed"]:
                done.append(
                    f"{sid}: +{report['rows_new']:,} new, {report['rows_revised']:,} revised"
                )
            else:
                done.append(f"{sid}: already current ({report['rows_total']:,} rows)")

    # Held in session_state rather than printed here, because the rerun below
    # rebuilds the page and would wipe anything written before it. A pull that
    # reports nothing is indistinguishable from a pull that did nothing.
    st.session_state["wb_pull"] = " · ".join(done)

    # Clear every cached read: the store's own stat token would invalidate
    # cache.read on its own, but _market_frame and market_summary are keyed on the
    # market code and would otherwise keep serving pre-pull frames.
    st.cache_data.clear()
    st.rerun()


@st.fragment
def _panel() -> None:
    """Controls plus chart. A fragment, so a checkbox does not rerun the page."""
    top = st.columns([2.6, 1.05, 1.05])
    with top[0]:
        source = st.selectbox(
            "Report",
            [sid for sid, _ in FAMILIES],
            format_func=lambda s: dict(FAMILIES)[s],
            key="wb_family",
            label_visibility="collapsed",
        )
    with top[1]:
        tier = st.selectbox("Universe", TIERS, index=TIERS.index(DEFAULT_TIER),
                            key="wb_tier", label_visibility="collapsed")
    with top[2]:
        _pull_button(source)

    # Survives the rerun the pull triggers; cleared once shown so it does not
    # linger as a stale claim about data that has since changed.
    if note := st.session_state.pop("wb_pull", None):
        st.caption(note)

    summary = cache.market_summary(source)
    if summary.empty:
        st.info(f"`{source}` has no data yet. Use the pull button, or run the ingest job.")
        return

    keep = {"CORE": ("CORE",), "CORE + WIDE": ("CORE", "WIDE")}.get(tier)
    pool = summary if keep is None else summary[summary["tier"].isin(keep)]
    # A superseded E-mini/Micro leg is already inside its Consolidated series, so
    # offering both invites reading one exposure twice.
    pool = pool[pool["superseded_by"].isna()]
    pool = pool.sort_values("oi_window_max", ascending=False)
    if pool.empty:
        st.info("No markets in that universe.")
        return

    codes = list(pool["market_code"].astype(str))
    labels = {
        str(r.market_code): f"{r.market}  ·  {r.market_code}"
        for r in pool.itertuples()
    }

    # A per-family widget key, because the market list and the cohort vocabulary both
    # change with the family: a shared key would carry a NASDAQ code into the
    # commodity report and land on "no rows for that market".
    #
    # Financials open on NASDAQ-100 rather than on whatever carries the most open
    # interest, since pure OI ranking puts 3-month SOFR first -- a fine market and a
    # strange front page. The other families just take their most liquid.
    prefer = {"cftc_tff_fut": ("20974+", "1170E1")}.get(source, ())
    default_code = next((c for c in prefer if c in codes), codes[0])
    key = f"wb_code_{source}"
    if st.session_state.get(key) not in codes:
        st.session_state[key] = default_code
    code = st.selectbox(
        "Market",
        codes,
        format_func=lambda c: labels.get(c, c),
        key=key,
        label_visibility="collapsed",
    )

    # Cohorts come from the chosen report's own spec, because the three families
    # classify traders differently and there is no shared vocabulary: "commercial"
    # in legacy is not "producer + swap dealer" in disaggregated, and treating them
    # as the same field would be a definitional error wearing a checkbox.
    cohorts = tuple(
        (c.id, _COHORT_LABEL.get(c.id, c.label)) for c in cftc_spec.spec_for(source).cohorts
    )
    defaults = _DEFAULT_COHORTS.get(source, (cohorts[0][0],))
    boxes = st.columns(len(cohorts))
    chosen: list[str] = []
    for (cid, label), col in zip(cohorts, boxes):
        with col:
            ckey = f"wb_c_{source}_{cid}"
            if st.checkbox(label, value=st.session_state.get(ckey, cid in defaults),
                           key=ckey):
                chosen.append(cid)

    sub = _market_frame(source, code)
    if sub.empty:
        st.info("No rows for that market.")
        return
    if not chosen:
        st.caption("Pick at least one cohort to group.")
        return

    frame = _measure(sub, tuple(chosen))
    if frame["share"].notna().sum() < 2:
        st.caption("Not enough history in this market to plot.")
        return

    who = " + ".join(dict(cohorts)[c] for c in chosen)
    market_name = str(sub["market"].iloc[-1])
    st.markdown(
        f'<div class="card-head"><h2>{market_name}</h2>'
        f'<div class="card-sub">{who} &nbsp;·&nbsp; net as a share of open interest'
        f" &nbsp;·&nbsp; {code}</div></div>",
        unsafe_allow_html=True,
    )
    _hero(frame, who)

    ref = pricemap.for_code(code)
    price = _price_at(ref.symbol, tuple(frame.index)) if ref else pd.Series(dtype=float)

    slug = f"{code}-{'-'.join(chosen)}".replace("+", "plus").replace(" ", "")
    st.plotly_chart(
        charts.spotlight(
            frame["share"],
            frame["open_interest"],
            unit="% of open interest",
            kind="net_share",
            # NO `breaks=`: the dotted red rule at each contract re-specification is
            # not drawn, by request. Nothing it carried is lost from the NUMBERS --
            # the same segmentation still splits the open-interest percentile, so
            # that rank is computed within the current contract definition rather
            # than across a re-basing (see _oi_pctile), and the step in the open
            # interest line remains visible in the strip. The line was labelling a
            # step that is already on screen.
            price=price if not price.empty else None,
            price_label=ref.label if ref else "",
            # Log for anything that has compounded over decades -- an equity index
            # or a coin on a linear axis spends its first fifteen years flat on the
            # floor, which hides exactly the early positioning history the chart is
            # there to sit beside.
            pctile=frame["pctile"],
            oi_pctile=_oi_pctile(frame, sub.groupby("report_date")["contract_units"].first()),
            price_log=bool(ref and ref.symbol in _LOG_PRICE),
            # Taller with a price panel: three panels in 620px left the price
            # squeezed, and it is the one a reader orients by.
            height=760 if not price.empty else 460,
        ),
        width="stretch",
        config=charts.png_config(f"watchboard-{slug}"),
        key=f"chart-{slug}",
    )
    _copy_button(slug)
    # Slots 0-2 are fixed by lib.charts.spotlight, which packs (price, net, open
    # interest) into customdata in that order and appends the ranks after them.
    overlay.crosshair(
        f"chart-{slug}",
        [
            # Omitted rather than em-dashed: a market with no mapped price symbol
            # would otherwise carry a blank row on every single hover.
            overlay.Row(ref.label if ref else "price", 0, 2, omit_if_null=True),
            overlay.Row("net", 1, 2, suffix=" % of OI"),
            overlay.Row("open interest", 2, 0),
        ],
    )

    if ref:
        note = f"Price is **{ref.label}**, sampled as of each Tuesday report date."
        if ref.proxy:
            note += f" A proxy, not the reported contract: {ref.proxy}."
        st.caption(
            note + " Positions are as of Tuesday but published Friday 15:30 ET, so "
            "the price beside a position is contemporaneous with the position and "
            "not with the moment you could first have seen it."
        )
    else:
        st.caption(
            f"No price series is mapped to {code}. See `lib/pricemap.py` -- entries "
            "are added one at a time, after the symbol has been resolved."
        )

    with st.expander("The last 12 weeks, as numbers"):
        tail = frame.tail(12)[["net", "open_interest", "share", "pctile"]].copy()
        tail.index = [d.date() for d in tail.index]
        st.dataframe(
            tail.rename(
                columns={
                    "net": "net (contracts)",
                    "open_interest": "open interest",
                    "share": "% of OI",
                    "pctile": "percentile",
                }
            ).style.format(
                {
                    "net (contracts)": "{:,.0f}",
                    "open interest": "{:,.0f}",
                    "% of OI": "{:+.2f}",
                    "percentile": "{:.1f}",
                }
            ),
            width="stretch",
        )


def render() -> None:
    # The stamp reports the SELECTED family's newest report, not a fixed one --
    # legacy, financials and commodities are separate publications and can be a week
    # apart. Read outside the fragment so it is the same date the picker is showing.
    source = st.session_state.get("wb_family", DEFAULT_FAMILY)
    stats = cache.file_stats(source) or {}
    if stats.get("latest"):
        latest = pd.Timestamp(stats["latest"])
        age = (pd.Timestamp.today().normalize() - latest.normalize()).days
        st.markdown(
            f'<div class="stamp">Latest CFTC report &nbsp;<strong>{latest.date()}</strong>'
            f"&nbsp; · &nbsp;{age} days ago &nbsp; · &nbsp;Tuesday positions, "
            "published Friday 15:30 ET</div>",
            unsafe_allow_html=True,
        )
    _panel()
    caveat_block(source, PRICE_SOURCE)


BOARD = Board(
    id="positioning",
    title="Positioning",
    render=render,
    # Only the default family is a hard requirement: the others are selectable and a
    # missing one is handled inside the panel rather than blocking the whole board.
    sources=(DEFAULT_FAMILY,),
    order=10,
    blurb="Where the big cohorts are leaning, and how unusual that is.",
    group="CFTC",
)
