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

from lib import cache, cftc_spec, charts, metrics, pricemap, segments, theme
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


def _smooth_crosshair(slug: str, price_label: str = "") -> None:
    """Replace plotly's crosshair with one driven at frame rate.

    WHY THIS EXISTS, measured rather than assumed. plotly.js throttles hover to its
    HOVERMINTIME constant: on a mouse sweep across the chart, 201 mousemove events
    produced 31 hover updates, with the gap between them pinned at 57.6-60.0ms.
    So the crosshair redraws about 17 times a second while the pointer moves at
    60-120Hz, and roughly five sixths of the motion draws nothing. Rendering was
    never the problem -- frame times through the same sweep were a flat 8.3ms with
    zero long tasks. The stepping IS the throttle, and it is a module constant with
    no config hook.

    So: plotly's own hover is switched off and an overlay takes over, updated inside
    requestAnimationFrame, which tracks the pointer every frame.

    It also carries the second thing that has to happen in the browser: RESCALING
    THE Y AXES WHEN THE X RANGE CHANGES. plotly does that for a zoom box and not
    for a range button, and every control on this board changes x only. Both jobs
    live in one injected script because both need the same graph div, the same
    readiness polling and the same private-field guards; splitting them would mean
    a second hidden iframe doing the same three things.

    THE COST, stated because it is real. This reads two private plotly fields
    (`_fullLayout` axis `_offset`/`_length` and `p2d`/`d2p`), so a plotly upgrade
    could break it. It is therefore fully guarded: if anything it needs is missing
    it returns WITHOUT touching hovermode, and plotly's throttled-but-working
    crosshair stays. The failure mode is "back to 17Hz", never "no crosshair".
    """
    st.components.v1.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent.document;
          const W = window.parent;
          let tries = 0;
          const timer = setInterval(attach, 250);
          attach();

          function attach() {{
            if (++tries > 80) {{ clearInterval(timer); return; }}
            const gd = doc.querySelector('.js-plotly-plot');
            if (!gd || !W.Plotly || !gd._fullLayout) return;
            const card = gd.closest('[data-testid="stPlotlyChart"]');
            if (!card) return;
            if (card.querySelector('.wb-xhair')) {{ clearInterval(timer); return; }}

            const fl = gd._fullLayout;
            const xa = fl.xaxis;
            // Bail out rather than half-work if the private shape is not what we
            // expect. plotly's own crosshair is left switched on in that case.
            // p2l / l2p, NOT p2d / d2p. On a date axis p2d returns a formatted
            // STRING ("2018-07-20 12:00"), so a numeric nearest-point search against
            // it compares number < string, gets NaN, and silently returns index 0 --
            // the crosshair pinned itself to the first observation and never moved.
            // p2l returns linear ms and l2p takes it back to pixels.
            if (!xa || typeof xa.p2l !== 'function' || typeof xa.l2p !== 'function'
                || xa._offset == null || xa._length == null) {{
              clearInterval(timer); return;
            }}
            // The KEY NAMES are fixed for the life of the figure; the offsets behind
            // them are not, so only the names are captured. Measuring the band once
            // at attach was wrong: expanding the chart to full screen re-lays the
            // figure out taller, and a crosshair sized to the 760px version stopped
            // two thirds of the way down the enlarged one -- a line that ends in
            // mid-air over the panel it is supposed to be reading.
            const yKeys = Object.keys(fl).filter(k => /^yaxis\\d*$/.test(k));
            if (!yKeys.length) {{ clearInterval(timer); return; }}

            function band() {{
              const fl2 = gd._fullLayout;
              if (!fl2) return null;
              let t = Infinity, b = -Infinity;
              for (const k of yKeys) {{
                const a = fl2[k];
                if (!a || a._offset == null || a._length == null) continue;
                if (a._offset < t) t = a._offset;
                if (a._offset + a._length > b) b = a._offset + a._length;
              }}
              return (isFinite(t) && isFinite(b)) ? {{t: t, b: b}} : null;
            }}
            if (!band()) {{ clearInterval(timer); return; }}

            // The hover traces carry [price, net, open interest] per point. Take the
            // first one; they all share the same customdata.
            const src = (gd.data || []).find(t => t.customdata && t.customdata.length);
            if (!src) {{ clearInterval(timer); return; }}
            const xs = src.x.map(v => (v instanceof Date ? v.getTime() : +new Date(v)));
            const cd = src.customdata;

            const line = doc.createElement('div');
            line.className = 'wb-xhair';
            const tip = doc.createElement('div');
            tip.className = 'wb-xtip';
            card.appendChild(line);
            card.appendChild(tip);

            const fmt = (v, d) => (v == null || Number.isNaN(v)) ? '—'
              : v.toLocaleString(undefined, {{minimumFractionDigits: d,
                                              maximumFractionDigits: d}});
            const when = ms => new Date(ms).toLocaleDateString(undefined,
              {{year: 'numeric', month: 'short', day: 'numeric'}});

            let px = null, raf = null, shown = false;

            function nearest(xval) {{
              let lo = 0, hi = xs.length - 1;
              while (hi - lo > 1) {{
                const mid = (lo + hi) >> 1;
                if (xs[mid] < xval) lo = mid; else hi = mid;
              }}
              return (Math.abs(xs[lo] - xval) <= Math.abs(xs[hi] - xval)) ? lo : hi;
            }}

            function draw() {{
              raf = null;
              if (px == null) return;
              // Re-read the axis every frame rather than closing over the one
              // captured at attach time. A range button relayouts the figure and
              // plotly may hand back a REBUILT _fullLayout, so a captured axis
              // still answers p2l with the OLD range: the tooltip then reported a
              // date from the previous window (a zoom to 2025-2026 read "Oct 5,
              // 2010" under a pointer that had not moved). A property read per
              // frame is free next to the rAF itself.
              const ax = gd._fullLayout && gd._fullLayout.xaxis;
              if (!ax || typeof ax.p2l !== 'function') {{ hide(); return; }}
              const rel = px - ax._offset;
              if (rel < 0 || rel > ax._length) {{ hide(); return; }}
              const i = nearest(ax.p2l(rel));
              const snap = ax.l2p(xs[i]) + ax._offset;   // sit on the observation
              // The nearest observation to the pointer can lie OUTSIDE a zoomed
              // window (pointer at the left edge, last point of the prior year
              // just off it). Snapping to it would park the line off the panel.
              if (snap < ax._offset - 1 || snap > ax._offset + ax._length + 1) {{
                hide(); return;
              }}
              const bd = band();
              if (!bd) {{ hide(); return; }}
              line.style.transform = 'translateX(' + snap + 'px)';
              line.style.top = bd.t + 'px';
              line.style.height = (bd.b - bd.t) + 'px';

              const row = cd[i] || [];
              const parts = ['<b>' + when(xs[i]) + '</b>'];
              if (row[0] != null) parts.push('{price_label or "price"}  ' + fmt(row[0], 2));
              parts.push('net  ' + fmt(row[1], 2) + ' % of OI');
              parts.push('open interest  ' + fmt(row[2], 0));
              tip.innerHTML = parts.join('<br>');

              // Flip the tooltip to the other side near the right edge so it never
              // spills out of the card.
              const w = tip.offsetWidth || 150;
              const flip = snap + 14 + w > ax._offset + ax._length;
              tip.style.transform = 'translateX(' + (flip ? snap - w - 14 : snap + 14) + 'px)';
              tip.style.top = (bd.t + 10) + 'px';
              if (!shown) {{
                shown = true;
                line.style.opacity = '1';
                tip.style.opacity = '1';
              }}
            }}

            function hide() {{
              shown = false;
              line.style.opacity = '0';
              tip.style.opacity = '0';
            }}

            // mousemove, not pointermove: plotly's own drag layers sit on top and
            // swallow pointer events, so pointermove on the container fires once and
            // then stops. mousemove still reaches it.
            gd.addEventListener('mousemove', ev => {{
              px = ev.clientX - gd.getBoundingClientRect().left;
              if (!raf) raf = W.requestAnimationFrame(draw);
            }}, {{passive: true, capture: true}});
            gd.addEventListener('mouseleave', hide, {{passive: true}});

            // ---- Y FOLLOWS X ----------------------------------------------------
            // plotly rescales y on a ZOOM BOX but never when only the x range
            // changes, and every control here changes only x. So "1Y" left the
            // price panel showing one year of an index against sixteen years of
            // scale: a flat ribbon pinned to the top of the panel with the whole
            // middle empty. Nothing is wrong with the data in that picture, which
            // is what makes it worth fixing -- it reads as a broken chart.
            //
            // Each axis is rescaled by what it DECLARES, not by which panel it is,
            // so this keeps working if the panels are reordered or one is dropped:
            //   rangemode 'tozero'  -> floor stays exactly 0 (open interest)
            //   a trace filled to zero -> zero stays inside the range, because the
            //     baseline is the first thing read on a net position
            //   type 'log'          -> the range is in log10 units
            // READ gd._fullData, NOT gd.data. plotly.py ships numeric columns to
            // the browser as a BINARY typed-array spec -- gd.data[i].y is
            // {{dtype, bdata, _inputArray}}, an object with no numeric indices. The
            // first version of this loop iterated gd.data, read nothing, found no
            // finite values and quietly changed no range at all: a rescale that
            // silently does not happen. _fullData holds the decoded Float64Array.
            const xsCache = new Map();
            function msOf(i, t) {{
              if (!xsCache.has(i)) {{
                xsCache.set(i, Array.prototype.map.call(t.x || [],
                  v => (v instanceof Date ? v.getTime() : +new Date(v))));
              }}
              return xsCache.get(i);
            }}

            // EACH AXIS'S INTENT IS CAPTURED ONCE, HERE, BEFORE ANYTHING RELAYOUTS.
            // rangemode is only coerced onto _fullLayout while an axis is
            // AUTORANGING. The moment the rescale below writes an explicit range,
            // rangemode disappears from the live layout -- so re-reading it per
            // call made open interest lose its zero floor on the second zoom (a
            // strip that had read 0-436k came back as 232k-417k, turning a level
            // series into a magnified wiggle). Read at attach time it is still
            // there.
            const rules = {{}};
            Object.keys(fl).filter(k => /^yaxis\\d*$/.test(k)).forEach(k => {{
              rules[k] = {{toZero: fl[k].rangemode === 'tozero',
                          log: fl[k].type === 'log'}};
            }});

            function yUpdate() {{
              const fl2 = gd._fullLayout;
              const ax = fl2 && fl2.xaxis;
              const traces = gd._fullData;
              if (!ax || !ax.range || typeof ax.r2l !== 'function') return null;
              if (!traces || !traces.length) return null;
              const x0 = ax.r2l(ax.range[0]), x1 = ax.r2l(ax.range[1]);
              const seen = {{}};
              traces.forEach((t, i) => {{
                const name = t.yaxis || 'y';
                const g = seen[name] ||
                  (seen[name] = {{lo: Infinity, hi: -Infinity, zero: false}});
                if (t.fill === 'tozeroy') g.zero = true;
                const xs2 = msOf(i, t), ys = t.y || [];
                const n = Math.min(xs2.length, ys.length);
                for (let k = 0; k < n; k++) {{
                  if (xs2[k] < x0 || xs2[k] > x1) continue;
                  const v = ys[k];
                  if (v == null || !isFinite(v)) continue;
                  if (v < g.lo) g.lo = v;
                  if (v > g.hi) g.hi = v;
                }}
              }});
              const upd = {{}};
              Object.keys(seen).forEach(name => {{
                const g = seen[name];
                if (!isFinite(g.lo) || !isFinite(g.hi)) return;
                const key = 'yaxis' + name.slice(1);   // 'y'->'yaxis', 'y2'->'yaxis2'
                const rule = rules[key];
                if (!rule) return;
                const toZero = rule.toZero;
                let lo = g.lo, hi = g.hi;
                if (toZero || g.zero) {{ lo = Math.min(0, lo); hi = Math.max(0, hi); }}
                if (rule.log) {{
                  if (lo <= 0 || hi <= 0) return;   // a log axis cannot show these
                  const l0 = Math.log10(lo), l1 = Math.log10(hi);
                  const p = Math.max((l1 - l0) * 0.08, 0.01);
                  upd[key + '.range'] = [l0 - p, l1 + p];
                  // Ticks have to follow the span, because no fixed rule labels
                  // both ends of this zoom. "D2" is the 1-2-5 ladder: clean over
                  // sixteen years (2k/5k/10k/20k) and EMPTY over one, where the
                  // window 22.7k-31.4k contains no rung. null hands it back to
                  // plotly's adaptive log ticks, which label a narrow window
                  // evenly (23k...31k) but crowd a wide one (2k,3k,4k...9k,20k).
                  // One decade is the crossover.
                  // tickmode MUST be set alongside dtick. Supplying dtick at all
                  // flips tickmode to 'linear', and it does not flip back when
                  // dtick goes to null -- the axis then kept a one-tick-per-decade
                  // rule and a 22.7k-31.4k window came back with NO labels at all.
                  const wide = (l1 - l0) >= 1;
                  upd[key + '.tickmode'] = wide ? 'linear' : 'auto';
                  upd[key + '.dtick'] = wide ? 'D2' : null;
                }} else {{
                  const p = (Math.abs(hi - lo) || Math.abs(hi) || 1) * 0.08;
                  upd[key + '.range'] = [toZero ? lo : lo - p, hi + p];
                }}
              }});
              return Object.keys(upd).length ? upd : null;
            }}

            let selfUpdate = false;
            if (typeof gd.on === 'function') {{
              gd.on('plotly_relayout', ev => {{
                // Guard the recursion: the relayout below fires this same event.
                if (selfUpdate) return;
                if (!Object.keys(ev || {{}}).some(k => k.indexOf('xaxis') === 0)) return;
                // The pointer has not moved, so nothing would redraw the overlay,
                // and what it is showing describes the window that just went away.
                hide();
                const upd = yUpdate();
                if (!upd) return;
                selfUpdate = true;
                Promise.resolve(W.Plotly.relayout(gd, upd))
                  .catch(() => {{}})
                  .then(() => {{ selfUpdate = false; }});
              }});
            }}

            // Normalise the FIRST paint through the same rule the buttons use, so
            // that the view on load and the view after clicking "All" are the same
            // picture rather than two nearly-identical ones -- and so the log tick
            // rule above has one home instead of being restated server-side.
            // Folded into the relayout that silences plotly's own throttled
            // crosshair (safe now that the overlay is proven installed) because
            // one relayout is cheaper than two and neither carries an xaxis key,
            // so the handler above ignores both.
            const first = yUpdate() || {{}};
            first.hovermode = false;
            W.Plotly.relayout(gd, first);
            clearInterval(timer);
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
    _smooth_crosshair(slug, ref.label if ref else "")

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
