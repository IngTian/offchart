"""The browser-side half of a chart: a frame-rate crosshair, and y following x.

Both jobs are here because both are things plotly will not do and Streamlit cannot
help with, and both need the same three preliminaries -- find the graph div, wait
until plotly has laid it out, and bail out safely if plotly's private shape is not
what we expect. Splitting them would mean two hidden iframes doing the same work.

This started life inside panels/positioning.py and moved out when a second board
needed it. The only board-specific part was the three lines that built the tooltip
rows, so that is now a passed-in spec; everything else was already general.

WHY A CROSSHAIR AT ALL, measured rather than assumed. plotly.js throttles hover to
its HOVERMINTIME constant: on a mouse sweep across the chart, 201 mousemove events
produced 31 hover updates, with the gap between them pinned at 57.6-60.0ms. So
plotly's crosshair redraws about 17 times a second while the pointer moves at
60-120Hz, and roughly five sixths of the motion draws nothing. Rendering was never
the problem -- frame times through the same sweep were a flat 8.3ms with zero long
tasks. The stepping IS the throttle, and it is a module constant with no config hook.
So plotly's own hover is switched off and this overlay takes over inside
requestAnimationFrame, tracking the pointer every frame.

THE COST, stated because it is real. This reads private plotly fields (`_fullLayout`
axis `_offset`/`_length`/`p2l`/`l2p`, and `_fullData`), so a plotly upgrade could
break it. It is therefore fully guarded: if anything it needs is missing it returns
WITHOUT touching hovermode, and plotly's throttled-but-working crosshair stays. The
failure mode is "back to 17Hz", never "no crosshair".
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Row:
    """One line in the tooltip, read out of the hover trace's customdata.

    `slot` is the index into customdata. The builder that made the figure decides
    the slot order, so this spec has to agree with it -- which is why both live
    beside each other in the panel that draws the chart.
    """

    label: str
    slot: int
    decimals: int = 2
    #: Appended after the number, e.g. " % of OI".
    suffix: str = ""
    #: Drop the row entirely when the value is null, rather than showing an em dash.
    #: For a series that may be absent for the whole market (a price with no mapped
    #: symbol), an em-dash row is noise on every single hover.
    omit_if_null: bool = False


def crosshair(key: str, rows: list[Row]) -> None:
    """Install the overlay on the chart Streamlit rendered with `key=key`."""
    spec = json.dumps(
        [
            {
                "label": r.label,
                "slot": r.slot,
                "dp": r.decimals,
                "suffix": r.suffix,
                "omit": r.omit_if_null,
            }
            for r in rows
        ]
    )

    st.components.v1.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent.document;
          const W = window.parent;
          const ROWS = {spec};
          let tries = 0;
          const timer = setInterval(attach, 250);
          attach();

          function findChart() {{
            // SCOPED TO THIS CHART'S OWN CONTAINER. Streamlit stamps st-key-<key> on
            // the element container when st.plotly_chart is given a key, so with two
            // boards in the app a bare querySelector('.js-plotly-plot') could attach
            // to whichever chart happens to be first in the document.
            const holder = doc.querySelector('[class*="st-key-{key}"]');
            if (holder) {{
              const gd = holder.querySelector('.js-plotly-plot');
              if (gd) return gd;
            }}
            return null;
          }}

          function attach() {{
            if (++tries > 80) {{ clearInterval(timer); return; }}
            const gd = findChart();
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

            // The hover traces all carry the same customdata; take the first.
            const src = (gd.data || []).find(t => t.customdata && t.customdata.length);
            if (!src) {{ clearInterval(timer); return; }}
            const xs = Array.prototype.map.call(src.x,
              v => (v instanceof Date ? v.getTime() : +new Date(v)));
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
              for (const r of ROWS) {{
                const v = row[r.slot];
                if (r.omit && (v == null || Number.isNaN(v))) continue;
                parts.push(r.label + '  ' + fmt(v, r.dp) + r.suffix);
              }}
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
