"""Chart helpers that make the board's display rules the path of least resistance.

Four rules, each of them the residue of a real error rather than a style
preference. The point of this module is that following them should be less work
than breaking them.

RULE 1 -- NO DUAL AXES. Two series on one panel with two y-scales lets the author
choose the scaling that makes a correlation look however they want, and the
reader cannot see that a choice was made. So `stacked()` builds independent
subplots sharing only the x-axis, and there is no parameter anywhere in this
module that produces a secondary y-axis. tests/test_display_rules.py greps the
panels for `secondary_y`, `make_subplots` and direct plotly imports, which is the
actual enforcement -- a panel can always import plotly itself, so the test is the
mechanism and this module is the convenience.

RULE 2 -- CAUSAL RANKINGS ONLY. Enforced in lib/metrics, not here.

RULE 3 -- SIGNED QUANTITIES KEEP THEIR OWN UNITS. `Series` carries a `kind`, and
`pct_change()` refuses to format a percent change for a kind that can change
sign. The gate is semantic: it asks what the quantity IS, not whether this
particular sample happens to have crossed zero. An earlier empirical version
returned "safe" for 14% of cohort-net series, including the board's own default
market.

RULE 4 -- OPEN INTEREST IS ALWAYS ON SCREEN NEXT TO POSITIONING. A cohort's share
of open interest can rise because that cohort bought or because open interest
shrank, and the share alone cannot distinguish them. `stacked()` therefore
requires that any figure containing a share-of-open-interest series also contains
an open-interest series, and raises if it does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import metrics, theme

# Validated categorical palette, in slot order. Slots are assigned in this order
# and never cycled -- a 9th series folds into "other" or becomes a small multiple.
# Worst adjacent CVD delta-E is 9.1 on the light surface against a target of 8.
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SERIES_3 = "#1baf7a"  # aqua
SERIES_4 = "#eda100"  # yellow
SERIES_COLORS = (SERIES_1, SERIES_2, SERIES_3, SERIES_4, "#e87ba4", "#008300")

# Diverging poles for a signed quantity: warm/cool, so they read as opposite.
POS = SERIES_1
NEG = "#e34948"

# Chrome and ink come from lib/theme, which reads them off ingtian.github.io.
# Gridlines and axis rules are SOLID hairlines one shade off the surface --
# dashing them adds noise and reads as "threshold" when it is just a grid. Dashes
# are reserved here for actual reference levels.
SURFACE = theme.c("surface")
INK = theme.c("ink")
INK_SECONDARY = theme.c("ink_2")
MUTED = theme.c("muted")
GRID = theme.c("grid")
BASELINE = theme.c("baseline")
ACCENT_LINE = theme.c("accent")
ACCENT_FILL = theme.c("accent_soft")
SEAL = theme.c("seal")

# Kept for the parked boards, which were written against these names.
ACCENT = SERIES_1
WARN = SERIES_2
POSITIVE = SERIES_3
ZERO_LINE = BASELINE

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: Client-side zoom presets. These are the smoothness win in a Streamlit app:
#: plotly handles them in the browser, so the range changes instantly and the
#: server never reruns the script. A widget doing the same job would round-trip.
#:
#: "All" IS A BACKWARD SPAN, NOT step="all", AND THAT IS LOAD-BEARING. step="all"
#: relayouts xaxis.autorange=true, and autorange here is not tight: a one-point
#: marker trace (the endpoint dot) has zero span, so plotly pads it by a fraction
#: of the axis LENGTH IN PIXELS, which on sixteen years of weekly data resolved to
#: 347 extra days -- measured, and measured again with the marker at size 0 to rule
#: the marker size out. Every other button is stepmode="backward" and so measures
#: from range[1]; with a padded range[1] "1Y" meant "a year ending eleven months
#: in the future" and put three weeks of data against the left edge. Keeping every
#: button on an explicit backward count means no button can reintroduce autorange,
#: so the axis end stays on the last observation for all four.
def range_buttons(span_days: int) -> list[dict]:
    """The four presets, with "All" spelled as an exact backward span in days."""
    return [
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=5, label="5Y", step="year", stepmode="backward"),
        dict(count=10, label="10Y", step="year", stepmode="backward"),
        dict(count=max(span_days, 1), label="All", step="day", stepmode="backward"),
    ]

#: Passed to st.plotly_chart. No scroll-hijack, no logo, responsive.
#:
#: The modebar is trimmed to the PNG download alone. Everything else it offers
#: (lasso, box select, autoscale, the plotly logo) is either meaningless on a time
#: series or duplicates the range buttons, and a full modebar is most of what makes
#: an embedded plot look like a developer tool.
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom", "pan", "select", "lasso2d", "zoomIn", "zoomOut",
        "autoScale", "resetScale", "toggleSpikelines",
        "hoverClosestCartesian", "hoverCompareCartesian",
    ],
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
    "toImageButtonOptions": {
        # scale 3 so the export is usable in a document rather than a screenshot of
        # a screen. Plotly renders at this multiple rather than upscaling.
        "format": "png",
        "scale": 3,
        "filename": "watchboard",
    },
}


def png_config(filename: str, scale: int = 3) -> dict:
    """PLOTLY_CONFIG with the download named after what is on screen.

    Worth doing: a folder of `newplot.png`, `newplot(1).png` is unusable a week
    later, and the filename is the only label an exported image carries.
    """
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PLOTLY_CONFIG.items()}
    cfg["toImageButtonOptions"].update(filename=filename, scale=scale)
    return cfg


def _ordinal(p: float | None) -> str:
    """A percentile as an ordinal: 40.2 -> '40th', 1.4 -> '1st', NaN -> em dash.

    Rounded with the SAME format string the hero text uses (`:.0f`), because the
    bubble and the sentence above the chart print the same number and reading
    "40th" beside "41st percentile" would look like two different statistics.
    That inherits banker's rounding at exactly .5, which is a price worth paying
    for the two agreeing.

    An em dash for a missing rank rather than 'nan': a percentile is genuinely
    blank during a series' warm-up (the first observation is the max of a
    one-element set), and 'nanth' next to a real figure reads as a bug.
    """
    if p is None or pd.isna(p):
        return "—"
    v = float(p)
    # FLOOR, not round-to-nearest, and 100 reserved for exactly 100.
    #
    # Rounding made two symmetric false claims. 99.75 printed "100th", which means
    # "highest ever" -- and four real market/cohort series in the committed data hit
    # that band without being a record. Below the other end, 0.14 printed "0th",
    # which an expanding percentile can never be: the minimum attainable is 100/n,
    # since every observation ranks at least against itself. Flooring costs nothing
    # legible and cannot manufacture a record in either direction.
    n = 100 if v >= 100.0 else max(1, int(v))
    suffix = (
        "th" if n % 100 in (11, 12, 13)
        else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def _align_pctile(pct: pd.Series | None, index: pd.Index) -> pd.Series | None:
    """A caller's percentile series put onto the figure's own x grid.

    Returns None for "nothing to draw", which covers both `pct is None` and an
    all-NaN percentile (a market with one observation, where every rank is
    warm-up). Callers can then test one thing instead of three.

    Out-of-range values are dropped rather than drawn: `clamp_percentage` sends
    anything outside [0, 100] to NaN, so a caller that hands over a z-score or a
    0-1 fraction by mistake gets no bubble instead of a confident "0th".

    The positional fallback exists because lib.metrics returns a couple of its
    series on a RangeIndex (see how positioning.py handles `flow`), and silently
    reindexing one of those onto dates yields all-NaN -- the bubble would just
    not appear, which is the worst failure mode: invisible.
    """
    if pct is None:
        return None
    s = pd.to_numeric(pd.Series(pct), errors="coerce")
    if not s.index.equals(index):
        if len(s) == len(index) and not s.index.isin(index).any():
            s = pd.Series(s.to_numpy(), index=index)  # positional, see docstring
        else:
            s = s.reindex(index)
    # A 0-1 FRACTION IS THE LIKELY CALLER MISTAKE, and clamp_percentage cannot catch
    # it: 0.402 sits comfortably inside [0, 100] and would render a confident "0th",
    # which reads as record-low positioning. Refuse the series instead of drawing the
    # most alarming possible label from a unit error. A genuine percentile series
    # whose every value is under 1 would need every observation to be a near-record
    # low, which is not a thing worth pinning a bubble to either.
    finite = s.dropna()
    if len(finite) > 1 and float(finite.max()) <= 1.0:
        return None
    s = metrics.clamp_percentage(s)
    return s if s.notna().any() else None


def _endpoint(values: pd.Series, pct: pd.Series) -> tuple | None:
    """(x, y, percentile) at the LATEST observation, or None.

    Anchored on the last non-NaN of `values` -- the last point the line actually
    reaches -- and then abandoned if the rank there is missing. Drawing the
    bubble at the last point that happens to have BOTH would silently slide it
    back into the middle of the chart, where "this is where the series sits now"
    stops being true and nothing on screen says so.

    Positional throughout, not label lookups: `pct` has already been put on
    `values`' own grid by _align_pctile, and a label lookup would additionally
    have to cope with a duplicated report date, which returns a Series and makes
    `pd.isna` raise.
    """
    v = pd.to_numeric(values, errors="coerce")
    if len(v) != len(pct):
        return None
    live = v.notna().to_numpy().nonzero()[0]
    if live.size == 0:
        return None
    i = int(live[-1])
    p = pct.to_numpy(dtype=float)[i]
    if pd.isna(p):
        return None
    return v.index[i], float(v.to_numpy(dtype=float)[i]), float(p)


@dataclass
class Series:
    """One line on a panel, tagged with what kind of quantity it is."""

    values: pd.Series
    label: str
    #: A key from lib.metrics.SIGNED_KINDS or UNSIGNED_KINDS. Drives the
    #: percent-change gate and the zero line.
    kind: str = "gross"
    color: str | None = None
    dash: str | None = None
    fill: bool = False

    @property
    def signed(self) -> bool:
        return not metrics.is_ratio_safe(self.kind)


@dataclass
class Panel:
    """One subplot row."""

    series: list[Series]
    unit: str
    title: str = ""
    height: float = 1.0
    #: Horizontal reference lines, e.g. (10, 90) for percentile bands.
    guides: tuple[float, ...] = ()
    y_range: tuple[float, float] | None = None
    note: str = ""
    #: Vertical dividers, used to mark segment boundaries.
    breaks: tuple = field(default_factory=tuple)


def _is_oi_share(p: Panel) -> bool:
    return any(s.kind in ("net_share", "gross_share", "share") for s in p.series)


def _is_oi(p: Panel) -> bool:
    return any(s.kind == "open_interest" for s in p.series)


def stacked(
    panels: list[Panel],
    *,
    height_per_panel: int = 200,
    require_open_interest: bool = True,
    hovermode: str = "x unified",
) -> go.Figure:
    """Independent subplots sharing an x-axis. The only chart builder panels use.

    Raises if a figure shows a share of open interest without showing open
    interest itself (rule 4). Pass require_open_interest=False only for figures
    that contain no share-of-OI series at all -- it is a declaration, not an
    escape hatch, and the tests check that no panel passes it alongside a share.
    """
    panels = [p for p in panels if p.series]
    if not panels:
        raise ValueError("stacked() needs at least one panel with at least one series")

    if require_open_interest and any(_is_oi_share(p) for p in panels):
        if not any(_is_oi(p) for p in panels):
            raise ValueError(
                "display rule 4: this figure plots a share of open interest but no "
                "open-interest series. A cohort's share can rise because the cohort "
                "bought or because open interest fell, and the share alone cannot "
                "tell you which. Add a Panel with a Series(kind='open_interest')."
            )

    total = sum(p.height for p in panels)
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.06, 0.16 / max(len(panels), 1)),
        row_heights=[p.height / total for p in panels],
        subplot_titles=[p.title for p in panels],
    )

    ci = 0
    for r, p in enumerate(panels, start=1):
        for s in p.series:
            color = s.color
            if color is None:
                color = SERIES_COLORS[ci % len(SERIES_COLORS)]
                ci += 1
            fig.add_trace(
                go.Scatter(
                    x=s.values.index,
                    y=s.values.to_numpy(),
                    name=s.label,
                    mode="lines",
                    line=dict(color=color, width=1.6, dash=s.dash),
                    fill="tozeroy" if s.fill else None,
                    connectgaps=False,  # a gap in the data must look like a gap
                    hovertemplate=f"{s.label}: %{{y:,.2f}} {p.unit}<extra></extra>",
                ),
                row=r,
                col=1,
            )

        if any(s.signed for s in p.series):
            fig.add_hline(
                y=0, line=dict(color=ZERO_LINE, width=1, dash="dot"), row=r, col=1
            )
        for g in p.guides:
            fig.add_hline(
                y=g, line=dict(color=MUTED, width=1, dash="dot"), row=r, col=1
            )
        for b in p.breaks:
            fig.add_vline(
                x=b, line=dict(color=WARN, width=1, dash="dot"), row=r, col=1
            )

        fig.update_yaxes(
            title_text=p.unit,
            row=r,
            col=1,
            range=list(p.y_range) if p.y_range else None,
            gridcolor=GRID,
            zeroline=False,
        )
        fig.update_xaxes(row=r, col=1, gridcolor=GRID)

    fig.update_layout(
        height=height_per_panel * len(panels) + 90,
        hovermode=hovermode,
        showlegend=any(len(p.series) > 1 for p in panels),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=76, r=28, t=56, b=42),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12.5
    return fig


def spotlight(
    values: pd.Series,
    open_interest: pd.Series,
    *,
    unit: str,
    kind: str = "net_share",
    guides: tuple[float, ...] = (),
    breaks: tuple = (),
    height: int = 460,
    price: pd.Series | None = None,
    price_label: str = "",
    price_log: bool = False,
    pctile: pd.Series | None = None,
    oi_pctile: pd.Series | None = None,
) -> go.Figure:
    """One market, one measure. The polished single-series figure.

    Deliberately different from stacked(): that builder is for reading five things
    at once and its subplot titles, legend row and dense axis furniture are what
    made the board feel like a slide deck. This one is built to be GLANCED at.

    ONE X-AXIS, SEVERAL Y-AXES -- AND THAT IS THE WHOLE TRICK

    Built by hand rather than with make_subplots, because make_subplots gives each
    stacked panel its OWN x-axis and links them with `matches`. Linked axes zoom
    together but they are still separate hover targets: the crosshair is drawn only
    inside the panel the pointer happens to be in, and the tooltip lists only that
    panel's series. Reading positioning against price then means hovering twice and
    holding the date in your head. (plotly's `hoversubplots` is meant to fix that;
    it did not, on either hovermode, with either matches wiring.)

    Here there is exactly ONE x-axis and three y-axes at different vertical
    domains. All three traces live on that single axis, so "x unified" hover has
    nothing to reconcile -- one date, one tooltip, every series in it -- and the
    spike is a single line through the whole figure because there is only one
    x-axis for it to cross.

    Design decisions, each from the anti-pattern list rather than from taste:

    - ONE SERIES PER PANEL, SO NO LEGEND. A legend box for a single line is ink
      doing no work; the heading and the axis titles name the series instead.
    - SOLID HAIRLINE GRID, HORIZONTAL ONLY. Dashed gridlines read as a threshold
      when they are just a grid, and vertical grid on a dense time series is noise.
      Dashes are kept for real reference levels.
    - THE ZERO LINE IS THE BASELINE, not a gridline. For a net position, which side
      of zero you are on is the first thing to read.
    - A FILL TO ZERO at a tenth opacity, encoding distance-from-flat without adding
      a mark, and working for both signs.
    - PRICE ON TOP AND OPEN INTEREST AS A STRIP, never a twin axis. Price and
      percent-of-open-interest are dimensionally unrelated, so a shared y-scale
      would let the scaling choice manufacture whatever correlation the author
      wanted. Separate panels show the same comparison and cannot overstate it.
      Open interest is on screen because a share can move when the cohort trades or
      when open interest does, and the share alone cannot say which (rule 4).
    - CLIENT-SIDE RANGE BUTTONS, so zooming never touches the server.

    PERCENTILE BUBBLES, and why they are not a second scale

    `pctile` and `oi_pctile` are optional CAUSAL percentiles (lib.metrics) of
    `values` and `open_interest`. Each is drawn twice:

      - as a filled dot on that panel's LAST observation with a small label
        carrying the rank ("40th"), so the eye finishes the line and lands on
        where that endpoint sits in its own history;
      - as a row in the single tooltip, so the rank is readable at ANY date and
        not only at the end.

    Only the endpoint is labelled. A number on all 846 observations is a wall of
    digits that no one reads and that hides the line it annotates.

    RULE 1: the dot's y coordinate is the SERIES' OWN VALUE in the panel's own
    units, read straight off the line, and the percentile travels as TEXT in the
    label beside it. Nothing is mapped to a y position, so the panel still has
    exactly one scale, and there is no scaling choice for an author to hide. The
    alternative -- a 0-100 percentile axis on a panel measured in "% of open
    interest" or in contracts -- is precisely the twin axis rule 1 forbids.

    WHAT THE CALLER OWNS, since this function cannot check it: whether the rank is
    causal (use lib.metrics; `.rank(pct=True)` is rule 2) and whether it is
    computed within a segment. An expanding percentile of a raw open-interest
    LEVEL across a contract re-specification is a real trap -- NASDAQ-100 open
    interest steps 49,531 -> 255,954 on 2023-05-02 with no change in positioning,
    so every post-break week ranks near the 100th against pre-break history and
    the bubble would read "record high" for a units change. Segment it, or pass a
    trailing percentile, or accept that the OI bubble is ranking the contract
    definition as much as the crowd.
    """
    has_price = price is not None and price.notna().any()

    # Vertical domains, top to bottom, with gaps between them. Price takes the most
    # room: it is what a reader orients by. Open interest needs only enough to show
    # its shape and its steps.
    if has_price:
        dom_price = (0.58, 1.0)
        dom_main = (0.20, 0.50)
        dom_oi = (0.0, 0.12)
    else:
        dom_price = None
        dom_main = (0.30, 1.0)
        dom_oi = (0.0, 0.20)

    signed = not metrics.is_ratio_safe(kind)
    axis_font = dict(size=12, color=MUTED)
    tick_font = dict(size=11.5, color=MUTED)
    fig = go.Figure()

    # The three drawn lines carry no hover of their own. plotly builds one unified
    # label PER (x, y) subplot, so three traces on three y-axes give three separate
    # tooltips no matter how the axes are wired -- you would read price in one box
    # and positioning in another. Instead they are hover-silent and a single
    # invisible trace below carries every value for the hovered date.
    if has_price:
        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price.to_numpy(),
                mode="lines",
                line=dict(color=INK_SECONDARY, width=1.6),
                connectgaps=False,
                hoverinfo="skip",
                name=price_label or "price",
                yaxis="y3",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=values.index,
            y=values.to_numpy(),
            mode="lines",
            line=dict(color=ACCENT_LINE, width=2),
            fill="tozeroy",
            fillcolor=ACCENT_FILL,
            connectgaps=False,  # a gap in the data must look like a gap
            hoverinfo="skip",
            name=unit,
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=open_interest.index,
            y=open_interest.to_numpy(),
            mode="lines",
            line=dict(color=MUTED, width=1.2),
            connectgaps=False,
            hoverinfo="skip",
            name="open interest",
            yaxis="y2",
        )
    )

    # ONE TOOLTIP, AND IT MUST FIRE WHEREVER THE POINTER IS.
    #
    # plotly only raises a hover for the subplot the pointer is inside, so a single
    # invisible trace on the middle axis meant hovering the PRICE panel -- the
    # largest one, the one you look at first -- produced no crosshair and no tooltip
    # at all. Measured: a mouse sweep across the price panel fired 0 hover events.
    # You had to find the narrow middle strip to get a reading, which is most of
    # what "not smooth" felt like.
    #
    # So there is one zero-width hover trace PER AXIS. Each carries the same
    # customdata and the same template, so the box is identical wherever you are;
    # each takes its own panel's y-values so the label anchors near the line you are
    # actually looking at.
    px_at = (
        price.reindex(values.index).to_numpy()
        if has_price
        else [None] * len(values)
    )
    net_at = values.to_numpy()
    oi_at = open_interest.reindex(values.index).to_numpy()

    # Percentiles on the figure's own x grid. values.index IS that grid -- the
    # hover traces and everything derived from them are indexed by it -- so both
    # ranks are aligned to it once here, and the bubble below reads the same
    # aligned series the tooltip does. A pill and a tooltip disagreeing about the
    # same endpoint is the bug that alignment-per-consumer invites.
    pct_at = _align_pctile(pctile, values.index)
    oi_pct_at = _align_pctile(oi_pctile, values.index)

    # The rank goes into customdata as an ALREADY-FORMATTED ordinal, not as a
    # number with a `:.0f` in the template. Two reasons, both about NaN: plotly
    # renders a null through a numeric format as an empty string and a NaN as
    # "NaN", and neither is the em dash a blank warm-up rank should read as; and
    # "40th" needs the suffix rule anyway, which a format string cannot express.
    # Cost: ~850 short strings per hover trace instead of 850 floats.
    pct_txt = [_ordinal(p) for p in (pct_at if pct_at is not None else [])]
    oi_pct_txt = [_ordinal(p) for p in (oi_pct_at if oi_pct_at is not None else [])]
    blank = [""] * len(values)

    # A muted trailing chip, so the rank reads as a note on the row above it
    # rather than as a fourth quantity. f-string, not .format(): the template is
    # full of plotly's own %{...} braces and str.format cannot see the difference.
    def _rank_chip(slot: int) -> str:
        return f'  <span style="color:{MUTED}">%{{customdata[{slot}]}} pctile</span>'

    rows_tpl = []
    if has_price:
        rows_tpl.append(f"{price_label or 'price'}  <b>%{{customdata[0]:,.2f}}</b>")
    net_row = f"net  <b>%{{customdata[1]:,.2f}}</b> {unit}"
    oi_row = "open interest  <b>%{customdata[2]:,.0f}</b>"
    if pct_at is not None:
        net_row += _rank_chip(3)
    if oi_pct_at is not None:
        oi_row += _rank_chip(4)
    rows_tpl += [net_row, oi_row]
    tpl = "<br>".join(rows_tpl) + "<extra></extra>"
    # Slots 0-2 stay put: panels/positioning.py's frame-rate crosshair overlay
    # reads (price, net, open interest) by position off the first trace that
    # carries customdata, so the ranks are APPENDED rather than interleaved.
    combined = list(zip(
        px_at, net_at, oi_at,
        pct_txt or blank, oi_pct_txt or blank,
    ))

    hover_axes = [("y", net_at), ("y2", oi_at)]
    if has_price:
        hover_axes.append(("y3", px_at))
    for axis, yvals in hover_axes:
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=yvals,
                mode="lines",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                customdata=combined,
                hovertemplate=tpl,
                name="",
                yaxis=axis,
                showlegend=False,
            )
        )

    # THE PERCENTILE BUBBLE, ONE PER POSITIONING PANEL.
    #
    # NOT A SECOND SCALE (rule 1). The dot's y is the series' own last value, in
    # the panel's own units; the rank is TEXT in the label next to it. No 0-100
    # quantity is mapped to a y position anywhere, so each panel still has one
    # scale and there is no second scaling for an author to choose.
    #
    # Colour is the LINE'S OWN colour in both cases -- accent for the share,
    # near-gray for open interest -- so the bubble reads as belonging to the line
    # it terminates rather than as a third series that appears only at the right
    # edge. The dot gets a surface-coloured ring so it separates from the line
    # underneath instead of dissolving into it.
    #
    # Cost of the label being an annotation rather than trace text: annotations
    # are not clipped to the plot area, which is exactly what is wanted here (the
    # pill lives in the right margin, beyond the last observation) and is also
    # why the right margin below has to grow to make room for it.
    #
    # THE PILL IS ANCHORED TO PAPER x, NOT TO ITS DATE, AND THAT IS A BUG FIX.
    # An annotation on a data axis expands that axis's autorange to contain the
    # annotation's BOX, in pixels. A pill hung 9px past the final observation
    # therefore pushed the x range end ~11.5 months past the last report -- the
    # padding was right-side-only, which is what gave it away, since range[0] sat
    # exactly on the first observation. That alone was survivable in the "All"
    # view (a little trailing white space), but the range buttons are
    # stepmode="backward" and measure from range[1], so "1Y" resolved to a window
    # ending a year in the FUTURE and showed three weeks of data crushed against
    # the left edge under a full-history y scale. Paper x=1 puts the pill in the
    # same place on screen -- the right margin -- while touching no axis range.
    # It stays truthful because dragmode is False and scrollZoom is off, so the
    # only way to move x is the four range buttons and every one of them ends on
    # the last observation; the right edge IS the last observation. y stays in
    # data coordinates so the pill still sits at its own line's terminal height.
    annotations = []
    for axis, panel_values, panel_pct, colour in (
        ("y", values, pct_at, ACCENT_LINE),
        # Open interest on the shared grid, not its own index: this is the series
        # the tooltip reports, so pinning the pill to it keeps the two identical.
        ("y2", pd.Series(oi_at, index=values.index), oi_pct_at, MUTED),
    ):
        if panel_pct is None:
            continue
        end = _endpoint(panel_values, panel_pct)
        if end is None:
            continue  # no live endpoint, or its rank is warm-up: draw nothing
        at, y, p = end
        fig.add_trace(
            go.Scatter(
                x=[at],
                y=[y],
                mode="markers",
                marker=dict(size=8, color=colour,
                            line=dict(color=SURFACE, width=1.6)),
                cliponaxis=False,  # the last point sits ON the right edge
                hoverinfo="skip",  # one tooltip only; the rank is a row in it
                name="",
                yaxis=axis,
                showlegend=False,
            )
        )
        annotations.append(
            dict(
                xref="paper", yref=axis, x=1, y=y,
                text=f"<b>{_ordinal(p)}</b>",
                showarrow=False,
                xanchor="left", xshift=9, yanchor="middle",
                font=dict(family=FONT, size=11.5, color=colour),
                bgcolor=theme.c("chip"),
                bordercolor=theme.c("baseline"),
                borderwidth=1,
                borderpad=3,
            )
        )

    # Reference levels, drawn against a specific y-axis rather than a subplot row.
    shapes = []
    if signed:
        shapes.append(
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0,
                 line=dict(color=BASELINE, width=1.2), layer="below")
        )
    for g in guides:
        shapes.append(
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=g, y1=g,
                 line=dict(color=MUTED, width=1, dash="dot"), layer="below")
        )
    for b in breaks:
        # A contract re-specification. Spanning the full paper height rather than
        # one panel, because the break applies to every series at once.
        shapes.append(
            dict(type="line", xref="x", x0=b, x1=b, yref="paper", y0=0, y1=1,
                 line=dict(color=SEAL, width=1, dash="dot"), layer="below")
        )

    # The plotted x extent, taken from every series rather than from `values`
    # alone, so a price history that starts earlier or ends later is not cropped.
    x_union = values.index.union(open_interest.index)
    if has_price:
        x_union = x_union.union(price.index)
    x_first, x_last = x_union.min(), x_union.max()
    x_span_days = max(int((x_last - x_first).days), 1)

    # NO dtick ON THE LOG PRICE AXIS -- plotly picks, and that is deliberate,
    # because this axis is rescaled per zoom by the browser (see
    # panels/positioning._smooth_crosshair) and no fixed decade rule survives both
    # ends of that. dtick=1 is one tick per DECADE: an index running 2.4k to 26k
    # crosses one decade boundary, so the whole 42%-tall panel was labelled "10k"
    # and nothing else. dtick="D2" is the 1-2-5 ladder, which fixes the full-history
    # view (2k/5k/10k/20k) and then labels a zoomed 22.7k-31.4k window with NOTHING,
    # since no rung falls inside it. Plotly's own log autotick adapts to the span,
    # which is the only thing that can. tickformat "~s" keeps them as 2k/20k/30k.
    log_ticks = dict(tickformat="~s", minor=dict(showgrid=False)) if price_log else {}

    layout = dict(
        height=height,
        # MARGINS, AND THE ONE CLIPPING BUG THEY EXIST TO SURVIVE.
        #
        # Top carries the range buttons.
        #
        # Bottom is 66, up from 52. The x-axis tick band lives in it, and the card
        # around this figure is not guaranteed to be as tall as the figure asks:
        # Streamlit pins the chart card's height to the figure height with
        # box-sizing:border-box, so a 1px border plus vertical padding leaves a
        # content box ~9-12px SHORTER than the plot -- measured 758px of box for a
        # 760px plot, which macOS Chrome renders as a scrollbar sawn across the
        # bottom of the chart and, once that is suppressed, as the year labels
        # being eaten by the card edge. The CSS in app.py is the real fix (zero
        # border and zero padding, so the content box equals the figure height
        # exactly -- NOT height:auto, which was tried and broke the layout worse);
        # this margin is the belt to that braces, so a future off-by-one costs
        # empty pixels instead of the axis.
        #
        # Right grows when a percentile pill is drawn, because the pill hangs
        # PAST the last observation and the last observation can be flush with the
        # plot's right edge (any range-button zoom that ends at today puts it
        # there). Measured in the browser with the last point pinned to the edge:
        # the widest pill occupies 9px of offset plus ~40px of box, so 24 would
        # cut it in half and 52 still spilled 4px off the paper. 60 clears the
        # five-character worst case ("100th") with a few px to spare. The cost is
        # ~36px of plot width, ~3% of a full-width card, paid only by figures that
        # actually carry a rank.
        margin=dict(l=64, r=60 if annotations else 24, t=48, b=66),
        annotations=annotations,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family=FONT, size=12.5, color=INK_SECONDARY),
        showlegend=False,
        # One tooltip listing every series at the hovered date. With a single
        # x-axis this needs no hoversubplots gymnastics -- the traces already share
        # the axis that "unified" groups by.
        hovermode="x unified",
        # SNAP ALWAYS, NEVER BLINK. The defaults only show the spike and tooltip
        # within ~20px of a point, so moving along a weekly series makes the
        # crosshair strobe between observations. -1 means "always take the nearest
        # point on the x-axis", so it tracks the pointer continuously.
        spikedistance=-1,
        hoverdistance=-1,
        hoverlabel=dict(
            bgcolor=theme.c("chip"),
            bordercolor=theme.c("baseline"),
            font=dict(family=FONT, size=12.5, color=INK),
            align="left",
        ),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=theme.c("faint"),
                     activecolor=ACCENT_LINE),
        dragmode=False,
        transition=dict(duration=250, easing="cubic-in-out"),
        shapes=shapes,
        xaxis=dict(
            domain=(0.0, 1.0),
            anchor="y2",           # sits under the bottom panel
            showgrid=False,
            showline=True,
            linecolor=BASELINE,
            linewidth=1,
            ticks="",
            tickfont=tick_font,
            # The crosshair. One x-axis means one line through the whole figure.
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor=MUTED,
            spikedash="solid",
            # Explicit, so autorange's marker padding never applies. See
            # range_buttons() for the 347-day measurement behind this.
            range=[x_first, x_last],
            rangeselector=dict(
                buttons=range_buttons(x_span_days),
                bgcolor=theme.c("chip"),
                activecolor=ACCENT_FILL,
                bordercolor=theme.c("baseline"),
                borderwidth=1,
                font=dict(family=FONT, size=11.5, color=INK_SECONDARY),
                x=0, xanchor="left", y=1.0, yanchor="bottom",
            ),
        ),
        yaxis=dict(
            domain=dom_main,
            anchor="x",
            title=dict(text=unit, font=axis_font),
            gridcolor=GRID,
            zeroline=False,
            showline=False,
            ticks="",
            tickfont=tick_font,
        ),
        yaxis2=dict(
            domain=dom_oi,
            anchor="x",
            title=dict(text="open interest", font=axis_font),
            gridcolor=GRID,
            zeroline=False,
            showline=False,
            ticks="",
            rangemode="tozero",
            tickfont=tick_font,
        ),
    )
    if has_price:
        layout["yaxis3"] = dict(
            domain=dom_price,
            anchor="x",
            title=dict(text=price_label or "price", font=axis_font),
            gridcolor=GRID,
            zeroline=False,
            showline=False,
            ticks="",
            type="log" if price_log else "linear",
            tickfont=tick_font,
            **log_ticks,
        )

    fig.update_layout(**layout)
    return fig


def ranked_bars(
    labels: list[str],
    values: list[float],
    *,
    unit: str,
    kind: str = "share",
    hover: list[str] | None = None,
    height_per_bar: int = 21,
    center_on_zero: bool | None = None,
) -> go.Figure:
    """Horizontal ranked bars, for a cross-market screen.

    Diverging colour when the quantity can change sign, because a signed bar
    chart where positive and negative look the same is unreadable at a glance.
    """
    signed = not metrics.is_ratio_safe(kind)
    if center_on_zero is None:
        center_on_zero = signed
    colors = [WARN if v < 0 else ACCENT for v in values] if signed else ACCENT

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            customdata=hover or [""] * len(labels),
            hovertemplate="%{y}<br>%{x:,.2f} " + unit + "<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(240, height_per_bar * len(labels) + 96),
        xaxis_title=unit,
        margin=dict(l=8, r=24, t=16, b=44),
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.24,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=center_on_zero, zerolinecolor=ZERO_LINE)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


#: Kinds measured in whole contracts. Everything else here is a rate or a ratio
#: and needs decimals to be readable at all.
_COUNT_KINDS = frozenset(
    {"net", "gross", "flow", "change", "long", "short", "spread", "open_interest", "traders"}
)


def fmt_change(delta: float, kind: str, unit: str) -> str:
    """Format a change, refusing a percent for a sign-changing quantity.

    This is the formatting-time half of rule 3. `metrics.is_ratio_safe` raises on
    an unknown kind, so a new quantity cannot slip through by being unclassified.

    Resolution follows the QUANTITY, not the sign. Two earlier bugs, opposite
    directions, both from tying decimals to signedness:

      - a gross book change printed '-11,659.00 contracts' -- two decimals on an
        integer count
      - a net share change of +0.1555 percentage points printed '+0', because the
        signed branch used zero decimals. 21% of one market's weekly share moves
        are under 0.5pp, so they all rendered as a literal zero.

    A NaN is stated rather than printed as 'nan': a missing comparison is a fact
    about the data, and '+nan contracts' next to a real number reads as a bug.
    """
    safe = metrics.is_ratio_safe(kind)  # raises on an unknown kind -- fail closed
    if delta is None or pd.isna(delta):
        return "no comparable period"
    body = f"{delta:+,.0f}" if kind in _COUNT_KINDS else f"{delta:+,.2f}"
    if safe:
        return f"{body} {unit}"
    # Signed: absolute units only. "+3,913" on a short position means "less
    # short", not "more" -- pair this with signed_direction() for the word.
    return f"{body} {unit}"


def signed_direction(previous: float, current: float) -> str:
    """Plain-language description of a move in a sign-changing quantity.

    Exists because the number alone is ambiguous: "+3,913" on a short position
    means the cohort got LESS short, not more anything. Returns "" when either
    end is missing, so a caller can omit the parenthetical rather than print an
    assertion about a comparison it could not make.
    """
    if pd.isna(previous) or pd.isna(current):
        return ""
    if current == previous:
        return "unchanged"
    # Landing exactly on zero has no side, so a comparative is unreadable --
    # "less flat" was the old output and it grades a closed position by
    # magnitude against a side that no longer exists.
    if current == 0:
        return "closed to flat"
    crossed = (previous < 0 < current) or (current < 0 < previous)
    side = "long" if current > 0 else "short"
    if crossed:
        return f"flipped to net {side}"
    if previous == 0:
        return f"opened net {side}"
    return f"{'more' if abs(current) > abs(previous) else 'less'} {side}"
