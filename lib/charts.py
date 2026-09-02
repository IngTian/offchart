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

from lib import metrics

# A restrained, colour-blind-safe set. Order is the assignment order.
ACCENT = "#2E6F9E"
MUTED = "#9AA5B1"
WARN = "#B4623A"
POSITIVE = "#3F7F6E"
SERIES_COLORS = (ACCENT, WARN, POSITIVE, "#7A6B9E", "#8A7250", MUTED)

GRID = "rgba(154,165,177,0.22)"
ZERO_LINE = "rgba(154,165,177,0.85)"


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
