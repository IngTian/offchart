"""Invariants of the browser-side overlay, enforced by reading its source.

These are source scans rather than behavioural tests, for the same reason the panel
scans in test_display_rules are: the code under test is JavaScript injected into a
Streamlit page, so exercising it needs a real browser and a running server. What a
scan CAN do is stop a specific mistake from coming back, and each one below is a
mistake that actually happened and cost a round of "the chart is broken".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import overlay  # noqa: E402

SRC = (ROOT / "lib" / "overlay.py").read_text()


def _code_only(text: str) -> str:
    """Source with comment lines dropped.

    Every scan here needs this. The comments deliberately quote the broken patterns
    they warn about, so a scan that cannot tell code from an explanation of the bug
    fails on the very comment documenting it -- which has already happened twice.
    """
    return "\n".join(
        ln for ln in text.splitlines()
        if not ln.lstrip().startswith("#") and not ln.lstrip().startswith("//")
    )


CODE = _code_only(SRC)


# ------------------------------------------------- the exit-fullscreen geometry

def test_geometry_is_repaired_after_a_resize() -> None:
    """Leaving plotly's fullscreen leaves the figure DRAWN at fullscreen size inside
    a normal-size card: _fullLayout said 1022x760 while the rendered box was
    1022x1042 and the x axis was 1858px long against the 880 it should be, so the
    series was painted across more than twice the visible width."""
    assert "ResizeObserver" in CODE, "nothing watches for the figure being resized"
    assert "function drift(" in CODE, "the drift check must exist to be testable"


def test_the_resize_observer_watches_the_figure_not_only_the_card() -> None:
    """In the broken state the CARD never changes size -- it is 1022x760 throughout.
    Only the figure's own box moves, so a card-only observer sees nothing at all."""
    observed = re.findall(r"ro\.observe\((\w+)\)", CODE)
    assert "gd" in observed, f"the figure must be observed, only saw {observed}"


def test_the_repair_relayouts_width_and_height() -> None:
    """Measured: Plotly.Plots.resize(gd) does NOT repair this state -- called it on a
    reproduced failure and nothing changed. A relayout carrying width and height
    does, even though those values already match _fullLayout, because it forces the
    size recompute that react's diff skips."""
    assert "Plots.resize" not in CODE, (
        "Plots.resize was measured not to fix this; if it is back, re-measure first"
    )
    body = CODE[CODE.index("function drift("):]
    assert "width:" in body and "height:" in body, (
        "the repair must relayout width and height"
    )


def test_the_repair_cannot_loop_forever() -> None:
    """It corrects the figure, which resizes the figure, which notifies the observer.
    Without a cap that is a spin, and a scrollbar is better than a spin."""
    assert "repairs" in CODE, "no attempt counter"
    assert re.search(r"repairs\s*>=?\s*\d", CODE), "the counter must bound something"


def test_the_repair_is_debounced() -> None:
    """Entering and leaving fullscreen both pass through frames where the box and the
    layout legitimately disagree; correcting mid-transition fights the transition."""
    assert "setTimeout" in CODE, "the repair must not fire on every observer tick"


def test_the_repair_only_acts_on_a_real_mismatch() -> None:
    """An unconditional relayout on every resize would re-enter plotly constantly and
    throw away the y ranges the zoom handler just computed."""
    body = CODE[CODE.index("function drift("):CODE.index("let repairing")]
    assert "return null" in body, "drift() must be able to report 'nothing wrong'"
    assert "Math.abs" in body, "the mismatch has to be measured, not assumed"


# --------------------------------------------------- things established earlier

def test_the_crosshair_reads_full_data_not_the_binary_spec() -> None:
    """gd.data[i].y is a binary typed-array spec ({dtype, bdata}) when plotly.py
    ships numeric columns, so iterating it reads nothing and silently rescales
    nothing. _fullData holds the decoded array."""
    assert "_fullData" in CODE
    assert "gd.data[i].y" not in CODE


def test_the_axis_is_read_per_frame_not_captured() -> None:
    """A relayout can hand back a REBUILT _fullLayout, so an axis captured at attach
    answers p2l with the old range -- the tooltip reported a date from the previous
    zoom window under a motionless pointer."""
    assert "gd._fullLayout && gd._fullLayout.xaxis" in CODE or \
           "gd._fullLayout.xaxis" in CODE


def test_it_uses_p2l_and_not_p2d() -> None:
    """On a date axis p2d returns a formatted STRING, so a numeric nearest-point
    search against it compares number < string, gets NaN, and pins the crosshair to
    the first observation forever."""
    assert "p2l" in CODE and "l2p" in CODE
    assert "p2d" not in CODE and "d2p" not in CODE


def test_the_chart_is_found_by_key_not_by_first_plot_in_the_document() -> None:
    """With two boards there can be more than one figure in the document, and a bare
    querySelector('.js-plotly-plot') attaches to whichever comes first."""
    assert "st-key-" in CODE, "the lookup must be scoped to the chart's own key"


def test_row_carries_its_own_formatting() -> None:
    """The tooltip rows are a passed-in spec, which is what let a second board reuse
    this instead of copying it."""
    row = overlay.Row("net", 1, 2, suffix=" % of OI")
    assert (row.label, row.slot, row.decimals, row.suffix) == ("net", 1, 2, " % of OI")
    assert row.omit_if_null is False
