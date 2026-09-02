"""The four display rules, enforced rather than documented.

lib/charts.py and lib/metrics.py make the correct thing the path of least
resistance, which is not the same as enforcement: a panel can always `import
plotly` and build whatever it likes, and a percentile is one keystroke away from
`.rank(pct=True)`. This file is the mechanism. Three questions:

  1. Does any panel reach past the library? (static analysis of panels/*.py --
     direct plotly imports, secondary_y, make_subplots, look-ahead ranking,
     waivers of the open-interest requirement, a missing BOARD.)
  2. Does the library keep its promises? (behavioural tests of lib/charts --
     rule 4 raises, no twin axes, gaps stay gaps, no percent on a signed kind.)
  3. Does every discovered board render? (AppTest over panels.all_boards(),
     plus the assertion that no module failed to import -- a board that silently
     vanishes looks like a design decision.)

It does NOT check that a chart is informative. The rules govern what a figure is
allowed to claim, not whether the claim is interesting.

The static half is ast/tokenize rather than regex because the banned things are
syntax: `secondary_y` is a keyword argument in one plotly idiom and a dict key in
another. Comments and docstrings are exempt from the identifier bans, because the
house style is to explain a rule beside the code obeying it and lib/charts.py's
own docstring names `secondary_y` twice. Explaining it is allowed; using it is not.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # self-contained: works without a conftest bootstrap
    sys.path.insert(0, str(ROOT))

import panels  # noqa: E402
import sources  # noqa: E402
from lib import charts, metrics  # noqa: E402

PANEL_DIR = ROOT / "panels"
APP = ROOT / "app.py"

#: Identifiers that only exist to build a second y-axis on one panel (rule 1).
BANNED_IDENTIFIERS = ("secondary_y", "make_subplots")


def panel_files() -> list[Path]:
    return sorted(PANEL_DIR.glob("*.py"))


def board_files() -> list[Path]:
    """Panel modules that must register a board. __init__.py is the registry."""
    return [p for p in panel_files() if p.name != "__init__.py"]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Line numbers covered by a module/class/function docstring."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", []) if isinstance(node, DOC_OWNERS) else []
        head = body[0] if body else None
        if isinstance(head, ast.Expr) and isinstance(getattr(head.value, "value", None), str):
            lines.update(range(head.lineno, (head.end_lineno or head.lineno) + 1))
    return lines


def _code_tokens(path: Path):
    """Tokens that are code: comments and docstrings dropped, other strings kept.

    Non-docstring strings stay in scope because plotly's own way to ask for a twin
    axis is a string key -- specs=[[{"secondary_y": True}]].
    """
    skip = _docstring_lines(_tree(path))
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.start[0] in skip:
            continue
        yield tok


def _where(path: Path, hits) -> str:
    return "; ".join(f"{path.name}:{line} {what}" for line, what in hits)


# --------------------------------------------------------------------------- #
# 1. Static analysis of panels/
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", panel_files(), ids=lambda p: p.name)
def test_panel_does_not_import_plotly(path: Path) -> None:
    """Rule 1. Routing every figure through lib.charts is what makes a second
    y-axis unreachable; a panel holding its own plotly handle undoes that."""
    def is_plotly(name: str) -> bool:
        return name == "plotly" or name.startswith("plotly.")

    hits = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            hits += [(node.lineno, f"import {a.name}") for a in node.names if is_plotly(a.name)]
        elif isinstance(node, ast.ImportFrom) and is_plotly(node.module or ""):
            hits.append((node.lineno, f"from {node.module} import ..."))
    assert not hits, (
        f"panel imports plotly directly -- {_where(path, hits)}. Build figures with "
        "lib.charts.stacked()/ranked_bars(); if charts.py genuinely cannot express "
        "the figure, extend charts.py rather than bypassing it."
    )


@pytest.mark.parametrize("path", panel_files(), ids=lambda p: p.name)
def test_panel_has_no_secondary_axis_machinery(path: Path) -> None:
    """Rule 1. Two series on one panel with two y-scales lets the author pick the
    scaling that makes a correlation look however they want, invisibly."""
    hits = [
        (tok.start[0], tok.string)
        for tok in _code_tokens(path)
        if any(b in tok.string for b in BANNED_IDENTIFIERS)
    ]
    assert not hits, (
        f"panel uses secondary-axis machinery -- {_where(path, hits)}. "
        "charts.stacked() gives independent subplots sharing only the x-axis, "
        "which is the only comparison this board is allowed to draw."
    )


def _is_literal(node: ast.expr | None, value: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


@pytest.mark.parametrize("path", board_files(), ids=lambda p: p.name)
def test_panel_has_no_lookahead_ranking(path: Path) -> None:
    """Rule 2. Both of these rank an observation against data that did not exist
    yet: .rank(pct=True) uses the whole sample, a centred window straddles t."""
    hits = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if name == "rank" and "pct" in kwargs:
            hits.append((node.lineno, ".rank(pct=...) ranks against the full sample"))
        centred = "center" in kwargs and not _is_literal(kwargs["center"], False)
        if name in ("rolling", "expanding") and centred:
            hits.append((node.lineno, f"{name}(center=...) straddles each date"))
    assert not hits, (
        f"look-ahead ranking -- {_where(path, hits)}. lib.metrics is causal by "
        "construction: expanding_percentile / trailing_percentile / trailing_zscore."
    )


@pytest.mark.parametrize("path", board_files(), ids=lambda p: p.name)
def test_panel_never_waives_the_open_interest_requirement(path: Path) -> None:
    """Rule 4, checked more strictly than charts.stacked() enforces it.

    The runtime rule is conditional -- the waiver is only wrong when the figure
    also carries a share-of-OI series -- and deciding that statically means
    resolving which `kind=` each Series was built with, data flow an ast walk
    cannot follow. So the waiver is banned outright: stricter than the rule, and
    free, since a figure with no share-of-OI series never trips the runtime check
    and loses nothing by leaving the default alone.
    """
    hits = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg == "require_open_interest" and not _is_literal(k.value, True):
                    hits.append((node.lineno, "require_open_interest waived"))
        # A kwargs dict is the other route in: stacked(**{"require_...": False}).
        elif isinstance(node, ast.Constant) and node.value == "require_open_interest":
            hits.append((node.lineno, "require_open_interest passed as a dict key"))
    assert not hits, (
        f"open-interest requirement waived -- {_where(path, hits)}. A cohort's "
        "share of OI can rise because the cohort bought or because OI fell, and "
        "the share alone cannot tell you which, so OI stays on screen."
    )


@pytest.mark.parametrize("path", board_files(), ids=lambda p: p.name)
def test_panel_defines_module_level_board(path: Path) -> None:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    assert "BOARD" in names, (
        f"{path.name} defines no module-level BOARD, so panels._discover() imports "
        "it, finds nothing to register and drops it without an error."
    )


# --------------------------------------------------------------------------- #
# 2. Behaviour of lib/charts.py
# --------------------------------------------------------------------------- #
def _weekly(values: tuple[float, ...]) -> pd.Series:
    idx = pd.date_range("2024-01-05", periods=len(values), freq="7D")
    return pd.Series(list(values), index=idx, dtype="float64")


def _panel(kind: str, values: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)) -> charts.Panel:
    return charts.Panel(series=[charts.Series(_weekly(values), kind, kind=kind)], unit="x")


@pytest.mark.parametrize("share_kind", ["net_share", "gross_share", "share"])
def test_stacked_requires_open_interest_beside_a_share_of_it(share_kind: str) -> None:
    """Rule 4. Refused alone, accepted next to an open-interest panel."""
    with pytest.raises(ValueError, match="display rule 4"):
        charts.stacked([_panel(share_kind)])
    assert len(charts.stacked([_panel(share_kind), _panel("open_interest")]).data) == 2


def test_stacked_gives_one_y_axis_per_panel_and_no_twin() -> None:
    """Rule 1, verified on the object rather than on the source that built it."""
    fig = charts.stacked([_panel("net"), _panel("open_interest"), _panel("share")])
    layout = fig.to_plotly_json()["layout"]
    yaxes = {k: v for k, v in layout.items() if k.startswith("yaxis")}
    assert len(yaxes) == 3, f"expected one y-axis per panel, got {sorted(yaxes)}"

    twin = [k for k, v in yaxes.items() if v.get("overlaying") or v.get("side") == "right"]
    assert not twin, f"twinned y-axis in a stacked figure: {twin}"

    # And no two traces share a scale, which is the same rule seen from the data.
    assert len({t["yaxis"] for t in fig.to_plotly_json()["data"]}) == 3


def test_every_trace_renders_a_gap_as_a_gap() -> None:
    """CFTC has integer-week holes and codes with multi-year ones. A joined line
    across a 12,341-day hole asserts continuity the data does not have."""
    fig = charts.stacked([_panel("net", (1.0, float("nan"), 3.0, 4.0))])
    for trace in fig.data:
        assert trace.connectgaps is False, f"{trace.name} would bridge a data gap"
    assert pd.isna(fig.data[0].y[1]), "the NaN must survive into the trace"


@pytest.mark.parametrize("kind", sorted(metrics.SIGNED_KINDS))
def test_fmt_change_refuses_a_percent_for_a_signed_kind(kind: str) -> None:
    """Rule 3. -100,640 -> -96,727 is '+3,913 contracts, less short'. As
    '+3.89%' it reads as growth while the position shrank toward zero."""
    out = charts.fmt_change(-96_727 - -100_640, kind, "contracts")
    assert "%" not in out, f"kind={kind!r} formatted as a percent: {out}"
    assert "3,913" in out


def test_fmt_change_raises_on_an_unknown_kind() -> None:
    """Failing closed is the point: a new quantity cannot slip past the gate by
    being unclassified."""
    with pytest.raises(ValueError, match="unknown quantity kind"):
        charts.fmt_change(1.0, "basis_points", "bp")


def test_ranked_bars_diverges_only_for_a_signed_kind() -> None:
    signed = charts.ranked_bars(["a", "b"], [12.0, -8.0], unit="contracts", kind="net")
    colors = signed.data[0].marker.color
    assert not isinstance(colors, str), "a signed bar chart needs per-bar colour"
    assert colors[0] == charts.ACCENT and colors[1] == charts.WARN
    assert signed.layout.xaxis.zeroline is True

    unsigned = charts.ranked_bars(["a", "b"], [12.0, 8.0], unit="%", kind="share")
    assert unsigned.data[0].marker.color == charts.ACCENT
    assert unsigned.layout.xaxis.zeroline is False


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        (-100_640, -96_727, "less short"),   # the real S&P print that started the rule
        (-96_727, -100_640, "more short"),
        (-10, 10, "flipped to net long"),
        (10, -10, "flipped to net short"),
        (5, 5, "unchanged"),
    ],
)
def test_signed_direction(previous: float, current: float, expected: str) -> None:
    assert charts.signed_direction(previous, current) == expected


# --------------------------------------------------------------------------- #
# 3. Every discovered board renders
# --------------------------------------------------------------------------- #
BOARDS = panels.all_boards()


@pytest.mark.parametrize(
    "kind,errors", [("board", panels.IMPORT_ERRORS), ("source", sources.IMPORT_ERRORS)]
)
def test_no_module_failed_to_import(kind: str, errors: dict[str, str]) -> None:
    """A module that raised on import is absent from its registry. It must fail
    the suite rather than quietly disappear from the sidebar."""
    assert errors == {}, f"{kind} module(s) failed to import: " + "; ".join(errors)


def test_boards_were_discovered() -> None:
    assert BOARDS, "panels/ registered no boards, so the smoke tests below are vacuous"


def _label_for(options: list[str], title: str) -> str:
    """app.py labels a board '<group> · <title>' when more than one group exists."""
    matches = [o for o in options if o == title or o.endswith(f"· {title}")]
    assert len(matches) == 1, f"{title!r} matched {matches} in the sidebar"
    return matches[0]


def _fresh_app():
    """A run of the real app.py, timeout generous because boards scan 4.1M rows."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_app_runs_clean_before_any_selection() -> None:
    """The first load, which renders whichever board sorts first."""
    at = _fresh_app()
    assert not at.exception, [str(e.value) for e in at.exception]


@pytest.mark.parametrize("board", BOARDS, ids=lambda b: b.id)
def test_board_renders_without_exception(board) -> None:
    """Selects each board in the real app and asserts it raised nothing.

    Parameterised over the registry, so a board added later is covered here
    without editing this file. Only the post-selection run is asserted on: the
    first run necessarily renders the default board (AppTest must run once before
    the sidebar radio exists to be set) and each run rebuilds the element tree, so
    scoping to the second run makes a broken default board fail its own case plus
    test_app_runs_clean_before_any_selection rather than every case at once, which
    would hide which file is at fault.
    """
    at = _fresh_app()
    radio = at.sidebar.radio[0]
    radio.set_value(_label_for(list(radio.options), board.title)).run()
    raised = [str(e.value) for e in at.exception]
    assert not raised, f"board {board.id!r} raised: {raised}"
    # Guards against the vacuous pass: if the selection did not take, every case
    # would be re-testing the default board and reporting success.
    assert board.title in [t.value for t in at.title], (
        f"selected {board.title!r} but the page rendered {[t.value for t in at.title]}"
    )
