"""The four display rules, enforced rather than documented.

lib/charts.py and lib/metrics.py make the correct thing the path of least
resistance, which is not the same as enforcement: a panel can always `import
plotly` and build whatever it likes, and a percentile is one keystroke away from
`.rank(pct=True)`. This file is the mechanism. Four questions:

  1. Does any panel reach past the library? (ast scan of panels/*.py -- a plotly
     handle however it is spelled, secondary-axis machinery, look-ahead ranking, a
     self-computed percent change, waivers of the open-interest requirement, a
     missing BOARD.)
  2. Does the library keep its promises? (behavioural tests of lib/charts --
     rule 4 raises, no twin axes, gaps stay gaps, no percent on a signed kind.)
  3. Does every discovered board actually draw something, and does what it drew
     obey rules 1 and 4? (AppTest over panels.all_boards(), reading the plotly
     JSON of every figure the page produced.)
  4. Do the scans in (1) and (3) still catch anything at all? (negative controls:
     the shortest source that reaches each banned outcome must be flagged, and the
     prose that merely names it must not be. Every scan below passes on every
     panel today, so without these controls a scan that stopped working would look
     exactly like a codebase that complies.)

It does NOT check that a chart is informative. The rules govern what a figure is
allowed to claim, not whether the claim is interesting.

WHY AST AND NOT TOKENS OR REGEX

The banned things are syntax. `secondary_y` is a keyword argument in one plotly
idiom (`add_trace(..., secondary_y=True)`) and a dict key in another
(`specs=[[{"secondary_y": True}]]`); `yaxis2` arrives either as a keyword to
`update_layout` or as a key in a `**kwargs` dict. So the scan looks at keyword
names, attribute names, plain names, and string constants -- but a string constant
counts only when it is EXACTLY a banned name, because that is how plotly spells
the option. A caption or a comment that discusses `secondary_y` inside a sentence
is not a hit: explaining a rule beside the code obeying it is the house style, and
lib/charts.py's own docstring names `secondary_y` twice.

Two limits of the static half, stated rather than implied. Rule 2: it catches the
per-observation ranking idioms (`.rank`, `np.percentile`, a centred window) but
not `s.quantile([0.1, 0.9])` used as a bucket edge -- panels/base_rates.py does
exactly that, as a description of the whole sample's distribution rather than as a
rank of one observation, and an ast walk cannot tell those two uses apart. Rule 3:
it catches `pct_change` but not a hand-rolled `(now - prev) / prev`.

WHY THE RENDER HALF READS tests/fixtures/ AND NEVER data/

The load-bearing reason: app.py writes `st.title(board.title)` BEFORE it checks
whether the board's sources have any rows, so on a checkout where data/ has not
been ingested every board falls through to app.py's "not yet ingested" st.info,
and a smoke test that asserts only "raised nothing, right title" passes having
drawn nothing at all. The assertions here are therefore positive -- a figure must
exist, and the skip message must be absent -- and the data they run on is the
committed 0.5 MB slice, so they hold on a fresh clone and in CI before ingest.
(To see the guard bite, swap `fixture_data_dir` for conftest's empty
`tmp_data_dir`.) Second reason, the same one conftest gives for the rest of the
suite: which boards get covered should not vary with what today's ingest contains.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # self-contained: works without a conftest bootstrap
    sys.path.insert(0, str(ROOT))

import panels  # noqa: E402
import sources  # noqa: E402
from lib import charts, metrics, store  # noqa: E402

PANEL_DIR = ROOT / "panels"
APP = ROOT / "app.py"

#: Names that exist only to build, or to retrofit, a second y-axis (rule 1).
#: `overlaying` and `yaxis<n>` are the two halves of the shortest bypass there is:
#: take a figure charts.stacked() already built and
#: `update_layout(yaxis2=dict(overlaying="y", side="right"))`.
TWIN_AXIS_NAMES = frozenset({"secondary_y", "make_subplots", "overlaying"})
TWIN_AXIS_PATTERN = re.compile(r"^yaxis\d+$")

#: Live plotly handles that lib/charts.py holds at module level. `from lib.charts
#: import go` is a plotly import wearing a different name, and `charts.go` is the
#: same handle by attribute -- neither one mentions plotly.
PLOTLY_HANDLES = frozenset({"go", "px", "graph_objects", "graph_objs", "subplots", "make_subplots"})
CHARTS_MODULES = frozenset({"charts", "lib.charts"})

#: Whole-sample statistics that rank an observation against data that did not
#: exist yet. `quantile` is here only under a numpy/scipy handle: the pandas
#: method of the same name has a legitimate whole-sample use (see module docstring).
NUMPY_NAMES = frozenset({"np", "numpy", "scipy", "stats"})
NUMPY_LOOKAHEAD = frozenset({"percentile", "quantile", "percentileofscore"})


def panel_files() -> list[Path]:
    return sorted(PANEL_DIR.glob("*.py"))


def board_files() -> list[Path]:
    """Panel modules that must register a board. __init__.py is the registry."""
    return [p for p in panel_files() if p.name != "__init__.py"]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _where(path: Path, hits) -> str:
    return "; ".join(f"{path.name}:{line} {what}" for line, what in hits)


def _line(node: ast.AST) -> int:
    return getattr(node, "lineno", 0)


def _is_literal(node: ast.expr | None, value: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def _dotted(node: ast.expr) -> str:
    """The dotted source of an attribute chain, or "" if it is not one."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _called_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


# --------------------------------------------------------------------------- #
# The scans. Each returns [(lineno, what)] so the negative controls below can
# call it on a source string instead of on a file.
# --------------------------------------------------------------------------- #
def plotly_handle_hits(tree: ast.Module) -> list[tuple[int, str]]:
    """Rule 1. Any live plotly object a panel could build a figure with."""
    def is_plotly(name: str) -> bool:
        return name == "plotly" or name.startswith("plotly.")

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [(node.lineno, f"import {a.name}") for a in node.names if is_plotly(a.name)]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if is_plotly(module):
                hits.append((node.lineno, f"from {module} import ..."))
            elif module in CHARTS_MODULES:
                hits += [
                    (node.lineno, f"from {module} import {a.name} (a plotly handle)")
                    for a in node.names
                    if a.name in PLOTLY_HANDLES
                ]
        elif isinstance(node, ast.Attribute) and node.attr in PLOTLY_HANDLES:
            owner = _dotted(node.value)
            if owner in CHARTS_MODULES:
                hits.append((node.lineno, f"{owner}.{node.attr} (a plotly handle)"))
    return hits


def twin_axis_hits(tree: ast.Module) -> list[tuple[int, str]]:
    """Rule 1. Secondary-axis machinery, in every spelling that reaches plotly."""
    def banned(name: str) -> bool:
        return name in TWIN_AXIS_NAMES or bool(TWIN_AXIS_PATTERN.match(name))

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg and banned(node.arg):
            hits.append((_line(node) or _line(node.value), f"{node.arg}= keyword argument"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and banned(node.value):
            # Exactly the name, so a string used as a key or an attribute name --
            # prose that merely mentions it is a longer string and does not match.
            hits.append((node.lineno, f"{node.value!r} used as a key or name"))
        elif isinstance(node, ast.Name) and banned(node.id):
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and banned(node.attr):
            hits.append((node.lineno, f".{node.attr}"))
    return hits


def lookahead_hits(tree: ast.Module) -> list[tuple[int, str]]:
    """Rule 2. Ranking an observation against data that did not exist yet."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if name == "rank":
            how = "pct=..." if "pct" in kwargs else "no window"
            hits.append((node.lineno, f".rank({how}) ranks against the full sample"))
        if name in ("rolling", "expanding"):
            if "center" in kwargs and not _is_literal(kwargs["center"], False):
                hits.append((node.lineno, f"{name}(center=...) straddles each date"))
        if name in NUMPY_LOOKAHEAD and isinstance(node.func, ast.Attribute):
            owner = _dotted(node.func.value)
            if owner.split(".")[0] in NUMPY_NAMES:
                hits.append((node.lineno, f"{owner}.{name}() over the whole sample"))
    return hits


def percent_change_hits(tree: ast.Module) -> list[tuple[int, str]]:
    """Rule 3. A percent change computed in the panel bypasses the formatting gate
    in charts.fmt_change, which is the only thing that knows whether the quantity
    can change sign."""
    return [
        (node.lineno, "pct_change() -- route the change through charts.fmt_change")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == "pct_change"
    ]


def open_interest_waiver_hits(tree: ast.Module) -> list[tuple[int, str]]:
    """Rule 4, checked more strictly than charts.stacked() enforces it.

    The runtime rule is conditional -- the waiver is only wrong when the figure
    also carries a share-of-OI series -- and deciding that statically means
    resolving which `kind=` each Series was built with, data flow an ast walk
    cannot follow. So the waiver is banned outright: stricter than the rule, and
    free, since a figure with no share-of-OI series never trips the runtime check
    and loses nothing by leaving the default alone.
    """
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg == "require_open_interest" and not _is_literal(k.value, True):
                    hits.append((node.lineno, "require_open_interest waived"))
        # A kwargs dict is the other route in: stacked(**{"require_...": False}).
        elif isinstance(node, ast.Constant) and node.value == "require_open_interest":
            hits.append((node.lineno, "require_open_interest passed as a dict key"))
    return hits


# --------------------------------------------------------------------------- #
# 1. Static analysis of panels/
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", panel_files(), ids=lambda p: p.name)
def test_panel_does_not_import_plotly(path: Path) -> None:
    """Rule 1. Routing every figure through lib.charts is what makes a second
    y-axis unreachable; a panel holding its own plotly handle undoes that --
    including a handle borrowed from lib.charts, which re-exports two of them."""
    hits = plotly_handle_hits(_tree(path))
    assert not hits, (
        f"panel holds a plotly handle -- {_where(path, hits)}. Build figures with "
        "lib.charts.stacked()/ranked_bars(); if charts.py genuinely cannot express "
        "the figure, extend charts.py rather than bypassing it."
    )


@pytest.mark.parametrize("path", panel_files(), ids=lambda p: p.name)
def test_panel_has_no_secondary_axis_machinery(path: Path) -> None:
    """Rule 1. Two series on one panel with two y-scales lets the author pick the
    scaling that makes a correlation look however they want, invisibly."""
    hits = twin_axis_hits(_tree(path))
    assert not hits, (
        f"panel uses secondary-axis machinery -- {_where(path, hits)}. "
        "charts.stacked() gives independent subplots sharing only the x-axis, "
        "which is the only comparison this board is allowed to draw."
    )


@pytest.mark.parametrize("path", board_files(), ids=lambda p: p.name)
def test_panel_has_no_lookahead_ranking(path: Path) -> None:
    """Rule 2. Every one of these ranks an observation against data that did not
    exist yet: `.rank()` uses the whole sample with or without pct=, a centred
    window straddles t, np.percentile sees the end of the series from the start."""
    hits = lookahead_hits(_tree(path))
    assert not hits, (
        f"look-ahead ranking -- {_where(path, hits)}. lib.metrics is causal by "
        "construction: expanding_percentile / trailing_percentile / trailing_zscore."
    )


@pytest.mark.parametrize("path", board_files(), ids=lambda p: p.name)
def test_panel_does_not_compute_its_own_percent_change(path: Path) -> None:
    """Rule 3. charts.fmt_change is where a kind is checked before a percent is
    printed; a panel calling pct_change() itself has already decided the answer."""
    hits = percent_change_hits(_tree(path))
    assert not hits, (
        f"panel computes a percent change itself -- {_where(path, hits)}. A net "
        "position going -100,640 -> -96,727 is '+3,913 contracts, less short'; as "
        "'+3.9%' it reads as growth while the position shrank toward zero."
    )


@pytest.mark.parametrize("path", board_files(), ids=lambda p: p.name)
def test_panel_never_waives_the_open_interest_requirement(path: Path) -> None:
    """Rule 4. See open_interest_waiver_hits for why the ban is unconditional."""
    hits = open_interest_waiver_hits(_tree(path))
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
# 1b. Negative controls for the scans above.
#
# Every scan passes on every panel in the repo, which is indistinguishable from a
# scan that has stopped working. Each case below is the shortest source that
# reaches the banned outcome -- the `charts.go` and `update_layout` cases are the
# routes to a twin axis that an earlier version of this file did not see -- plus
# the near-misses that must NOT be flagged, because a false positive on prose is
# how a scan gets deleted.
# --------------------------------------------------------------------------- #
STATIC_CONTROLS = [
    ("import plotly", "import plotly.graph_objects as go", plotly_handle_hits, True),
    ("from plotly", "from plotly.subplots import make_subplots", plotly_handle_hits, True),
    ("charts re-export", "from lib.charts import go", plotly_handle_hits, True),
    ("charts attribute", "from lib import charts\ncharts.go.Figure()", plotly_handle_hits, True),
    ("charts used properly", "from lib import charts\ncharts.stacked([p])", plotly_handle_hits, False),
    ("secondary_y kwarg", "fig.add_trace(t, secondary_y=True)", twin_axis_hits, True),
    ("secondary_y spec key", 'make_subplots(specs=[[{"secondary_y": True}]])', twin_axis_hits, True),
    ("retrofit via update_layout",
     'fig.update_layout(yaxis2=dict(overlaying="y", side="right"))', twin_axis_hits, True),
    ("retrofit via kwargs dict",
     'fig.update_layout(**{"yaxis2": {"overlaying": "y"}})', twin_axis_hits, True),
    ("prose naming the rule",
     'st.caption("charts.stacked() never builds a secondary_y axis.")', twin_axis_hits, False),
    ("rank(pct=True)", "df[\"p\"] = s.rank(pct=True)", lookahead_hits, True),
    ("bare rank()", "df[\"p\"] = s.rank() / len(s)", lookahead_hits, True),
    ("np.percentile", "hi = np.percentile(s, 90)", lookahead_hits, True),
    ("np.quantile", "hi = np.quantile(s, 0.9)", lookahead_hits, True),
    ("centred window", "m = s.rolling(52, center=True).mean()", lookahead_hits, True),
    ("trailing window", "m = s.rolling(52).mean()", lookahead_hits, False),
    ("pandas quantile", "edges = s.quantile([0.1, 0.9])", lookahead_hits, False),
    ("pct_change", 'st.metric("net", f"{net.pct_change().iloc[-1]:.1%}")', percent_change_hits, True),
    ("fmt_change", 'charts.fmt_change(d, "net", "contracts")', percent_change_hits, False),
    ("waiver", "charts.stacked(ps, require_open_interest=False)", open_interest_waiver_hits, True),
    ("no waiver", "charts.stacked(ps)", open_interest_waiver_hits, False),
]


@pytest.mark.parametrize(
    "source,scan,should_flag",
    [(src, scan, flag) for _, src, scan, flag in STATIC_CONTROLS],
    ids=[label for label, *_ in STATIC_CONTROLS],
)
def test_static_scan_flags_what_it_claims_to(source, scan, should_flag) -> None:
    hits = scan(ast.parse(source))
    assert bool(hits) == should_flag, (
        f"{scan.__name__} {'missed' if should_flag else 'wrongly flagged'} "
        f"{source!r}: hits={hits}"
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
    spec = fig.to_plotly_json()
    assert len({k for k in spec["layout"] if k.startswith("yaxis")}) == 3
    problems = twin_axis_problems(spec)
    assert not problems, problems
    # And no two traces share a scale, which is the same rule seen from the data.
    assert len({t["yaxis"] for t in spec["data"]}) == 3


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
    # The same refusal at a magnitude that is in range for the two signed kinds
    # that are ratios rather than contract counts (purity lives in [-1, 1]), so
    # this test is not resting entirely on a delta three of the five kinds could
    # carry and two of them arithmetically could not.
    small = charts.fmt_change(0.42, kind, "of open interest")
    assert "%" not in small, f"kind={kind!r} formatted as a percent: {small}"


def test_fmt_change_keeps_a_percent_and_its_resolution_for_an_unsigned_kind() -> None:
    """The permitted half of the gate, asserted directly rather than inferred from
    a bar colour: the signed cases above would also pass if fmt_change returned
    one string for every kind."""
    assert charts.fmt_change(0.1234, "share", "%") == "+0.12 %"
    # Resolution follows the QUANTITY, not the sign. A contract count is an
    # integer, so it gets no decimals whichever branch of the gate it takes --
    # this used to render '+3,913.00 contracts'. The mirror bug was worse: the
    # signed branch used zero decimals, so a +0.1555pp share move printed '+0'.
    assert charts.fmt_change(-96_727 - -100_640, "gross", "contracts") == "+3,913 contracts"
    assert charts.fmt_change(0.1555, "net_share", "pp") == "+0.16 pp"


def test_fmt_change_raises_on_an_unknown_kind() -> None:
    """Failing closed is the point: a new quantity cannot slip past the gate by
    being unclassified."""
    with pytest.raises(ValueError, match="unknown quantity kind"):
        charts.fmt_change(1.0, "basis_points", "bp")


def test_fmt_change_does_not_render_a_nan_as_a_figure() -> None:
    """Rule 3 with the value metrics.flow actually returns for a broken span."""
    assert "nan" not in charts.fmt_change(float("nan"), "flow", "contracts").lower()


@pytest.mark.parametrize(
    "value, expected",
    [
        # The suffix rule itself. The hero line used to paste a hardcoded "th" onto
        # the number, so EVERY rank ending 1, 2 or 3 was printed wrong -- "53th"
        # sat next to a chart bubble reading "53rd", the same statistic disagreeing
        # with itself on one screen.
        (1.0, "1st"), (2.0, "2nd"), (3.0, "3rd"), (4.0, "4th"),
        (21.0, "21st"), (22.0, "22nd"), (23.0, "23rd"), (53.0, "53rd"),
        (91.0, "91st"), (92.0, "92nd"),
        # The teens are the exception to the exception.
        (11.0, "11th"), (12.0, "12th"), (13.0, "13th"),
        # Floors rather than rounds, so a near-record cannot print as a record and
        # a small positive rank cannot print as the impossible "0th".
        (99.75, "99th"), (100.0, "100th"), (0.14, "1st"), (40.2, "40th"),
    ],
)
def test_ordinal_suffix_and_flooring(value: float, expected: str) -> None:
    assert charts._ordinal(value) == expected
    number, suffix = charts.ordinal_parts(value)
    assert number + suffix == expected, "the split rendering must agree with the joined one"


def test_ordinal_of_a_missing_rank_is_not_a_figure() -> None:
    """A warm-up rank is genuinely blank, and 'nanth' beside a real number reads as
    a bug rather than as an absence."""
    assert charts._ordinal(float("nan")) == "—"
    assert charts.ordinal_parts(float("nan")) == ("—", "")
    assert charts.ordinal_parts(None) == ("—", "")


def test_hero_renders_its_rank_through_the_shared_ordinal() -> None:
    """The hero must not build its own ordinal. It did, and printed '53th'.

    Read as source rather than rendered, because rendering needs a Streamlit script
    context; the defect was a hardcoded suffix in the f-string, which is visible
    here.
    """
    src = (ROOT / "panels" / "positioning.py").read_text()
    hero = src[src.index("def _hero("):]
    hero = hero[: hero.index("\ndef ")]
    # Comments are stripped before scanning. The fix carries a comment quoting the
    # old broken format string, and a scan that cannot tell code from a description
    # of the bug would forbid explaining it -- so it would fail on the fix itself.
    code = "\n".join(
        ln for ln in hero.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "ordinal_parts" in code, "the hero must use the shared ordinal rule"
    assert '"k-unit">th<' not in code.replace(" ", ""), "hardcoded 'th' suffix is back"
    assert ":.0f}" not in code, "the rank must not be rounded back into a false record"


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
        # metrics.flow returns NaN for a span that is not one week, so both ends
        # of this arrive NaN in practice. Saying nothing is the right answer.
        (float("nan"), 10, ""),
        (10, float("nan"), ""),
    ],
)
def test_signed_direction(previous: float, current: float, expected: str) -> None:
    assert charts.signed_direction(previous, current) == expected


def test_signed_direction_of_a_position_closed_to_zero() -> None:
    """Landing on zero has no side, so a comparative is unreadable there.

    Was an xfail: both directions returned 'less flat', grading a closed position
    by magnitude against a side it no longer has. The mirror case matters too --
    leaving zero is an opening, not a 'more'.
    """
    assert charts.signed_direction(-5, 0) == "closed to flat"
    assert charts.signed_direction(5, 0) == "closed to flat"
    assert charts.signed_direction(0, -5) == "opened net short"
    assert charts.signed_direction(0, 5) == "opened net long"
    # The ordinary cases must still read correctly, including the one this whole
    # helper exists for: a short position shrinking is "less short", not "more".
    assert charts.signed_direction(-100_640, -96_727) == "less short"
    assert charts.signed_direction(-10, 10) == "flipped to net long"
    assert charts.signed_direction(5, 5) == "unchanged"


# --------------------------------------------------------------------------- #
# 3. Every discovered board renders, and what it rendered obeys the rules
# --------------------------------------------------------------------------- #
BOARDS = panels.all_boards()


def _axis_title(axis: dict) -> str:
    title = axis.get("title") if isinstance(axis, dict) else None
    if isinstance(title, dict):
        title = title.get("text")
    return title or ""


def twin_axis_problems(spec: dict) -> list[str]:
    """Rule 1 read off a rendered figure rather than off the source that built it.

    Closes the gap the static scan cannot: a panel that post-processes a figure
    charts.stacked() handed back, through any indirection an ast walk loses track
    of, still has to put the second scale in this JSON.
    """
    layout = spec.get("layout", {})
    data = spec.get("data", [])
    yaxes = {k: v for k, v in layout.items() if k.startswith("yaxis")}
    problems = [
        f"{k} is overlaid or right-hand: overlaying={v.get('overlaying')!r} side={v.get('side')!r}"
        for k, v in yaxes.items()
        if isinstance(v, dict) and (v.get("overlaying") or v.get("side") == "right")
    ]
    scales = {d.get("yaxis", "y") for d in data}
    if data and len(scales) > max(len(yaxes), 1):
        problems.append(f"{len(scales)} trace scales {sorted(scales)} but {len(yaxes)} y-axes")
    return problems


def open_interest_problems(spec: dict) -> list[str]:
    """Rule 4 on a one-panel figure -- the case charts.stacked() cannot police.

    stacked() raises when a share-of-OI panel appears with no OI panel, and
    test_stacked_requires_open_interest_beside_a_share_of_it covers that. A ranked
    bar chart has no panels, so ranked_bars() has no equivalent gate and the only
    place open interest can be is the hover. panels/crowding.py does put it there
    ("<n> spreads of <oi> OI"); nothing checked that it did until here.
    """
    layout = spec.get("layout", {})
    if len({k for k in layout if k.startswith("yaxis")}) > 1:
        return []  # multi-panel: stacked() enforced this when it built the figure
    axes = [_axis_title(v) for k, v in layout.items() if k.startswith(("xaxis", "yaxis"))]
    share_axes = [t for t in axes if "%" in t and "open interest" in t.lower()]
    if not share_axes:
        return []
    for i, trace in enumerate(spec.get("data", [])):
        hover = " ".join(str(c) for c in (trace.get("customdata") or []))
        if not re.search(r"open interest|\bOI\b", hover, re.I):
            return [
                f"axis {share_axes} plots a share of open interest but trace {i} "
                "never states the open interest it is a share of"
            ]
    return []


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


def _fresh_app(data_dir: Path):
    """A run of the real app.py against a given data directory.

    Takes the directory rather than reading store.DATA_DIR itself so a caller that
    forgot to request `fixture_data_dir` fails here instead of silently reading
    data/ -- see the module docstring for why that matters.
    """
    from streamlit.testing.v1 import AppTest

    assert store.DATA_DIR == data_dir, (
        f"lib.store.DATA_DIR is {store.DATA_DIR}, not {data_dir}: request the "
        "fixture_data_dir fixture so this renders the committed slice"
    )
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def _figure_specs(at) -> list[dict]:
    """The plotly JSON of every chart on the page.

    AppTest has no typed accessor for st.plotly_chart, but the untyped
    at.get("plotly_chart") returns the elements and each carries the figure it was
    handed as `proto.spec`.
    """
    return [json.loads(el.proto.spec) for el in at.get("plotly_chart")]


def _assert_the_board_actually_drew_something(at, what: str) -> list[dict]:
    """The anti-vacuity guard, and the reason the render tests read fixtures.

    app.py emits st.title(board.title) before it checks for data, so "no exception
    and the right title" is also what a board that never ran looks like.

    Both halves are needed. The skip message covers the six boards that declare a
    source; the Integrity board declares none (it reconciles whatever is present)
    so app.py never skips it and it prints its own "nothing to reconcile" notice
    instead -- only "a figure exists" catches that one.
    """
    skipped = [i.value for i in at.info if "not been ingested" in i.value]
    assert not skipped, f"{what} never ran -- app.py skipped it for missing data: {skipped}"
    specs = _figure_specs(at)
    assert specs, f"{what} rendered no figure, so this case asserted nothing"
    return specs


def test_app_runs_clean_before_any_selection(fixture_data_dir) -> None:
    """The first load, which renders whichever board sorts first."""
    at = _fresh_app(fixture_data_dir)
    assert not at.exception, [str(e.value) for e in at.exception]
    _assert_the_board_actually_drew_something(at, "the default board")


@pytest.mark.parametrize("board", BOARDS, ids=lambda b: b.id)
def test_board_renders_without_exception(board, fixture_data_dir) -> None:
    """Selects each board in the real app, asserts it raised nothing, drew
    something, and that what it drew obeys rules 1 and 4.

    Parameterised over the registry, so a board added later is covered here
    without editing this file. Only the post-selection run is asserted on: the
    first run necessarily renders the default board (AppTest must run once before
    the sidebar radio exists to be set) and each run rebuilds the element tree, so
    scoping to the second run makes a broken default board fail its own case plus
    test_app_runs_clean_before_any_selection rather than every case at once, which
    would hide which file is at fault.
    """
    at = _fresh_app(fixture_data_dir)
    if at.sidebar.radio:
        radio = at.sidebar.radio[0]
        radio.set_value(_label_for(list(radio.options), board.title)).run()
    else:
        # app.py renders no board selector when there is only one board, because a
        # picker with one option is chrome that costs a rerun. Assert that is
        # actually why the radio is missing, rather than treating any absent
        # selector as fine -- otherwise a broken sidebar would read as a pass.
        assert len(BOARDS) == 1, (
            f"no board selector, but {len(BOARDS)} boards are registered: "
            f"{[b.id for b in BOARDS]}"
        )
    raised = [str(e.value) for e in at.exception]
    assert not raised, f"board {board.id!r} raised: {raised}"
    specs = _assert_the_board_actually_drew_something(at, f"board {board.id!r}")
    problems = [
        f"figure {i}: {p}"
        for i, spec in enumerate(specs)
        for p in twin_axis_problems(spec) + open_interest_problems(spec)
    ]
    assert not problems, f"board {board.id!r} broke a display rule on screen: {problems}"


#: Negative controls for the two figure scans, for the same reason as the static
#: ones: every figure the boards draw today passes both, so a scan that stopped
#: working would be invisible. Hand-built specs, because producing a twin axis
#: needs the plotly call this repo does not allow a panel to make.
FIGURE_CONTROLS = [
    (
        "overlaid right-hand axis",
        {
            "layout": {"yaxis": {}, "yaxis2": {"overlaying": "y", "side": "right"}},
            "data": [{"yaxis": "y"}, {"yaxis": "y2"}],
        },
        twin_axis_problems,
        True,
    ),
    (
        "more trace scales than axes",
        {"layout": {"yaxis": {}}, "data": [{"yaxis": "y"}, {"yaxis": "y2"}]},
        twin_axis_problems,
        True,
    ),
    (
        "one scale per panel",
        {"layout": {"yaxis": {}, "yaxis2": {}}, "data": [{"yaxis": "y"}, {"yaxis": "y2"}]},
        twin_axis_problems,
        False,
    ),
    (
        "share of OI with no OI in the hover",
        {
            "layout": {"xaxis": {"title": {"text": "% of open interest"}}, "yaxis": {}},
            "data": [{"type": "bar", "customdata": ["40.2% of the side"]}],
        },
        open_interest_problems,
        True,
    ),
    (
        "share of OI with OI in the hover",
        {
            "layout": {"xaxis": {"title": {"text": "% of open interest"}}, "yaxis": {}},
            "data": [{"type": "bar", "customdata": ["34,081 spreads of 322,190 OI"]}],
        },
        open_interest_problems,
        False,
    ),
    (
        "not a share of OI at all",
        {
            "layout": {"xaxis": {"title": {"text": "contracts"}}, "yaxis": {}},
            "data": [{"type": "bar", "customdata": ["net -67,709 to -44,316"]}],
        },
        open_interest_problems,
        False,
    ),
]


@pytest.mark.parametrize(
    "spec,scan,should_flag",
    [(spec, scan, flag) for _, spec, scan, flag in FIGURE_CONTROLS],
    ids=[label for label, *_ in FIGURE_CONTROLS],
)
def test_figure_scan_flags_what_it_claims_to(spec, scan, should_flag) -> None:
    problems = scan(spec)
    assert bool(problems) == should_flag, (
        f"{scan.__name__} {'missed' if should_flag else 'wrongly flagged'} "
        f"{spec}: {problems}"
    )
