"""The dashboard JSON, linted. This is what keeps a config display layer honest.

Moving the renderer to Grafana costs the repo its strongest check: today
tests/test_display_rules reads the RENDERED plotly figure and asserts on what was
actually drawn. There is no equivalent for a Grafana panel from Python. What there IS
is a committed JSON file that is authoritative -- provisioned with
allowUiUpdates: false, so Grafana refuses UI saves -- and these tests lint it.

Two of them matter more than the rest and are worth reading before editing a
dashboard:

  test_no_panel_query_ranks_non_causally
      PERCENT_RANK() OVER (ORDER BY share) is textbook look-ahead and reads as
      perfectly reasonable SQL. Every rank on these boards is precomputed in pandas by
      scripts/build precisely so a panel never has to; this test is the negative
      control that keeps it that way.

  test_every_series_declares_its_axis_side
      Written as a POSITIVE assertion on purpose. Grafana omits default-valued keys
      from saved JSON and axisPlacement defaults to "auto", which means "first field
      left, everything else right". So a lint that looks for axisPlacement == "right"
      passes on a dashboard that visibly renders a right-hand axis -- the dangerous
      case is exactly the one an absence-based check cannot see.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DASH_DIR = ROOT / "grafana" / "dashboards"
DASHBOARDS = sorted(DASH_DIR.glob("*.json"))

#: SQL that ranks a whole partition rather than a causal window. Any of these in a
#: panel query means the number on screen used information from the future.
NON_CAUSAL_SQL = re.compile(
    r"\b(percent_rank|cume_dist|ntile)\s*\(|\brank\s*\(\s*\)\s*over\b", re.I
)

#: Tables a panel may read. The serving tables are built by scripts/build with the
#: ranks already computed; the archive tables are long-format and reading them from a
#: panel invites recomputing a statistic in SQL.
SERVING_TABLES = {"board_positioning", "board_inflation"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _panels(dash: dict) -> list[dict]:
    """Every panel, including any nested in a collapsed row."""
    out = []
    for panel in dash.get("panels", []):
        out.append(panel)
        out.extend(panel.get("panels", []) or [])
    return out


def _queries(dash: dict) -> list[str]:
    sql = []
    for panel in _panels(dash):
        for target in panel.get("targets", []) or []:
            for field in ("rawQueryText", "queryText", "rawSql"):
                if target.get(field):
                    sql.append(target[field])
    for var in (dash.get("templating", {}) or {}).get("list", []) or []:
        if isinstance(var.get("query"), str) and var["query"]:
            sql.append(var["query"])
    return sql


def test_there_is_at_least_one_dashboard() -> None:
    """Otherwise every parametrised test below silently covers nothing."""
    assert DASHBOARDS, f"no dashboards found in {DASH_DIR}"


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_dashboard_is_valid_json_with_a_stable_uid(path: Path) -> None:
    dash = _load(path)
    assert dash.get("uid"), "a dashboard without a uid gets a new one on every provision"
    assert dash.get("title")


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_no_panel_query_ranks_non_causally(path: Path) -> None:
    """The most important test in this file. See the module docstring."""
    for sql in _queries(_load(path)):
        hit = NON_CAUSAL_SQL.search(sql)
        assert not hit, (
            f"{path.name} ranks in SQL with {hit.group(0)!r}. Every rank on these "
            "boards is causal and precomputed by scripts/build; a window function over "
            "the whole partition is look-ahead bias."
        )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_panel_queries_read_only_the_serving_tables(path: Path) -> None:
    """SQLite has no per-role grants, so this test IS the read/serve separation.

    Reading an archive table from a panel means recomputing a derived quantity in SQL,
    which is how the causal guarantee gets lost one convenient query at a time.
    """
    for sql in _queries(_load(path)):
        for table in re.findall(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.I):
            assert table in SERVING_TABLES, (
                f"{path.name} queries {table!r}; panels may read only "
                f"{sorted(SERVING_TABLES)}"
            )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_no_panel_query_sums_a_market_week_column(path: Path) -> None:
    """open_interest is a market-week quantity repeated across cohort rows, so summing
    it multiplies by the cohort count -- a plausible number three to five times too
    big. The serving tables are already one row per grid timestamp, so a panel has no
    reason to aggregate at all."""
    for sql in _queries(_load(path)):
        assert not re.search(r"\bsum\s*\(\s*open_interest", sql, re.I), (
            f"{path.name} sums open_interest; it is already per market-week"
        )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_every_series_declares_its_axis_side(path: Path) -> None:
    """Positive assertion, because absence is the dangerous case. See the docstring."""
    for panel in _panels(_load(path)):
        if panel.get("type") != "timeseries":
            continue
        custom = ((panel.get("fieldConfig") or {}).get("defaults") or {}).get("custom") or {}
        placement = custom.get("axisPlacement")
        assert placement == "left", (
            f"{path.name} panel {panel.get('title')!r} has axisPlacement="
            f"{placement!r}. It must say 'left' explicitly: the default is 'auto', "
            "which puts the first field left and EVERYTHING ELSE RIGHT, and Grafana "
            "does not write defaults into the JSON -- so a missing key renders a "
            "second y axis that no absence-based lint can see."
        )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_gaps_are_drawn_as_gaps(path: Path) -> None:
    """A missing observation must not be joined across. The series here genuinely have
    holes -- a survey question not asked, a market with no report that week."""
    for panel in _panels(_load(path)):
        if panel.get("type") != "timeseries":
            continue
        custom = ((panel.get("fieldConfig") or {}).get("defaults") or {}).get("custom") or {}
        assert custom.get("spanNulls") is False, (
            f"{path.name} panel {panel.get('title')!r} must set spanNulls false "
            "explicitly rather than relying on the default"
        )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_a_share_of_open_interest_is_shown_beside_open_interest(path: Path) -> None:
    """Rule 4, and it is easier to express here than against a plotly figure: a share
    can move because the cohort traded or because open interest did, and the share
    alone cannot say which."""
    panels = _panels(_load(path))
    titles = " ".join((p.get("title") or "").lower() for p in panels)
    queries = " ".join(_queries(_load(path))).lower()
    shows_share = "share" in queries or "% of open interest" in titles
    if not shows_share:
        return
    assert "open_interest" in queries, (
        f"{path.name} charts a share of open interest without charting open interest"
    )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_the_crosshair_is_shared_across_panels(path: Path) -> None:
    """graphTooltip 2 is shared crosshair AND tooltip; 0 is neither. Stacked panels
    that do not share a cursor are three charts, not one reading."""
    dash = _load(path)
    assert dash.get("graphTooltip") == 2, (
        f"{path.name} has graphTooltip={dash.get('graphTooltip')!r}; 2 shares the "
        "crosshair and tooltip across panels"
    )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_the_dashboard_is_utc(path: Path) -> None:
    """Every timestamp in this data is a calendar date at UTC midnight. A
    browser-local dashboard west of Greenwich shifts every point back a day."""
    assert _load(path).get("timezone") == "utc"


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_time_is_projected_in_seconds_not_milliseconds(path: Path) -> None:
    """The SQLite plugin reads a time column as epoch SECONDS. Measured: passing the
    stored millisecond value renders 1953-11-19 instead of 2026-09-08 -- a wrong chart
    that looks like a real one. ts is stored in ms so the WHERE clause can compare
    against $__from/$__to directly and use the index, so every projection must divide.
    """
    for sql in _queries(_load(path)):
        if " as time" not in sql.lower():
            continue
        assert re.search(r"ts\s*/\s*1000\s+as\s+time", sql, re.I), (
            f"{path.name} projects a time column without dividing ts by 1000: {sql[:120]}"
        )


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_the_time_filter_uses_the_indexed_column_directly(path: Path) -> None:
    """The plugin has no $__timeFilter macro, so the bound is hand-written. It must
    compare ts itself -- `ts/1000 BETWEEN ...` would not use the index."""
    for sql in _queries(_load(path)):
        if "$__from" not in sql:
            continue
        assert re.search(r"\bts\s+between\s+\$__from\s+and\s+\$__to", sql, re.I), (
            f"{path.name} filters time in a form that cannot use the ts index: {sql[:120]}"
        )
