"""Watchboard -- CFTC positioning, reproduced from the primary source.

    streamlit run app.py

Reads only from data/*.parquet. It never fetches, so it works offline and it is
impossible for a chart to show a number that is not in the committed dataset.
Ingest is a separate job (scripts/ingest.py, run by CI).

This file is deliberately almost empty. Boards live in panels/ and sources in
sources/, both discovered, so adding either touches no dispatch table here. The
display rules are enforced in lib/charts.py and lib/metrics.py rather than
restated per board.

WHY THERE IS SO LITTLE CHROME

Streamlit reruns the whole script on every widget interaction. A page with a row
of controls therefore feels like a slide advancing, because that is structurally
what it is. So the board carries no widgets: the zoom presets and the crosshair
are plotly's and run in the browser, and this page never reruns while it is read.
The sidebar appears only when there is more than one board to choose between.
"""
from __future__ import annotations

import streamlit as st

import panels
from lib import cache
from lib.ui import caveat_block, freshness  # noqa: F401  -- re-exported for convenience
from sources import IMPORT_ERRORS as SOURCE_IMPORT_ERRORS
from sources import all_sources

st.set_page_config(
    page_title="Watchboard", page_icon="◧", layout="centered",
    initial_sidebar_state="collapsed",
)

BOARDS = panels.all_boards()

# Chrome off, type and spacing set once. Streamlit's defaults are what make a
# board read as a slide: a toolbar, a footer, a 6rem top pad and a heading scale
# built for demos. Everything below is either removing that or setting the
# typographic scale the charts are drawn against.
st.markdown(
    """
    <style>
      #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
      [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

      .stApp { background: #f9f9f7; }
      .block-container { padding: 2.6rem 1.2rem 4rem; max-width: 1080px; }

      html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
      }

      .masthead { margin: 0 0 2.2rem; }
      .masthead .mark { font-size: .95rem; letter-spacing: .14em; text-transform: uppercase;
                        color: #898781; }
      .masthead .sub  { font-size: .95rem; color: #52514e; margin-top: .35rem; }

      /* One card per chart: a hairline ring on the chart surface, not a shadow. */
      .card-head { margin: 0 0 .2rem; }
      .card-head h2 { font-size: 1.5rem; font-weight: 650; color: #0b0b0b;
                      margin: 0; letter-spacing: -.01em; }
      .card-sub { font-size: .88rem; color: #898781; margin-top: .25rem; }
      .card-gap { height: 3.4rem; }

      /* Hero figure: proportional digits on purpose -- tabular-nums makes a large
         standalone number look loose. */
      .hero { display: flex; align-items: baseline; gap: .9rem; margin: 1.1rem 0 .1rem; }
      .hero-figure { font-size: 3.4rem; line-height: 1; font-weight: 620;
                     color: #0b0b0b; letter-spacing: -.025em; }
      .hero-unit { font-size: 1.15rem; font-weight: 500; color: #898781;
                   margin-left: .35rem; letter-spacing: 0; }
      .hero-side { font-size: 1rem; color: #52514e; }
      .hero-side strong { color: #0b0b0b; font-weight: 600; }

      .hero-row { display: flex; gap: 2.2rem; flex-wrap: wrap;
                  margin: .55rem 0 .2rem; align-items: baseline; }
      .hero-row .k { font-size: 1.35rem; font-weight: 600; color: #0b0b0b; }
      .hero-row .k-unit { font-size: .8rem; font-weight: 500; color: #898781; }
      .hero-row .k-sm { font-size: 1rem; font-weight: 550; color: #52514e; }
      .hero-row .v { font-size: .88rem; color: #898781; margin-left: .45rem; }

      /* Plotly sits on the chart surface, ringed rather than shadowed. */
      [data-testid="stPlotlyChart"] { background: #fcfcfb; border-radius: 10px;
        border: 1px solid rgba(11,11,11,.10); padding: .5rem .35rem .2rem; margin-top: .6rem; }

      [data-testid="stCaptionContainer"] p { font-size: .88rem; color: #52514e;
        line-height: 1.55; }
      .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        font-size: .86rem; color: #52514e; }
      [data-testid="stExpander"] { border: none; }
      hr { border-color: #e1e0d9; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _broken_module_warnings() -> None:
    """A module that failed to import is shown, never silently dropped.

    A board that vanishes looks like a design decision; one that says it is broken
    gets fixed.
    """
    for label, errors in (("source", SOURCE_IMPORT_ERRORS), ("board", panels.IMPORT_ERRORS)):
        for name, tb in errors.items():
            st.error(f"The {label} module `{name}` failed to import, so it is missing here.")
            with st.expander(f"traceback for {name}"):
                st.code(tb, language="text")


def _pick_board():
    """The sidebar exists only to choose between boards, so with one board there
    is nothing to choose and no sidebar."""
    if len(BOARDS) <= 1:
        return BOARDS[0] if BOARDS else None

    groups: dict[str, list] = {}
    for b in BOARDS:
        groups.setdefault(b.group, []).append(b)
    labels, lookup = [], {}
    for group in sorted(groups, key=lambda g: min(b.order for b in groups[g])):
        for b in groups[group]:
            label = f"{group} · {b.title}" if len(groups) > 1 else b.title
            labels.append(label)
            lookup[label] = b
    st.sidebar.title("◧ Watchboard")
    return lookup[st.sidebar.radio("Board", labels, label_visibility="collapsed")]


_broken_module_warnings()
board = _pick_board()

st.markdown(
    '<div class="masthead"><div class="mark">◧ Watchboard</div>'
    "<div class=\"sub\">CFTC Commitments of Traders, straight from the primary "
    "source. A number you cannot reproduce yourself can raise a question; it "
    "should not answer one.</div></div>",
    unsafe_allow_html=True,
)

if board is None:
    st.error("No boards were discovered in panels/. Nothing to show.")
else:
    missing = [
        sid for sid in board.sources if (cache.file_stats(sid) or {}).get("rows", 0) == 0
    ]
    if missing:
        st.info(
            "Not ingested yet: " + ", ".join(f"`{m}`" for m in missing)
            + ". Run `python -m scripts.ingest --backfill` (first run pulls full "
            "history and takes a few minutes)."
        )
    else:
        board.render()
