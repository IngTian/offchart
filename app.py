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

Streamlit reruns the whole script on every widget interaction. A page with a row of
controls therefore feels like a slide advancing, because that is structurally what
it is. So everything that responds to the pointer runs in the BROWSER and never
reaches the server: the zoom presets are plotly's, and the crosshair is this repo's
own (see panels/positioning._smooth_crosshair -- plotly throttles its hover to
about 17 updates a second, which is what "not smooth" turned out to mean). The
board's few controls live in an st.fragment, so they rerun that function rather
than this page. The sidebar appears only when there is more than one board.
"""
from __future__ import annotations

import streamlit as st

import panels
from lib import cache, theme
from lib.ui import caveat_block, freshness  # noqa: F401  -- re-exported for convenience
from sources import IMPORT_ERRORS as SOURCE_IMPORT_ERRORS
from sources import all_sources


def all_sources_by_id() -> dict:
    return {s.id: s for s in all_sources()}


# Discovered BEFORE set_page_config, because the sidebar's initial state depends on
# how many boards there are. With one board the sidebar holds nothing and starts
# collapsed; with two or more it is the only way to reach the second one, and a
# collapsed sidebar hides that there is a second one at all.
BOARDS = panels.all_boards()

st.set_page_config(
    page_title="Watchboard", page_icon="◧", layout="centered",
    initial_sidebar_state="expanded" if len(BOARDS) > 1 else "collapsed",
)

T = theme.ACTIVE

# Chrome off, type and spacing set once. Streamlit's defaults are what make a
# board read as a slide: a toolbar, a footer, a 6rem top pad and a heading scale
# built for demos. Everything below is either removing that or setting the
# typographic scale the charts are drawn against.
st.markdown(
    f"""
    <style>
      #MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
      [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

      .stApp {{ background: {T['page']}; }}
      .block-container {{ padding: 2.4rem 1.2rem 4rem; max-width: 1060px; }}

      html, body, [class*="css"] {{
        font-family: {theme.FONT_SANS};
        -webkit-font-smoothing: antialiased;
      }}

      /* The source site pairs a serif display face with a system sans body and a
         mono for small labels. Headings and the hero figure take the serif. */
      .masthead {{ margin: 0 0 1.9rem; }}
      .masthead .mark {{ font-family: {theme.FONT_MONO}; font-size: .78rem;
        letter-spacing: .2em; text-transform: uppercase; color: {T['accent']}; }}
      .masthead .sub {{ font-size: .95rem; color: {T['muted']}; margin-top: .5rem;
        max-width: 78ch; line-height: 1.6; }}

      .stamp {{ font-family: {theme.FONT_MONO}; font-size: .74rem; letter-spacing: .06em;
        text-transform: uppercase; color: {T['faint']}; margin: 0 0 1.6rem;
        padding-bottom: 1.1rem; border-bottom: 1px solid {T['hairline']}; }}
      .stamp strong {{ color: {T['ink_2']}; font-weight: 600; }}

      .card-head {{ margin: 1.4rem 0 .2rem; }}
      .card-head h2 {{ font-family: {theme.FONT_DISPLAY}; font-size: 1.85rem;
        font-weight: 400; color: {T['ink']}; margin: 0; letter-spacing: -.01em; }}
      .card-sub {{ font-size: .84rem; color: {T['faint']}; margin-top: .35rem; }}

      /* Hero figure in the display serif, proportional digits -- tabular-nums
         makes a large standalone number look loose. */
      .hero {{ display: flex; align-items: baseline; gap: 1rem; margin: 1rem 0 .1rem;
        flex-wrap: wrap; }}
      .hero-figure {{ font-family: {theme.FONT_DISPLAY}; font-size: 3.5rem; line-height: 1;
        font-weight: 400; color: {T['accent']}; letter-spacing: -.02em; }}
      .hero-unit {{ font-family: {theme.FONT_SANS}; font-size: 1rem; font-weight: 500;
        color: {T['faint']}; margin-left: .4rem; letter-spacing: 0; }}
      .hero-side {{ font-size: .98rem; color: {T['muted']}; }}
      .hero-side strong {{ color: {T['ink']}; font-weight: 600; }}

      .hero-row {{ display: flex; gap: 2rem; flex-wrap: wrap; margin: .5rem 0 .2rem;
        align-items: baseline; }}
      .hero-row .k {{ font-family: {theme.FONT_DISPLAY}; font-size: 1.5rem;
        color: {T['ink']}; }}
      .hero-row .k-unit {{ font-family: {theme.FONT_SANS}; font-size: .72rem;
        color: {T['faint']}; }}
      .hero-row .k-sm {{ font-size: .95rem; font-weight: 550; color: {T['ink_2']}; }}
      .hero-row .v {{ font-size: .85rem; color: {T['faint']}; margin-left: .4rem; }}

      /* Plotly on the chart plane, ringed with a hairline rather than shadowed.
         NO PADDING AND NO BORDER, and that is a bug fix rather than a taste call.
         Streamlit pins this element's height to the figure height and box-sizing is
         border-box, so any padding or border steals from the content box and the
         plot overflows by exactly that much -- which macOS Chrome draws as a
         scrollbar across the chart. Zero of both makes the content box equal the
         figure height and the overflow zero.
         DO NOT "fix" this with height:auto. That was tried and it is far worse: the
         container loses its definite height, plotly's responsive autosize recomputes
         against a moving target, and the figure renders with its panels at the wrong
         vertical positions and the x-axis labels stranded in the middle, settling
         only seconds later when the resize observer catches up.
         The card still reads as a card: the surface colour lifts it off the page and
         the radius shapes it. The 1px ring was never doing much work. */
      [data-testid="stPlotlyChart"] {{ background: {T['surface']}; border-radius: 12px;
        border: none; padding: 0; overflow: hidden;
        margin: 0; position: relative; }}

      /* NO MARGIN ON THE CARD ITSELF -- the gap above the chart is set on the
         element container instead, and that is the same bug as the padding above,
         one box further out. Streamlit pins the CONTAINER's height to the figure
         height too (760px) and gives it overflow:auto. A .7rem top margin on the
         card therefore put 760px of content 11px down inside a 760px box:
         scrollHeight 771 against clientHeight 760. Those 11px were charged twice --
         a scrollbar drawn across the bottom of the chart, AND the card's own bottom
         11px pushed outside the container and clipped off, which is what "the graph
         is still cut off" was. Measured: container box top 626, card box top 637.
         A margin on the container is OUTSIDE its box, so it moves the card down
         without adding anything to scroll. overflow:visible is the belt to that
         brace: if some future off-by-one overflows again it costs a stray pixel
         rather than a scrollbar sawn through the x-axis. */
      [data-testid="stElementContainer"]:has(> * > [data-testid="stPlotlyChart"]),
      [data-testid="stElementContainer"]:has(> [data-testid="stPlotlyChart"]) {{
        margin-top: .7rem; overflow: visible; }}
      /* The modebar is trimmed to the PNG download; keep it quiet until hover. */
      [data-testid="stPlotlyChart"] .modebar {{ opacity: 0; transition: opacity .18s; }}
      [data-testid="stPlotlyChart"]:hover .modebar {{ opacity: 1; }}

      /* Injected by the copy-image button; see panels/positioning._copy_button.
         right: must clear the modebar, which holds two ~28px icons at the top
         right. At 3.1rem this button sat ON TOP of the PNG download and silently
         swallowed its clicks -- the download was unreachable and looked broken. */
      .wb-copy {{ position: absolute; top: .5rem; right: 5.6rem; z-index: 5;
        font-family: {theme.FONT_SANS}; font-size: .72rem; letter-spacing: .02em;
        color: {T['faint']}; background: {T['chip']};
        border: 1px solid {T['hairline']}; border-radius: 6px;
        padding: .2rem .5rem; cursor: pointer; opacity: 0; transition: opacity .18s; }}
      [data-testid="stPlotlyChart"]:hover .wb-copy {{ opacity: 1; }}
      .wb-copy:hover {{ color: {T['accent']}; border-color: {T['baseline']}; }}

      /* The frame-rate crosshair, injected by panels/positioning._smooth_crosshair.
         No CSS transition on transform: the whole point is that it follows the
         pointer every frame, and easing 16ms-apart updates would reintroduce
         exactly the lag it was built to remove. Only opacity eases, on enter/exit. */
      .wb-xhair {{ position: absolute; width: 1px; left: 0; top: 0;
        background: {T['muted']}; pointer-events: none; z-index: 3;
        opacity: 0; transition: opacity .12s ease-out; will-change: transform; }}
      .wb-xtip {{ position: absolute; left: 0; top: 0; pointer-events: none; z-index: 4;
        background: {T['chip']}; border: 1px solid {T['baseline']}; border-radius: 7px;
        padding: .4rem .6rem; font-family: {theme.FONT_SANS}; font-size: .8rem;
        line-height: 1.5; color: {T['ink']}; white-space: nowrap;
        opacity: 0; transition: opacity .12s ease-out; will-change: transform;
        box-shadow: 0 4px 16px rgba(0,0,0,.35); }}
      .wb-xtip b {{ color: {T['accent']}; font-weight: 600; }}

      [data-testid="stCaptionContainer"] p {{ font-size: .85rem; color: {T['faint']};
        line-height: 1.6; }}
      [data-testid="stExpander"] {{ border: none; }}
      [data-testid="stExpander"] summary {{ font-size: .82rem; color: {T['faint']}; }}
      [data-testid="stExpander"] summary:hover {{ color: {T['accent']}; }}
      hr {{ border-color: {T['hairline']}; }}

      /* Controls: quieter than Streamlit's defaults, which are sized for demos. */
      [data-testid="stSelectbox"] label, [data-testid="stCheckbox"] label p {{
        font-size: .82rem !important; color: {T['muted']}; }}
      [data-testid="stCheckbox"] {{ margin-top: .1rem; }}
      div[data-baseweb="select"] > div {{ background: {T['chip']};
        border-color: {T['hairline']}; font-size: .85rem; }}
      .stButton button {{ font-size: .8rem; padding: .3rem .8rem;
        background: {T['chip']}; color: {T['ink_2']};
        border: 1px solid {T['hairline']}; }}
      .stButton button:hover {{ color: {T['accent']}; border-color: {T['accent']}; }}
      iframe[title="st.iframe"] {{ display: none; }}
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
    # Deliberately not "CFTC ...": there is more than one source now, and a
    # masthead that names only the first one makes the others look like guests.
    '<div class="masthead"><div class="mark">◧ Watchboard</div>'
    "<div class=\"sub\">Positioning and expectations, straight from the primary "
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
        # A source we may not redistribute is GITIGNORED, so "not ingested" is the
        # normal state of a fresh clone rather than a mistake, and the notice has to
        # say so or it reads as a broken repo. See sources/base.redistributable.
        restricted = []
        for sid in missing:
            try:
                src = all_sources_by_id()[sid]
            except KeyError:
                continue
            if not src.redistributable:
                restricted.append((sid, src.citation))
        st.info(
            "Not ingested yet: " + ", ".join(f"`{m}`" for m in missing)
            + ". Run `python -m scripts.ingest"
            + (f" --source {missing[0]}" if len(missing) == 1 else " --backfill")
            + "`."
        )
        for sid, citation in restricted:
            st.caption(
                f"`{sid}` is deliberately not committed: its terms permit use but "
                f"not redistribution, so the data is fetched rather than shipped. "
                f"Source: {citation}."
            )
    else:
        board.render()
