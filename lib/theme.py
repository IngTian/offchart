"""Palette and type, taken from ingtian.github.io rather than invented.

The values below are read straight out of that site's compiled CSS custom
properties (`:root` for light, `[data-theme=dark]` for dark), so the board matches
it instead of approximating it.

WHY DARK IS THE ONLY MODE SHIPPED

Both modes are defined here, and switching is one constant. Only dark is wired up,
and the reason is measured rather than aesthetic: the site's light-mode accent
`#c8a36a` sits at 1.95:1 against its own cream paper `#efe9dd`. That is fine for
the large serif headings it was chosen for and too low for a 2px data line, which
needs 3:1 to stay legible. The dark accent `#66c28c` clears 3:1 on the dark
surface, so dark is the mode where the same palette also works for charts.

If light is wanted later, the honest fix is a darker step of the ochre for the
series colour only -- not shipping the heading colour as a line colour.

WHAT THE CHART COLOURS ARE DOING, since a validator run flags this palette if you
treat it as four categorical slots:

    series   #66c28c   the one data line. Contrast 3:1+ on the surface: PASS.
    context  #8b938c   open interest. Deliberately near-gray (chroma 0.014) --
                       it is context, not a competing series, and it lives in its
                       own panel with its own axis label, which is the secondary
                       encoding that carries the distinction rather than hue.
    marker   #e0574a   a contract re-specification. A status colour, not a series.

The categorical adjacent-pair floors do not apply to that set: there is exactly
one series, so there is no adjacent pair to separate. The lightness-band check
flags `#66c28c` as brighter than the dark-mode band, which exists to keep eight
mutually-separable slots apart; with one series the only real risk is glare, and a
2px line at this step is how the source site renders its own accent.
"""
from __future__ import annotations

DARK = {
    "mode": "dark",
    "page": "#08090b",       # --bg
    "surface": "#14171b",    # --reading-grad start, the chart plane
    "raised": "#111417",
    "ink": "#dce1dc",        # --ink-1
    "ink_2": "#b7beb8",
    "muted": "#8b938c",      # --ink-3
    "faint": "#646b64",      # --ink-4
    "grid": "#2a2f2c",       # between ink-5 and the surface: a hairline, not a rule
    "baseline": "#4a4f4a",   # --ink-5
    "accent": "#66c28c",     # --ochre
    "accent_soft": "rgba(102,194,140,0.13)",
    "cool": "#5fb2c9",       # --indigo
    "seal": "#e0574a",       # --seal
    "hairline": "rgba(184,177,161,0.14)",
    "chip": "#191d21",
}

LIGHT = {
    "mode": "light",
    "page": "#efe9dd",       # --paper
    "surface": "#f4efe4",
    "raised": "#f1ebe0",
    "ink": "#16140f",        # --ink-1
    "ink_2": "#2f2b24",
    "muted": "#5a544a",      # --ink-3
    "faint": "#8c8576",      # --ink-4
    "grid": "#ddd5c6",
    "baseline": "#b8b1a1",   # --ink-5
    # NOT --ochre: see the module docstring. 1.95:1 is too low for a thin line, so
    # light mode would need its own darker step before it could ship.
    "accent": "#8a6a2f",
    "accent_soft": "rgba(138,106,47,0.12)",
    "cool": "#6d7689",
    "seal": "#b23a2e",
    "hairline": "rgba(90,84,74,0.26)",
    "chip": "#e7dfd0",
}

#: The mode the board renders in. One line to change.
ACTIVE = DARK

# The site pairs a system sans for body with Georgia for display and JetBrains
# Mono for labels. Charts use the sans; headings use the serif.
FONT_SANS = 'system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif'
FONT_DISPLAY = 'Georgia, "Times New Roman", "Nimbus Roman", serif'
FONT_MONO = 'ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, monospace'


def c(key: str) -> str:
    """A colour from the active mode."""
    return ACTIVE[key]


def is_dark() -> bool:
    return ACTIVE["mode"] == "dark"
