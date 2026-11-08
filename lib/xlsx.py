"""Read an XLSX into rows of strings, with the standard library and nothing else.

Two issuers publish their NAV history as XLSX and offer no CSV or JSON: State Street
for SPY and VanEck for SMH. openpyxl would be the first new dependency in this repo --
which runs on pandas and pytest alone -- for about forty lines of zipfile and
ElementTree, so this is the forty lines.

Deliberately dumb: everything comes back as a string and the caller decides what is a
date and what is a number. A reader that guessed would have to guess about
'05-Nov-2026' (SSGA writes dates as locale text, not Excel serials) and about
'-' (SSGA's placeholder for a shares-outstanding figure that does not exist yet), and
both guesses are wrong in a way that produces numbers rather than errors.

Two details carry the whole correctness burden, and tests/test_xlsx.py pins both:

  SHARED STRINGS. A cell with t="s" holds an INDEX into xl/sharedStrings.xml, not a
  value. Reading it literally yields '0', '1', '2' -- which looks like data.

  GAP COLUMNS. A row omits cells that are empty rather than writing them, so position
  must come from each cell's r="C7" reference. Appending in document order instead
  shifts every value left of a blank, turning a NAV into a share count with no error
  anywhere. That is the exact shape of failure this repo exists to avoid, so it is a
  test rather than a comment.
"""
from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree

#: The one namespace in a SpreadsheetML worksheet.
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

_CELL_REF = re.compile(r"^([A-Z]+)")


def _column_index(ref: str) -> int:
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26. Base-26 with no zero digit.

    Spelled out because the obvious shortcut -- treating the letters as a positional
    base-26 number -- puts AA at index 1 instead of 26, which silently lands a column
    in the wrong field for any sheet wider than Z.
    """
    match = _CELL_REF.match(ref)
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(raw)
    out = []
    for si in root.findall("m:si", NS):
        # A shared string may be split across several <t> runs when parts of it are
        # formatted differently; joining them is the only way to get the whole value.
        out.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    return out


def _sheet_name(archive: zipfile.ZipFile, index: int) -> str:
    names = sorted(
        n for n in archive.namelist()
        if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
    )
    if not names:
        raise ValueError(f"no worksheet in this workbook: {archive.namelist()[:8]}")
    return names[min(index, len(names) - 1)]


def rows(data: bytes, sheet: int = 0) -> list[list[str]]:
    """Every row of one sheet, as strings, with gaps preserved as empty cells.

    `data` is the workbook's bytes rather than a path: both callers fetch over HTTP and
    never land the file on disk, which is also what keeps them inside the repo's rule
    that no source writes data anywhere but the gitignored store.
    """
    archive = zipfile.ZipFile(io.BytesIO(data))
    shared = _shared_strings(archive)
    root = ElementTree.fromstring(archive.read(_sheet_name(archive, sheet)))

    out: list[list[str]] = []
    for row in root.iter(f"{{{NS['m']}}}row"):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            position = _column_index(cell.get("r") or "")
            kind = cell.get("t")
            if kind == "s":
                raw = cell.findtext("m:v", default="", namespaces=NS)
                try:
                    cells[position] = shared[int(raw)]
                except (ValueError, IndexError):
                    cells[position] = ""
            elif kind == "inlineStr":
                cells[position] = "".join(
                    t.text or "" for t in cell.iter(f"{{{NS['m']}}}t")
                )
            else:
                cells[position] = cell.findtext("m:v", default="", namespaces=NS)
        if cells:
            width = max(cells) + 1
            out.append([cells.get(i, "") for i in range(width)])
        else:
            out.append([])
    return out


def find_header(table: list[list[str]], wanted: tuple[str, ...]) -> int:
    """Index of the row containing every name in `wanted`. Raises if absent.

    Anchoring on the header's TEXT rather than on a row offset, because both issuers
    put disclosure prose in the first rows and SSGA's header currently sits at row 4 --
    a number that is true today and is not a contract. A layout tweak then shifts the
    whole series silently; this raises instead.
    """
    lowered = tuple(w.strip().lower() for w in wanted)
    for index, row in enumerate(table):
        cells = {c.strip().lower() for c in row}
        if all(w in cells for w in lowered):
            return index
    raise ValueError(
        f"no header row containing {list(wanted)} in the first {len(table)} rows -- "
        "the publisher changed the layout, so the parser must be re-read rather than "
        "nudged"
    )
