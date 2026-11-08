"""The stdlib XLSX reader, which exists so two issuers' workbooks need no dependency.

Both plain-ETF sources publish XLSX and nothing else, and openpyxl would be the first
new dependency in this repo for 40 lines of zipfile plus ElementTree. The parsing is
boring but the FAILURE is not: a shared-string index read as a literal, or a gap column
silently shifting every value one place left, both produce a full series of plausible
wrong numbers.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import xlsx  # noqa: E402

SHARED = """<?xml version="1.0"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4" uniqueCount="4">
  <si><t>Date</t></si><si><t>NAV</t></si><si><t>Shares Outstanding</t></si><si><t>-</t></si>
</sst>"""


def _book(sheet_xml: str, shared: str = SHARED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/sharedStrings.xml", shared)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return buf.getvalue()


def test_shared_strings_resolve_instead_of_reading_as_indexes() -> None:
    """A cell with t="s" holds an INDEX. Printing the index would give '0', '1', '2' --
    which looks like data and is the failure this test exists for."""
    sheet = """<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
  <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>
             <c r="C1" t="s"><v>2</v></c></row>
</sheetData></worksheet>"""
    assert xlsx.rows(_book(sheet)) == [["Date", "NAV", "Shares Outstanding"]]


def test_a_gap_column_holds_its_place_rather_than_shifting_the_row() -> None:
    """The one that corrupts a whole series quietly.

    A row with no cell for column B must read ['x', '', 'y'] and not ['x', 'y'] --
    otherwise every value right of a blank lands in the wrong field, and a NAV becomes
    a share count with no error anywhere.
    """
    sheet = """<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
  <row r="1"><c r="A1"><v>1</v></c><c r="C1"><v>3</v></c></row>
</sheetData></worksheet>"""
    assert xlsx.rows(_book(sheet)) == [["1", "", "3"]]


def test_columns_past_z_are_placed_correctly() -> None:
    """AA is column 27, not column 2. Naive two-letter handling puts it at index 1."""
    sheet = """<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
  <row r="1"><c r="A1"><v>first</v></c><c r="AA1"><v>twenty-seventh</v></c></row>
</sheetData></worksheet>"""
    got = xlsx.rows(_book(sheet))[0]
    assert len(got) == 27
    assert got[0] == "first"
    assert got[26] == "twenty-seventh"


def test_inline_strings_and_numbers_both_read() -> None:
    sheet = """<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>05-Nov-2026</t></is></c>
             <c r="B1"><v>733.080722</v></c></row>
</sheetData></worksheet>"""
    assert xlsx.rows(_book(sheet)) == [["05-Nov-2026", "733.080722"]]


def test_a_workbook_with_no_shared_strings_still_reads() -> None:
    """VanEck ships one; SSGA ships one; a future issuer might not."""
    sheet = """<?xml version="1.0"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
  <row r="1"><c r="A1"><v>42</v></c></row>
</sheetData></worksheet>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    assert xlsx.rows(buf.getvalue()) == [["42"]]
