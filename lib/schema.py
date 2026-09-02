"""Declared shape for each source, checked between fetch and write.

The failure this exists to prevent is not a crash -- it is an upstream change
that leaves the pipeline running and the charts wrong. A renamed column becomes
a null column becomes a chart that quietly flatlines, and nobody notices for
weeks because the board still renders.

Layered on purpose, because the layers catch different things:

1. AT THE NETWORK BOUNDARY. Asking Socrata for named columns via $select turns
   an upstream rename into HTTP 400 with the column name in the body, before a
   single row is parsed. This is the earliest and cheapest detector and it lives
   in lib/cftc_api, not here.
2. HERE, ON THE TIDY FRAME. Required columns present, no all-null required
   column, dtypes coercible, row count sane against what is already stored.
3. AT THE ARITHMETIC. The accounting identities in lib/cftc_spec. A field name
   that exists but means the wrong thing -- the old-crop bucket instead of the
   all-maturity column -- passes 1 and 2 and fails only here.

Checks return findings rather than raising, so ingest can report every problem
with a source in one run and decide per-source whether to write. A broken feed
must not be able to take down the sources that are fine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Severity(str, Enum):
    #: Do not write. The data is wrong in a way that would produce wrong charts.
    FATAL = "fatal"
    #: Write, but say so loudly. Real but expected-in-the-tails.
    WARN = "warn"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.detail}"


@dataclass(frozen=True)
class TableSchema:
    """Columns a source promises to produce.

    `required` columns must be present and not entirely null. `numeric` columns
    must be coercible to a number. `optional` documents columns that may be
    absent -- listing them is how a reader tells "this source never had trader
    counts" from "trader counts vanished".
    """

    required: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    #: Reject a pull that is drastically smaller than what we already hold.
    #: Expressed as a fraction and only applied to a FULL pull -- an incremental
    #: window is legitimately tiny, so ingest passes stored_rows=None for those.
    min_rows_vs_stored: float = 0.90

    def check(self, df: pd.DataFrame, *, stored_rows: int | None = None) -> list[Finding]:
        out: list[Finding] = []

        if df is None or df.empty:
            return [Finding(Severity.FATAL, "EMPTY", "fetch returned no rows")]

        missing = [c for c in self.required if c not in df.columns]
        if missing:
            out.append(
                Finding(
                    Severity.FATAL,
                    "MISSING_COLUMN",
                    f"required columns absent: {missing}. Upstream schema changed.",
                )
            )

        for col in self.required:
            if col in df.columns and df[col].isna().all():
                out.append(
                    Finding(
                        Severity.FATAL,
                        "ALL_NULL_REQUIRED",
                        f"{col!r} is present but entirely null -- the usual shape of a "
                        "silent upstream rename.",
                    )
                )

        for col in self.numeric:
            if col not in df.columns:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = int(coerced.isna().sum() - df[col].isna().sum())
            if bad > 0:
                out.append(
                    Finding(
                        Severity.FATAL,
                        "NOT_NUMERIC",
                        f"{col!r} has {bad:,} values that will not parse as numbers.",
                    )
                )

        if stored_rows and len(df) < self.min_rows_vs_stored * stored_rows:
            out.append(
                Finding(
                    Severity.FATAL,
                    "ROW_COUNT_DROP",
                    f"full pull returned {len(df):,} rows against {stored_rows:,} stored "
                    f"(< {self.min_rows_vs_stored:.0%}). Markets do get delisted, so this "
                    "may be legitimate -- confirm, then backfill with --force.",
                )
            )

        unexpected = [
            c
            for c in df.columns
            if c not in set(self.required) | set(self.numeric) | set(self.optional)
        ]
        if unexpected:
            out.append(
                Finding(
                    Severity.WARN,
                    "UNDECLARED_COLUMN",
                    f"columns produced but not declared: {unexpected}. Harmless, but "
                    "declare them so the next reader knows they are intentional.",
                )
            )
        return out


def worst(findings: list[Finding]) -> Severity | None:
    if any(f.severity is Severity.FATAL for f in findings):
        return Severity.FATAL
    if findings:
        return Severity.WARN
    return None
