"""Which cohort combinations get a materialised rank, and why not all of them.

A percentile has to be precomputed (see scripts/build for why it cannot be SQL), and
it depends on WHICH cohorts are summed. So the serving table is keyed by a named
group, not by an arbitrary subset -- and that is a deliberate narrowing, measured:

    family            cohorts   non-empty subsets   markets   subsets x markets x weeks
    cftc_tff_fut            5                  31       147               4,816,749
    cftc_disagg_fut         5                  31       655              21,462,385
    cftc_legacy_fut         3                   7       955              12,935,475

Thirty-nine million rows to serve combinations nobody will ask for. The groups below
are the ones a person actually reaches for: each cohort on its own, plus the
speculative-money grouping that each board defaults to.

WHAT THIS COSTS, STATED PLAINLY. An arbitrary subset can still be charted -- cohort
shares are additive over a common denominator, so the LEVEL is exact for any
combination. What it cannot have is a rank, because the rank is not additive. So an
undeclared combination gets a line and a blank percentile, which is the same rule
panels/positioning._hero already follows for a series too short to rank. Adding a
group is one line here plus a rebuild.
"""
from __future__ import annotations

#: The grouping each family defaults to: the cohorts whose position is a view rather
#: than a hedge against physical or a market-making book. Mirrors
#: panels/positioning._DEFAULT_COHORTS -- if these ever disagree, the Grafana board
#: and the Streamlit board are answering different questions.
SPECULATIVE = {
    "cftc_tff_fut": ("asset_mgr", "lev_money"),
    "cftc_disagg_fut": ("m_money",),
    "cftc_legacy_fut": ("noncomm",),
}

#: Human labels, borrowed from the same place the checkbox row gets them.
LABEL = {
    "dealer": "Dealers",
    "asset_mgr": "Asset managers",
    "lev_money": "Hedge funds",
    "prod_merc": "Producers",
    "swap": "Swap dealers",
    "m_money": "Managed money",
    "noncomm": "Non-commercial",
    "comm": "Commercial",
    "other_rept": "Other reportables",
    "nonrept": "Small traders",
}


def groups_for(dataset: str, cohorts: list[str]) -> dict[str, tuple[str, ...]]:
    """group name -> member cohorts, for one family.

    `cohorts` is what the data actually contains, not a hardcoded list, so a family
    that gains a cohort upstream gains a group here rather than silently dropping it.
    """
    out: dict[str, tuple[str, ...]] = {c: (c,) for c in sorted(cohorts)}
    combo = tuple(c for c in SPECULATIVE.get(dataset, ()) if c in cohorts)
    if len(combo) > 1:
        # Named for its members so the group is self-describing in a Grafana
        # dropdown, where there is no docstring to explain "speculative".
        out["+".join(combo)] = combo
    return out


def label_for(group: str) -> str:
    """A group name as a reader should see it."""
    return " + ".join(LABEL.get(c, c) for c in group.split("+"))
