# Parked boards

These are built, tested and working. They are parked, not abandoned: the board
they were part of showed six views with dense explanatory captions above every
chart, and that was the wrong surface — an audit instrument rather than something
you can read in five seconds.

`panels/__init__.py` skips any module whose name starts with `_`, so nothing in
this directory is discovered or rendered. Moving a file back up one level is the
only step needed to bring it back.

| file | what it does | why it might come back |
|---|---|---|
| `screen.py` | ranks every market in a family by how stretched its positioning is | the "what should I look at this week" view; the most likely first return |
| `market.py` | one market in full: level, causal rank, open interest, spread share, purity | the natural drill-down once you click a name on the screen |
| `flows.py` | week-over-week cohort flows and the zero-sum check across cohorts | answers "who moved", which a level chart cannot |
| `crowding.py` | CR4/CR8 concentration, trader counts, spread share | the fragility view — a net held by 5 firms is not the same as by 60 |
| `base_rates.py` | conditions forward positioning on today's percentile bucket | the honesty check on whether an extreme precedes anything |
| `integrity.py` | the accounting identities across all 1,046,369 market-weeks | proves the numbers reconcile; worth having before trusting any of the above |
| `tokens.py` | OpenRouter token prices | explicitly deferred, to be rebuilt later |

Two cautions if you unpark one:

- `tests/test_display_rules.py` scans `panels/*.py` and does not recurse, so
  nothing here is currently checked against the no-dual-axis and causal-ranking
  rules. Moving a file back up puts it under that scan again — which is the point.
- They were written against `lib/charts.stacked()`, which is the multi-panel
  builder. The single-market board uses `lib/charts.spotlight()` instead. Both
  still exist; `stacked()` was not removed.
