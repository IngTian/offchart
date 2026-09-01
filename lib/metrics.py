"""Derived measures. Every function here is deliberately boring and testable.

Two conventions are enforced by design rather than by comment, because both were
real errors before they were rules:

1. **Expanding percentiles only.** A percentile ranked against the *full* history
   including future observations carries look-ahead bias — it tells you where a
   level sat relative to data that did not exist yet. Every percentile here ranks
   observation t against observations 0..t only.

2. **Signed quantities are reported in their native unit, never as a percent
   change.** A net position going -100,640 -> -96,727 is "+3,913 contracts, less
   short". Calling it "+3.89%" reads as growth while the position shrank toward
   zero. `pct_change_is_safe()` exists to be called before formatting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_percentile(values: pd.Series) -> pd.Series:
    """Percentile of each observation against history up to and including itself.

    No look-ahead: observation t is ranked only against 0..t. Returns 0-100.
    The first observation is always 100.0 (it is the max of a one-element set),
    which is why `min_history` filtering matters at the display layer.
    """
    v = values.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        window = v[: i + 1]
        window = window[~np.isnan(window)]
        if len(window):
            out[i] = 100.0 * float((window <= v[i]).sum()) / len(window)
    return pd.Series(out, index=values.index, name=f"{values.name}_pctile")


def net(long: pd.Series, short: pd.Series) -> pd.Series:
    """Directional lean. CAN CHANGE SIGN -- report in contracts, never in percent."""
    return long - short


def gross(long: pd.Series, short: pd.Series) -> pd.Series:
    """Total book size: how much there is to unwind. Never negative, so a percent
    change is meaningful here (unlike net). This is a fragility measure, not a
    direction measure."""
    return long + short


def share_of(part: pd.Series, whole: pd.Series) -> pd.Series:
    """Percent of a total, guarding division by zero."""
    return 100.0 * part / whole.replace(0, np.nan)


def pct_change_is_safe(values: pd.Series) -> bool:
    """False if the series changes sign anywhere, i.e. a percent change on it
    would be misleading. Call this before formatting any change as a percent."""
    v = values.dropna()
    if v.empty:
        return True
    return bool((v > 0).all() or (v < 0).all())


def change_over(values: pd.Series, periods: int) -> pd.Series:
    """Absolute change over N periods, in the series' own units."""
    return values.diff(periods)
