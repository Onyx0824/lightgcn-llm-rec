"""Shared statistics helpers: paired difference over seeds with t-based CI and two-sided paired t-test."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_ci(ctrl, treat, level: float = 0.95) -> tuple[float, float, float, float]:
    """Return (mean difference treat - ctrl, CI low, CI high, p). Needs >= 2 paired values for CI / p (else NaN)."""
    c = np.asarray(ctrl, dtype=float)
    a = np.asarray(treat, dtype=float)
    d = a - c
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, np.nan, np.nan, np.nan
    sd = float(d.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n))
    p = float(stats.ttest_rel(a, c).pvalue) if sd > 0 else np.nan
    return mean, mean - half, mean + half, p
