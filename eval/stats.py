"""Bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from eval import config as C

_CHUNK = 500  # resamples per vectorised block (bounds memory for large n)


def _boot_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        means[start : start + m] = values[idx].mean(axis=1)
    return means


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = C.BOOTSTRAP_RESAMPLES,
    seed: int = C.BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI (NaNs dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    means = _boot_means(v, n_boot, seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    ci_lo: float
    ci_hi: float
    p_bootstrap: float
    p_wilcoxon: float
    cohens_dz: float


def paired_test(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_boot: int = C.BOOTSTRAP_RESAMPLES,
    seed: int = C.BOOTSTRAP_SEED,
) -> PairedResult:
    """Compare config ``b`` against baseline ``a`` on the same queries (positive = b better)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    d = y - x
    n = len(d)
    if n == 0:
        nan = float("nan")
        return PairedResult(0, nan, nan, nan, nan, nan, nan, nan, nan)
    means = _boot_means(d, n_boot, seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled mean difference crosses zero
    p_boot = float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))
    p_wil = float("nan")
    if np.any(d != 0):
        try:
            p_wil = float(sp_stats.wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            p_wil = float("nan")
    else:
        p_wil = 1.0
    sd = d.std(ddof=1) if n > 1 else 0.0
    dz = float(d.mean() / sd) if sd > 0 else 0.0
    return PairedResult(
        n, float(x.mean()), float(y.mean()), float(d.mean()), float(lo), float(hi),
        p_boot, p_wil, dz,
    )  # fmt: skip


def percentile(values: list[float], q: float) -> float:
    v = [x for x in values if x == x]
    return float(np.percentile(v, q)) if v else float("nan")
