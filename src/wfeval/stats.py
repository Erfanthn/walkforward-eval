"""t-statistics and intervals for mean returns that may be autocorrelated."""
from __future__ import annotations

import numpy as np


def naive_tstat(x: np.ndarray) -> float:
    """Ordinary t-statistic. Overstated when observations overlap."""
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def newey_west_tstat(x: np.ndarray, lags: int) -> float:
    """t-statistic with Newey-West (Bartlett kernel) standard error."""
    x = np.asarray(x, dtype=float)
    if not 0 <= lags < len(x):
        raise ValueError("lags must be in [0, len(x))")
    e = x - x.mean()
    n = len(x)
    long_run_var = e @ e / n
    for k in range(1, lags + 1):
        long_run_var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(long_run_var / n))


def non_overlapping_tstat(x: np.ndarray, horizon: int, offset: int = 0) -> float:
    """t-statistic using every `horizon`-th observation, so that windows do not overlap."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return naive_tstat(np.asarray(x, dtype=float)[offset::horizon])


def block_bootstrap_ci(
    x: np.ndarray, block: int, level: float = 0.95, draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Circular block bootstrap percentile interval for the mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not 1 <= block <= n:
        raise ValueError("block must be in [1, len(x)]")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, (draws, blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    means = x[idx].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)
