"""Control for testing many hypotheses on the same data."""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses rejected at the given false discovery rate."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    passed = p[order] <= fdr * np.arange(1, len(p) + 1) / len(p)
    rejected = np.zeros(len(p), dtype=bool)
    if passed.any():
        rejected[order[: np.nonzero(passed)[0].max() + 1]] = True
    return rejected
