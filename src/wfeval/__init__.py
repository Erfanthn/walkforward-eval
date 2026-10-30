"""Honest evaluation tools for time-series research."""
from .multiple import benjamini_hochberg
from .splits import Split, walk_forward_splits
from .stats import block_bootstrap_ci, naive_tstat, newey_west_tstat, non_overlapping_tstat

__all__ = [
    "Split", "benjamini_hochberg", "block_bootstrap_ci", "naive_tstat",
    "newey_west_tstat", "non_overlapping_tstat", "walk_forward_splits",
]
