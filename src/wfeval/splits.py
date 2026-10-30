"""Walk-forward splits with a gap that stops labels leaking across the boundary."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train: range
    test: range


def walk_forward_splits(
    n: int, train_size: int, test_size: int, horizon: int = 1, expanding: bool = False
) -> Iterator[Split]:
    """Yield chronological splits, leaving `horizon - 1` bars between train and test.

    A label at bar t looks `horizon` bars ahead, so the last training label must be
    fully known before the first test bar. Test windows never overlap.
    """
    if min(train_size, test_size, horizon) < 1:
        raise ValueError("train_size, test_size and horizon must be positive")
    gap = horizon - 1
    train_end = train_size
    while train_end + gap + test_size <= n:
        start = 0 if expanding else train_end - train_size
        test_start = train_end + gap
        yield Split(range(start, train_end), range(test_start, test_start + test_size))
        train_end += test_size
