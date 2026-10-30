import numpy as np
import pytest

from wfeval import (
    benjamini_hochberg, block_bootstrap_ci, naive_tstat, newey_west_tstat,
    non_overlapping_tstat, walk_forward_splits,
)


def test_train_labels_end_before_test_starts():
    for s in walk_forward_splits(200, 50, 10, horizon=5):
        last_label_known_at = s.train[-1] + 5
        assert last_label_known_at <= s.test[0]


def test_test_windows_are_chronological_and_disjoint():
    splits = list(walk_forward_splits(100, 40, 10))
    tests = [i for s in splits for i in s.test]
    assert tests == sorted(set(tests))
    assert tests[-1] < 100


def test_expanding_train_keeps_the_start():
    assert all(s.train[0] == 0 for s in walk_forward_splits(100, 40, 10, expanding=True))


def test_rolling_train_keeps_its_size():
    assert all(len(s.train) == 40 for s in walk_forward_splits(100, 40, 10))


def test_too_little_data_gives_no_splits():
    assert list(walk_forward_splits(10, 40, 10)) == []


def test_naive_tstat_hand_calculation():
    assert naive_tstat(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0 / (1.0 / np.sqrt(3)))


def test_newey_west_with_zero_lags_is_close_to_naive():
    x = np.random.default_rng(0).normal(0.1, 1, 5000)
    assert newey_west_tstat(x, 0) == pytest.approx(naive_tstat(x), rel=1e-3)


def test_newey_west_shrinks_t_under_positive_autocorrelation():
    noise = np.random.default_rng(1).normal(0.02, 1, 5000)
    x = np.convolve(noise, np.ones(20), mode="valid")
    assert abs(newey_west_tstat(x, 40)) < abs(naive_tstat(x))


def test_non_overlapping_uses_every_horizon_th_point():
    x = np.arange(1.0, 11.0)
    assert non_overlapping_tstat(x, 2) == pytest.approx(naive_tstat(x[::2]))


def test_bootstrap_interval_contains_sample_mean():
    x = np.random.default_rng(2).normal(size=500)
    low, high = block_bootstrap_ci(x, block=10)
    assert low < x.mean() < high


def test_bootstrap_is_reproducible():
    x = np.random.default_rng(3).normal(size=100)
    assert block_bootstrap_ci(x, 5, seed=9) == block_bootstrap_ci(x, 5, seed=9)


def test_benjamini_hochberg_step_up():
    # Sorted thresholds are 0.0125, 0.025, 0.0375, 0.05. 0.03 fails its own threshold
    # but is still rejected, because the larger 0.035 passes its threshold.
    p = np.array([0.01, 0.035, 0.03, 0.20])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, True, False]


def test_benjamini_hochberg_rejects_nothing_when_all_large():
    assert not benjamini_hochberg(np.array([0.5, 0.9])).any()
