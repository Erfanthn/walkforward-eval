# walkforward-eval

Tools for testing whether a time-series signal is real: leakage-safe walk-forward splits,
t-statistics that account for overlapping returns, block bootstrap intervals and false
discovery control.

## Result

`examples/overlap_false_positives.py` tests a signal with no edge: 2,000 trials of pure
noise, with 2,000 overlapping 24-bar forward returns in each trial. A valid 5% test should
report significance in about 5% of trials.

| Method | False positive rate |
|---|---:|
| Naive t-statistic | 68.3% |
| Newey-West (48 lags) | 7.9% |
| Non-overlapping sample | 5.6% |

Treating overlapping returns as independent makes noise look significant most of the time.
Newey-West removes most of the problem but stays somewhat above 5% in this setup, while
non-overlapping sampling is close to the target at the cost of using fewer observations.

## Contents

| Function | Purpose |
|---|---|
| `walk_forward_splits` | Chronological train/test windows, with a gap so that no training label overlaps the test period |
| `naive_tstat` | Baseline t-statistic, for comparison |
| `newey_west_tstat` | t-statistic with an autocorrelation-robust standard error |
| `non_overlapping_tstat` | t-statistic on every `horizon`-th observation |
| `block_bootstrap_ci` | Circular block bootstrap interval for the mean |
| `benjamini_hochberg` | False discovery control when many hypotheses are tested on the same data |

## Limitations

- The example uses Gaussian noise. Real returns have fat tails and changing volatility,
  which can change these rates.
- These tools reduce false discoveries but do not remove them. Choosing among many
  variants still needs a held-out period.

## Usage

```bash
pip install -e .[dev]
pytest
python examples/overlap_false_positives.py
```

```python
from wfeval import newey_west_tstat, walk_forward_splits

for split in walk_forward_splits(n=5000, train_size=2000, test_size=250, horizon=24):
    ...  # fit on split.train, evaluate on split.test
```
