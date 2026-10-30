"""How often does a signal with no edge look significant?

Returns are pure noise with a true mean of zero. Each observation is a forward return
over `horizon` bars, so neighbouring observations share most of their data. A valid
5% test should report |t| > 1.96 in roughly 5% of trials.
"""
import numpy as np

from wfeval import naive_tstat, newey_west_tstat, non_overlapping_tstat

TRIALS, N, HORIZON = 2000, 2000, 24
rng = np.random.default_rng(42)
tests = {
    "naive t-stat": lambda x: naive_tstat(x),
    "Newey-West t-stat": lambda x: newey_west_tstat(x, lags=2 * HORIZON),
    "non-overlapping t-stat": lambda x: non_overlapping_tstat(x, HORIZON),
}
hits = dict.fromkeys(tests, 0)
for _ in range(TRIALS):
    noise = rng.normal(size=N + HORIZON - 1)
    forward = np.convolve(noise, np.ones(HORIZON), mode="valid")
    for name, test in tests.items():
        hits[name] += abs(test(forward)) > 1.96

print(f"{TRIALS} trials, {N} observations, {HORIZON}-bar overlapping returns, true mean 0\n")
for name, count in hits.items():
    print(f"{name:<26}{count / TRIALS:>8.1%} false positives")
