"""Bootstrap confidence intervals for aggregate ranking metrics.

A single aggregate number ("mean AUC = 0.68") says nothing about how much
that number would wobble under a different sample of impressions.
Bootstrapping estimates that uncertainty without assuming any particular
distribution: resample the per-impression values with replacement (same
size as the original) many times, recompute the mean each time, and read
off percentiles of the resulting distribution. It's the data demonstrating
its own uncertainty, rather than a formula that assumes normality.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a (lower, upper) confidence interval for the mean of `values`.

    Args:
        values: One float per impression (already filtered to impressions
            where the metric was defined -- see news_reco.evaluation.metrics).
        n_resamples: Number of bootstrap resamples. 1000 is the standard
            default: enough to estimate the 2.5th/97.5th percentiles
            stably, cheap enough to run in well under a second at this
            project's scale.
        confidence: e.g. 0.95 for a 95% CI.
        seed: Makes the resampling reproducible -- rerunning this on the
            same values must give the same interval every time, not a
            slightly different one.
    """
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty array of values.")

    rng = np.random.default_rng(seed)
    n = len(values)
    # Vectorized: build all resamples' indices as one (n_resamples, n)
    # matrix rather than looping n_resamples times in Python.
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = values[resample_indices].mean(axis=1)

    alpha = 1 - confidence
    lower = float(np.percentile(resampled_means, 100 * alpha / 2))
    upper = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return lower, upper
