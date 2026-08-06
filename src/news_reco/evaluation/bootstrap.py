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


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """CI for the mean difference (a - b) between two systems on the same impressions.

    Unlike calling bootstrap_ci() independently on each system and eyeballing
    whether the two intervals overlap, this resamples *the same impression
    indices* for both arrays on every resample, preserving the pairing --
    values_a[i] and values_b[i] must come from the same impression. That
    matters because the two systems' errors are correlated (an impression
    that's easy for one method tends to be easy for the other too); treating
    them as independent overstates the apparent uncertainty and can hide a
    real, consistent difference. If the returned interval excludes zero, the
    difference is significant at the given confidence level -- this is the
    comparison that non-overlapping independent CIs can only suggest, not prove.
    """
    if len(values_a) != len(values_b):
        raise ValueError(
            f"values_a and values_b must be paired (same length): {len(values_a)} != {len(values_b)}"
        )
    if len(values_a) == 0:
        raise ValueError("Cannot bootstrap an empty array of values.")

    rng = np.random.default_rng(seed)
    n = len(values_a)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))

    resampled_diffs = (values_a[resample_indices] - values_b[resample_indices]).mean(axis=1)

    alpha = 1 - confidence
    lower = float(np.percentile(resampled_diffs, 100 * alpha / 2))
    upper = float(np.percentile(resampled_diffs, 100 * (1 - alpha / 2)))
    return lower, upper
