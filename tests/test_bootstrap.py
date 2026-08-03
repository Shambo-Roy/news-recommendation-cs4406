"""Unit tests for the bootstrap CI utility (Milestone 5).

A full bootstrap result can't easily be hand-computed exactly (it's a
randomized estimation procedure), so these tests check properties any
correct implementation must satisfy, rather than exact numeric output:
reproducibility, degenerate-input behavior, and that the interval widens
with variance and contains the sample mean.

Run with: pytest tests/test_bootstrap.py -v
"""

import numpy as np
import pytest

from news_reco.evaluation.bootstrap import bootstrap_ci


def test_bootstrap_ci_is_reproducible_with_same_seed():
    values = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.7, 0.3, 0.5])
    result_a = bootstrap_ci(values, seed=42)
    result_b = bootstrap_ci(values, seed=42)
    assert result_a == result_b


def test_bootstrap_ci_collapses_for_a_constant_array():
    values = np.full(50, 0.7)
    lower, upper = bootstrap_ci(values)
    # Every possible resample of a constant array has the same mean, so the
    # interval must collapse to that constant (up to floating point).
    assert lower == pytest.approx(0.7)
    assert upper == pytest.approx(0.7)


def test_bootstrap_ci_contains_the_sample_mean():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=200)
    lower, upper = bootstrap_ci(values, seed=1)
    assert lower <= values.mean() <= upper


def test_bootstrap_ci_widens_with_more_variance():
    rng = np.random.default_rng(0)
    low_variance = rng.normal(loc=0.5, scale=0.01, size=200)
    high_variance = rng.normal(loc=0.5, scale=0.3, size=200)

    low_lo, low_hi = bootstrap_ci(low_variance, seed=1)
    high_lo, high_hi = bootstrap_ci(high_variance, seed=1)

    assert (high_hi - high_lo) > (low_hi - low_lo)


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))
