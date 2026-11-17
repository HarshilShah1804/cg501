"""
Tests for pattern generation.
"""

import numpy as np
import pytest
from src.patterns import generate_random_bipolar_patterns


def test_bipolar_values():
    """Patterns must contain only -1 and +1."""
    rng = np.random.default_rng(123)
    patterns = generate_random_bipolar_patterns(N=100, P=10, rng=rng)
    assert set(np.unique(patterns)).issubset({-1, 1})
    assert patterns.shape == (10, 100)


def test_exact_balance():
    """When exact_balance=True, each pattern has equal number of +1 and -1."""
    rng = np.random.default_rng(456)
    patterns = generate_random_bipolar_patterns(N=100, P=5, rng=rng, exact_balance=True)
    assert np.all(np.sum(patterns, axis=1) == 0)


def test_reproducibility():
    """Identical seeds produce identical patterns."""
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    p1 = generate_random_bipolar_patterns(N=50, P=4, rng=rng1)
    p2 = generate_random_bipolar_patterns(N=50, P=4, rng=rng2)
    assert np.array_equal(p1, p2)
