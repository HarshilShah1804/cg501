"""
Tests for synaptic lesions and cue corruption.
"""

import numpy as np
import pytest
from src.patterns import generate_random_bipolar_patterns
from src.hopfield import HopfieldNetwork
from src.lesions import apply_random_lesion, apply_targeted_lesion, corrupt_cue


def test_lesion_fraction_and_symmetry():
    """Verify deleted undirected edges count and symmetry preservation."""
    N = 50
    total_undirected = N * (N - 1) // 2
    P = 4
    rng = np.random.default_rng(42)
    patterns = generate_random_bipolar_patterns(N, P, rng=rng)

    net = HopfieldNetwork(N)
    net.train(patterns)

    for frac in [0.0, 0.1, 0.25, 0.5]:
        W_lesioned, num_deleted = apply_random_lesion(net.W, frac, rng=rng)
        expected_deleted = int(round(frac * total_undirected))
        assert num_deleted == expected_deleted

        # Must maintain symmetry
        assert np.allclose(W_lesioned, W_lesioned.T)
        # Diagonal must remain zero
        assert np.all(np.diag(W_lesioned) == 0.0)


def test_targeted_lesion_removes_largest_weights():
    """Targeted lesion should remove the connections with highest absolute magnitude."""
    N = 20
    P = 5
    rng = np.random.default_rng(99)
    patterns = generate_random_bipolar_patterns(N, P, rng=rng)

    net = HopfieldNetwork(N)
    net.train(patterns)

    W_lesioned, num_deleted = apply_targeted_lesion(net.W, lesion_fraction=0.1)
    assert np.allclose(W_lesioned, W_lesioned.T)

    # Check that removed edges originally had larger or equal abs weights compared to any remaining edge
    triu_i, triu_j = np.triu_indices(N, k=1)
    original_abs = np.abs(net.W[triu_i, triu_j])
    remaining_abs = np.abs(W_lesioned[triu_i, triu_j])

    # Zeroed out positions
    zeroed = (original_abs > 0) & (remaining_abs == 0)
    survived = (remaining_abs > 0)

    if np.any(zeroed) and np.any(survived):
        assert np.min(original_abs[zeroed]) >= np.max(remaining_abs[survived]) - 1e-12


def test_cue_corruption():
    """Cue corruption flips bits independently with probability q."""
    N = 1000
    p = np.ones(N, dtype=np.int8)
    rng = np.random.default_rng(123)

    q = 0.2
    corrupted = corrupt_cue(p, flip_probability=q, rng=rng)
    diff = np.mean(corrupted != p)
    # With N=1000, diff should be very close to 0.2
    assert 0.15 < diff < 0.25
