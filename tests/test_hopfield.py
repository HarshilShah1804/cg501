"""
Tests for Hopfield network core logic, Hebbian weights, retrieval, and energy descent.
"""

import numpy as np
import pytest
from src.patterns import generate_random_bipolar_patterns
from src.hopfield import HopfieldNetwork
from src.lesions import corrupt_cue


def test_hebbian_weights_properties():
    """Weight matrix must be symmetric with zero diagonal."""
    N = 100
    P = 5
    rng = np.random.default_rng(42)
    patterns = generate_random_bipolar_patterns(N, P, rng=rng)

    net = HopfieldNetwork(N)
    net.train(patterns)

    # Check zero diagonal
    assert np.all(np.diag(net.W) == 0.0)
    # Check symmetry
    assert np.allclose(net.W, net.W.T)


def test_energy_non_increasing():
    """Energy should never increase after an individual asynchronous update."""
    N = 100
    P = 3
    rng = np.random.default_rng(7)
    patterns = generate_random_bipolar_patterns(N, P, rng=rng)

    net = HopfieldNetwork(N)
    net.train(patterns)

    # Pick pattern 0 and corrupt 25% of bits
    cue = corrupt_cue(patterns[0], flip_probability=0.25, rng=rng)
    _, _, _, energy_history = net.retrieve(cue, max_sweeps=30, track_energy=True, rng=rng)

    assert len(energy_history) > 0
    # Every subsequent energy must be <= previous energy
    for i in range(1, len(energy_history)):
        assert energy_history[i] <= energy_history[i - 1] + 1e-12, (
            f"Energy increased from {energy_history[i-1]} to {energy_history[i]} at step {i}"
        )


def test_small_scale_perfect_retrieval():
    """Milestone 1 & 2: Small network (N=100, P=3) should reliably retrieve from a 10% corrupted cue."""
    N = 100
    P = 3
    rng = np.random.default_rng(101)
    patterns = generate_random_bipolar_patterns(N, P, rng=rng)

    net = HopfieldNetwork(N)
    net.train(patterns)

    for mu in range(P):
        cue = corrupt_cue(patterns[mu], flip_probability=0.10, rng=rng)
        retrieved, converged, _, _ = net.retrieve(cue, max_sweeps=20, rng=rng)
        assert converged
        assert np.array_equal(retrieved, patterns[mu])
        assert net.is_stable(retrieved)
