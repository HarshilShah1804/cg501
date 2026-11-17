"""
Tests for metrics, outcome classification, and Hamming similarity.
"""

import numpy as np
import pytest
from src.metrics import (
    normalized_hamming_similarity,
    classify_outcome,
    evaluate_trial,
)


def test_hamming_similarity():
    s1 = np.array([1, 1, -1, -1, 1], dtype=np.int8)
    s2 = np.array([1, -1, -1, 1, 1], dtype=np.int8)
    # matching: indices 0, 2, 4 (3 out of 5)
    sim = normalized_hamming_similarity(s1, s2)
    assert pytest.approx(sim) == 0.6


def test_outcome_classification_hierarchy():
    stored = np.array([
        [1, 1, 1, 1],
        [-1, -1, -1, -1],
        [1, -1, 1, -1],
    ], dtype=np.int8)

    target_idx = 0
    target = stored[target_idx]

    # 1. Exact target
    assert classify_outcome(target, target, stored, target_idx, is_stable=True, converged=True) == "exact_target"

    # 2. Other stored memory
    other = stored[1]
    assert classify_outcome(other, target, stored, target_idx, is_stable=True, converged=True) == "other_stored"

    # 3. Stable spurious state
    spurious = np.array([1, 1, -1, -1], dtype=np.int8)
    assert classify_outcome(spurious, target, stored, target_idx, is_stable=True, converged=True) == "stable_spurious"

    # 4. Non-converged: incorrect state that is NOT stable must NOT be classified as spurious!
    assert classify_outcome(spurious, target, stored, target_idx, is_stable=False, converged=True) == "non_converged"
    assert classify_outcome(spurious, target, stored, target_idx, is_stable=True, converged=False) == "non_converged"
    assert classify_outcome(spurious, target, stored, target_idx, is_stable=False, converged=False) == "non_converged"
