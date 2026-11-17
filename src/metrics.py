"""
Metrics and outcome classification for Hopfield network retrieval trials.
"""

from typing import Dict, Any
import numpy as np


def normalized_hamming_similarity(state: np.ndarray, target: np.ndarray) -> float:
    """
    Compute normalized Hamming similarity between retrieved state and target:
    similarity = (1 / N) * sum_i 1[s_i == xi_i]
    
    Parameters
    ----------
    state : np.ndarray
        Retrieved bipolar state vector of shape (N,).
    target : np.ndarray
        Target bipolar memory vector of shape (N,).
        
    Returns
    -------
    float
        Fraction of matching neurons in [0.0, 1.0].
    """
    return float(np.mean(state == target))


def classify_outcome(
    state: np.ndarray,
    target: np.ndarray,
    stored_patterns: np.ndarray,
    target_idx: int,
    is_stable: bool,
    converged: bool,
) -> str:
    """
    Classify the retrieval outcome into one of 4 mutually exclusive categories:
    1. 'exact_target': final state exactly equals the target memory.
    2. 'other_stored': final state equals another stored memory (not target).
    3. 'stable_spurious': final state is stable under the update rule, but does not match any stored memory.
    4. 'non_converged': run did not stabilize or reached sweep limit without being stable.
    
    Parameters
    ----------
    state : np.ndarray
        Retrieved bipolar state vector of shape (N,).
    target : np.ndarray
        Target bipolar memory vector of shape (N,).
    stored_patterns : np.ndarray
        All stored patterns of shape (P, N).
    target_idx : int
        Index of the target memory in stored_patterns.
    is_stable : bool
        Whether the final state is a verified fixed point under the network update rule.
    converged : bool
        Whether the asynchronous retrieval loop completed with zero flips in the final sweep.
        
    Returns
    -------
    str
        Outcome classification label.
    """
    # 1. Check exact target recall
    if np.array_equal(state, target):
        return "exact_target"

    # 2. Check retrieval of any other stored pattern
    P = stored_patterns.shape[0]
    for mu in range(P):
        if mu != target_idx and np.array_equal(state, stored_patterns[mu]):
            return "other_stored"

    # 3. Check stability before spurious classification
    if is_stable and converged:
        return "stable_spurious"
    else:
        # Non-convergence if not stable or did not converge
        return "non_converged"


def evaluate_trial(
    state: np.ndarray,
    target: np.ndarray,
    stored_patterns: np.ndarray,
    target_idx: int,
    is_stable: bool,
    converged: bool,
    sweeps_taken: int,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for a single retrieval trial.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with outcome classification, similarity, stability, and sweeps.
    """
    outcome = classify_outcome(
        state=state,
        target=target,
        stored_patterns=stored_patterns,
        target_idx=target_idx,
        is_stable=is_stable,
        converged=converged,
    )
    similarity = normalized_hamming_similarity(state, target)
    is_inverted_target = bool(np.array_equal(state, -target))

    return {
        "outcome": outcome,
        "exact_recall": int(outcome == "exact_target"),
        "similarity": similarity,
        "is_stable": is_stable,
        "converged": converged,
        "sweeps_taken": sweeps_taken,
        "is_inverted_target": is_inverted_target,
    }
