"""
Synaptic lesioning and cue corruption modules for Hopfield networks.
"""

from typing import Optional, Tuple
import numpy as np


def apply_random_lesion(
    W: np.ndarray,
    lesion_fraction: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """
    Remove a random fraction of unique undirected off-diagonal synaptic connections.
    
    Total undirected connections: N * (N - 1) / 2.
    When connection (i, j) is removed, W_ij = W_ji = 0.
    
    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (N, N).
    lesion_fraction : float
        Fraction of undirected connections to delete (between 0.0 and 1.0).
    rng : np.random.Generator, optional
        Random generator for reproducibility.
        
    Returns
    -------
    W_lesioned : np.ndarray
        Lesioned symmetric weight matrix copy.
    num_deleted : int
        Number of undirected connections removed.
    """
    if not (0.0 <= lesion_fraction <= 1.0):
        raise ValueError(f"lesion_fraction must be in [0.0, 1.0], got {lesion_fraction}")

    N = W.shape[0]
    total_undirected = N * (N - 1) // 2
    num_to_delete = int(round(lesion_fraction * total_undirected))

    W_lesioned = W.copy()
    if num_to_delete == 0:
        return W_lesioned, 0

    if rng is None:
        rng = np.random.default_rng()

    # Upper triangular indices (i < j)
    triu_i, triu_j = np.triu_indices(N, k=1)
    chosen_indices = rng.choice(total_undirected, size=num_to_delete, replace=False)

    del_i = triu_i[chosen_indices]
    del_j = triu_j[chosen_indices]

    W_lesioned[del_i, del_j] = 0.0
    W_lesioned[del_j, del_i] = 0.0

    return W_lesioned, num_to_delete


def apply_targeted_lesion(
    W: np.ndarray,
    lesion_fraction: float,
) -> Tuple[np.ndarray, int]:
    """
    Remove undirected off-diagonal connections with the largest absolute weights |W_ij|.
    
    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (N, N).
    lesion_fraction : float
        Fraction of undirected connections to delete (between 0.0 and 1.0).
        
    Returns
    -------
    W_lesioned : np.ndarray
        Lesioned symmetric weight matrix copy.
    num_deleted : int
        Number of undirected connections removed.
    """
    if not (0.0 <= lesion_fraction <= 1.0):
        raise ValueError(f"lesion_fraction must be in [0.0, 1.0], got {lesion_fraction}")

    N = W.shape[0]
    total_undirected = N * (N - 1) // 2
    num_to_delete = int(round(lesion_fraction * total_undirected))

    W_lesioned = W.copy()
    if num_to_delete == 0:
        return W_lesioned, 0

    triu_i, triu_j = np.triu_indices(N, k=1)
    abs_weights = np.abs(W[triu_i, triu_j])

    # Find indices of largest absolute weights
    top_indices = np.argsort(abs_weights)[-num_to_delete:]

    del_i = triu_i[top_indices]
    del_j = triu_j[top_indices]

    W_lesioned[del_i, del_j] = 0.0
    W_lesioned[del_j, del_i] = 0.0

    return W_lesioned, num_to_delete


def corrupt_cue(
    pattern: np.ndarray,
    flip_probability: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Independently flip each neuron in the pattern with probability q:
    s_i = -xi_i with probability q, xi_i otherwise.
    
    Parameters
    ----------
    pattern : np.ndarray
        Target bipolar pattern of shape (N,).
    flip_probability : float
        Probability q of flipping each bit.
    rng : np.random.Generator, optional
        Random generator for bit flipping.
        
    Returns
    -------
    corrupted_cue : np.ndarray
        Noisy cue vector of shape (N,).
    """
    if not (0.0 <= flip_probability <= 1.0):
        raise ValueError(f"flip_probability must be in [0.0, 1.0], got {flip_probability}")

    if rng is None:
        rng = np.random.default_rng()

    cue = pattern.astype(np.int8).copy()
    flips = rng.random(size=cue.shape) < flip_probability
    cue[flips] = -cue[flips]
    return cue
