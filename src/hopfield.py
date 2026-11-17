"""
Hopfield Network implementation with asynchronous dynamics and energy tracking.
"""

from typing import List, Optional, Tuple
import numpy as np


class HopfieldNetwork:
    """
    Classical fully-connected Hopfield Neural Network with bipolar states {-1, +1}.
    
    Attributes
    ----------
    N : int
        Number of neurons.
    W : np.ndarray
        Synaptic weight matrix of shape (N, N).
    stored_patterns : Optional[np.ndarray]
        Array of stored patterns of shape (P, N).
    """

    def __init__(self, N: int):
        self.N = N
        self.W = np.zeros((N, N), dtype=np.float64)
        self.stored_patterns: Optional[np.ndarray] = None

    def train(self, patterns: np.ndarray) -> None:
        """
        Train the network using standard Hebbian learning with zero diagonal.
        
        W_ij = (1 / N) * sum_{mu=1}^P xi_i^mu * xi_j^mu  (for i != j)
        W_ii = 0
        
        Parameters
        ----------
        patterns : np.ndarray
            Array of shape (P, N) with values in {-1, +1}.
        """
        P, N = patterns.shape
        if N != self.N:
            raise ValueError(f"Pattern dimension {N} does not match network size {self.N}")

        patterns_float = patterns.astype(np.float64)
        # Outer-product Hebbian learning
        self.W = (patterns_float.T @ patterns_float) / float(self.N)
        np.fill_diagonal(self.W, 0.0)
        self.stored_patterns = patterns.copy()

    def energy(self, state: np.ndarray) -> float:
        """
        Compute the Lyapunov energy of a network state:
        E(s) = -0.5 * sum_{i,j} W_{ij} s_i s_j
        
        Parameters
        ----------
        state : np.ndarray
            Bipolar state vector of shape (N,).
            
        Returns
        -------
        float
            Energy of the state.
        """
        s = state.astype(np.float64)
        return float(-0.5 * (s @ self.W @ s))

    def compute_local_field(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the local field h = W @ s for all neurons.
        
        Parameters
        ----------
        state : np.ndarray
            Bipolar state vector of shape (N,).
            
        Returns
        -------
        np.ndarray
            Local field vector of shape (N,).
        """
        return self.W @ state.astype(np.float64)

    def is_stable(self, state: np.ndarray) -> bool:
        """
        Check if a given state is a fixed point (attractor) under the deterministic update rule:
        s_i -> +1 if h_i > 0, -1 if h_i < 0, s_i if h_i == 0.
        
        Parameters
        ----------
        state : np.ndarray
            Bipolar state vector of shape (N,).
            
        Returns
        -------
        bool
            True if state is completely stable, False otherwise.
        """
        h = self.compute_local_field(state)
        next_state = np.where(h > 0, 1, np.where(h < 0, -1, state)).astype(np.int8)
        return bool(np.array_equal(state, next_state))

    def retrieve(
        self,
        cue: np.ndarray,
        max_sweeps: int = 50,
        track_energy: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, bool, int, List[float]]:
        """
        Perform asynchronous retrieval starting from an initial cue.
        
        In each sweep, all N neurons are visited in a randomized permutation order.
        Updates are applied sequentially:
        s_i <- +1 if h_i > 0, -1 if h_i < 0, s_i if h_i == 0.
        
        Parameters
        ----------
        cue : np.ndarray
            Initial bipolar cue vector of shape (N,).
        max_sweeps : int, default 50
            Maximum number of full sweeps across all N neurons.
        track_energy : bool, default False
            Whether to record the energy after every flip / sweep.
        rng : np.random.Generator, optional
            Random generator for sweep permutations.
            
        Returns
        -------
        final_state : np.ndarray
            Retrieved bipolar state vector of shape (N,).
        converged : bool
            Whether a full sweep passed with zero neuron flips.
        sweeps_taken : int
            Number of sweeps executed.
        energy_history : List[float]
            List of energy values over time (if track_energy=True).
        """
        if rng is None:
            rng = np.random.default_rng()

        s = cue.astype(np.int8).copy()
        h = self.compute_local_field(s)
        
        energy_history: List[float] = []
        if track_energy:
            energy_history.append(self.energy(s))

        converged = False
        sweeps_taken = 0

        for sweep in range(1, max_sweeps + 1):
            sweeps_taken = sweep
            flips_in_sweep = 0
            order = rng.permutation(self.N)

            for idx in order:
                h_i = h[idx]
                if h_i > 0:
                    s_new = 1
                elif h_i < 0:
                    s_new = -1
                else:
                    s_new = s[idx]

                if s_new != s[idx]:
                    delta_s = s_new - s[idx]
                    s[idx] = s_new
                    # Incrementally update local fields: h_j = h_j + W_ji * delta_s
                    h += self.W[:, idx] * delta_s
                    flips_in_sweep += 1
                    
                    if track_energy:
                        energy_history.append(self.energy(s))

            if flips_in_sweep == 0:
                converged = True
                break

        return s, converged, sweeps_taken, energy_history
