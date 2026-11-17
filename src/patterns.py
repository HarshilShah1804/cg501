"""
Pattern generation module for Hopfield associative memory.
"""

from typing import Optional
import numpy as np


def generate_random_bipolar_patterns(
    N: int,
    P: int,
    rng: Optional[np.random.Generator] = None,
    exact_balance: bool = False,
) -> np.ndarray:
    """
    Generate P independent random bipolar patterns of dimension N.
    
    Parameters
    ----------
    N : int
        Dimension of each pattern (number of neurons).
    P : int
        Number of patterns to generate.
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility.
    exact_balance : bool, default False
        If True, each pattern has exactly N//2 ones and N - N//2 minus ones.
        If False, each bit is independently drawn from {-1, +1} with p=0.5.
        
    Returns
    -------
    patterns : np.ndarray
        Array of shape (P, N) with values in {-1, +1}.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if exact_balance:
        patterns = np.zeros((P, N), dtype=np.int8)
        half = N // 2
        base = np.array([1] * half + [-1] * (N - half), dtype=np.int8)
        for mu in range(P):
            patterns[mu] = rng.permutation(base)
        return patterns
    else:
        # Independent coin flips: {-1, +1} with equal probability
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, N))


def load_binarized_mnist(
    num_patterns: int,
    data_dir: str = "./data",
    rng: Optional[np.random.Generator] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Load and binarize MNIST handwritten digits as bipolar patterns in {-1, +1}^784.
    
    Parameters
    ----------
    num_patterns : int
        Number of digit patterns to extract.
    data_dir : str, default "./data"
        Directory to download/cache MNIST dataset.
    rng : np.random.Generator, optional
        Random generator for digit sampling.
    threshold : float, default 0.5
        Intensity threshold (in [0, 1]) for binarization.
        
    Returns
    -------
    patterns : np.ndarray
        Array of shape (num_patterns, 784) with values in {-1, +1}.
    """
    try:
        from torchvision import datasets, transforms
        transform = transforms.ToTensor()
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        if rng is None:
            rng = np.random.default_rng()
            
        indices = rng.choice(len(dataset), size=num_patterns, replace=False)
        patterns = []
        for idx in indices:
            img, _ = dataset[int(idx)]
            img_arr = img.squeeze().numpy()  # 28x28
            flat = (img_arr > threshold).astype(np.int8)
            bipolar = np.where(flat == 1, 1, -1).astype(np.int8)
            patterns.append(bipolar.flatten())
            
        return np.array(patterns, dtype=np.int8)
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST dataset: {e}")
