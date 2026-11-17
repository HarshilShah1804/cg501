"""
Experiment execution module: parameter sweeps, multiprocessing, and raw data logging.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor

from src.patterns import generate_random_bipolar_patterns
from src.hopfield import HopfieldNetwork
from src.lesions import apply_random_lesion, apply_targeted_lesion, corrupt_cue
from src.metrics import evaluate_trial


def run_trial(
    replicate_id: int,
    N: int,
    P: int,
    load: float,
    lesion_fraction: float,
    cue_noise: float,
    max_sweeps: int,
    base_seed: int,
    lesion_type: str = "random",
    cues_per_network: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run an independent network/lesion configuration replicate across designated cues.
    
    Parameters
    ----------
    replicate_id : int
        Replicate index (1 to num_replicates).
    N : int
        Number of neurons.
    P : int
        Number of patterns.
    load : float
        P / N ratio.
    lesion_fraction : float
        Fraction of undirected connections deleted.
    cue_noise : float
        Probability of flipping each bit of the cue.
    max_sweeps : int
        Maximum asynchronous sweeps.
    base_seed : int
        Seed offset for reproducibility.
    lesion_type : str
        'random' or 'targeted'.
    cues_per_network : int
        Number of target patterns to test per independent network.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of trial outcome dictionaries.
    """
    # Deterministic independent seed per configuration and replicate
    seed = (
        int(base_seed)
        + int(replicate_id) * 100003
        + int(round(load * 1000)) * 7919
        + int(round(lesion_fraction * 100)) * 997
        + int(round(cue_noise * 100)) * 101
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    # 1. Generate patterns
    patterns = generate_random_bipolar_patterns(N=N, P=P, rng=rng)

    # 2. Train network
    net = HopfieldNetwork(N=N)
    net.train(patterns)

    # 3. Apply lesion
    if lesion_type == "targeted":
        W_lesioned, deleted_count = apply_targeted_lesion(net.W, lesion_fraction=lesion_fraction)
    else:
        W_lesioned, deleted_count = apply_random_lesion(net.W, lesion_fraction=lesion_fraction, rng=rng)
    net.W = W_lesioned

    results: List[Dict[str, Any]] = []
    # Test up to min(cues_per_network, P) target patterns
    num_targets = min(cues_per_network, P)
    target_indices = rng.choice(P, size=num_targets, replace=False)

    for target_idx in target_indices:
        target = patterns[target_idx]
        cue = corrupt_cue(target, flip_probability=cue_noise, rng=rng)

        final_state, converged, sweeps_taken, _ = net.retrieve(
            cue=cue,
            max_sweeps=max_sweeps,
            track_energy=False,
            rng=rng,
        )
        is_stable = net.is_stable(final_state)

        trial_metrics = evaluate_trial(
            state=final_state,
            target=target,
            stored_patterns=patterns,
            target_idx=int(target_idx),
            is_stable=is_stable,
            converged=converged,
            sweeps_taken=sweeps_taken,
        )

        record = {
            "replicate_id": replicate_id,
            "N": N,
            "P": P,
            "load": load,
            "lesion_fraction": lesion_fraction,
            "cue_noise": cue_noise,
            "lesion_type": lesion_type,
            "deleted_edges": deleted_count,
            "target_idx": int(target_idx),
            "seed": seed,
            **trial_metrics,
        }
        results.append(record)

    return results


def _worker_task(task_args: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_trial(**task_args)


def run_experiment(
    config_path: str = "configs/baseline.yaml",
    num_replicates: Optional[int] = None,
    memory_loads: Optional[List[float]] = None,
    lesion_fractions: Optional[List[float]] = None,
    cue_noise_levels: Optional[List[float]] = None,
    num_workers: Optional[int] = None,
    output_path: Optional[str] = None,
    cues_per_network: int = 3,
) -> pd.DataFrame:
    """
    Run full or customized parameter sweep and save raw results.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    net_cfg = config.get("network", {})
    exp_cfg = config.get("experiment", {})
    paths_cfg = config.get("paths", {})

    N = int(net_cfg.get("N", 400))
    max_sweeps = int(net_cfg.get("max_sweeps", 50))
    base_seed = int(exp_cfg.get("base_seed", 42))

    loads = memory_loads or exp_cfg.get("memory_loads", [0.02, 0.05, 0.10, 0.14])
    lesions = lesion_fractions or exp_cfg.get("lesion_fractions", [0.0, 0.10, 0.30, 0.50])
    cues = cue_noise_levels or exp_cfg.get("cue_noise_levels", [0.10, 0.20])
    replicates = num_replicates or exp_cfg.get("num_replicates", 30)

    tasks: List[Dict[str, Any]] = []
    for rep in range(1, replicates + 1):
        for load in loads:
            P = int(round(load * N))
            for lesion in lesions:
                for cue_noise in cues:
                    tasks.append({
                        "replicate_id": rep,
                        "N": N,
                        "P": P,
                        "load": float(load),
                        "lesion_fraction": float(lesion),
                        "cue_noise": float(cue_noise),
                        "max_sweeps": max_sweeps,
                        "base_seed": base_seed,
                        "lesion_type": "random",
                        "cues_per_network": cues_per_network,
                    })

    total_tasks = len(tasks)
    print(f"Executing {total_tasks} network configurations across workers...")

    if num_workers is None:
        num_workers = min(16, os.cpu_count() or 4)

    start_time = time.time()
    all_records: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_results in executor.map(_worker_task, tasks):
            all_records.extend(chunk_results)

    elapsed = time.time() - start_time
    print(f"Finished {len(all_records)} total retrieval trials in {elapsed:.2f}s ({len(all_records)/elapsed:.1f} trials/s).")

    df = pd.DataFrame(all_records)

    # Save raw CSV
    if output_path is None:
        raw_dir = paths_cfg.get("raw_results", "results/raw")
        os.makedirs(raw_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(raw_dir, f"raw_results_{timestamp}.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved raw trial data to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hopfield Memory Resilience Experiments")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--replicates", type=int, default=None, help="Override number of replicates")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--cues", type=int, default=3, help="Cues per network configuration")
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        num_replicates=args.replicates,
        num_workers=args.workers,
        output_path=args.output,
        cues_per_network=args.cues,
    )
