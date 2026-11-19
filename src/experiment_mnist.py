"""
MNIST extension experiment: tests Hopfield network on binarized MNIST digits (N=784)
and compares recall performance against synthetic random bipolar patterns.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.patterns import load_binarized_mnist, generate_random_bipolar_patterns
from src.hopfield import HopfieldNetwork
from src.lesions import apply_random_lesion, corrupt_cue
from src.metrics import evaluate_trial
from src.plots import set_plot_style


def run_mnist_experiment(
    p_values: list = [2, 4, 6, 8, 10, 15, 20],
    num_replicates: int = 10,
    cue_noise: float = 0.10,
    lesion_fractions: list = [0.0, 0.10, 0.20, 0.30],
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run comparison experiment between MNIST digits and Synthetic random patterns on N=784.
    """
    N = 784
    records = []

    print(f"Running MNIST vs Synthetic experiment on N={N} across P={p_values}...")

    for P in p_values:
        for rep in range(1, num_replicates + 1):
            seed = 1000 + rep * 17 + P * 3
            rng = np.random.default_rng(seed)

            # 1. Load MNIST patterns and Synthetic patterns
            mnist_patterns = load_binarized_mnist(num_patterns=P, rng=rng)
            synth_patterns = generate_random_bipolar_patterns(N=N, P=P, rng=rng)

            for dataset_name, patterns in [("MNIST", mnist_patterns), ("Synthetic", synth_patterns)]:
                # Train network
                net = HopfieldNetwork(N=N)
                net.train(patterns)
                orig_W = net.W.copy()

                for f in lesion_fractions:
                    if f > 0:
                        W_lesioned, _ = apply_random_lesion(orig_W, lesion_fraction=f, rng=rng)
                    else:
                        W_lesioned = orig_W.copy()
                    net.W = W_lesioned

                    # Test target pattern retrieval across up to 3 patterns
                    test_targets = min(3, P)
                    for mu in range(test_targets):
                        target = patterns[mu]
                        cue = corrupt_cue(target, flip_probability=cue_noise, rng=rng)

                        retrieved, converged, sweeps, _ = net.retrieve(cue, max_sweeps=30, rng=rng)
                        is_stable = net.is_stable(retrieved)

                        metrics = evaluate_trial(
                            state=retrieved,
                            target=target,
                            stored_patterns=patterns,
                            target_idx=mu,
                            is_stable=is_stable,
                            converged=converged,
                            sweeps_taken=sweeps,
                        )

                        records.append({
                            "dataset": dataset_name,
                            "P": P,
                            "load": P / N,
                            "lesion_fraction": f,
                            "cue_noise": cue_noise,
                            "replicate_id": rep,
                            "target_idx": mu,
                            **metrics,
                        })

    df = pd.DataFrame(records)

    # Replicate-level aggregation
    rep_grouped = df.groupby(
        ["dataset", "P", "load", "lesion_fraction", "cue_noise", "replicate_id"],
        as_index=False,
    ).agg({
        "exact_recall": "mean",
        "similarity": "mean",
        "is_stable": "mean",
        "converged": "mean",
    })

    # Summary aggregation
    summary = rep_grouped.groupby(
        ["dataset", "P", "load", "lesion_fraction", "cue_noise"],
        as_index=False,
    ).agg({
        "exact_recall": ["mean", "std"],
        "similarity": ["mean", "std"],
    })
    summary.columns = [
        "dataset", "P", "load", "lesion_fraction", "cue_noise",
        "accuracy_mean", "accuracy_std", "similarity_mean", "similarity_std"
    ]

    out_csv = os.path.join(output_dir, "processed", "mnist_vs_synthetic_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Summary saved to {out_csv}")

    # Generate comparison plot
    plot_mnist_comparison(summary, output_dir=os.path.join(output_dir, "figures"))
    return summary


def plot_mnist_comparison(summary_df: pd.DataFrame, output_dir: str = "results/figures"):
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Baseline Accuracy vs P (No lesions, f=0)
    base = summary_df[summary_df["lesion_fraction"] == 0.0]
    for dset, color, marker in [("MNIST", "#d62728", "s"), ("Synthetic", "#1f77b4", "o")]:
        sub = base[base["dataset"] == dset].sort_values("P")
        ax1.plot(
            sub["P"],
            sub["accuracy_mean"],
            marker=marker,
            color=color,
            linewidth=2.2,
            markersize=7,
            label=f"{dset} (Undamaged)",
        )

    ax1.set_xlabel("Number of Stored Patterns ($P$)")
    ax1.set_ylabel("Exact Recall Accuracy")
    ax1.set_title("Baseline Recall: MNIST vs. Synthetic Patterns ($N=784$)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=True)

    # Panel 2: Normalized Hamming Similarity vs P (showing partial pattern completion)
    for dset, color, marker in [("MNIST", "#d62728", "s"), ("Synthetic", "#1f77b4", "o")]:
        sub = base[base["dataset"] == dset].sort_values("P")
        ax2.plot(
            sub["P"],
            sub["similarity_mean"],
            marker=marker,
            color=color,
            linewidth=2.2,
            markersize=7,
            label=f"{dset} Similarity",
        )

    ax2.set_xlabel("Number of Stored Patterns ($P$)")
    ax2.set_ylabel("Normalized Hamming Similarity")
    ax2.set_title("Retrieval Quality (Hamming Similarity) vs. $P$")
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig7_mnist_vs_synthetic.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Comparison figure saved to {out_path}")


if __name__ == "__main__":
    run_mnist_experiment()
