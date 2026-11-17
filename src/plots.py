"""
Visualization module for Hopfield network memory resilience experiments.
Generates publication-quality figures:
1. Baseline accuracy vs. memory load (with theoretical 0.138 benchmark).
2. Accuracy vs. lesion fraction across memory loads.
3. Memory load x Lesion fraction accuracy heatmap.
4. Retrieval outcome composition breakdown (Target, Other, Spurious, Non-converged).
5. Cue noise interaction curves.
6. Energy descent trajectory verification.
"""

import os
import sys
from typing import Optional

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style():
    """Apply clean academic styling for matplotlib."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_baseline_load_vs_accuracy(
    summary_df: pd.DataFrame,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot baseline recall accuracy against memory load with no lesions (lesion_fraction = 0).
    Displays 95% confidence intervals and the classical P/N = 0.138 critical capacity line.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    baseline = summary_df[summary_df["lesion_fraction"] == 0.0]
    cue_noise_levels = sorted(baseline["cue_noise"].unique())

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "v"]

    for idx, cue_q in enumerate(cue_noise_levels):
        sub = baseline[baseline["cue_noise"] == cue_q].sort_values("load")
        ax.errorbar(
            sub["load"],
            sub["accuracy_mean"],
            yerr=sub["accuracy_ci95"],
            label=f"Cue noise $q={cue_q:.2f}$",
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            capsize=4,
            capthick=1.2,
        )

    # Theoretical capacity benchmark
    ax.axvline(
        x=0.138,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=r"Theoretical capacity benchmark $\alpha_c \approx 0.138$",
    )

    ax.set_xlabel("Memory Load ($P / N$)")
    ax.set_ylabel("Exact Recall Accuracy")
    ax.set_title("Baseline Recall Accuracy vs. Memory Load (No Lesions)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig1_baseline_load_vs_accuracy.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_lesion_vs_accuracy(
    summary_df: pd.DataFrame,
    target_cue_noise: float = 0.10,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot recall accuracy against synaptic lesion fraction for various memory loads.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    sub = summary_df[np.isclose(summary_df["cue_noise"], target_cue_noise)]
    loads = sorted(sub["load"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(loads)))

    for idx, load_val in enumerate(loads):
        load_sub = sub[sub["load"] == load_val].sort_values("lesion_fraction")
        ax.errorbar(
            load_sub["lesion_fraction"],
            load_sub["accuracy_mean"],
            yerr=load_sub["accuracy_ci95"],
            label=f"$P/N = {load_val:.2f}$ ($P={int(load_sub['P'].iloc[0])}$)",
            color=cmap[idx],
            marker="o",
            capsize=3,
        )

    ax.set_xlabel("Synaptic Lesion Fraction ($f$)")
    ax.set_ylabel("Exact Recall Accuracy")
    ax.set_title(f"Recall Accuracy vs. Synaptic Lesion Fraction (Cue noise $q={target_cue_noise}$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig2_lesions_vs_accuracy.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_load_lesion_heatmap(
    summary_df: pd.DataFrame,
    target_cue_noise: float = 0.10,
    output_dir: str = "results/figures",
) -> str:
    """
    Generate a 2D heatmap of recall accuracy across memory load x lesion fraction.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    sub = summary_df[np.isclose(summary_df["cue_noise"], target_cue_noise)]
    pivot = sub.pivot(index="load", columns="lesion_fraction", values="accuracy_mean")

    im = ax.imshow(
        pivot.values,
        cmap="coolwarm_r",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])

    ax.set_xlabel("Synaptic Lesion Fraction ($f$)")
    ax.set_ylabel("Memory Load ($P / N$)")
    ax.set_title(f"Accuracy Heatmap: Load × Lesions ($q={target_cue_noise}$)")

    # Overlay numeric cell values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text_color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Exact Recall Accuracy")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig3_heatmap_load_vs_lesion.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_outcome_composition(
    summary_df: pd.DataFrame,
    target_cue_noise: float = 0.10,
    selected_lesions: Optional[list] = None,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot the stacked breakdown of retrieval outcomes across memory load:
    Target, Other Stored, Stable Spurious, Non-converged.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    if selected_lesions is None:
        selected_lesions = [0.0, 0.20, 0.40]

    sub = summary_df[np.isclose(summary_df["cue_noise"], target_cue_noise)]
    available_lesions = [l for l in selected_lesions if any(np.isclose(sub["lesion_fraction"], l))]

    fig, axes = plt.subplots(
        1, len(available_lesions), figsize=(5 * len(available_lesions), 5), sharey=True
    )
    if len(available_lesions) == 1:
        axes = [axes]

    categories = [
        ("exact_target_mean", "Exact Target", "#2ca02c"),
        ("other_stored_mean", "Other Stored Memory", "#1f77b4"),
        ("stable_spurious_mean", "Stable Spurious State", "#ff7f0e"),
        ("non_converged_mean", "Non-converged", "#d62728"),
    ]

    for ax, lesion_val in zip(axes, available_lesions):
        l_sub = sub[np.isclose(sub["lesion_fraction"], lesion_val)].sort_values("load")
        loads = l_sub["load"].values
        bottom = np.zeros(len(loads))

        for col, label, color in categories:
            vals = l_sub[col].values
            ax.bar(
                [f"{x:.2f}" for x in loads],
                vals,
                bottom=bottom,
                label=label if lesion_val == available_lesions[0] else None,
                color=color,
                edgecolor="white",
                width=0.7,
            )
            bottom += vals

        ax.set_title(f"Lesion Fraction $f={lesion_val:.2f}$")
        ax.set_xlabel("Memory Load ($P / N$)")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Outcome Proportion")
    fig.suptitle(f"Retrieval Outcome Composition Breakdown (Cue Noise $q={target_cue_noise}$)", y=1.02)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig4_outcome_composition.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cue_corruption_interaction(
    summary_df: pd.DataFrame,
    selected_loads: Optional[list] = None,
    output_dir: str = "results/figures",
) -> str:
    """
    Plot how cue noise interacts with memory load and synaptic damage.
    """
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    if selected_loads is None:
        selected_loads = [0.05, 0.10, 0.14]

    loads = [l for l in selected_loads if any(np.isclose(summary_df["load"], l))]
    fig, axes = plt.subplots(1, len(loads), figsize=(5 * len(loads), 5), sharey=True)
    if len(loads) == 1:
        axes = [axes]

    cue_levels = sorted(summary_df["cue_noise"].unique())
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(cue_levels)))

    for ax, load_val in zip(axes, loads):
        sub = summary_df[np.isclose(summary_df["load"], load_val)]
        for idx, q_val in enumerate(cue_levels):
            q_sub = sub[np.isclose(sub["cue_noise"], q_val)].sort_values("lesion_fraction")
            ax.plot(
                q_sub["lesion_fraction"],
                q_sub["accuracy_mean"],
                marker="o",
                label=f"$q = {q_val:.2f}$",
                color=cmap[idx],
            )
        ax.set_title(f"Load $P/N = {load_val:.2f}$")
        ax.set_xlabel("Lesion Fraction ($f$)")
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Exact Recall Accuracy")
    axes[-1].legend(title="Cue Noise ($q$)", bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.suptitle("Cue Noise Interaction Across Memory Loads and Synaptic Lesions", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig5_cue_corruption_interaction.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_energy_trajectory(
    N: int = 100,
    P: int = 4,
    cue_noise: float = 0.25,
    seed: int = 42,
    output_dir: str = "results/figures",
) -> str:
    """
    Generate and plot an empirical Lyapunov energy trajectory during asynchronous updates.
    Demonstrates that E(s) monotonically decreases on every neuron flip.
    """
    from src.patterns import generate_random_bipolar_patterns
    from src.hopfield import HopfieldNetwork
    from src.lesions import corrupt_cue

    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    patterns = generate_random_bipolar_patterns(N, P, rng=rng)
    net = HopfieldNetwork(N)
    net.train(patterns)

    cue = corrupt_cue(patterns[0], flip_probability=cue_noise, rng=rng)
    _, _, _, energy_history = net.retrieve(cue, max_sweeps=30, track_energy=True, rng=rng)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(energy_history)), energy_history, marker=".", color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Neuron Flip Step")
    ax.set_ylabel(r"Lyapunov Energy $E(s) = -\frac{1}{2} s^T W s$")
    ax.set_title("Monotonic Energy Descent During Asynchronous Retrieval")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig6_energy_trajectory.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_all_plots(summary_df: pd.DataFrame, output_dir: str = "results/figures"):
    """Generate all standard figures from aggregated summary data."""
    print("Generating figures...")
    p1 = plot_baseline_load_vs_accuracy(summary_df, output_dir)
    print(f"Saved: {p1}")
    p2 = plot_lesion_vs_accuracy(summary_df, target_cue_noise=0.10, output_dir=output_dir)
    print(f"Saved: {p2}")
    p3 = plot_load_lesion_heatmap(summary_df, target_cue_noise=0.10, output_dir=output_dir)
    print(f"Saved: {p3}")
    p4 = plot_outcome_composition(summary_df, target_cue_noise=0.10, output_dir=output_dir)
    print(f"Saved: {p4}")
    p5 = plot_cue_corruption_interaction(summary_df, output_dir=output_dir)
    print(f"Saved: {p5}")
    p6 = plot_energy_trajectory(output_dir=output_dir)
    print(f"Saved: {p6}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        summary_path = sys.argv[1]
        summary = pd.read_csv(summary_path)
        generate_all_plots(summary)
    else:
        print("Please provide path to summary CSV file.")
