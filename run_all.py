"""
Master execution script for the Hopfield Memory Resilience Project.
Runs test suite, parameter sweep experiment, statistical analysis, and plot generation.
"""

import argparse
import os
import subprocess
import sys
import time

from src.experiment import run_experiment
from src.analysis import process_and_save_summary
from src.plots import generate_all_plots


def run_pipeline(
    replicates: int = 30,
    cues: int = 3,
    skip_tests: bool = False,
    config: str = "configs/baseline.yaml",
):
    print("=" * 70)
    print(" HOPFIELD MEMORY RESILIENCE & SYNAPTIC DAMAGE EXPERIMENT PIPELINE")
    print("=" * 70)

    # 1. Run unit test suite
    if not skip_tests:
        print("\n[Step 1/4] Running unit test suite via pytest...")
        test_res = subprocess.run([sys.executable, "-m", "pytest", "-v"], check=False)
        if test_res.returncode != 0:
            print("ERROR: Unit tests failed. Aborting pipeline.")
            sys.exit(1)
        print(">> All unit tests PASSED successfully.\n")
    else:
        print("\n[Step 1/4] Skipping tests as requested.\n")

    # 2. Run experiment parameter sweeps
    print(f"\n[Step 2/4] Executing experiment sweeps ({replicates} replicates, {cues} cues/network)...")
    raw_csv = f"results/raw/raw_results_rep{replicates}.csv"
    run_experiment(
        config_path=config,
        num_replicates=replicates,
        cues_per_network=cues,
        output_path=raw_csv,
    )

    # 3. Aggregate results and compute 95% CIs across independent replicates
    print(f"\n[Step 3/4] Aggregating results across independent replicates...")
    summary_csv = f"results/processed/summary_results_rep{replicates}.csv"
    summary_df = process_and_save_summary(raw_csv, summary_csv)

    # 4. Generate figures
    print(f"\n[Step 4/4] Generating publication-quality figures...")
    generate_all_plots(summary_df, output_dir="results/figures")

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Raw data:     {raw_csv}")
    print(f" Summary data: {summary_csv}")
    print(f" Figures:      results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Hopfield Experiment Pipeline")
    parser.add_argument("--replicates", type=int, default=30, help="Number of independent replicates (default: 30)")
    parser.add_argument("--cues", type=int, default=3, help="Cues tested per replicate network (default: 3)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest test suite")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Config file path")
    args = parser.parse_args()

    run_pipeline(
        replicates=args.replicates,
        cues=args.cues,
        skip_tests=args.skip_tests,
        config=args.config,
    )
