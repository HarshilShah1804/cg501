"""
Data analysis and aggregation module.
Computes means, standard errors, and 95% confidence intervals across independent network replicates.
"""

import os
import sys
from typing import Optional

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw trial results to independent network/lesion configuration replicates first,
    then compute condition-level means, standard errors, and 95% confidence intervals.
    
    Unit of replication: (load, lesion_fraction, cue_noise, replicate_id).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw trial data with replicate_id, load, lesion_fraction, cue_noise, outcome, etc.
        
    Returns
    -------
    pd.DataFrame
        Summary dataframe containing means and 95% CIs for accuracy, similarity,
        and outcome category fractions.
    """
    # 1. Indicator columns for outcome categories
    df = df.copy()
    df["is_exact_target"] = (df["outcome"] == "exact_target").astype(float)
    df["is_other_stored"] = (df["outcome"] == "other_stored").astype(float)
    df["is_stable_spurious"] = (df["outcome"] == "stable_spurious").astype(float)
    df["is_non_converged"] = (df["outcome"] == "non_converged").astype(float)

    # 2. Replicate-level aggregation (average across cues tested within each independent network)
    rep_group_cols = ["N", "P", "load", "lesion_fraction", "cue_noise", "lesion_type", "replicate_id"]
    rep_level = df.groupby(rep_group_cols, as_index=False).agg({
        "exact_recall": "mean",
        "similarity": "mean",
        "is_exact_target": "mean",
        "is_other_stored": "mean",
        "is_stable_spurious": "mean",
        "is_non_converged": "mean",
        "sweeps_taken": "mean",
    })

    # 3. Condition-level aggregation across independent replicates
    cond_group_cols = ["N", "P", "load", "lesion_fraction", "cue_noise", "lesion_type"]

    def compute_stats(series: pd.Series, prefix: str) -> dict:
        n = len(series)
        mean_val = float(series.mean())
        if n > 1:
            std_val = float(series.std(ddof=1))
            sem_val = std_val / np.sqrt(n)
            # 95% t-interval
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_val = t_crit * sem_val
        else:
            std_val = 0.0
            sem_val = 0.0
            ci_val = 0.0

        return {
            f"{prefix}_mean": mean_val,
            f"{prefix}_std": std_val,
            f"{prefix}_sem": sem_val,
            f"{prefix}_ci95": ci_val,
            f"{prefix}_ci95_low": max(0.0, mean_val - ci_val),
            f"{prefix}_ci95_high": min(1.0, mean_val + ci_val),
        }

    records = []
    for cond_keys, group in rep_level.groupby(cond_group_cols):
        cond_dict = dict(zip(cond_group_cols, cond_keys))
        cond_dict["num_replicates"] = len(group)

        for col, prefix in [
            ("exact_recall", "accuracy"),
            ("similarity", "similarity"),
            ("is_exact_target", "exact_target"),
            ("is_other_stored", "other_stored"),
            ("is_stable_spurious", "stable_spurious"),
            ("is_non_converged", "non_converged"),
            ("sweeps_taken", "sweeps"),
        ]:
            cond_dict.update(compute_stats(group[col], prefix))

        records.append(cond_dict)

    summary_df = pd.DataFrame(records)
    return summary_df


def process_and_save_summary(raw_csv_path: str, output_csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw results CSV, compute aggregate summary, and save to processed directory.
    """
    df = pd.read_csv(raw_csv_path)
    summary = aggregate_results(df)

    if output_csv_path is None:
        base_name = os.path.basename(raw_csv_path).replace("raw_results", "summary_results")
        output_csv_path = os.path.join("results/processed", base_name)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    summary.to_csv(output_csv_path, index=False)
    print(f"Summary results saved to {output_csv_path}")
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_and_save_summary(sys.argv[1])
    else:
        print("Please provide path to raw CSV file.")
