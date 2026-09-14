"""
Analysis module for extracting and aggregating experimental results.

Provides functions to:
- Load training logs (evaluation results from SB3 callbacks)
- Aggregate results across seeds
- Compute learning speed metrics
- Prepare data for visualization
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, PROJECT_ROOT


def load_eval_log(log_dir: Path) -> pd.DataFrame:
    """
    Load the evaluations.npz file saved by SB3's EvalCallback.

    Returns a DataFrame with columns: timesteps, mean_reward, std_reward.
    """
    npz_path = log_dir / "evaluations.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No evaluation log found at {npz_path}")

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    # results shape: (n_evals, n_episodes) — take mean/std across episodes
    results = data["results"]
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    return pd.DataFrame({
        "timesteps": timesteps,
        "mean_reward": mean_rewards,
        "std_reward": std_rewards,
    })


def load_training_curves(
    algo: str,
    seeds: list[int] | None = None,
    exploration_variant: str | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate training curves across multiple seeds.

    Parameters
    ----------
    algo : str
        Algorithm name.
    seeds : list of int or None
        Seeds to load. If None, uses seeds from config.
    exploration_variant : str or None
        Exploration variant name.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timesteps, mean_reward, std_reward, seed, algo.
    """
    config = load_config(algo.lower(), exploration_variant)
    if seeds is None:
        seeds = config["seeds"]

    variant_tag = f"_{exploration_variant}" if exploration_variant else ""
    dfs = []

    for seed in seeds:
        exp_name = f"{algo.lower()}{variant_tag}_seed{seed}"
        log_dir = Path(config["log_dir"]) / exp_name

        try:
            df = load_eval_log(log_dir)
            df["seed"] = seed
            df["algo"] = algo.lower()
            df["variant"] = exploration_variant or "default"
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: no log found for {exp_name}, skipping.")

    if not dfs:
        raise FileNotFoundError(f"No training logs found for {algo}")

    return pd.concat(dfs, ignore_index=True)


def aggregate_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate training curves across seeds.

    Groups by (algo, variant, timesteps) and computes mean ± std
    of the mean_reward.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns: timesteps, reward_mean,
        reward_std, reward_min, reward_max, algo, variant.
    """
    grouped = df.groupby(["algo", "variant", "timesteps"])["mean_reward"]
    agg = grouped.agg(["mean", "std", "min", "max"]).reset_index()
    agg.columns = ["algo", "variant", "timesteps", "reward_mean", "reward_std",
                    "reward_min", "reward_max"]
    return agg


def compute_time_to_threshold(
    df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute the number of timesteps needed to reach a reward threshold.

    For each (algo, variant, seed), finds the first timestep where
    mean_reward >= threshold. If never reached, returns NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Training curves (output of load_training_curves).
    threshold : float or None
        Reward threshold. If None, uses config default.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: algo, variant, seed, timesteps_to_threshold.
    """
    if threshold is None:
        config = load_config("ppo")  # Just to read threshold
        threshold = config["reward_threshold"]

    rows = []
    for (algo, variant, seed), group in df.groupby(["algo", "variant", "seed"]):
        above = group[group["mean_reward"] >= threshold]
        ts = above["timesteps"].min() if len(above) > 0 else np.nan
        rows.append({
            "algo": algo,
            "variant": variant,
            "seed": seed,
            "timesteps_to_threshold": ts,
            "threshold": threshold,
        })

    return pd.DataFrame(rows)


def load_all_eval_results(algos: list[str] | None = None) -> pd.DataFrame:
    """Load all evaluation CSV files and concatenate them."""
    if algos is None:
        algos = ["ppo", "ddpg", "sac"]

    config = load_config("ppo")
    results_dir = Path(config["results_dir"]) / "evaluation"
    dfs = []

    for csv_path in sorted(results_dir.glob("*_eval.csv")):
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No evaluation results found in {results_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_all_generalization_results() -> pd.DataFrame:
    """Load all generalization CSV files and concatenate them."""
    config = load_config("ppo")
    results_dir = Path(config["results_dir"]) / "generalization"
    dfs = []

    for csv_path in sorted(results_dir.glob("*_generalization.csv")):
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No generalization results found in {results_dir}")

    return pd.concat(dfs, ignore_index=True)


def build_summary_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table with mean ± std reward per algorithm/variant.

    Parameters
    ----------
    eval_df : pd.DataFrame
        Concatenated evaluation results.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    summary = eval_df.groupby(["algo", "variant", "seed"])["reward"].mean().reset_index()
    summary = summary.groupby(["algo", "variant"]).agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        n_seeds=("reward", "count"),
    ).reset_index()
    summary["reward_str"] = summary.apply(
        lambda r: f"{r['mean_reward']:.1f} ± {r['std_reward']:.1f}", axis=1
    )
    return summary
