"""
Visualization module for the RL comparison project.

Generates publication-quality plots for:
- Learning curves (Q1: learning speed)
- Seed stability (Q2: boxplots, variance)
- Generalization (Q3: train vs test tracks)
- Exploration effects (Q4: variant comparison)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

from src.config import load_config

# Use a clean, academic style
matplotlib.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

ALGO_COLORS = {"ppo": "#1f77b4", "ddpg": "#ff7f0e", "sac": "#2ca02c"}
ALGO_LABELS = {"ppo": "PPO", "ddpg": "DDPG", "sac": "SAC"}


def _get_figures_dir() -> Path:
    config = load_config("ppo")
    fig_dir = Path(config["results_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# =========================================================================
# Q1: Learning Speed
# =========================================================================

def plot_learning_curves(
    curves_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    save: bool = True,
    smooth_window: int = 5,
) -> plt.Figure:
    """
    Plot comparative learning curves (mean ± std across seeds).

    Parameters
    ----------
    curves_df : pd.DataFrame
        Raw per-seed training curves (from load_training_curves).
    agg_df : pd.DataFrame
        Aggregated curves (from aggregate_curves).
    save : bool
        Whether to save the figure.
    smooth_window : int
        Window size for smoothing.
    """
    fig, ax = plt.subplots()

    for algo in agg_df["algo"].unique():
        data = agg_df[agg_df["algo"] == algo].sort_values("timesteps")
        ts = data["timesteps"].values
        mean = data["reward_mean"].values
        std = data["reward_std"].values

        # Smooth
        if len(mean) > smooth_window:
            n_valid = len(mean) - smooth_window + 1
            ts_s = ts[:n_valid] if smooth_window > 1 else ts
            ts_s = ts[smooth_window - 1:]
            mean_s = _smooth(mean, smooth_window)
            std_s = _smooth(std, smooth_window)
        else:
            ts_s, mean_s, std_s = ts, mean, std

        color = ALGO_COLORS.get(algo, "gray")
        label = ALGO_LABELS.get(algo, algo.upper())
        ax.plot(ts_s, mean_s, color=color, label=label, linewidth=2)
        ax.fill_between(ts_s, mean_s - std_s, mean_s + std_s,
                         alpha=0.15, color=color)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("Learning Curves — PPO vs DDPG vs SAC")
    ax.legend()

    if save:
        fig.savefig(_get_figures_dir() / "learning_curves.png",
                    bbox_inches="tight")
    return fig


def plot_time_to_threshold(ttt_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot bar chart of timesteps to reach reward threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    summary = ttt_df.groupby("algo")["timesteps_to_threshold"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean")

    colors = [ALGO_COLORS.get(a, "gray") for a in summary["algo"]]
    labels = [ALGO_LABELS.get(a, a.upper()) for a in summary["algo"]]
    bars = ax.bar(labels, summary["mean"], yerr=summary["std"],
                  color=colors, capsize=5, edgecolor="black", linewidth=0.5)

    threshold = ttt_df["threshold"].iloc[0]
    ax.set_ylabel("Timesteps")
    ax.set_title(f"Timesteps to Reach Reward ≥ {threshold}")

    if save:
        fig.savefig(_get_figures_dir() / "time_to_threshold.png",
                    bbox_inches="tight")
    return fig


# =========================================================================
# Q2: Stability across seeds
# =========================================================================

def plot_seed_stability_boxplot(
    eval_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Boxplot of final evaluation rewards per algorithm, across seeds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    algos = sorted(eval_df["algo"].unique())
    data = [eval_df[eval_df["algo"] == a]["reward"].values for a in algos]
    labels = [ALGO_LABELS.get(a, a.upper()) for a in algos]
    colors = [ALGO_COLORS.get(a, "gray") for a in algos]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Performance Stability Across Seeds")

    if save:
        fig.savefig(_get_figures_dir() / "seed_stability_boxplot.png",
                    bbox_inches="tight")
    return fig


def plot_seed_curves(curves_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Plot individual seed learning curves per algorithm (subplots).
    """
    algos = sorted(curves_df["algo"].unique())
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        color = ALGO_COLORS.get(algo, "gray")
        for seed in sorted(curves_df[curves_df["algo"] == algo]["seed"].unique()):
            data = curves_df[(curves_df["algo"] == algo) & (curves_df["seed"] == seed)]
            data = data.sort_values("timesteps")
            ax.plot(data["timesteps"], data["mean_reward"],
                    alpha=0.5, label=f"Seed {seed}", linewidth=1)

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Reward")
        ax.set_title(ALGO_LABELS.get(algo, algo.upper()))
        ax.legend(fontsize=8)

    fig.suptitle("Learning Curves per Seed", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(_get_figures_dir() / "seed_curves.png",
                    bbox_inches="tight")
    return fig


# =========================================================================
# Q3: Generalization
# =========================================================================

def plot_generalization(gen_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Bar chart comparing train vs test track performance by algorithm.
    """
    summary = gen_df.groupby(["algo", "split"])["reward"].agg(["mean", "std"]).reset_index()

    algos = sorted(summary["algo"].unique())
    x = np.arange(len(algos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, split in enumerate(["train", "test"]):
        data = summary[summary["split"] == split].set_index("algo").loc[algos]
        offset = (i - 0.5) * width
        ax.bar(x + offset, data["mean"], width, yerr=data["std"],
               label=f"{'Seen tracks' if split == 'train' else 'Unseen tracks'}",
               capsize=4, edgecolor="black", linewidth=0.5,
               alpha=0.8 if split == "train" else 0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS.get(a, a.upper()) for a in algos])
    ax.set_ylabel("Mean Reward")
    ax.set_title("Generalization: Seen vs Unseen Tracks")
    ax.legend()

    if save:
        fig.savefig(_get_figures_dir() / "generalization.png",
                    bbox_inches="tight")
    return fig


def plot_generalization_per_track(
    gen_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Scatter plot of per-track-seed performance, colored by split."""
    algos = sorted(gen_df["algo"].unique())
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        data = gen_df[gen_df["algo"] == algo]
        per_track = data.groupby(["split", "track_seed"])["reward"].mean().reset_index()

        for split, marker, alpha in [("train", "o", 0.8), ("test", "x", 0.6)]:
            subset = per_track[per_track["split"] == split]
            label = "Seen tracks" if split == "train" else "Unseen tracks"
            ax.scatter(range(len(subset)), subset["reward"],
                       marker=marker, alpha=alpha, label=label, s=60)

        ax.set_xlabel("Track index")
        ax.set_ylabel("Mean Reward")
        ax.set_title(ALGO_LABELS.get(algo, algo.upper()))
        ax.legend(fontsize=8)

    fig.suptitle("Per-Track Generalization", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(_get_figures_dir() / "generalization_per_track.png",
                    bbox_inches="tight")
    return fig


# =========================================================================
# Q4: Exploration
# =========================================================================

def plot_exploration_effect(
    exploration_curves: dict[str, pd.DataFrame],
    algo: str,
    save: bool = True,
) -> plt.Figure:
    """
    Plot learning curves for different exploration variants of one algorithm.

    Parameters
    ----------
    exploration_curves : dict
        Maps variant name -> aggregated DataFrame.
    algo : str
        Algorithm name (for title/filename).
    """
    fig, ax = plt.subplots()

    cmap = plt.cm.Set2
    for i, (variant, agg_df) in enumerate(exploration_curves.items()):
        data = agg_df.sort_values("timesteps")
        ts = data["timesteps"].values
        mean = data["reward_mean"].values
        std = data["reward_std"].values

        color = cmap(i)
        ax.plot(ts, mean, color=color, label=variant, linewidth=2)
        ax.fill_between(ts, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title(f"Exploration Effect — {ALGO_LABELS.get(algo.lower(), algo.upper())}")
    ax.legend()

    if save:
        fig.savefig(_get_figures_dir() / f"exploration_{algo.lower()}.png",
                    bbox_inches="tight")
    return fig


def plot_exploration_comparison_bar(
    eval_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing final performance across exploration variants.
    """
    summary = eval_df.groupby(["algo", "variant"])["reward"].agg(["mean", "std"]).reset_index()

    algos = sorted(summary["algo"].unique())
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        data = summary[summary["algo"] == algo]
        x = np.arange(len(data))
        colors = plt.cm.Set2(np.linspace(0, 0.6, len(data)))
        ax.bar(x, data["mean"], yerr=data["std"], color=colors,
               capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(data["variant"], rotation=30, ha="right")
        ax.set_ylabel("Mean Reward")
        ax.set_title(ALGO_LABELS.get(algo, algo.upper()))

    fig.suptitle("Exploration Variants — Final Performance", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        fig.savefig(_get_figures_dir() / "exploration_comparison.png",
                    bbox_inches="tight")
    return fig
