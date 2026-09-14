"""
Configuration loader for the RL comparison project.

Loads and merges YAML configuration files (common + algorithm-specific).
"""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(algo: str, exploration_variant: str | None = None) -> dict[str, Any]:
    """
    Load the merged configuration for a given algorithm.

    Merges common.yaml with the algorithm-specific YAML file.
    If an exploration_variant is specified, override the relevant keys.

    Parameters
    ----------
    algo : str
        Algorithm name: "ppo", "ddpg", or "sac".
    exploration_variant : str or None
        Name of the exploration variant (e.g. "low_entropy").
        If None, use default hyperparameters.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    common = load_yaml(CONFIGS_DIR / "common.yaml")
    algo_cfg = load_yaml(CONFIGS_DIR / f"{algo.lower()}.yaml")

    # Merge: algo-specific overrides common
    config = {**common, **algo_cfg}

    # Apply exploration variant overrides
    if exploration_variant is not None:
        variants = config.get("exploration_variants", {})
        if exploration_variant not in variants:
            raise ValueError(
                f"Unknown exploration variant '{exploration_variant}' "
                f"for {algo}. Available: {list(variants.keys())}"
            )
        config.update(variants[exploration_variant])

    return config


def get_algo_hyperparams(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract only the hyperparameters relevant to the SB3 algorithm constructor.

    Filters out project-level keys (paths, seeds, etc.) and returns
    only the keys that SB3 accepts.
    """
    # Keys that are project-level, not SB3 constructor args
    excluded_keys = {
        "algo", "total_timesteps", "eval_freq", "n_eval_episodes",
        "checkpoint_freq", "seeds", "n_final_eval_episodes",
        "train_track_seeds", "test_track_seeds",
        "n_generalization_episodes_per_seed", "reward_threshold",
        "model_dir", "results_dir", "log_dir", "env_id", "frame_stack",
        "exploration_variants", "action_noise_type", "action_noise_sigma",
    }
    return {k: v for k, v in config.items() if k not in excluded_keys}
