"""
Training pipeline for PPO, DDPG, and SAC on CarRacing-v2.

Provides a unified `train()` function that handles:
- Environment creation
- Algorithm instantiation (with action noise for DDPG)
- Evaluation callbacks
- Checkpoint saving
- Metric logging
"""

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from src.env_factory import make_vec_env, make_eval_env
from src.config import load_config, get_algo_hyperparams


ALGO_MAP = {
    "ppo": PPO,
    "ddpg": DDPG,
    "sac": SAC,
}


def _build_action_noise(config: dict[str, Any], n_actions: int):
    """Build action noise for DDPG from config."""
    noise_type = config.get("action_noise_type", "normal")
    sigma = config.get("action_noise_sigma", 0.2)

    if noise_type == "ornstein-uhlenbeck":
        return OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma * np.ones(n_actions),
        )
    # Default: Gaussian noise
    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=sigma * np.ones(n_actions),
    )


def train(
    algo: str,
    seed: int,
    exploration_variant: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> Path:
    """
    Train a single agent and return the path to the saved model.

    Parameters
    ----------
    algo : str
        Algorithm name: "ppo", "ddpg", or "sac".
    seed : int
        Random seed for reproducibility.
    exploration_variant : str or None
        Exploration variant name (e.g. "low_entropy"). None = default.
    config_override : dict or None
        Additional overrides applied on top of the config.

    Returns
    -------
    Path
        Path to the saved final model.
    """
    algo_lower = algo.lower()
    config = load_config(algo_lower, exploration_variant)
    if config_override:
        config.update(config_override)

    # Build experiment name
    variant_tag = f"_{exploration_variant}" if exploration_variant else ""
    exp_name = f"{algo_lower}{variant_tag}_seed{seed}"

    # Paths
    model_dir = Path(config["model_dir"]) / exp_name
    log_dir = Path(config["log_dir"]) / exp_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Environments
    env = make_vec_env(
        env_id=config["env_id"],
        seed=seed,
        frame_stack=config["frame_stack"],
    )
    eval_env = make_eval_env(
        env_id=config["env_id"],
        seed=seed + 1000,  # Different seed for evaluation
        frame_stack=config["frame_stack"],
    )

    # Extract SB3-compatible hyperparameters
    hyperparams = get_algo_hyperparams(config)

    # Build action noise for DDPG
    action_noise = None
    if algo_lower == "ddpg":
        n_actions = env.action_space.shape[0]
        action_noise = _build_action_noise(config, n_actions)
        hyperparams["action_noise"] = action_noise

    # Instantiate algorithm
    AlgoClass = ALGO_MAP[algo_lower]
    model = AlgoClass(
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(log_dir),
        **hyperparams,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=str(model_dir / "checkpoints"),
        name_prefix=exp_name,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Train
    print(f"\n{'='*60}")
    print(f"Training {algo.upper()} | seed={seed} | variant={exploration_variant or 'default'}")
    print(f"Timesteps: {config['total_timesteps']}")
    print(f"Model dir: {model_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"\nModel saved to {final_path}")

    env.close()
    eval_env.close()

    return final_path
