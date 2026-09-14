"""
Evaluation module for trained RL agents.

Provides functions to:
- Evaluate a trained model on N episodes
- Evaluate generalization across different track seeds
- Save results to CSV
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import VecTransposeImage
from tqdm import tqdm

from src.env_factory import make_vec_env
from src.config import load_config


ALGO_MAP = {
    "ppo": PPO,
    "ddpg": DDPG,
    "sac": SAC,
}


def load_model(algo: str, model_path: str | Path):
    """Load a trained SB3 model from disk."""
    AlgoClass = ALGO_MAP[algo.lower()]
    return AlgoClass.load(str(model_path))


def evaluate_model(
    model,
    env: VecTransposeImage,
    n_episodes: int = 50,
) -> dict[str, Any]:
    """
    Evaluate a model for n_episodes and return statistics.

    Parameters
    ----------
    model : BaseAlgorithm
        Trained SB3 model.
    env : VecTransposeImage
        Evaluation environment (vectorized).
    n_episodes : int
        Number of episodes to run.

    Returns
    -------
    dict
        Dictionary with keys: rewards, lengths, mean_reward, std_reward,
        mean_length, std_length.
    """
    rewards = []
    lengths = []

    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            done = dones[0]

        rewards.append(ep_reward)
        lengths.append(ep_length)

    return {
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }


def evaluate_agent(
    algo: str,
    seed: int,
    exploration_variant: str | None = None,
    n_episodes: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate a trained agent and save results to CSV.

    Parameters
    ----------
    algo : str
        Algorithm name.
    seed : int
        Training seed.
    exploration_variant : str or None
        Exploration variant name.
    n_episodes : int or None
        Number of evaluation episodes. If None, uses config default.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-episode results.
    """
    config = load_config(algo.lower(), exploration_variant)
    if n_episodes is None:
        n_episodes = config["n_final_eval_episodes"]

    variant_tag = f"_{exploration_variant}" if exploration_variant else ""
    exp_name = f"{algo.lower()}{variant_tag}_seed{seed}"
    model_path = Path(config["model_dir"]) / exp_name / "final_model"

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(algo, model_path)

    env = make_vec_env(
        env_id=config["env_id"],
        seed=seed + 2000,  # Different seed for final evaluation
        frame_stack=config["frame_stack"],
    )

    results = evaluate_model(model, env, n_episodes)
    env.close()

    # Build per-episode DataFrame
    df = pd.DataFrame({
        "algo": algo.lower(),
        "seed": seed,
        "variant": exploration_variant or "default",
        "episode": range(n_episodes),
        "reward": results["rewards"],
        "length": results["lengths"],
    })

    # Save
    results_dir = Path(config["results_dir"]) / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{exp_name}_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    return df


def evaluate_generalization(
    algo: str,
    seed: int,
    exploration_variant: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate generalization by testing on seen and unseen track seeds.

    The agent is evaluated on:
    - train_track_seeds: tracks seen during training (same random seeds)
    - test_track_seeds:  tracks never seen during training

    Note: In CarRacing-v2, the track layout is determined by the seed
    passed to env.reset(seed=...). By using specific seeds, we can
    control which tracks the agent encounters.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-track-seed results, including a 'split' column
        ('train' or 'test') for easy comparison.
    """
    config = load_config(algo.lower(), exploration_variant)
    variant_tag = f"_{exploration_variant}" if exploration_variant else ""
    exp_name = f"{algo.lower()}{variant_tag}_seed{seed}"
    model_path = Path(config["model_dir"]) / exp_name / "final_model"

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(algo, model_path)
    n_ep = config["n_generalization_episodes_per_seed"]
    rows = []

    for split, track_seeds in [
        ("train", config["train_track_seeds"]),
        ("test", config["test_track_seeds"]),
    ]:
        for track_seed in tqdm(track_seeds, desc=f"Generalization ({split})"):
            env = make_vec_env(
                env_id=config["env_id"],
                seed=track_seed,
                frame_stack=config["frame_stack"],
            )
            results = evaluate_model(model, env, n_ep)
            env.close()

            for i in range(n_ep):
                rows.append({
                    "algo": algo.lower(),
                    "seed": seed,
                    "variant": exploration_variant or "default",
                    "split": split,
                    "track_seed": track_seed,
                    "episode": i,
                    "reward": results["rewards"][i],
                    "length": results["lengths"][i],
                })

    df = pd.DataFrame(rows)

    # Save
    results_dir = Path(config["results_dir"]) / "generalization"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{exp_name}_generalization.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generalization results saved to {csv_path}")

    # Summary
    summary = df.groupby("split")["reward"].agg(["mean", "std"])
    print(f"\n{summary}")

    return df
