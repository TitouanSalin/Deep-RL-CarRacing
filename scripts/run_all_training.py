"""
Batch training script: train all algorithms across all seeds.

Usage:
    python scripts/run_all_training.py
    python scripts/run_all_training.py --algos ppo sac --seeds 0 1 2
    python scripts/run_all_training.py --timesteps 500000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train all algorithms across multiple seeds"
    )
    parser.add_argument(
        "--algos", nargs="+", default=["ppo", "ddpg", "sac"],
        help="Algorithms to train."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to use. Default: from config."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps."
    )
    args = parser.parse_args()

    # Get seeds from config if not provided
    seeds = args.seeds
    if seeds is None:
        config = load_config("ppo")
        seeds = config["seeds"]

    override = {}
    if args.timesteps is not None:
        override["total_timesteps"] = args.timesteps

    total = len(args.algos) * len(seeds)
    current = 0

    for algo in args.algos:
        for seed in seeds:
            current += 1
            print(f"\n{'#'*60}")
            print(f"# Experiment {current}/{total}: {algo.upper()} seed={seed}")
            print(f"{'#'*60}")
            train(
                algo=algo,
                seed=seed,
                config_override=override if override else None,
            )

    print(f"\n{'='*60}")
    print(f"All {total} training runs completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
