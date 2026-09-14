"""
Batch training for exploration experiments.

Trains each algorithm with all its exploration variants.

Usage:
    python scripts/run_exploration_experiments.py
    python scripts/run_exploration_experiments.py --algos ppo sac --seeds 0 1
    python scripts/run_exploration_experiments.py --timesteps 500000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Run exploration experiments for all algorithms"
    )
    parser.add_argument(
        "--algos", nargs="+", default=["ppo", "ddpg", "sac"],
        help="Algorithms to run."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds. Default: from config."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps."
    )
    args = parser.parse_args()

    override = {}
    if args.timesteps is not None:
        override["total_timesteps"] = args.timesteps

    for algo in args.algos:
        config = load_config(algo)
        seeds = args.seeds or config["seeds"]
        variants = config.get("exploration_variants", {})

        if not variants:
            print(f"No exploration variants defined for {algo.upper()}, skipping.")
            continue

        for variant_name in variants:
            for seed in seeds:
                print(f"\n{'#'*60}")
                print(f"# Exploration: {algo.upper()} | {variant_name} | seed={seed}")
                print(f"{'#'*60}")
                train(
                    algo=algo,
                    seed=seed,
                    exploration_variant=variant_name,
                    config_override=override if override else None,
                )

    print(f"\n{'='*60}")
    print("All exploration experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
