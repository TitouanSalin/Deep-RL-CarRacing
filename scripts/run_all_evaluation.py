"""
Batch evaluation script: evaluate all trained models.

Usage:
    python scripts/run_all_evaluation.py
    python scripts/run_all_evaluation.py --generalization
    python scripts/run_all_evaluation.py --algos ppo sac --seeds 0 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluate import evaluate_agent, evaluate_generalization


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained agents"
    )
    parser.add_argument(
        "--algos", nargs="+", default=["ppo", "ddpg", "sac"],
        help="Algorithms to evaluate."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to evaluate. Default: from config."
    )
    parser.add_argument(
        "--generalization", action="store_true",
        help="Run generalization evaluation."
    )
    args = parser.parse_args()

    seeds = args.seeds
    if seeds is None:
        config = load_config("ppo")
        seeds = config["seeds"]

    total = len(args.algos) * len(seeds)
    current = 0

    for algo in args.algos:
        for seed in seeds:
            current += 1
            print(f"\n{'#'*60}")
            print(f"# Evaluation {current}/{total}: {algo.upper()} seed={seed}")
            print(f"{'#'*60}")

            try:
                if args.generalization:
                    evaluate_generalization(algo=algo, seed=seed)
                else:
                    evaluate_agent(algo=algo, seed=seed)
            except FileNotFoundError as e:
                print(f"Skipping: {e}")

    print(f"\n{'='*60}")
    print("All evaluations completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
