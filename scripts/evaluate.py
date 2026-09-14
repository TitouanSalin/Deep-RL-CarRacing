"""
CLI script to evaluate a trained agent.

Usage:
    python scripts/evaluate.py --algo ppo --seed 0
    python scripts/evaluate.py --algo sac --seed 1 --variant high_entropy
    python scripts/evaluate.py --algo ddpg --seed 0 --generalization
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_agent, evaluate_generalization


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL agent on CarRacing-v2"
    )
    parser.add_argument(
        "--algo", type=str, required=True,
        choices=["ppo", "ddpg", "sac"],
        help="Algorithm to evaluate."
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Training seed of the model to evaluate."
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Exploration variant name."
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--generalization", action="store_true",
        help="Run generalization evaluation (train vs test tracks)."
    )
    args = parser.parse_args()

    if args.generalization:
        evaluate_generalization(
            algo=args.algo,
            seed=args.seed,
            exploration_variant=args.variant,
        )
    else:
        evaluate_agent(
            algo=args.algo,
            seed=args.seed,
            exploration_variant=args.variant,
            n_episodes=args.n_episodes,
        )


if __name__ == "__main__":
    main()
