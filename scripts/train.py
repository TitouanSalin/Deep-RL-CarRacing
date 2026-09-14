"""
CLI script to train a single agent.

Usage:
    python scripts/train.py --algo ppo --seed 0
    python scripts/train.py --algo sac --seed 1 --variant high_entropy
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent on CarRacing-v2"
    )
    parser.add_argument(
        "--algo", type=str, required=True,
        choices=["ppo", "ddpg", "sac"],
        help="Algorithm to train."
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Random seed."
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Exploration variant name (e.g. 'low_entropy')."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps."
    )
    args = parser.parse_args()

    override = {}
    if args.timesteps is not None:
        override["total_timesteps"] = args.timesteps

    train(
        algo=args.algo,
        seed=args.seed,
        exploration_variant=args.variant,
        config_override=override if override else None,
    )


if __name__ == "__main__":
    main()
