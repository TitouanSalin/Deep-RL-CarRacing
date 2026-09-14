"""
Watch a trained RL agent drive in CarRacing with on-screen rendering.

Usage examples:
    python scripts/watch_agent.py --algo ppo --seed 0
    python scripts/watch_agent.py --algo ppo --seed 0 --model best --episodes 3
    python scripts/watch_agent.py --algo ppo --seed 0 --track-seed 200 --vary-track
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.env_factory import make_vec_env
from src.evaluate import load_model


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained RL agent drive on CarRacing."
    )
    parser.add_argument(
        "--algo", type=str, required=True,
        choices=["ppo", "ddpg", "sac"],
        help="Algorithm of the trained model."
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Training seed used for the model name."
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Exploration variant name (if not default)."
    )
    parser.add_argument(
        "--model", type=str, default="final", choices=["final", "best"],
        help="Which checkpoint to watch: final or best."
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to render."
    )
    parser.add_argument(
        "--track-seed", type=int, default=200,
        help="Base track seed used for rendering."
    )
    parser.add_argument(
        "--vary-track", action="store_true",
        help="If set, increments the track seed at each episode."
    )
    args = parser.parse_args()

    config = load_config(args.algo.lower(), args.variant)
    variant_tag = f"_{args.variant}" if args.variant else ""
    exp_name = f"{args.algo.lower()}{variant_tag}_seed{args.seed}"

    model_dir = Path(config["model_dir"]) / exp_name
    if args.model == "best":
        model_path = model_dir / "best" / "best_model"
    else:
        model_path = model_dir / "final_model"

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = load_model(args.algo, model_path)

    print(f"Watching {args.algo.upper()} | model={args.model} | episodes={args.episodes}")
    print("Close the render window or press Ctrl+C in terminal to stop.")

    for ep in range(args.episodes):
        ep_seed = args.track_seed + ep if args.vary_track else args.track_seed
        env = make_vec_env(
            env_id=config["env_id"],
            seed=ep_seed,
            frame_stack=config["frame_stack"],
            render_mode="human",
        )

        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1
            done = bool(dones[0])

        env.close()
        print(f"Episode {ep + 1}/{args.episodes} | track_seed={ep_seed} | reward={ep_reward:.2f} | length={ep_len}")


if __name__ == "__main__":
    main()