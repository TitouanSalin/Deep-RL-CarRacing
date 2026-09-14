"""
Master script: runs the FULL experimental pipeline end-to-end.

This script sequentially:
1. Trains PPO, DDPG, SAC across all seeds (default config)
2. Evaluates all trained models
3. Runs generalization evaluation (train vs test tracks)
4. Trains exploration variants for each algorithm
5. Evaluates exploration variants
6. Generates all plots

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --timesteps 100000 --seeds 0 1 2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import train
from src.evaluate import evaluate_agent, evaluate_generalization


def main():
    parser = argparse.ArgumentParser(description="Run full experimental pipeline")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override training timesteps (default: from config)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to use (default: from config)")
    parser.add_argument("--algos", nargs="+", default=["ppo", "ddpg", "sac"],
                        help="Algorithms to run")
    parser.add_argument("--skip-exploration", action="store_true",
                        help="Skip exploration experiments")
    parser.add_argument("--skip-generalization", action="store_true",
                        help="Skip generalization evaluation")
    args = parser.parse_args()

    config = load_config("ppo")
    seeds = args.seeds or config["seeds"]
    override = {}
    if args.timesteps is not None:
        override["total_timesteps"] = args.timesteps

    t0 = time.time()

    # =====================================================================
    # PHASE 1: Main training (all algos × all seeds)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING (default hyperparameters)")
    print("=" * 70)
    total_train = len(args.algos) * len(seeds)
    i = 0
    for algo in args.algos:
        for seed in seeds:
            i += 1
            print(f"\n--- [{i}/{total_train}] {algo.upper()} seed={seed} ---")
            try:
                train(algo=algo, seed=seed,
                      config_override=override if override else None)
            except Exception as e:
                print(f"ERROR training {algo} seed={seed}: {e}")

    # =====================================================================
    # PHASE 2: Evaluation
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: EVALUATION")
    print("=" * 70)
    for algo in args.algos:
        for seed in seeds:
            try:
                evaluate_agent(algo=algo, seed=seed)
            except FileNotFoundError as e:
                print(f"Skipping eval {algo} seed={seed}: {e}")

    # =====================================================================
    # PHASE 3: Generalization
    # =====================================================================
    if not args.skip_generalization:
        print("\n" + "=" * 70)
        print("PHASE 3: GENERALIZATION EVALUATION")
        print("=" * 70)
        for algo in args.algos:
            for seed in seeds:
                try:
                    evaluate_generalization(algo=algo, seed=seed)
                except FileNotFoundError as e:
                    print(f"Skipping generalization {algo} seed={seed}: {e}")

    # =====================================================================
    # PHASE 4: Exploration experiments
    # =====================================================================
    if not args.skip_exploration:
        print("\n" + "=" * 70)
        print("PHASE 4: EXPLORATION EXPERIMENTS")
        print("=" * 70)
        for algo in args.algos:
            algo_config = load_config(algo)
            variants = algo_config.get("exploration_variants", {})
            for variant_name in variants:
                for seed in seeds:
                    print(f"\n--- {algo.upper()} / {variant_name} / seed={seed} ---")
                    try:
                        train(algo=algo, seed=seed,
                              exploration_variant=variant_name,
                              config_override=override if override else None)
                    except Exception as e:
                        print(f"ERROR: {e}")

        # Evaluate exploration variants
        print("\nEvaluating exploration variants...")
        for algo in args.algos:
            algo_config = load_config(algo)
            variants = algo_config.get("exploration_variants", {})
            for variant_name in variants:
                for seed in seeds:
                    try:
                        evaluate_agent(algo=algo, seed=seed,
                                       exploration_variant=variant_name)
                    except FileNotFoundError as e:
                        print(f"Skipping: {e}")

    # =====================================================================
    # PHASE 5: Generate plots
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: GENERATING PLOTS")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")

    from src.analysis import (
        load_training_curves, aggregate_curves, compute_time_to_threshold,
        load_all_eval_results, load_all_generalization_results,
    )
    from src.plots import (
        plot_learning_curves, plot_time_to_threshold,
        plot_seed_stability_boxplot, plot_seed_curves,
        plot_generalization, plot_generalization_per_track,
        plot_exploration_effect, plot_exploration_comparison_bar,
    )
    import pandas as pd

    # Learning curves
    try:
        all_curves = []
        for algo in args.algos:
            all_curves.append(load_training_curves(algo))
        curves_df = pd.concat(all_curves, ignore_index=True)
        agg_df = aggregate_curves(curves_df)
        plot_learning_curves(curves_df, agg_df)
        ttt_df = compute_time_to_threshold(curves_df)
        plot_time_to_threshold(ttt_df)
        plot_seed_curves(curves_df)
        print("  -> Learning curves + seed curves saved")
    except Exception as e:
        print(f"  Skipping learning curves: {e}")

    # Stability
    try:
        eval_df = load_all_eval_results()
        default_eval = eval_df[eval_df["variant"] == "default"]
        if len(default_eval) > 0:
            plot_seed_stability_boxplot(default_eval)
            print("  -> Stability boxplot saved")
    except Exception as e:
        print(f"  Skipping stability: {e}")

    # Generalization
    try:
        gen_df = load_all_generalization_results()
        plot_generalization(gen_df)
        plot_generalization_per_track(gen_df)
        print("  -> Generalization plots saved")
    except Exception as e:
        print(f"  Skipping generalization plots: {e}")

    # Exploration
    for algo in args.algos:
        try:
            algo_config = load_config(algo)
            variants = algo_config.get("exploration_variants", {})
            exploration_curves = {}
            for v in variants:
                c = load_training_curves(algo, exploration_variant=v)
                exploration_curves[v] = aggregate_curves(c)
            if exploration_curves:
                plot_exploration_effect(exploration_curves, algo)
                print(f"  -> exploration_{algo}.png saved")
        except Exception as e:
            print(f"  Skipping exploration {algo}: {e}")

    try:
        results_dir = Path(config["results_dir"]) / "evaluation"
        dfs = [pd.read_csv(p) for p in sorted(results_dir.glob("*_eval.csv"))]
        if dfs:
            all_eval = pd.concat(dfs, ignore_index=True)
            variant_eval = all_eval[all_eval["variant"] != "default"]
            if len(variant_eval) > 0:
                plot_exploration_comparison_bar(variant_eval)
                print("  -> exploration_comparison.png saved")
    except Exception as e:
        print(f"  Skipping exploration bar: {e}")

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE! Total time: {hours}h {minutes}m")
    print(f"Results: results/")
    print(f"Figures: results/figures/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
