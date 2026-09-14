"""
Generate all analysis plots from saved results.

Usage:
    python scripts/generate_plots.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.analysis import (
    load_training_curves,
    aggregate_curves,
    compute_time_to_threshold,
    load_all_eval_results,
    load_all_generalization_results,
)
from src.plots import (
    plot_learning_curves,
    plot_time_to_threshold,
    plot_seed_stability_boxplot,
    plot_seed_curves,
    plot_generalization,
    plot_generalization_per_track,
    plot_exploration_effect,
    plot_exploration_comparison_bar,
)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script usage


def main():
    algos = ["ppo", "ddpg", "sac"]
    config = load_config("ppo")

    # ------------------------------------------------------------------
    # Q1: Learning curves
    # ------------------------------------------------------------------
    print("Generating learning curve plots...")
    try:
        all_curves = []
        for algo in algos:
            curves = load_training_curves(algo)
            all_curves.append(curves)
        import pandas as pd
        curves_df = pd.concat(all_curves, ignore_index=True)
        agg_df = aggregate_curves(curves_df)

        plot_learning_curves(curves_df, agg_df)
        print("  -> learning_curves.png")

        ttt_df = compute_time_to_threshold(curves_df)
        plot_time_to_threshold(ttt_df)
        print("  -> time_to_threshold.png")
    except FileNotFoundError as e:
        print(f"  Skipping learning curves: {e}")

    # ------------------------------------------------------------------
    # Q2: Seed stability
    # ------------------------------------------------------------------
    print("\nGenerating stability plots...")
    try:
        eval_df = load_all_eval_results()
        plot_seed_stability_boxplot(eval_df)
        print("  -> seed_stability_boxplot.png")

        # Per-seed curves
        if 'curves_df' in dir():
            plot_seed_curves(curves_df)
            print("  -> seed_curves.png")
    except FileNotFoundError as e:
        print(f"  Skipping stability plots: {e}")

    # ------------------------------------------------------------------
    # Q3: Generalization
    # ------------------------------------------------------------------
    print("\nGenerating generalization plots...")
    try:
        gen_df = load_all_generalization_results()
        plot_generalization(gen_df)
        print("  -> generalization.png")
        plot_generalization_per_track(gen_df)
        print("  -> generalization_per_track.png")
    except FileNotFoundError as e:
        print(f"  Skipping generalization plots: {e}")

    # ------------------------------------------------------------------
    # Q4: Exploration
    # ------------------------------------------------------------------
    print("\nGenerating exploration plots...")
    for algo in algos:
        algo_config = load_config(algo)
        variants = algo_config.get("exploration_variants", {})
        if not variants:
            continue

        try:
            exploration_curves = {}
            for variant_name in variants:
                curves = load_training_curves(algo, exploration_variant=variant_name)
                agg = aggregate_curves(curves)
                exploration_curves[variant_name] = agg

            plot_exploration_effect(exploration_curves, algo)
            print(f"  -> exploration_{algo}.png")
        except FileNotFoundError as e:
            print(f"  Skipping exploration for {algo}: {e}")

    # Exploration comparison bar (from eval results of variants)
    try:
        import pandas as pd
        results_dir = Path(config["results_dir"]) / "evaluation"
        dfs = []
        for csv_path in sorted(results_dir.glob("*_eval.csv")):
            df = pd.read_csv(csv_path)
            dfs.append(df)
        if dfs:
            all_eval = pd.concat(dfs, ignore_index=True)
            variant_eval = all_eval[all_eval["variant"] != "default"]
            if len(variant_eval) > 0:
                plot_exploration_comparison_bar(variant_eval)
                print("  -> exploration_comparison.png")
    except Exception as e:
        print(f"  Skipping exploration comparison: {e}")

    print("\nAll plots generated! Check results/figures/")


if __name__ == "__main__":
    main()
