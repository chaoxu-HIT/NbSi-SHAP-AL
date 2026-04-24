#!/usr/bin/env python3
"""
main.py – End-to-end pipeline for dynamic SHAP-guided active learning
         on Nb-Si alloy fracture toughness data.

Usage
-----
    python main.py [--data PATH] [--n-initial N] [--n-iter N]
                   [--alpha A] [--seed S] [--results-dir DIR]

Defaults
--------
    --data        data/nbsi_data.csv
    --n-initial   15
    --n-iter      50
    --alpha       0.5  (equal weight to uncertainty and SHAP exploration)
    --seed        42
    --results-dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root is on sys.path when run directly.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.active_learning import ActiveLearner
from src.model import NbSiModel
from src.shap_analysis import SHAPAnalyser
from src.utils import (
    RESULTS_DIR,
    load_data,
    plot_final_shap_bar,
    plot_kic_discovery,
    plot_learning_curve,
    plot_shap_importance_heatmap,
    split_initial_pool,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic SHAP-guided active learning for Nb-Si alloys."
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "nbsi_data.csv"),
        help="Path to the Nb-Si CSV dataset (default: data/nbsi_data.csv).",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=15,
        dest="n_initial",
        help="Number of initially labelled samples (default: 15).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        dest="n_iter",
        help="Number of active-learning query iterations (default: 50).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Acquisition weight: alpha*uncertainty + (1-alpha)*SHAP (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initial pool selection (default: 42).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        dest="results_dir",
        help=f"Directory for output figures (default: {RESULTS_DIR}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    print(f"\n{'='*60}")
    print("Dynamic SHAP-Guided Active Learning – Nb-Si Alloys")
    print(f"{'='*60}")
    print(f"\n[1/5] Loading data from {args.data} …")
    X, y = load_data(args.data)
    print(f"      Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"      Features: {list(X.columns)}")
    print(f"      KIC range: [{y.min():.2f}, {y.max():.2f}] MPa·m^0.5")

    # ---- 2. Initial train/eval of full-data model ----
    print("\n[2/5] Cross-validating full-data baseline model …")
    baseline = NbSiModel(random_state=args.seed)
    cv_results = baseline.cross_validate(X, y, cv=5)
    print(
        f"      Baseline (5-fold CV) – "
        f"R²={cv_results['r2_mean']:.3f}±{cv_results['r2_std']:.3f}  "
        f"RMSE={cv_results['rmse_mean']:.3f}±{cv_results['rmse_std']:.3f}"
    )

    # ---- 3. Active learning ----
    print(f"\n[3/5] Running active learning …")
    print(f"      Initial labelled pool : {args.n_initial} samples")
    print(f"      Iterations            : {args.n_iter}")
    print(f"      Alpha                 : {args.alpha}")

    initial_idx = split_initial_pool(len(X), args.n_initial, random_state=args.seed)
    initial_best = float(y.iloc[initial_idx].max())
    print(f"      Best KIC in initial pool: {initial_best:.2f} MPa·m^0.5")

    learner = ActiveLearner(
        alpha=args.alpha,
        model_kwargs={"random_state": args.seed},
    )
    result = learner.run(
        X, y,
        initial_indices=initial_idx,
        n_iterations=args.n_iter,
        verbose=True,
    )

    # ---- 4. Summary ----
    print(f"\n[4/5] Summary …")
    df_history = result.to_dataframe()
    final = df_history.iloc[-1]
    print(f"      Final labelled pool  : {int(final['n_labelled'])} samples")
    print(f"      Final RMSE           : {final['rmse']:.3f} MPa·m^0.5")
    print(f"      Final R²             : {final['r2']:.3f}")
    print(f"      Best KIC discovered  : {final['best_kic']:.2f} MPa·m^0.5")
    print(
        f"      KIC improvement      : "
        f"{final['best_kic'] - initial_best:+.2f} MPa·m^0.5 "
        f"({(final['best_kic']/initial_best - 1)*100:+.1f} %)"
    )

    # ---- 5. Plots ----
    print(f"\n[5/5] Generating figures in {results_dir} …")
    plot_learning_curve(result, save_path=results_dir / "learning_curve.png")
    plot_kic_discovery(result, save_path=results_dir / "kic_discovery.png")
    plot_shap_importance_heatmap(result, save_path=results_dir / "shap_heatmap.png")
    plot_final_shap_bar(result, save_path=results_dir / "shap_bar.png")

    print(f"\nDone.  Results written to {results_dir}/\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args)
