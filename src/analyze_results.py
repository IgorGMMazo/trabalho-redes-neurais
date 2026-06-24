#!/usr/bin/env python3
"""
Tomato Leaf Dataset Scaling Law Analyzer.
Author: Data Scientist
Description: Analyzes training metrics from a CSV log, fits a logarithmic
             regression model (Scaling Law) on smaller subsets, predicts
             performance on subsequent scales, and plots actual vs. predicted curves.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

# Configure matplotlib for headless and interactive environments
import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[WARNING] Matplotlib is not installed. Visualization will be skipped.")
    plt = None


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fit scaling laws on training metrics CSV and predict performance."
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="results/training_results.csv",
        help="Path to the training results CSV file (default: training_results.csv).",
    )
    parser.add_argument(
        "--fit-points",
        type=int,
        default=3,
        help="Number of initial data points (subsets) to use for plotting the predicted curve (default: 3, e.g. 1%%, 2%%, 5%%).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="results/scaling_law_plot.png",
        help="Filename to save the generated plot (default: scaling_law_plot.png).",
    )
    return parser.parse_args()


def fit_logarithmic_curve(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fits a logarithmic curve y = a * ln(x) + b.
    Returns:
        a (float), b (float): coefficients of the fit.
    """
    # y = a * ln(x) + b is equivalent to linear regression of y on ln(x)
    log_x = np.log(x)
    a, b = np.polyfit(log_x, y, 1)
    return a, b


def predict_logarithmic(x_pred: float, a: float, b: float) -> float:
    """
    Predicts the value for x_pred using coefficients a and b.
    """
    return a * np.log(x_pred) + b


def load_and_aggregate_metrics(csv_path: Path) -> pd.DataFrame:
    """
    Loads the CSV file, filters for the final epoch of each run,
    extracts the numerical percentage, and averages metrics across runs.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)

    # 1. Get the last epoch of each run/subset combination
    # We group by subset and run_index, then take the row with the maximum epoch
    idx = df.groupby(["subset", "run_index"])["epoch"].idxmax()
    df_final = df.loc[idx].copy()

    # 2. Extract numerical percentage from subset name (e.g. "subset_5%" -> 5.0)
    df_final["percentage"] = df_final["subset"].str.extract(r"(\d+)").astype(float)

    # 3. Group by percentage and calculate the average metric values across all runs
    agg_df = df_final.groupby("percentage").agg({
        "train_loss": "mean",
        "test_accuracy": "mean",
        "test_precision": "mean"
    }).reset_index()

    # Sort by percentage to ensure curves plot correctly
    agg_df = agg_df.sort_values("percentage")

    return agg_df


def run_sequential_predictions(df: pd.DataFrame):
    """
    Iteratively fits curves on P[0..k-1] to predict P[k] for all k >= 2.
    Displays predictions and errors.
    """
    percentages = df["percentage"].values
    losses = df["train_loss"].values
    accuracies = df["test_accuracy"].values

    print("\n" + "=" * 80)
    print("SEQUENTIAL SCALING LAW PREDICTIONS")
    print("=" * 80)
    print(f"{'Target %':<10} | {'Metric':<10} | {'Actual':<10} | {'Predicted':<10} | {'Abs Error':<10} | {'Rel Error %':<10}")
    print("-" * 80)

    for k in range(2, len(percentages)):
        target_pct = percentages[k]
        
        # Fit on all previous subsets
        x_train = percentages[:k]
        
        # Fit & Predict Loss
        y_train_loss = losses[:k]
        a_loss, b_loss = fit_logarithmic_curve(x_train, y_train_loss)
        pred_loss = predict_logarithmic(target_pct, a_loss, b_loss)
        act_loss = losses[k]
        abs_err_loss = abs(act_loss - pred_loss)
        rel_err_loss = (abs_err_loss / act_loss) * 100 if act_loss != 0 else 0.0
        
        print(f"{target_pct:>8.1f}% | {'Loss':<10} | {act_loss:>10.4f} | {pred_loss:>10.4f} | {abs_err_loss:>10.4f} | {rel_err_loss:>9.2f}%")

        # Fit & Predict Accuracy
        y_train_acc = accuracies[:k]
        a_acc, b_acc = fit_logarithmic_curve(x_train, y_train_acc)
        pred_acc = predict_logarithmic(target_pct, a_acc, b_acc)
        # Accuracy cannot exceed 1.0 (100%)
        pred_acc_clipped = np.clip(pred_acc, 0.0, 1.0)
        act_acc = accuracies[k]
        abs_err_acc = abs(act_acc - pred_acc_clipped)
        rel_err_acc = (abs_err_acc / act_acc) * 100 if act_acc != 0 else 0.0
        
        print(f"{'':<10} | {'Accuracy':<10} | {act_acc:>10.4f} | {pred_acc_clipped:>10.4f} | {abs_err_acc:>10.4f} | {rel_err_acc:>9.2f}%")
        print("-" * 80)


def generate_plot(df: pd.DataFrame, fit_points: int, output_plot_path: Path):
    """
    Fits a logarithmic curve on the first `fit_points` subsets and plots
    them against all actual subsets.
    """
    if plt is None:
        return

    percentages = df["percentage"].values
    losses = df["train_loss"].values
    accuracies = df["test_accuracy"].values

    if len(percentages) < fit_points:
        print(f"[ERROR] Cannot fit on {fit_points} points when only {len(percentages)} points exist in the CSV.")
        return

    # 1. Fit curves on the first M points
    x_fit = percentages[:fit_points]
    y_fit_loss = losses[:fit_points]
    y_fit_acc = accuracies[:fit_points]

    a_loss, b_loss = fit_logarithmic_curve(x_fit, y_fit_loss)
    a_acc, b_acc = fit_logarithmic_curve(x_fit, y_fit_acc)

    # 2. Generate smooth curve for plotting
    x_curve = np.linspace(percentages[0], percentages[-1], 200)
    y_curve_loss = predict_logarithmic(x_curve, a_loss, b_loss)
    y_curve_acc = np.clip(predict_logarithmic(x_curve, a_acc, b_acc), 0.0, 1.0)

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Tomato Leaf Scaling Law Analysis\n(Curves fitted on first {fit_points} subsets)", fontsize=14, fontweight='bold')

    # Subplot 1: Train Loss
    ax1.scatter(percentages, losses, color="red", label="Actual Mean Loss", zorder=5, s=60)
    ax1.plot(x_curve, y_curve_loss, color="salmon", linestyle="--", label="Logarithmic Prediction Curve")
    # Highlight fit points
    ax1.scatter(x_fit, y_fit_loss, facecolors='none', edgecolors='black', s=150, linewidths=2, label="Fitting Bases", zorder=6)
    ax1.set_xscale("log")
    ax1.set_xlabel("Subset Percentage (Log Scale)", fontsize=11)
    ax1.set_ylabel("Average Cross Entropy Loss", fontsize=11)
    ax1.set_title("Training Loss vs. Dataset Scale", fontsize=12)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend()

    # Subplot 2: Test Accuracy
    ax2.scatter(percentages, accuracies, color="blue", label="Actual Mean Accuracy", zorder=5, s=60)
    ax2.plot(x_curve, y_curve_acc, color="skyblue", linestyle="--", label="Logarithmic Prediction Curve")
    # Highlight fit points
    ax2.scatter(x_fit, y_fit_acc, facecolors='none', edgecolors='black', s=150, linewidths=2, label="Fitting Bases", zorder=6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Subset Percentage (Log Scale)", fontsize=11)
    ax2.set_ylabel("Average Test Accuracy (Ratio)", fontsize=11)
    ax2.set_title("Test Accuracy vs. Dataset Scale", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig(output_plot_path, dpi=300)
        print(f"\n[PLOT SAVED] Learning curve comparison saved to: {output_plot_path.resolve()}")
    except Exception as e:
        print(f"[ERROR] Failed to save plot to {output_plot_path}: {e}")

    # Try to display (will show if GUI or Jupyter backend is active)
    try:
        plt.show()
    except Exception:
        pass


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)

    print("=" * 80)
    print("DATA SCIENCE ANALYZER: DATASET SCALING LAWS")
    print(f"Reading CSV from: {csv_path.resolve()}")
    print("=" * 80)

    try:
        agg_df = load_and_aggregate_metrics(csv_path)
        
        print("\nAggregated Metrics by Subset Scale:")
        for idx, row in agg_df.iterrows():
            print(f"  Subset {row['percentage']:>5.1f}% | Train Loss: {row['train_loss']:.4f} | Test Acc: {row['test_accuracy']:.4f} | Test Prec: {row['test_precision']:.4f}")

        # Execute predictions sequentially
        run_sequential_predictions(agg_df)

        # Plot scaling curves
        generate_plot(agg_df, args.fit_points, Path(args.output_plot))

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
