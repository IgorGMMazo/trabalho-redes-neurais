#!/usr/bin/env python3
"""
Scaling Laws Analyzer — Tomato Leaf Dataset.
Reads training_results.csv, fits power-law curves per model size
(error = a * D^-beta + E_inf, Rosenfeld et al. 2020 §2.6), fits the joint
surface (error = a * N^-alpha * D^-beta + E_inf), detects saturation
points (§3.3), and generates publication-quality plots.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    print("[WARNING] Matplotlib not installed — plots skipped.")
    HAS_MPL = False


# ── Saturation detection §3.3 ─────────────────────────────────────────────────

def find_saturation_point(
    data_pcts: np.ndarray, accuracies: np.ndarray, threshold: float = 0.01
) -> Optional[float]:
    """Return the first data_pct where acc gain vs previous point < threshold (1pp)."""
    for i in range(1, len(data_pcts)):
        if (accuracies[i] - accuracies[i - 1]) < threshold:
            return float(data_pcts[i])
    return None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_final_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
        "model_size_pct", "data_subset_pct", "step",
        "train_loss", "test_accuracy", "test_error_rate",
        "test_f1", "test_precision",
    }
    missing = required - set(df.columns)
    if missing:
        # Backward compat: derive missing columns
        if "test_error_rate" not in df.columns and "test_accuracy" in df.columns:
            df["test_error_rate"] = 1.0 - df["test_accuracy"]
        if "test_f1" not in df.columns:
            df["test_f1"] = np.nan

    idx = df.groupby(["model_size_pct", "data_subset_pct"])["step"].idxmax()
    return (
        df.loc[idx]
        .sort_values(["model_size_pct", "data_subset_pct"])
        .reset_index(drop=True)
    )


# ── Console report ────────────────────────────────────────────────────────────

def report_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FINAL-STEP METRICS PER (model_size, data_subset)")
    print("=" * 80)
    prev = None
    for _, row in df.iterrows():
        if row["model_size_pct"] != prev:
            print(f"\n  Model {row['model_size_pct']}% params:")
            prev = row["model_size_pct"]
        f1_str = f"f1={row['test_f1']:.4f}  " if not np.isnan(row.get("test_f1", float("nan"))) else ""
        print(
            f"    data={row['data_subset_pct']:>3}%  "
            f"loss={row['train_loss']:.4f}  "
            f"acc={row['test_accuracy']:.4f}  "
            f"{f1_str}"
            f"err={row['test_error_rate']:.4f}"
        )



def report_saturation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SATURATION POINTS  (first data% where accuracy gain < 1pp)")
    print("=" * 80)
    for model_size in sorted(df["model_size_pct"].unique()):
        sub   = df[df["model_size_pct"] == model_size].sort_values("data_subset_pct")
        D_pct = sub["data_subset_pct"].values.astype(float)
        accs  = sub["test_accuracy"].values.astype(float)
        sat   = find_saturation_point(D_pct, accs)
        if sat is not None:
            print(f"  Model {model_size}%: saturates at data={sat:.0f}%")
        else:
            print(f"  Model {model_size}%: no saturation detected in measured range")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _model_label(pct: float) -> str:
    return f"{pct:.0f}% params"


def generate_plots(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if not HAS_MPL:
        return

    model_sizes = sorted(df["model_size_pct"].unique())
    n_models    = len(model_sizes)
    palette     = [cm.tab10(i / 10) for i in range(n_models)]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(
        "Scaling Laws — Tomato Leaf (ResNet-18 width variants)",
        fontsize=14, fontweight="bold",
    )
    ax_acc, ax_err, ax_f1 = axes

    D_curve = np.geomspace(1.0, 100.0, 300)

    for color, model_size in zip(palette, model_sizes):
        sub   = df[df["model_size_pct"] == model_size].sort_values("data_subset_pct")
        D_pct = sub["data_subset_pct"].values.astype(float)
        accs  = sub["test_accuracy"].values.astype(float)
        errs  = sub["test_error_rate"].values.astype(float)
        f1s   = sub["test_f1"].values.astype(float) if "test_f1" in sub.columns else None
        label = _model_label(model_size)

        # Scatter
        ax_acc.scatter(D_pct, accs, color=color, s=60, zorder=5)
        ax_err.scatter(D_pct, errs, color=color, s=60, zorder=5)
        if f1s is not None and not np.all(np.isnan(f1s)):
            ax_f1.scatter(D_pct, f1s,  color=color, s=60, zorder=5)

        else:
            ax_acc.plot(D_pct, accs, color=color, lw=1.5, label=label)
            ax_err.plot(D_pct, errs, color=color, lw=1.5, label=label)

        # Mark saturation point
        sat = find_saturation_point(D_pct, accs)
        if sat is not None:
            sat_acc = float(accs[D_pct == sat][0]) if sat in D_pct else None
            if sat_acc is not None:
                ax_acc.axvline(sat, color=color, lw=0.8, linestyle=":", alpha=0.6)

    for ax, ylabel, title in [
        (ax_acc, "Test Accuracy",    "Accuracy vs. Data Volume"),
        (ax_err, "Test Error Rate",  "Error Rate vs. Data Volume (log-log = power-law)"),
        (ax_f1,  "Test F1 Macro",    "F1-Score Macro vs. Data Volume"),
    ]:
        ax.set_xscale("log")
        if ax is ax_err:
            ax.set_yscale("log")
        ax.set_xlabel("Data Subset % of Train Pool (log scale)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(title="Model size", fontsize=8, loc="best")

    ax_acc.set_ylim(-0.05, 1.05)
    ax_f1.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\n[PLOT SAVED] {output_path.resolve()}")
    except Exception as e:
        print(f"[WARNING] Could not save plot: {e}")
    plt.close()

    # Individual plot per model size
    for color, model_size in zip(palette, model_sizes):
        sub   = df[df["model_size_pct"] == model_size].sort_values("data_subset_pct")
        D_pct = sub["data_subset_pct"].values.astype(float)
        accs  = sub["test_accuracy"].values.astype(float)
        errs  = sub["test_error_rate"].values.astype(float)
        f1s   = sub["test_f1"].values.astype(float) if "test_f1" in sub.columns else None

        fig_ind, (ax_a, ax_e, ax_f) = plt.subplots(1, 3, figsize=(21, 6))
        fig_ind.suptitle(
            f"Scaling Laws — Model {model_size:.0f}% params (ResNet-18 width variant)",
            fontsize=14, fontweight="bold",
        )

        ax_a.scatter(D_pct, accs, color=color, s=60, zorder=5)
        ax_e.scatter(D_pct, errs, color=color, s=60, zorder=5)
        if f1s is not None and not np.all(np.isnan(f1s)):
            ax_f.scatter(D_pct, f1s, color=color, s=60, zorder=5)

        sat = find_saturation_point(D_pct, accs)
        if sat is not None:
            ax_a.axvline(sat, color=color, lw=0.8, linestyle=":", alpha=0.6,
                         label=f"saturação @ {sat:.0f}%")
            ax_a.legend(fontsize=8)

        for ax, ylabel, title in [
            (ax_a, "Test Accuracy",   "Accuracy vs. Data Volume"),
            (ax_e, "Test Error Rate", "Error Rate vs. Data Volume (log-log)"),
            (ax_f, "Test F1 Macro",   "F1-Score Macro vs. Data Volume"),
        ]:
            ax.set_xscale("log")
            if ax is ax_e:
                ax.set_yscale("log")
            ax.set_xlabel("Data Subset % of Train Pool (log scale)", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)

        ax_a.set_ylim(-0.05, 1.05)
        ax_f.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        ind_path = output_path.parent / f"{output_path.stem}_{model_size}pct{output_path.suffix}"
        try:
            plt.savefig(ind_path, dpi=200, bbox_inches="tight")
            print(f"[PLOT SAVED] {ind_path.resolve()}")
        except Exception as e:
            print(f"[WARNING] Could not save individual plot for {model_size}%: {e}")
        plt.close()


def generate_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Accuracy heatmap: model size (rows) × data fraction (cols)."""
    if not HAS_MPL:
        return
    pivot = df.pivot(index="model_size_pct", columns="data_subset_pct", values="test_accuracy")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Test Accuracy")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}%" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.0f}%\nparams" for r in pivot.index], fontsize=9)
    ax.set_xlabel("Data Subset", fontsize=10)
    ax.set_title("Accuracy Heatmap: Model Capacity × Data Volume", fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.7 else "white")

    plt.tight_layout()
    heatmap_path = output_path.parent / (output_path.stem + "_heatmap.png")
    try:
        plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        print(f"[PLOT SAVED] {heatmap_path.resolve()}")
    except Exception as e:
        print(f"[WARNING] Could not save heatmap: {e}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze scaling law results from training CSV.")
    p.add_argument("--csv-file",    type=str, default="resultados_earlystop/training_results.csv")
    p.add_argument("--output-plot", type=str, default="resultados_earlystop/scaling_law_plot.png")
    p.add_argument("--no-plot",     action="store_true", help="Skip plot generation.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    csv_path = Path(args.csv_file)

    print("=" * 80)
    print("SCALING LAWS ANALYZER — TOMATO LEAF DATASET")
    print(f"CSV: {csv_path.resolve()}")
    print("=" * 80)

    try:
        df = load_final_metrics(csv_path)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    n_runs = len(df)
    n_models = df["model_size_pct"].nunique()
    n_subsets = df["data_subset_pct"].nunique()
    print(f"\nLoaded {n_runs} final-step records  "
          f"({n_models} model sizes x {n_subsets} data subsets)")

    if not args.no_plot:
        output_path = Path(args.output_plot)
        generate_plots(df, output_path)
        if n_models > 1 and n_subsets > 1:
            generate_heatmap(df, output_path)


if __name__ == "__main__":
    main()
