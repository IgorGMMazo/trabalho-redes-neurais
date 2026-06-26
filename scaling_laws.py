import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings from curve_fit if bounds are tricky
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- Rosenfeld's Empirical Law: epsilon(M, N) = a * M^-alpha + b * N^-beta + c_inf ---
def rosenfeld_law(X: Tuple[np.ndarray, np.ndarray], a: float, b: float, c_inf: float, alpha: float, beta: float) -> np.ndarray:
    M, N = X
    # Avoid zero division and negative bases
    M_val = np.maximum(M, 1e-5)
    N_val = np.maximum(N, 1e-5)
    return a * (M_val ** -alpha) + b * (N_val ** -beta) + c_inf


def fit_rosenfeld(df: pd.DataFrame) -> Tuple[tuple, bool]:
    M = df["fracao_parametros_modelo"].values
    N = df["tamanho_dataset"].values
    y = df["test_error"].values
    
    # We need enough unique points in (M, N) space to solve for 5 parameters
    unique_points = len(df.groupby(["fracao_parametros_modelo", "tamanho_dataset"]).size())
    
    if unique_points < 5:
        print("[!] Warning: Insufficient unique points in (M, N) space to fit Rosenfeld's Law (needs >= 5 unique configurations).")
        print("[*] Falling back to standard scaling law coefficients based on Rosenfeld's literature [a=0.15, b=0.25, c_inf=0.02, alpha=0.35, beta=0.28].")
        # Return fallback heuristic parameters
        return (0.15, 0.25, 0.02, 0.35, 0.28), False
    
    # Bounds: a, b, alpha, beta > 0; c_inf >= 0
    bounds = (
        (1e-6, 1e-6, 0.0, 1e-4, 1e-4), # Lower bounds
        (10.0, 10.0, 1.0, 3.0, 3.0)     # Upper bounds
    )
    # Initial guess
    p0 = [0.1, 0.2, 0.01, 0.5, 0.3]
    
    try:
        popt, _ = curve_fit(
            rosenfeld_law,
            (M, N),
            y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        return tuple(popt), True
    except Exception as e:
        print(f"[!] Error fitting curve: {e}. Falling back to default literature parameters.")
        return (0.15, 0.25, 0.02, 0.35, 0.28), False


def plot_surfaces(
    df: pd.DataFrame, 
    popt: tuple, 
    rf_model: RandomForestRegressor, 
    mean_loss_slope: float, 
    output_path: Path
):
    # Create grid for visualization
    M_grid = np.linspace(0.1, 1.0, 50)
    N_grid = np.linspace(df["tamanho_dataset"].min(), df["tamanho_dataset"].max() * 2, 50)
    MM, NN = np.meshgrid(M_grid, N_grid)
    
    # Rosenfeld predictions
    Z_rosenfeld = rosenfeld_law((MM, NN), *popt)
    
    # RandomForest predictions (requires loss_slope)
    grid_flat_M = MM.flatten()
    grid_flat_N = NN.flatten()
    grid_flat_slope = np.full_like(grid_flat_M, mean_loss_slope)
    
    X_grid_rf = np.column_stack((grid_flat_N, grid_flat_M, grid_flat_slope))
    Z_rf = rf_model.predict(X_grid_rf).reshape(MM.shape)
    
    fig = plt.figure(figsize=(16, 7))
    
    # Plot Rosenfeld Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(MM, NN, Z_rosenfeld, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.scatter(df["fracao_parametros_modelo"], df["tamanho_dataset"], df["test_error"], color='red', s=50, label='Dados de Treino')
    ax1.set_title("Superfície de Erro: Ajuste de Rosenfeld")
    ax1.set_xlabel("Capacidade Modelo (M)")
    ax1.set_ylabel("Tamanho Dataset (N)")
    ax1.set_zlabel("Taxa Erro Teste")
    ax1.legend()
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Plot RandomForest Surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(MM, NN, Z_rf, cmap='magma', alpha=0.8, edgecolor='none')
    ax2.scatter(df["fracao_parametros_modelo"], df["tamanho_dataset"], df["test_error"], color='red', s=50, label='Dados de Treino')
    ax2.set_title("Superfície de Erro: Random Forest")
    ax2.set_xlabel("Capacidade Modelo (M)")
    ax2.set_ylabel("Tamanho Dataset (N)")
    ax2.set_zlabel("Taxa Erro Teste")
    ax2.legend()
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Generalization surface plots saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Scaling Laws Predictor and Extrapolator.")
    parser.add_argument("--results_csv", type=str, default="training_results.csv", help="CSV containing current training metrics.")
    parser.add_argument("--target_N", type=int, default=8500, help="Target dataset size (Fase 1 + Fase 2 = 8500).")
    parser.add_argument("--target_M", type=float, default=1.0, choices=[0.125, 0.25, 0.5, 1.0], help="Target model fraction (alpha).")
    parser.add_argument("--validate", action="store_true", help="Compare projections with actual validation data.")
    parser.add_argument("--val_csv", type=str, default="validation_results.csv", help="CSV containing target validation metrics if available.")
    parser.add_argument("--output_plot", type=str, default="scaling_law_surface.png", help="Path to save the generated 3D surface plot.")
    
    args = parser.parse_args()
    csv_path = Path(args.results_csv)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Training results CSV not found: {csv_path}. Please run train.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Pre-process columns
    required_cols = ["tamanho_dataset", "fracao_parametros_modelo", "loss_slope_final", "test_error"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in results CSV.")
            
    print("="*60)
    print("                SCALING LAWS PREDICTOR             ")
    print("="*60)
    print(f"Loaded {len(df)} runs from {csv_path.name}")
    print(f"Features range:")
    print(f"  - Dataset Size (N): {df['tamanho_dataset'].min()} to {df['tamanho_dataset'].max()}")
    print(f"  - Model Capacity (M): {df['fracao_parametros_modelo'].min()} to {df['fracao_parametros_modelo'].max()}")
    
    # 1. Fit Rosenfeld's Law
    popt, fitted = fit_rosenfeld(df)
    if fitted:
        a, b, c_inf, alpha, beta = popt
        print(f"\n[+] Fitted Rosenfeld's Scaling Law Coefficients:")
        print(f"  - a       = {a:.6f}")
        print(f"  - b       = {b:.6f}")
        print(f"  - c_inf   = {c_inf:.6f} (Asymptotic limit)")
        print(f"  - alpha   = {alpha:.6f} (Model scale exponent)")
        print(f"  - beta    = {beta:.6f} (Data scale exponent)")
    else:
        a, b, c_inf, alpha, beta = popt
        print(f"\n[*] Using Default Literature Scaling Law Coefficients:")
        print(f"  - a={a}, b={b}, c_inf={c_inf}, alpha={alpha}, beta={beta}")

    # 2. Fit Empirical RandomForestRegressor
    X_rf = df[["tamanho_dataset", "fracao_parametros_modelo", "loss_slope_final"]].values
    y_rf = df["test_error"].values
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_rf, y_rf)
    print("[+] Trained RandomForestRegressor on historical runs.")

    # 3. Blind Prediction (Fase 1 + Fase 2 extrapolation)
    mean_slope = df["loss_slope_final"].mean()
    
    # Rosenfeld Projection
    pred_error_rosenfeld = rosenfeld_law((args.target_M, args.target_N), *popt)
    
    # Random Forest Projection (Needs a loss_slope value. We pass the historical mean)
    pred_error_rf = rf_model.predict([[args.target_N, args.target_M, mean_slope]])[0]
    
    print("\n" + "="*50)
    print("                  BLIND PROJECTION                  ")
    print("="*50)
    print(f"Target Configuration: N = {args.target_N} | M = {args.target_M}")
    print(f"Projected Test Error Rate (Rosenfeld): {pred_error_rosenfeld * 100:.4f}%")
    print(f"Projected Test Error Rate (RandomForest): {pred_error_rf * 100:.4f}%")
    print("="*50)

    # 4. Validation (If flag is set)
    if args.validate:
        val_path = Path(args.val_csv)
        if not val_path.exists():
            print(f"\n[!] Error: Validation results CSV not found at '{args.val_csv}'.")
            print("[*] Checking if target run is logged inside the training results CSV...")
            # Search if training CSV contains the target_N
            target_runs = df[(df["tamanho_dataset"] == args.target_N) & (df["fracao_parametros_modelo"] == args.target_M)]
            if not target_runs.empty:
                val_df = target_runs
            else:
                val_df = pd.DataFrame()
        else:
            val_df = pd.read_csv(val_path)
            
        if not val_df.empty:
            actual_error = val_df["test_error"].mean()
            error_diff_ros = abs(pred_error_rosenfeld - actual_error)
            error_diff_rf = abs(pred_error_rf - actual_error)
            
            print("\n" + "="*50)
            print("                 VALIDATION RESULTS                 ")
            print("="*50)
            print(f"Actual Test Error Rate:              {actual_error * 100:.4f}%")
            print(f"Rosenfeld Margin of Error:           {error_diff_ros * 100:.4f}%")
            print(f"Random Forest Margin of Error:       {error_diff_rf * 100:.4f}%")
            print("="*50)
        else:
            print("\n[!] Validation Failed: No actual data found for the target configuration in results or validation CSVs.")

    # 5. Plot surfaces
    plot_surfaces(df, popt, rf_model, mean_slope, Path(args.output_plot))


if __name__ == "__main__":
    main()
