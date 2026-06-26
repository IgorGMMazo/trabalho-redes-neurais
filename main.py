import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log_banner():
    print("┌────────────────────────────────────────────────────────────┐")
    print("│  COBALT SHIELD // AUTOMATED SCALING LAWS PIPELINE          │")
    print("│  SYSTEM STATUS: ONLINE // OPERATIONAL MODULES: ACTIVE      │")
    print("└────────────────────────────────────────────────────────────┘")


def log_section(name: str):
    print("\n" + "═"*60)
    print(f" ⚓ [SYSTEM] REGISTRY: {name.upper()}")
    print("═"*60)


def log_info(module: str, message: str):
    print(f" [{module.upper():<8}] {message}")


def run_command(cmd: List[str], cwd: Path) -> bool:
    try:
        # Run subprocess forwarding logs dynamically to stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd)
        )
        
        # Read stdout line by line
        for line in process.stdout:
            print(f"    │ {line.strip()}")
            
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"    │ [ERROR] Subprocess launch failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Orchestrator for Scaling Laws Experiments.")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per run (for Grid Search).")
    parser.add_argument("--img_size", type=int, default=128, help="Resolution for images.")
    parser.add_argument("--max_images_per_class", type=int, default=1000, help="Enforced balance constraint.")
    parser.add_argument("--target_steps", type=int, default=10000, help="Steps for the Phase 1+2 final model.")
    parser.add_argument("--dry_run", action="store_true", help="If set, executes script dry runs only.")
    
    args = parser.parse_args()
    
    cwd = Path(__file__).parent.resolve()
    
    # Filenames for pipeline outputs
    training_csv = "training_results.csv"
    validation_csv = "validation_results.csv"
    
    log_banner()
    
    # ----------------------------------------------------
    # PHASE 1: DATA PREPARATION
    # ----------------------------------------------------
    log_section("Phase 1: Dataset Partitioning & Stratification")
    
    cmd_prep = [
        sys.executable,
        "process_dataset.py",
        "--input_dir", "dataset/orginal/dataset-plantas-com-augmentation",
        "--output_dir", "dataset/processed",
        "--max_images_per_class", str(args.max_images_per_class)
    ]
    log_info("data", f"Invoking dataset compiler script (OS-agnostic pathlib)...")
    if not run_command(cmd_prep, cwd):
        log_info("data", "Dataset compilation aborted due to error.")
        sys.exit(1)
    log_info("data", "Dataset structures validated and partitioned successfully.")

    # ----------------------------------------------------
    # PHASE 2: GRID SEARCH (140 RUNS)
    # ----------------------------------------------------
    log_section("Phase 2: Grid Search Experimentation (140 Trainings)")
    log_info("model", "Sweeping 4 model capacities (alpha) x 7 cumulative subsets x 5 seeds")
    
    # Clear out any previous training results to start fresh
    csv_file = cwd / training_csv
    if csv_file.exists():
        csv_file.unlink()
        log_info("system", f"Purged historical results CSV: {training_csv}")

    capacities = [0.125, 0.25, 0.5, 1.0]
    subsets = [1, 2, 5, 10, 20, 50, 100]
    
    total_runs = len(capacities) * len(subsets)
    run_idx = 1
    
    start_grid_time = time.time()
    
    for alpha in capacities:
        for pct in subsets:
            log_info("grid", f"Progress: {run_idx}/{total_runs} ({(run_idx/total_runs)*100:.1f}%) | Config: alpha={alpha} | subset={pct}%")
            
            # Execute train.py (which internally runs the 5 seeds and appends metrics)
            cmd_train = [
                sys.executable,
                "train.py",
                "--train_dir", f"dataset/processed/subsets_fase_1/pct_{pct}",
                "--test_dir", "dataset/processed/global_test",
                "--alpha", str(alpha),
                "--steps", str(args.steps),
                "--img_size", str(args.img_size),
                "--output_csv", training_csv
            ]
            
            if args.dry_run:
                log_info("dry_run", f"Would run: {' '.join(cmd_train)}")
            else:
                if not run_command(cmd_train, cwd):
                    log_info("grid", f"Experiment failed at configuration alpha={alpha}, subset={pct}")
                    sys.exit(1)
            
            run_idx += 1

    elapsed_grid = time.time() - start_grid_time
    log_info("system", f"Grid search finished. Total elapsed time: {elapsed_grid/60:.1f} minutes.")

    # ----------------------------------------------------
    # PHASE 3: META-MODEL SCALING LAWS EXTRAPOLATION
    # ----------------------------------------------------
    log_section("Phase 3: Meta-Model Fitting & Extrapolation")
    log_info("math", "Extrapolating test error to consolidated Phase 1+2 dataset (N=8500, M=1.0)...")
    
    cmd_scale = [
        sys.executable,
        "scaling_laws.py",
        "--results_csv", training_csv,
        "--target_N", "8500",
        "--target_M", "1.0",
        "--output_plot", "scaling_law_surface.png"
    ]
    
    if args.dry_run:
        log_info("dry_run", f"Would run: {' '.join(cmd_scale)}")
    else:
        if not run_command(cmd_scale, cwd):
            log_info("math", "Failed to fit scaling law curves.")
            sys.exit(1)

    # ----------------------------------------------------
    # PHASE 4: TRAINING ON PHASE 1 + PHASE 2 (REAL TEST)
    # ----------------------------------------------------
    log_section("Phase 4: Empirical Ground Truth Training (Phase 1+2)")
    log_info("train", f"Training target model on consolidated data (fase_1_mais_fase_2, {args.target_steps} steps)...")
    
    # Purge any previous validation CSV
    val_csv_file = cwd / validation_csv
    if val_csv_file.exists():
        val_csv_file.unlink()

    cmd_val_train = [
        sys.executable,
        "train.py",
        "--train_dir", "dataset/processed/fase_1_mais_fase_2",
        "--test_dir", "dataset/processed/global_test",
        "--alpha", "1.0",
        "--steps", str(args.target_steps),
        "--img_size", str(args.img_size),
        "--output_csv", validation_csv
    ]
    
    if args.dry_run:
        log_info("dry_run", f"Would run: {' '.join(cmd_val_train)}")
    else:
        if not run_command(cmd_val_train, cwd):
            log_info("train", "Target model training failed.")
            sys.exit(1)
        log_info("train", "Ground truth training completed successfully.")

    # ----------------------------------------------------
    # PHASE 5: VALIDATION (METAMODEL VS EMPIRICAL)
    # ----------------------------------------------------
    log_section("Phase 5: Projection Accuracy Validation")
    log_info("validate", "Comparing mathematical projection with observed empirical truth...")
    
    cmd_validate = [
        sys.executable,
        "scaling_laws.py",
        "--results_csv", training_csv,
        "--target_N", "8500",
        "--target_M", "1.0",
        "--validate",
        "--val_csv", validation_csv
    ]
    
    if args.dry_run:
        log_info("dry_run", f"Would run: {' '.join(cmd_validate)}")
    else:
        if not run_command(cmd_validate, cwd):
            log_info("validate", "Validation evaluation failed.")
            sys.exit(1)
            
    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│  PIPELINE COMPLETION: ALL STAGES PROCESSED SUCCESSFULLY    │")
    print("│  REPORT LOGS AND CSV EXPORTS SAVED IN EXPERIMENT STORAGE  │")
    print("└────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
