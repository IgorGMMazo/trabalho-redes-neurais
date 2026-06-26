# Bloco consolidado para rodar direto em uma unica celula do Google Colab
# Este script pode ser copiado e colado direto em uma celula do Colab para executar o pipeline completo.

import os
import sys
import subprocess
import time
from pathlib import Path

# --- Configurações do Pipeline ---
STEPS = 1000            # Passos de treino para as 140 rodadas do Grid Search
TARGET_STEPS = 10000    # Passos de treino para o modelo final (Fase 1+2)
IMG_SIZE = 128          # Resolução das imagens
MAX_IMAGES_PER_CLASS = 1000

# Arquivos de resultados
training_csv = "training_results.csv"
validation_csv = "validation_results.csv"

# --- Logs Estilizados Cyber-Industrial Minimalistas ---
def log_banner():
    print("┌────────────────────────────────────────────────────────────┐")
    print("│  COBALT SHIELD // AUTOMATED SCALING LAWS PIPELINE          │")
    print("│  SYSTEM STATUS: ONLINE // OPERATIONAL MODULES: ACTIVE      │")
    print("└────────────────────────────────────────────────────────────┘")

def log_section(name):
    print("\n" + "═"*60)
    print(f" ⚓ [SYSTEM] REGISTRY: {name.upper()}")
    print("═"*60)

def log_info(module, message):
    print(f" [{module.upper():<8}] {message}")

def run_cmd(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in process.stdout:
        print(f"    │ {line.strip()}")
    process.wait()
    return process.returncode == 0

# --- Execução do Pipeline ---
log_banner()

# 1. Particionamento do Dataset
log_section("Phase 1: Dataset Partitioning & Stratification")
log_info("data", "Compilando e dividindo conjuntos de dados...")
cmd_prep = [
    sys.executable, "process_dataset.py",
    "--input_dir", "dataset/orginal/dataset-plantas-com-augmentation",
    "--output_dir", "dataset/processed",
    "--max_images_per_class", str(MAX_IMAGES_PER_CLASS)
]
if not run_cmd(cmd_prep):
    log_info("data", "Erro ao compilar o dataset.")
    sys.exit(1)

# 2. Grid Search (140 Treinamentos)
log_section("Phase 2: Grid Search Experimentation (140 Trainings)")
log_info("model", "Sweeping 4 capacities x 7 cumulative subsets x 5 seeds...")

if os.path.exists(training_csv):
    os.remove(training_csv)
    log_info("system", f"Purge: {training_csv} deletado.")

capacities = [0.125, 0.25, 0.5, 1.0]
subsets = [1, 2, 5, 10, 20, 50, 100]
total_runs = len(capacities) * len(subsets)
run_idx = 1
start_grid_time = time.time()

for alpha in capacities:
    for pct in subsets:
        log_info("grid", f"Run {run_idx}/{total_runs} ({(run_idx/total_runs)*100:.1f}%) | alpha={alpha} | pct={pct}%")
        cmd_train = [
            sys.executable, "train.py",
            "--train_dir", f"dataset/processed/subsets_fase_1/pct_{pct}",
            "--test_dir", "dataset/processed/global_test",
            "--alpha", str(alpha),
            "--steps", str(STEPS),
            "--img_size", str(IMG_SIZE),
            "--output_csv", training_csv
        ]
        if not run_cmd(cmd_train):
            log_info("grid", "Falha na execucao do treino.")
            sys.exit(1)
        run_idx += 1

log_info("system", f"Grid search concluido em {(time.time() - start_grid_time)/60:.1f} minutos.")

# 3. Ajuste de Curva e Extrapolacao
log_section("Phase 3: Meta-Model Fitting & Extrapolation")
log_info("math", "Ajustando curva de Rosenfeld e Random Forest para N=8500, M=1.0...")
cmd_scale = [
    sys.executable, "scaling_laws.py",
    "--results_csv", training_csv,
    "--target_N", "8500",
    "--target_M", "1.0",
    "--output_plot", "scaling_law_surface.png"
]
if not run_cmd(cmd_scale):
    log_info("math", "Erro ao ajustar as Scaling Laws.")
    sys.exit(1)

# 4. Treinamento Ground Truth (Fase 1+2)
log_section("Phase 4: Empirical Ground Truth Training (Phase 1+2)")
log_info("train", f"Treinando modelo alvo em fase_1_mais_fase_2 ({TARGET_STEPS} steps)...")

if os.path.exists(validation_csv):
    os.remove(validation_csv)

cmd_val_train = [
    sys.executable, "train.py",
    "--train_dir", "dataset/processed/fase_1_mais_fase_2",
    "--test_dir", "dataset/processed/global_test",
    "--alpha", "1.0",
    "--steps", str(TARGET_STEPS),
    "--img_size", str(IMG_SIZE),
    "--output_csv", validation_csv
]
if not run_cmd(cmd_val_train):
    log_info("train", "Falha no treino final.")
    sys.exit(1)

# 5. Cruzamento de Dados e Validacao
log_section("Phase 5: Projection Accuracy Validation")
log_info("validate", "Calculando a margem de erro entre a previsao matematica e a realidade...")
cmd_validate = [
    sys.executable, "scaling_laws.py",
    "--results_csv", training_csv,
    "--target_N", "8500",
    "--target_M", "1.0",
    "--validate",
    "--val_csv", validation_csv
]
if not run_cmd(cmd_validate):
    log_info("validate", "Erro ao validar projecoes.")
    sys.exit(1)

print("\n┌────────────────────────────────────────────────────────────┐")
print("│  PIPELINE COMPLETION: ALL STAGES PROCESSED SUCCESSFULLY    │")
print("└────────────────────────────────────────────────────────────┘")
