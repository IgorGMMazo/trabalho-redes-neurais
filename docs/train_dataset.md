# Tomato Leaf Classification - Training Pipeline Documentation

This document describes the design, implementation, and execution guidelines for training PyTorch image classification models on balanced, cumulative subsets of tomato leaf images.

---

## 📋 Objective
The goal is to train a **ResNet-18** neural network architecture from scratch (with random weight initialization) across multiple data scales (`[1%, 2%, 5%, 10%, 20%, 50%, 100%]`). Each subset training is run **5 times** using unique random initialization seeds to evaluate the performance stability and variance of learning curves as data scales grow.

---

## ⚙️ Core Technical Specifications

### 1. Universal Hardware Detection (`get_device`)
The training script includes a robust selection layer to enable hardware acceleration across different student machines. Devices are resolved in the following priority order:
1. **NVIDIA CUDA**: Detected via `torch.cuda.is_available()`. Recommended for NVIDIA cards and Google Colab environments.
2. **AMD / DirectML**: Detected via `torch_directml.is_available()` (provided by the `torch-directml` library). Recommended for AMD GPUs running on Windows environments.
3. **CPU Fallback**: Selected as the final safe default if no accelerated runtime is found.

### 2. Data Transformations (Strictly No Augmentation)
To isolate data scaling factors and initialization seeds from stochastic regularization, data augmentation is omitted. Images are normalized to ImageNet statistics:
* **Resizing**: Capped to standard `(224, 224)` pixels.
* **Standardization**:
  $$\text{Mean} = [0.485, 0.456, 0.406]$$
  $$\text{Std} = [0.229, 0.224, 0.225]$$

### 3. Training & Evaluation Runs
* **Architecture**: Standard ResNet-18 (`torchvision.models.resnet18(weights=None, num_classes=10)`).
* **Optimizer**: Adam with learning rate $0.001$.
* **Criterion**: CrossEntropyLoss.
* **Multi-Run Setup**: Performs 5 runs per subset. A specific seed in `[42, 43, 44, 45, 46]` is set before weights initialization for each run.

---

## 🚀 Execution Guide

### Local Terminal (CLI):
Execute the orchestration script using Python:
```bash
python train.py --data-dir "dataset/preprocessed" --epochs 10 --batch-size 32
```

### CLI Command Options:
* `--data-dir`: Directory containing the preprocessed subsets folders (default: `dataset/preprocessed`).
* `--output-csv`: Destination of the logging CSV file (default: `training_results.csv`).
* `--epochs`: Epoch count per seed execution (default: `10`).
* `--batch-size`: Mini-batch size (default: `32`).
* `--lr`: Learning rate for the Adam optimizer (default: `0.001`).
* `--subsets`: Comma-separated list of subset percentages to evaluate (default: `1,2,5,10,20,50,100`).
* `--runs-per-subset`: Number of different seed runs to execute per subset (default: `5`).

### Running a Quick Smoke Test:
To verify PyTorch, dataset paths, and hardware bindings run:
```bash
python train.py --subsets 1 --epochs 1 --runs-per-subset 1
```

---

## 📊 Logging Schema

Epoch metrics are written incrementally to the output CSV (`training_results.csv`) with the following headers:

| Header | Data Type | Description |
|:---|:---:|:---|
| `subset` | `string` | Target subset folder name (e.g. `subset_1%`). |
| `run_index` | `int` | Execution run index per subset ($0$ to $4$). |
| `seed` | `int` | Initialization seed value ($42$ to $46$). |
| `epoch` | `int` | Current epoch index. |
| `train_loss` | `float` | Averaged Cross-Entropy loss over the training split. |
| `test_accuracy` | `float` | Accuracy ratio computed on the test split. |
| `test_precision` | `float` | Macro-averaged Precision score computed on the test split. |
| `epoch_time_sec` | `float` | Computational time in seconds taken by the epoch. |
