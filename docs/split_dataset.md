# Tomato Leaf Preprocessing and Training Pipeline

This repository contains tools to preprocess the tomato leaf disease dataset and run training experiments using PyTorch.

---

## 📂 Repository Structure

* **Preprocessing**:
  * [split_dataset.py](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/split_dataset.py): Python CLI to balance classes and extract cumulative subsets.
  * [preprocess_colab.ipynb](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/preprocess_colab.ipynb): Google Colab notebook for preprocessing.
  * [dataset_metadata.json](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/dataset/preprocessed/dataset_metadata.json): JSON mapping files to subsets and splits.
* **Model Training**:
  * [train.py](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/train.py): Python CLI to run ResNet-18 training runs across subsets.
  * [train_colab.ipynb](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/train_colab.ipynb): Google Colab notebook for model training.
  * `training_results.csv`: Generated output recording loss, accuracy, precision, and training times.
* **Documentation**:
  * [split_dataset.md](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/split_dataset.md): Standard engineering documentation (this file).
  * [sliptt_dataset_explicação_informal.md](file:///C:/Users/Herik/Documents/faculdade/redes%20neurais/trabalho-redes-neurais/sliptt_dataset_explica%C3%A7%C3%A3o_informal.md): Informal student README.

---

## 🛠️ Part 1: Preprocessing Pipeline

Preprocesses the 10 tomato leaf classes. It balances folders to a maximum limit (default: 1,000 images per class) and partitions subsets (`[1%, 2%, 5%, 10%, 20%, 50%, 100%]`) using strict cumulative inclusion.

### Command-Line Usage (Local):
```bash
python split_dataset.py --src-dir "dataset/original_path" --max-images-per-class 1000
```
### Splitting Strategy:
1. **Balance**: Downsamples classes containing more than 1,000 images to exactly 1,000.
2. **Fixed Pools**: Randomly partitions the 1,000 images into a Train Pool (850 images) and a Test Pool (150 images) per class.
3. **Cumulative Slices**: Extracts subsets based on logarithmic percentages, ensuring smaller subsets are strict subsets of larger ones.

---

## 🚀 Part 2: Model Training Pipeline

Trains a **ResNet-18** from scratch (without pre-trained weights) to classify tomato leaves. The training uses no data augmentation (only 224x224 resize and standard normalization).

### Features:
1. **Universal Hardware Detection**: Autodetects hardware priority: NVIDIA CUDA -> AMD DirectML (via `torch-directml`) -> CPU fallback.
2. **Multi-Run Seed Evaluations**: Runs training 5 times per subset utilizing 5 different seeds (`[42, 43, 44, 45, 46]`) to assess stability.
3. **Metrics CSV**: Appends epoch-level train loss, test accuracy (macro), test precision (macro), and time.

### Command-Line Usage (Local):
```bash
python train.py --data-dir "dataset/preprocessed" --epochs 10 --batch-size 32 --lr 0.001
```

#### Argument Parameters:
* `--data-dir`: Directory containing the preprocessed subsets folders.
* `--output-csv`: Destination path of the metrics log (default: `training_results.csv`).
* `--epochs`: Training epochs per run (default: `10`).
* `--batch-size`: Mini-batch size for training/eval (default: `32`).
* `--lr`: Learning rate for Adam optimizer (default: `0.001`).
* `--subsets`: Comma-separated list of subsets (default: `"1,2,5,10,20,50,100"`).
* `--runs-per-subset`: Number of unique seeds evaluated per subset (default: `5`).

### Sample CSV Log Format:
```csv
subset,run_index,seed,epoch,train_loss,test_accuracy,test_precision,epoch_time_sec
subset_1%,0,42,1,2.74448,0.1,0.01,3.9
...
```
