#!/usr/bin/env python3
"""
Tomato Leaf Disease Model Training Script.
Author: AI Researcher
Description: Trains a ResNet-18 model from scratch on cumulative subsets,
             performing multiple runs with different seeds to measure stability.
             Supports CUDA, DirectML (AMD), and CPU hardware configurations.
             Logs training loss, test accuracy, and test precision to a CSV.
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2  # Using torchvision v2 transforms if available, otherwise fallback
from torchvision import transforms

# Import sklearn for precision metric
from sklearn.metrics import precision_score


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train image classification models on cumulative tomato leaf subsets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/preprocessed",
        help="Path to the preprocessed subsets directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/training_results.csv",
        help="Path to save the training results CSV.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for each run (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="1,2,5,10,20,50,100",
        help="Comma-separated list of subsets to train on (default: '1,2,5,10,20,50,100').",
    )
    parser.add_argument(
        "--runs-per-subset",
        type=int,
        default=5,
        help="Number of runs (different seeds) per subset (default: 5).",
    )
    return parser.parse_args()


def get_device() -> Tuple[torch.device, str]:
    """
    Robust universal hardware detection.
    1st: NVIDIA/CUDA
    2nd: AMD/DirectML via torch-directml
    3rd: CPU fallback
    """
    # 1. NVIDIA / CUDA Check
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA/CUDA (GPU)"
    
    # 2. AMD / DirectML Check (Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            # torch_directml.device() returns a DirectML device object
            return torch_directml.device(), "AMD/DirectML (GPU via DirectML)"
    except ImportError:
        pass

    # 3. Fallback to CPU
    return torch.device("cpu"), "CPU"


def set_seed(seed: int):
    """
    Sets seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: torch-directml is seeded via torch.manual_seed


def get_transforms() -> transforms.Compose:
    """
    No data augmentation, only resizing and normalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
            std=[0.229, 0.224, 0.225]    # Standard ImageNet stds
        )
    ])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Trains the model for one epoch. Returns the average training loss.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the model on the test dataset.
    Returns:
        accuracy (float): Test accuracy.
        precision (float): Test macro-precision score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics calculation
    correct = (all_preds == all_labels).sum()
    accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0
    
    # Precision (macro-averaged for class-balanced results)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0.0)

    return float(accuracy), float(precision)


def main():
    args = parse_args()
    
    # 1. Device Selection & Identification
    device, hardware_name = get_device()
    print("=" * 60)
    print("TOMATO LEAF CLASSIFICATION TRAINING PIPELINE")
    print(f"Detected Hardware Device: {hardware_name}")
    print(f"PyTorch Device Object:   {device}")
    print("=" * 60)

    # 2. Setup folders and subsets
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Data directory '{data_dir}' does not exist. Run preprocessing first.")
        return

    subset_list = [s.strip() for s in args.subsets.split(",")]
    print(f"Subsets to train on: {subset_list}")
    print(f"Epochs per run: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print(f"Runs per subset: {args.runs_per_subset}")
    
    # Define a unique seed for each run index
    seeds = [42 + i for i in range(args.runs_per_subset)]
    print(f"Seeds to be used: {seeds}")
    
    # 3. Initialize CSV File
    output_csv = Path(args.output_csv)
    csv_header = ["subset", "run_index", "seed", "epoch", "train_loss", "test_accuracy", "test_precision", "epoch_time_sec"]
    
    # Write header if file doesn't exist, or overwrite
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
        
    img_transforms = get_transforms()

    # 4. Main training loop
    for subset in subset_list:
        subset_name = f"subset_{subset}%"
        subset_path = data_dir / subset_name
        
        if not subset_path.exists():
            print(f"\n[WARNING] Subset folder '{subset_name}' not found at {subset_path}. Skipping.")
            continue
            
        print(f"\n\n>>> Starting training on {subset_name} <<<")
        
        # Load dataset
        try:
            train_dataset = ImageFolder(root=subset_path / "train", transform=img_transforms)
            test_dataset = ImageFolder(root=subset_path / "test", transform=img_transforms)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        except Exception as e:
            print(f"[ERROR] Loading datasets failed for {subset_name}: {e}")
            continue

        print(f"Loaded {len(train_dataset)} train images, {len(test_dataset)} test images.")
        
        # Run iterations for different seeds
        for run_idx, seed in enumerate(seeds):
            print(f"\n  --- Run {run_idx + 1}/{args.runs_per_subset} (Seed: {seed}) ---")
            
            # Set seed before weights initialization to ensure reproducibility
            set_seed(seed)
            
            # Initialize ResNet-18 from scratch (weights=None ensures random init)
            # The only randomness is the initial weights
            model = torchvision.models.resnet18(weights=None, num_classes=10)
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            
            # Epoch loop
            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                
                # Train and evaluate
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                test_acc, test_prec = evaluate(model, test_loader, device)
                
                epoch_time = time.time() - start_time
                
                # Log epoch to console
                print(f"    Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | Test Prec: {test_prec:.4f} | Time: {epoch_time:.2f}s")
                
                # Append to CSV
                with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        subset_name,
                        run_idx,
                        seed,
                        epoch,
                        round(train_loss, 5),
                        round(test_acc, 5),
                        round(test_prec, 5),
                        round(epoch_time, 2)
                    ])
                    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Training complete! Results saved to: {output_csv.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
