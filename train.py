import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, accuracy_score


# --- Device Selection (NVIDIA CUDA, AMD DirectML, CPU) ---
def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[*] Device Selection: CUDA (NVIDIA GPU) detected.")
        return torch.device("cuda")
    
    try:
        import torch_directml
        if torch_directml.is_available():
            print("[*] Device Selection: DirectML (AMD/Intel GPU) detected.")
            return torch_directml.device()
    except ImportError:
        pass
    
    print("[*] Device Selection: CPU detected.")
    return torch.device("cpu")


# --- OS-Agnostic Dataset Class ---
class TomatoDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Discover and sort classes for OS-agnostic consistency
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG")
        for cls_name in self.classes:
            cls_dir = root_dir / cls_name
            class_images = []
            for ext in extensions:
                class_images.extend(list(cls_dir.glob(ext)))
            # Sort paths to guarantee exact same ordering on all OS
            for img_path in sorted(list(set(class_images))):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image via PIL and convert to RGB
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


# --- Custom ResNet-18 Architecture with Width Multiplier (Alpha) ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ScalableResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, alpha: float = 1.0):
        super(ScalableResNet18, self).__init__()
        # Calculate scaled channel widths
        self.in_planes = int(64 * alpha)
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, int(64 * alpha), 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, int(128 * alpha), 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, int(256 * alpha), 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, int(512 * alpha), 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512 * alpha) * BasicBlock.expansion, num_classes)

        # Apply Xavier initialization
        self._init_weights()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# --- Helper function to compute linear warmup + cosine annealing LR ---
def get_lr(step: int, total_steps: int, lr_max: float, warmup_pct: float = 0.05) -> float:
    warmup_steps = int(total_steps * warmup_pct)
    if step <= warmup_steps:
        # Linear Warmup
        return lr_max * (step / max(1, warmup_steps))
    else:
        # Cosine Annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * lr_max * (1.0 + math.cos(math.pi * progress))


# --- Calculate linear regression slope ---
def calculate_slope(y: List[float]) -> float:
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    denominator = (n * sum_xx) - (sum_x**2)
    if denominator == 0:
        return 0.0
    slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
    return float(slope)


# --- Seeding function for complete reproducibility ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Evaluation Function ---
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    test_error = 1.0 - acc
    f1 = f1_score(all_targets, all_preds, average="macro")
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    
    return test_error, f1, precision


# --- Training Loop ---
def train_single_run(
    seed: int,
    train_dir: Path,
    test_dir: Path,
    alpha: float,
    total_steps: int,
    batch_size: int,
    lr_max: float,
    weight_decay: float,
    img_size: int,
    device: torch.device
) -> Tuple[int, float, float, float, float]:
    
    set_seed(seed)
    print(f"\n[+] Starting run with Seed {seed} | Capacity Factor (alpha) = {alpha}")

    # Transform: resize and normalize ONLY (No Data Augmentation as requested)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    train_dataset = TomatoDataset(train_dir, transform=transform)
    test_dataset = TomatoDataset(test_dir, transform=transform)
    
    # Set drop_last dynamically to False if the dataset is smaller than the batch size,
    # otherwise we would drop the entire dataset and have 0 batches!
    drop_last = len(train_dataset) >= batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dataset_size = len(train_dataset)
    print(f"[*] Train dataset size: {dataset_size} images")
    print(f"[*] Test dataset size: {len(test_dataset)} images")

    # Model definition
    model = ScalableResNet18(num_classes=10, alpha=alpha).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model parameters count: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Cyclic iterator for steps budget
    train_iter = iter(train_loader)
    loss_history = []
    
    model.train()
    start_time = time.time()
    
    for step in range(1, total_steps + 1):
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Calculate learning rate for the step
        current_lr = get_lr(step, total_steps, lr_max)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Log training status
        if step % 500 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            print(f"Step {step:5d}/{total_steps:5d} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
            start_time = time.time()

    # Calculate final loss slope (over the last 10% of steps)
    slope_window = int(total_steps * 0.1)
    final_losses = loss_history[-slope_window:]
    loss_slope = calculate_slope(final_losses)
    print(f"[*] Final Loss Slope: {loss_slope:.6e}")

    # Evaluate on Global Test Set
    test_error, f1, precision = evaluate(model, test_loader, device)
    print(f"[+] Evaluation Result: Test Error = {test_error*100:.2f}% | F1-Score (Macro) = {f1:.4f} | Precision (Macro) = {precision:.4f}")

    return dataset_size, total_params, loss_slope, test_error, f1, precision


# --- Main Orchestrator ---
def main():
    parser = argparse.ArgumentParser(description="PyTorch Training Pipeline for Tomato Leaf Classifier.")
    parser.add_argument("--train_dir", type=str, default="dataset/processed/train_fase_1", help="Train dataset path.")
    parser.add_argument("--test_dir", type=str, default="dataset/processed/global_test", help="Global test dataset path.")
    parser.add_argument("--alpha", type=float, default=1.0, choices=[0.125, 0.25, 0.5, 1.0], help="Model capacity factor (alpha).")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps (updates).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size cravado em 64.")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay parameter.")
    parser.add_argument("--img_size", type=int, default=128, help="Resize dimension for train/test images.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="5 seeds for training iterations.")
    parser.add_argument("--output_csv", type=str, default="training_results.csv", help="CSV path to save results.")
    
    args = parser.parse_args()
    
    train_path = Path(args.train_dir)
    test_path = Path(args.test_dir)
    
    device = get_device()
    
    # Verify paths
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")
        
    print("="*60)
    print("           TRAINING PIPELINE CONFIGURATION          ")
    print("="*60)
    print(f"Train Dir:      {train_path}")
    print(f"Test Dir:       {test_path}")
    print(f"Alpha:          {args.alpha}")
    print(f"Steps:          {args.steps}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Seeds:          {args.seeds}")
    print(f"Output CSV:     {args.output_csv}")
    print("="*60)

    results = []

    # Iterate over the 5 seeds
    for seed in args.seeds:
        dataset_size, total_params, loss_slope, test_error, f1, precision = train_single_run(
            seed=seed,
            train_dir=train_path,
            test_dir=test_path,
            alpha=args.alpha,
            total_steps=args.steps,
            batch_size=args.batch_size,
            lr_max=args.lr,
            weight_decay=args.weight_decay,
            img_size=args.img_size,
            device=device
        )
        
        # Keep track of metrics
        results.append({
            "seed": seed,
            "tamanho_dataset": dataset_size,
            "fracao_parametros_modelo": args.alpha,
            "total_parametros": total_params,
            "loss_slope_final": f"{loss_slope:.6e}",
            "test_error": f"{test_error:.6f}",
            "f1_score_macro": f"{f1:.6f}",
            "precision_macro": f"{precision:.6f}"
        })

    # Save to CSV
    csv_file = Path(args.output_csv)
    fieldnames = [
        "seed", "tamanho_dataset", "fracao_parametros_modelo", 
        "total_parametros", "loss_slope_final", "test_error", 
        "f1_score_macro", "precision_macro"
    ]
    
    file_exists = csv_file.exists()
    with open(csv_file, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
        
    print(f"\n[+] Training complete for all 5 seeds! Results appended to {csv_file.resolve()}")


if __name__ == "__main__":
    main()
