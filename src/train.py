#!/usr/bin/env python3
"""
Tomato Leaf Disease Training — Scaling Laws Edition.
28-run grid: 4 width-scaled ResNet-18 variants x 7 data fractions (§2.5).

Performance strategy: entire dataset is loaded from disk once per run,
pre-processed (resize + normalize) and pinned directly in VRAM.
All training steps operate exclusively on VRAM tensors — zero per-step
disk I/O, zero CPU-GPU transfer, zero DataLoader overhead.
"""

import argparse
import concurrent.futures
import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score
from tqdm import tqdm


# ── Architecture §2.2 ────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[float, Dict] = {
    12.5: {"alpha": 0.35, "channels": [22,  22,  45,  90,  181]},
    25.0: {"alpha": 0.50, "channels": [32,  32,  64,  128, 256]},
    50.0: {"alpha": 0.71, "channels": [45,  45,  90,  181, 362]},
   100.0: {"alpha": 1.00, "channels": [64,  64,  128, 256, 512]},
}

DEFAULT_SEEDS: Dict[float, int] = {12.5: 42, 25.0: 43, 50.0: 44, 100.0: 45}


# ── Residual block ────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
    blocks: List[nn.Module] = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(BasicBlock(out_ch, out_ch))
    return nn.Sequential(*blocks)


class ScaledResNet18(nn.Module):
    """ResNet-18 with explicit channel counts for width scaling (§2.2)."""

    def __init__(self, channels: List[int], num_classes: int = 10):
        super().__init__()
        s, c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, s, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(s),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(s,  c1, num_blocks=2)
        self.layer2 = _make_layer(c1, c2, num_blocks=2, stride=2)
        self.layer3 = _make_layer(c2, c3, num_blocks=2, stride=2)
        self.layer4 = _make_layer(c3, c4, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_xavier_init(model: nn.Module) -> None:
    """Xavier/Glorot uniform for Conv/Linear; ones/zeros for BN (§2.3)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# ── VRAM Dataset Cache ────────────────────────────────────────────────────────

class CachedGPULoader:
    """
    Loads an ImageFolder dataset from disk once, applies all transforms,
    and stores the result as a single (N, C, H, W) tensor in VRAM.

    All subsequent batches are served via pure CUDA tensor indexing —
    no disk I/O, no PIL decoding, no CPU-GPU transfer per step.
    """

    def __init__(
        self,
        folder_dataset: ImageFolder,
        device: torch.device,
        transform: transforms.Compose,
        batch_size: int,
        shuffle: bool = True,
        num_load_workers: int = 8,
    ):
        self.device     = device
        self.batch_size = batch_size
        self.shuffle    = shuffle
        n = len(folder_dataset)

        # Load images in parallel using threads (I/O bound, GIL released)
        def load_one(item: Tuple[str, int]) -> Tuple[torch.Tensor, int]:
            path, label = item
            img = Image.open(path).convert("RGB")
            return transform(img), label

        tensors: List[torch.Tensor] = [None] * n  # type: ignore
        labels:  List[int]          = [0]   * n

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_load_workers) as ex:
            futures = {ex.submit(load_one, item): i
                       for i, item in enumerate(folder_dataset.samples)}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                tensors[i], labels[i] = fut.result()

        # float16 halves VRAM usage; model receives float32 via .float() on yield
        self.X = torch.stack(tensors).to(device, dtype=torch.float16)
        self.Y = torch.tensor(labels, dtype=torch.long, device=device)
        self.n = n
        vram_mb = self.X.nbytes / 1e6
        print(f"    Cached {n} imgs -> {vram_mb:.0f} MB VRAM (fp16)")

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.device)
        else:
            perm = torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start: start + self.batch_size]
            yield self.X[idx].float(), self.Y[idx]

    def free(self) -> None:
        """Release VRAM explicitly."""
        del self.X, self.Y
        torch.cuda.empty_cache()


def infinite_cached(loader: CachedGPULoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


# ── Hardware ─────────────────────────────────────────────────────────────────

def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "CPU (no CUDA)"


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── LR schedule: linear warmup + cosine annealing to 0 (§2.3) ───────────────

def get_lr_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: CachedGPULoader,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Returns (accuracy, f1_macro, precision_macro). Data is already on VRAM."""
    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for inputs, labels in test_loader:
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    preds_arr  = np.array(all_preds,  dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    if len(labels_arr) == 0:
        return 0.0, 0.0, 0.0

    accuracy  = float((preds_arr == labels_arr).mean())
    f1        = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0.0))
    precision = float(precision_score(labels_arr, preds_arr, average="macro", zero_division=0.0))
    return accuracy, f1, precision


# ── Transforms (no augmentation — §2.3) ──────────────────────────────────────

def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train width-scaled ResNet-18 variants under a fixed step budget."
    )
    p.add_argument("--data-dir",        type=str,   default="dataset/preprocessed")
    p.add_argument("--output-csv",      type=str,   default="resultados_earlustop/training_results.csv")
    p.add_argument("--total-steps",     type=int,   default=10_000,
                   help="Fixed gradient-step budget per run (§2.3).")
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=2e-3)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--log-every",       type=int,   default=500)
    p.add_argument("--load-workers",    type=int,   default=8,
                   help="Threads used for parallel image loading into VRAM cache.")
    p.add_argument("--model-sizes",     type=str,   default="12.5,25,50,100")
    p.add_argument("--subsets",         type=str,   default="1,2,5,10,20,50,100")
    p.add_argument("--model-seeds",     type=str,   default="",
                   help="Comma-separated seeds per model size. Defaults: 42,43,44,45.")
    p.add_argument("--no-amp",          action="store_true",
                   help="Disable Automatic Mixed Precision.")
    p.add_argument("--save-dir",        type=str,   default="checkpoints/best",
                   help="Diretório para salvar o melhor modelo de cada run.")
    p.add_argument("--patience",        type=int,   default=3,
                   help="Avaliações consecutivas sem ganho >= min-delta antes de parar.")
    p.add_argument("--min-delta",       type=float, default=0.01,
                   help="Ganho mínimo de accuracy para resetar o early stopping (padrão 1%%).")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device, hw_name = get_device()
    use_amp = (device.type == "cuda") and not args.no_amp

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print("=" * 70)
    print("TOMATO LEAF — SCALING LAWS TRAINING (28-run grid)")
    print(f"  GPU     : {hw_name}")
    print(f"  AMP     : {use_amp}  |  TF32 : {device.type == 'cuda'}")
    print(f"  Estratégia : dataset cacheado na VRAM (zero cópia RAM→VRAM por step)")
    print("=" * 70)

    data_dir   = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_sizes = [float(x) for x in args.model_sizes.split(",")]
    subset_pcts = [int(x)   for x in args.subsets.split(",")]

    if args.model_seeds:
        raw = [int(x) for x in args.model_seeds.split(",")]
        if len(raw) != len(model_sizes):
            raise ValueError("--model-seeds count must match --model-sizes count")
        seeds_map = dict(zip(model_sizes, raw))
    else:
        seeds_map = {sz: DEFAULT_SEEDS.get(sz, 42 + i) for i, sz in enumerate(model_sizes)}

    img_transforms = get_transforms()

    # Cache fixed test set into VRAM once — never reloaded
    test_dir = data_dir / "test"
    if not test_dir.exists():
        print(f"[ERROR] {test_dir} not found. Run split_dataset.py first.")
        return

    print(f"\nCarregando test set na VRAM...")
    test_folder = ImageFolder(root=test_dir)
    test_loader = CachedGPULoader(
        test_folder, device, img_transforms,
        batch_size=args.batch_size, shuffle=False,
        num_load_workers=args.load_workers,
    )
    print(f"  Test set: {test_loader.n} imgs | {len(test_folder.classes)} classes")
    print(f"  Budget  : {args.total_steps:,} steps | Batch {args.batch_size} | "
          f"LR {args.lr} | WD {args.weight_decay}")

    # CSV header
    csv_header = [
        "model_size_pct", "data_subset_pct", "step",
        "train_loss", "test_accuracy", "test_error_rate",
        "test_f1", "test_precision", "elapsed_sec",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_header)

    scaler     = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    total_runs = len(model_sizes) * len(subset_pcts)
    run_num    = 0

    for model_size in model_sizes:
        if model_size not in MODEL_CONFIGS:
            print(f"\n[WARNING] No config for model_size={model_size}%. Skipping.")
            continue

        cfg      = MODEL_CONFIGS[model_size]
        channels = cfg["channels"]
        seed     = seeds_map[model_size]

        print(f"\n{'='*70}")
        print(f"MODEL {model_size}%  (alpha={cfg['alpha']}, seed={seed})")
        print(f"Canais : stem={channels[0]} L1={channels[1]} L2={channels[2]} "
              f"L3={channels[3]} L4={channels[4]}")

        # Load init weights from checkpoint (herança de pesos)
        ckpt_path = Path("checkpoints/init") / f"init_{model_size}pct.pt"
        if not ckpt_path.exists():
            print(f"\n[ERROR] Checkpoint {ckpt_path} não encontrado. Rode init_weights.py primeiro.")
            continue

        ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        init_state = ckpt["state_dict"]

        tmp_model  = ScaledResNet18(channels=channels)
        n_params   = count_parameters(tmp_model)
        del tmp_model
        print(f"Params : {n_params:,}  ({n_params / 11_700_000 * 100:.1f}% of full ResNet-18)")

        for subset_pct in subset_pcts:
            run_num   += 1
            train_dir  = data_dir / f"subset_{subset_pct}%" / "train"

            if not train_dir.exists():
                print(f"\n  [WARNING] {train_dir} not found. Skipping.")
                continue

            print(f"\n  [{run_num}/{total_runs}] subset_{subset_pct}%")

            # Cache train subset into VRAM
            train_folder = ImageFolder(root=train_dir)
            train_loader = CachedGPULoader(
                train_folder, device, img_transforms,
                batch_size=args.batch_size, shuffle=True,
                num_load_workers=args.load_workers,
            )
            bpe = len(train_loader)
            print(f"    {train_loader.n} imgs — {bpe} batches/epoch")

            # Restore fixed init weights
            model = ScaledResNet18(channels=channels)
            model.load_state_dict(init_state)
            model = model.to(device)

            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = get_lr_scheduler(optimizer, args.total_steps, args.warmup_fraction)
            criterion = nn.CrossEntropyLoss()

            model.train()
            data_gen     = infinite_cached(train_loader)
            running_loss = 0.0
            running_n    = 0
            t_start      = time.time()
            best_acc     = 0.0
            no_improve   = 0

            pbar = tqdm(
                range(1, args.total_steps + 1),
                desc=f"  M{model_size:5.1f}% D{subset_pct:3d}%",
                ncols=95,
                leave=False,
            )

            for step in pbar:
                inputs, labels = next(data_gen)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(inputs), labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_n    += inputs.size(0)

                if step % args.log_every == 0 or step == args.total_steps:
                    avg_loss = running_loss / running_n
                    acc, f1, prec = evaluate(model, test_loader, use_amp)
                    elapsed   = time.time() - t_start
                    err_rate  = 1.0 - acc

                    # Best model checkpoint
                    if acc > best_acc + args.min_delta:
                        best_acc   = acc
                        no_improve = 0
                        best_path  = save_dir / f"best_M{model_size}pct_D{subset_pct}pct.pt"
                        torch.save({
                            "state_dict":  model.state_dict(),
                            "acc":         acc,
                            "step":        step,
                            "model_size":  model_size,
                            "subset_pct":  subset_pct,
                            "channels":    channels,
                        }, best_path)
                        improved_tag = f"  [BEST {acc:.4f} salvo]"
                    else:
                        no_improve  += 1
                        improved_tag = f"  [sem melhora {no_improve}/{args.patience}]"

                    pbar.write(
                        f"    step {step:>6}/{args.total_steps}  "
                        f"loss {avg_loss:.4f}  acc {acc:.4f}  "
                        f"f1 {f1:.4f}  err {err_rate:.4f}  {elapsed:.1f}s"
                        f"{improved_tag}"
                    )

                    with open(output_csv, "a", newline="", encoding="utf-8") as csv_f:
                        csv.writer(csv_f).writerow([
                            model_size, subset_pct, step,
                            round(avg_loss, 5), round(acc, 5), round(err_rate, 5),
                            round(f1, 5), round(prec, 5), round(elapsed, 2),
                        ])

                    running_loss = 0.0
                    running_n    = 0
                    model.train()

                    # Early stopping
                    if no_improve >= args.patience:
                        pbar.write(
                            f"    [EARLY STOP] {no_improve} avaliações sem ganho "
                            f">= {args.min_delta:.1%}. Melhor acc: {best_acc:.4f}"
                        )
                        break

            pbar.close()

            # Free VRAM before next run
            train_loader.free()
            del model, optimizer, scheduler

    print(f"\n{'='*70}")
    print(f"[SUCCESS] {run_num}/{total_runs} runs complete.")
    print(f"Resultados : {output_csv.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
