#!/usr/bin/env python3
"""
Dataset Preprocessing — Tomato Leaf Scaling Laws.
Produces fixed val/ and test/ splits (15% each) plus 7 nested cumulative
train subsets at [1, 2, 5, 10, 20, 50, 100]% of the 70% train pool.
All subsets are stratified per class and strictly aninhados (nested) — §2.1.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split balanced tomato dataset into fixed val/test + nested train subsets."
    )
    p.add_argument("--src-dir",              type=str, default="dataset/orginal/dataset-plantas-com-augmentation")
    p.add_argument("--dest-dir",             type=str, default="dataset/preprocessed")
    p.add_argument("--seed",                 type=int, default=42)
    p.add_argument("--train-ratio",          type=float, default=0.70)
    p.add_argument("--val-ratio",            type=float, default=0.15)
    p.add_argument("--max-images-per-class", type=int, default=1000)
    return p.parse_args()


# ── Dataset scan ─────────────────────────────────────────────────────────────

def scan_classes(src: Path) -> Dict[str, List[Path]]:
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes: Dict[str, List[Path]] = {}
    for d in sorted(src.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            imgs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
            if imgs:
                classes[d.name] = imgs
    return classes


# ── Copy helper ──────────────────────────────────────────────────────────────

def copy_images(images: List[Path], dest: Path) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    for img in images:
        shutil.copy2(img, dest / img.name)
        names.append(img.name)
    return names


# ── Main split logic ──────────────────────────────────────────────────────────

def build_dataset(
    classes: Dict[str, List[Path]],
    dest: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_per_class: int,
    subset_pcts: List[int],
) -> Dict:
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    random.seed(seed)

    # Balance: cap each class at max_per_class, shuffle deterministically
    balanced: Dict[str, List[Path]] = {}
    for cls, imgs in classes.items():
        shuffled = list(imgs)
        random.shuffle(shuffled)
        balanced[cls] = shuffled[:max_per_class]

    # Per-class deterministic split into train_pool / val / test
    pools: Dict[str, Tuple[List[Path], List[Path], List[Path]]] = {}
    for cls, imgs in balanced.items():
        n       = len(imgs)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        pools[cls] = (
            imgs[:n_train],
            imgs[n_train: n_train + n_val],
            imgs[n_train + n_val:],
        )

    metadata: Dict = {
        "seed": seed,
        "splits": {"train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": round(test_ratio, 4)},
        "balanced_per_class": max_per_class,
        "classes": list(balanced.keys()),
        "subset_percentages": subset_pcts,
        "fixed_splits": {},
        "subsets": {},
    }

    # ── val/ and test/ ───────────────────────────────────────────────────────
    for split in ("val", "test"):
        total = 0
        meta_split: Dict = {"total": 0, "per_class": {}}
        print(f"\nCreating {split}/...")
        for cls, (train_pool, val_pool, test_pool) in tqdm(pools.items(), desc=f"  {split}", ncols=80):
            src_pool = val_pool if split == "val" else test_pool
            copy_images(src_pool, dest / split / cls)
            meta_split["per_class"][cls] = len(src_pool)
            total += len(src_pool)
        meta_split["total"] = total
        metadata["fixed_splits"][split] = meta_split
        print(f"  -> {total} images across {len(pools)} classes")

    # ── Nested train subsets ─────────────────────────────────────────────────
    print("\nCreating nested train subsets...")
    for pct in subset_pcts:
        name = f"subset_{pct}%"
        total = 0
        meta_sub: Dict = {"pct": pct, "total": 0, "per_class": {}}
        for cls, (train_pool, _, _) in tqdm(pools.items(), desc=f"  {name}", ncols=80, leave=False):
            k = int(round(len(train_pool) * (pct / 100.0)))
            copy_images(train_pool[:k], dest / name / "train" / cls)
            meta_sub["per_class"][cls] = k
            total += k
        meta_sub["total"] = total
        metadata["subsets"][name] = meta_sub
        per_cls = total // len(pools) if pools else 0
        print(f"  {name}/train -> {total} images ({per_cls}/class)")

    return metadata


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dest_dir)

    subset_pcts = [1, 2, 5, 10, 20, 50, 100]

    print("=" * 60)
    print("TOMATO DATASET — 70/15/15 Split + Nested Subsets")
    print(f"  src  : {src_dir.resolve()}")
    print(f"  dest : {dst_dir.resolve()}")
    print(f"  seed : {args.seed} | max/class : {args.max_images_per_class}")
    print(f"  split: train {args.train_ratio:.0%} / val {args.val_ratio:.0%} / "
          f"test {1 - args.train_ratio - args.val_ratio:.0%}")
    print("=" * 60)

    classes = scan_classes(src_dir)
    print(f"\nFound {len(classes)} classes:")
    for cls, imgs in classes.items():
        capped = min(len(imgs), args.max_images_per_class)
        print(f"  {cls:<45} {len(imgs):>5} imgs  ->  {capped} kept")

    # Remove previous output
    print("\nCleaning old output...")
    for folder in ["val", "test"] + [f"subset_{p}%" for p in subset_pcts]:
        target = dst_dir / folder
        if target.exists():
            shutil.rmtree(target)
            print(f"  removed: {folder}")

    metadata = build_dataset(
        classes=classes,
        dest=dst_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_per_class=args.max_images_per_class,
        subset_pcts=subset_pcts,
    )

    meta_path = dst_dir / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Metadata saved: {meta_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
