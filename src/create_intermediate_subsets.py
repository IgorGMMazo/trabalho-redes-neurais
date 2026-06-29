#!/usr/bin/env python3
"""
Creates intermediate train subsets [3, 7, 15, 30, 70]% to validate the
meta-model prediction curve. Reproduces the same seed/shuffle from
split_dataset.py so the new subsets are correctly nested in the existing ones.
"""

import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

SRC_DIR         = Path("dataset/orginal/dataset-plantas-com-augmentation")
DEST_DIR        = Path("dataset/preprocessed")
SEED            = 42
MAX_PER_CLASS   = 1000
TRAIN_RATIO     = 0.70
NEW_SUBSET_PCTS = [3, 7, 15, 30, 70]


def main() -> None:
    # ── Scan source (same ordering as split_dataset.py) ──────────────────────
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = {}
    for d in sorted(SRC_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            imgs = sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
            if imgs:
                classes[d.name] = imgs

    print(f"Found {len(classes)} classes in source.")

    # ── Reproduce the exact same shuffle from split_dataset.py ───────────────
    random.seed(SEED)

    train_pools = {}
    for cls, imgs in classes.items():
        shuffled = list(imgs)
        random.shuffle(shuffled)
        capped = shuffled[:MAX_PER_CLASS]
        n_train = int(round(len(capped) * TRAIN_RATIO))
        train_pools[cls] = capped[:n_train]

    # ── Load existing metadata ────────────────────────────────────────────────
    meta_path = DEST_DIR / "dataset_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # ── Create new subsets ────────────────────────────────────────────────────
    print(f"\nCreating {len(NEW_SUBSET_PCTS)} new subsets: {NEW_SUBSET_PCTS}\n")

    for pct in NEW_SUBSET_PCTS:
        name   = f"subset_{pct}%"
        folder = DEST_DIR / name / "train"

        if folder.exists():
            print(f"  {name} already exists — skipping.")
            continue

        total    = 0
        meta_sub = {"pct": pct, "total": 0, "per_class": {}}

        for cls, pool in tqdm(train_pools.items(), desc=f"  {name}", ncols=80, leave=False):
            k        = int(round(len(pool) * (pct / 100.0)))
            dest_cls = folder / cls
            dest_cls.mkdir(parents=True, exist_ok=True)
            for img in pool[:k]:
                shutil.copy2(img, dest_cls / img.name)
            meta_sub["per_class"][cls] = k
            total += k

        meta_sub["total"] = total
        per_cls = total // len(train_pools)
        print(f"  {name}/train -> {total} images ({per_cls}/class)")

    # ── Update metadata ───────────────────────────────────────────────────────
    existing = metadata["configuration"].get("percentages", [1, 2, 5, 10, 20, 50, 100])
    metadata["configuration"]["percentages"] = sorted(set(existing + NEW_SUBSET_PCTS))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] metadata updated: {meta_path}")
    print(f"All subsets now: {metadata['configuration']['percentages']}")


if __name__ == "__main__":
    main()
