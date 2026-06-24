#!/usr/bin/env python3
"""
Dataset Preprocessing Script for Tomato Leaf Disease Classification.
Author: Data Engineer (Computer Vision)
Description: Generates cumulative subsets [1%, 2%, 5%, 10%, 20%, 50%, 100%]
             of the original dataset, capping each class to a balanced limit
             (default 1000 images), and splitting each subset into exactly
             85% train and 15% test, while ensuring cumulative inclusion.
             Saves a tracking metadata file (dataset_metadata.json).
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process a dataset of image classes into balanced, cumulative subsets."
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default="dataset/orginal/dataset-plantas-com-augmentation",
        help="Path to the source dataset containing class folders.",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="dataset/preprocessed",
        help="Path to the output folder where subsets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Ratio of images to allocate for the training set (default: 0.85).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=1000,
        help="Maximum images per class to select (default: 1000) to balance the dataset.",
    )
    return parser.parse_args()


def scan_dataset(src_path: Path) -> Dict[str, List[Path]]:
    """
    Scans the source directory for class subdirectories and lists all image files.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory '{src_path}' does not exist.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dataset_structure = {}

    for class_dir in src_path.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith("."):
            images = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            images.sort()  # Sort to ensure deterministic behavior across platforms
            if images:
                dataset_structure[class_dir.name] = images

    return dataset_structure


def generate_subsets(
    dataset_structure: Dict[str, List[Path]],
    dest_path: Path,
    seed: int,
    train_ratio: float,
    percentages: List[int],
    src_path: Path,
    max_images_per_class: int,
) -> Dict:
    """
    Splits the dataset classes into balanced pools, calculates cumulative subset sizes,
    copies the images, and builds the metadata dictionary.
    """
    random.seed(seed)
    
    # Pre-shuffle and cap classes to create the balanced base dataset
    balanced_dataset = {}
    for class_name, images in dataset_structure.items():
        shuffled = list(images)
        random.shuffle(shuffled)
        # Cap to max_images_per_class
        balanced_dataset[class_name] = shuffled[:max_images_per_class]

    metadata = {
        "original_dataset": {
            "path": str(src_path.resolve().as_posix()),
            "total_images": sum(len(imgs) for imgs in dataset_structure.values()),
            "classes": {c: len(imgs) for c, imgs in dataset_structure.items()}
        },
        "balanced_base": {
            "max_images_per_class": max_images_per_class,
            "total_images": sum(len(imgs) for imgs in balanced_dataset.values()),
            "classes": {c: len(imgs) for c, imgs in balanced_dataset.items()}
        },
        "configuration": {
            "seed": seed,
            "train_ratio": train_ratio,
            "test_ratio": round(1.0 - train_ratio, 2),
            "percentages": percentages
        },
        "subsets": {}
    }

    # Split the balanced base into fixed train/test pools per class
    pools = {}
    for class_name, images in balanced_dataset.items():
        n_total = len(images)
        n_train = int(round(n_total * train_ratio))
        
        train_pool = images[:n_train]
        test_pool = images[n_train:]
        pools[class_name] = (train_pool, test_pool)

    # Generate each subset in the list
    for p in percentages:
        subset_name = f"subset_{p}%"
        print(f"\nProcessing {subset_name}...")
        
        subset_meta = {
            "percentage": float(p),
            "summary": {
                "total_train": 0,
                "total_test": 0,
                "total_images": 0,
                "classes": {}
            },
            "files": {
                "train": {},
                "test": {}
            }
        }
        
        for class_name, (train_pool, test_pool) in pools.items():
            n_class_total = len(balanced_dataset[class_name])
            
            # Determine total images for this class in this subset percentage
            target_class_total = int(round(n_class_total * (p / 100.0)))
            
            # Train/test sizes for this subset (maintaining the strict ratio)
            target_train = int(round(target_class_total * train_ratio))
            target_test = target_class_total - target_train
            
            subset_train = train_pool[:target_train]
            subset_test = test_pool[:target_test]
            
            # Define destination paths
            train_dest_dir = dest_path / subset_name / "train" / class_name
            test_dest_dir = dest_path / subset_name / "test" / class_name
            
            train_dest_dir.mkdir(parents=True, exist_ok=True)
            test_dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy train images and track them
            train_rel_paths = []
            for img_path in subset_train:
                shutil.copy2(img_path, train_dest_dir / img_path.name)
                train_rel_paths.append(f"{class_name}/{img_path.name}")
                
            # Copy test images and track them
            test_rel_paths = []
            for img_path in subset_test:
                shutil.copy2(img_path, test_dest_dir / img_path.name)
                test_rel_paths.append(f"{class_name}/{img_path.name}")
                
            # Update class level metadata
            subset_meta["summary"]["classes"][class_name] = {
                "train": len(subset_train),
                "test": len(subset_test),
                "total": len(subset_train) + len(subset_test)
            }
            
            subset_meta["files"]["train"][class_name] = train_rel_paths
            subset_meta["files"]["test"][class_name] = test_rel_paths
            
            # Update subset level totals
            subset_meta["summary"]["total_train"] += len(subset_train)
            subset_meta["summary"]["total_test"] += len(subset_test)
            subset_meta["summary"]["total_images"] += (len(subset_train) + len(subset_test))
            
        metadata["subsets"][subset_name] = subset_meta
        print(f"  Saved {subset_meta['summary']['total_train']} train images, {subset_meta['summary']['total_test']} test images (Total: {subset_meta['summary']['total_images']}).")
        
    return metadata


def main():
    args = parse_args()
    
    src_dir = Path(args.src_dir)
    dest_dir = Path(args.dest_dir)
    
    print("Tomato Leaf Dataset Prep Script (Balanced Edition)")
    print(f"Source Directory: {src_dir.resolve()}")
    print(f"Destination Directory: {dest_dir.resolve()}")
    print(f"Seed: {args.seed} | Train Ratio: {args.train_ratio}")
    print(f"Max images per class (Balanced base): {args.max_images_per_class}")
    
    # Define rigid logarithmic percentage scale
    percentages = [1, 2, 5, 10, 20, 50, 100]
    
    try:
        print("\nScanning source dataset classes...")
        dataset_structure = scan_dataset(src_dir)
        print(f"Found {len(dataset_structure)} classes:")
        for cls_name, files in dataset_structure.items():
            print(f"  - {cls_name}: {len(files)} images (will be balanced to {min(len(files), args.max_images_per_class)})")
            
        if len(dataset_structure) == 0:
            print("[ERROR] No image classes found. Check --src-dir.")
            return
            
        print("\nChecking destination directory...")
        for p in percentages:
            subset_folder = dest_dir / f"subset_{p}%"
            if subset_folder.exists():
                print(f"Removing old subset folder: {subset_folder.name}")
                shutil.rmtree(subset_folder)
                
        # Generate the preprocessed subsets
        metadata = generate_subsets(
            dataset_structure=dataset_structure,
            dest_path=dest_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            percentages=percentages,
            src_path=src_dir,
            max_images_per_class=args.max_images_per_class
        )
        
        # Save metadata to json
        metadata_file = dest_dir / "dataset_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"\n[SUCCESS] Preprocessing completed! Metadata saved to {metadata_file.resolve()}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
