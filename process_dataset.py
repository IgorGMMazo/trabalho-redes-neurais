#!/usr/bin/env python3
"""
Tomato Leaf Dataset Processor - Computer Vision Data Pipeline
Author: Data Engineer (AI Assistant)
Description: OS-agnostic dataset processing script to split, stratify, and generate
             cumulative subsets for tomato leaf classification.
             Ensures strict class balancing by selecting a fixed number of images per class.
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process Tomato Leaf Dataset with strict stratification and cumulative subsets."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset/orginal/dataset-plantas-com-augmentation",
        help="Path to the directory containing original class folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/processed",
        help="Directory where processed datasets will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=1000,
        help="Number of images to select per class to ensure a perfectly balanced dataset.",
    )
    parser.add_argument(
        "--test_pct",
        type=float,
        default=0.15,
        help="Percentage of the dataset to allocate to the Global Test Set.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only metadata is generated without copying files (useful for validation).",
    )
    return parser.parse_args()


def process_dataset(
    input_dir_str: str,
    output_dir_str: str,
    seed: int,
    max_images_per_class: int,
    test_pct: float,
    dry_run: bool
):
    # Setup paths using pathlib (OS-agnostic)
    input_path = Path(input_dir_str).resolve()
    output_path = Path(output_dir_str).resolve()

    print(f"[*] Input directory: {input_path}")
    print(f"[*] Output directory: {output_path}")
    print(f"[*] Seed: {seed}")
    print(f"[*] Target images per class: {max_images_per_class}")
    print(f"[*] Test Percentage: {test_pct * 100}%")
    if dry_run:
        print("[!] DRY RUN MODE: No files will be copied.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Discover classes (subdirectories containing files)
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {input_path}")

    print(f"[*] Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    # Image extensions to search for
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG")

    # Structure to build metadata and statistics
    metadata: Dict[str, Any] = {
        "config": {
            "seed": seed,
            "max_images_per_class": max_images_per_class,
            "test_pct": test_pct,
            "input_dir": str(input_path),
            "output_dir": str(output_path),
        },
        "statistics": {
            "total_images_processed": 0,
            "classes": {},
            "splits": {
                "global_test": 0,
                "train_fase_1": 0,
                "train_fase_2": 0,
                "fase_1_mais_fase_2": 0,
                "subsets_fase_1": {}
            }
        },
        "file_lineage": {}
    }

    # Cumulative subset percentages to extract from train_fase_1
    cumulative_percentages = [1, 2, 5, 10, 20, 50, 100]
    for pct in cumulative_percentages:
        metadata["statistics"]["splits"]["subsets_fase_1"][f"pct_{pct}"] = 0

    # Main structure of output folders
    global_test_dir = output_path / "global_test"
    train_fase_1_dir = output_path / "train_fase_1"
    train_fase_2_dir = output_path / "train_fase_2"
    fase_1_plus_2_dir = output_path / "fase_1_mais_fase_2"
    subsets_dir = output_path / "subsets_fase_1"

    # Set up random engine with specified seed
    rng = random.Random(seed)

    # Process class by class for strict stratification and balancing
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Find all images
        images = []
        for ext in extensions:
            images.extend(list(class_dir.glob(ext)))
        
        # Deduplicate paths and sort alphabetically for OS-agnostic consistency
        images = sorted(list(set(images)))
        total_found = len(images)
        
        if total_found == 0:
            print(f"[!] Warning: No images found for class '{class_name}'. Skipping.")
            continue
            
        # Shuffle all available images deterministically to select a representative balanced subset
        shuffled_original = list(images)
        rng.shuffle(shuffled_original)

        # Select exactly target count to enforce perfect balancing (e.g. 1000)
        selected_images = shuffled_original[:max_images_per_class]
        actual_class_count = len(selected_images)

        if actual_class_count < max_images_per_class:
            print(f"[!] Warning: Class '{class_name}' only has {actual_class_count} images (requested {max_images_per_class}).")

        print(f"Processing '{class_name}': selected {actual_class_count} images (out of {total_found} available)...")
        
        metadata["statistics"]["classes"][class_name] = {
            "available_count": total_found,
            "selected_count": actual_class_count,
            "splits": {
                "global_test": 0,
                "train_fase_1": 0,
                "train_fase_2": 0,
                "fase_1_mais_fase_2": 0,
                "subsets_fase_1": {}
            }
        }
        for pct in cumulative_percentages:
            metadata["statistics"]["classes"][class_name]["splits"]["subsets_fase_1"][f"pct_{pct}"] = 0

        # Now split the selected balanced subset:
        # 1. Global Test Set (15% of selected images) -> 150 images
        test_size = int(math.ceil(actual_class_count * test_pct))
        test_images = selected_images[:test_size]
        train_pool_images = selected_images[test_size:]

        # 2. Global Train Pool (85%) split into train_fase_1 (50%) and train_fase_2 (50%)
        # For 850 images, this yields exactly 425 images for each phase
        half_pool = len(train_pool_images) // 2
        fase_1_images = train_pool_images[:half_pool]
        fase_2_images = train_pool_images[half_pool:]

        # Log splits info
        metadata["statistics"]["classes"][class_name]["splits"]["global_test"] = len(test_images)
        metadata["statistics"]["classes"][class_name]["splits"]["train_fase_1"] = len(fase_1_images)
        metadata["statistics"]["classes"][class_name]["splits"]["train_fase_2"] = len(fase_2_images)
        metadata["statistics"]["classes"][class_name]["splits"]["fase_1_mais_fase_2"] = len(fase_1_images) + len(fase_2_images)

        metadata["statistics"]["splits"]["global_test"] += len(test_images)
        metadata["statistics"]["splits"]["train_fase_1"] += len(fase_1_images)
        metadata["statistics"]["splits"]["train_fase_2"] += len(fase_2_images)
        metadata["statistics"]["splits"]["fase_1_mais_fase_2"] += len(fase_1_images) + len(fase_2_images)
        metadata["statistics"]["total_images_processed"] += actual_class_count

        # Helper function to safely copy a file to a destination directory
        def copy_image(src_path: Path, dest_dir: Path, rel_copied_paths: List[str]):
            dest_file_path = dest_dir / class_name / src_path.name
            rel_dest_path = dest_file_path.relative_to(output_path).as_posix()
            rel_copied_paths.append(rel_dest_path)
            
            if not dry_run:
                # Ensure destination folder for class exists
                (dest_dir / class_name).mkdir(parents=True, exist_ok=True)
                # Copy file preserving metadata
                shutil.copy2(src_path, dest_file_path)

        # Track destinations for selected files
        for img in selected_images:
            rel_original_path = img.relative_to(input_path.parent.parent).as_posix()
            metadata["file_lineage"][img.name] = {
                "original_path": rel_original_path,
                "class": class_name,
                "destinations": []
            }

        # Copy Global Test Set (Intouchable folder)
        for img in test_images:
            copy_image(img, global_test_dir, metadata["file_lineage"][img.name]["destinations"])
            metadata["file_lineage"][img.name]["split"] = "global_test"

        # Copy train_fase_1
        for img in fase_1_images:
            copy_image(img, train_fase_1_dir, metadata["file_lineage"][img.name]["destinations"])
            metadata["file_lineage"][img.name]["split"] = "train_fase_1"
            # Since train_fase_1 is part of fase_1_mais_fase_2, copy it there too
            copy_image(img, fase_1_plus_2_dir, metadata["file_lineage"][img.name]["destinations"])

        # Copy train_fase_2
        for img in fase_2_images:
            copy_image(img, train_fase_2_dir, metadata["file_lineage"][img.name]["destinations"])
            metadata["file_lineage"][img.name]["split"] = "train_fase_2"
            # Copy to fase_1_mais_fase_2 too
            copy_image(img, fase_1_plus_2_dir, metadata["file_lineage"][img.name]["destinations"])

        # 3. Create Cumulative Subsets over train_fase_1 (1%, 2%, 5%, 10%, 20%, 50%, 100%)
        # Here we follow the Cumulative Inclusion Rule. We slice the already shuffled 'fase_1_images'
        for pct in cumulative_percentages:
            # Calculate size for this percentage using math.ceil or round
            subset_size = int(round(len(fase_1_images) * (pct / 100.0)))
            # Slice the first N images. This guarantees that subset(p_a) is subset of subset(p_b) for p_a < p_b
            subset_images = fase_1_images[:subset_size]
            
            # Log statistic
            metadata["statistics"]["classes"][class_name]["splits"]["subsets_fase_1"][f"pct_{pct}"] = len(subset_images)
            metadata["statistics"]["splits"]["subsets_fase_1"][f"pct_{pct}"] += len(subset_images)
            
            # Directory for this subset
            subset_pct_dir = subsets_dir / f"pct_{pct}"
            
            # Copy files
            for img in subset_images:
                copy_image(img, subset_pct_dir, metadata["file_lineage"][img.name]["destinations"])

    # Write dataset_metadata.json
    metadata_file_path = output_path / "dataset_metadata.json"
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        # Clear out metadata if any left, then write new
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"\n[+] Processing complete! Metadata saved to {metadata_file_path}")
    else:
        print(f"\n[+] Dry run complete! Metadata structure generated successfully.")

    # Print summary statistics
    print("\n" + "="*50)
    print("                DATASET SUMMARY                 ")
    print("="*50)
    print(f"Total processed files: {metadata['statistics']['total_images_processed']}")
    print(f"Global Test Set:       {metadata['statistics']['splits']['global_test']} (15%)")
    print(f"Train Pool (85%):      {int(max_images_per_class * (1 - test_pct)) * len(class_dirs)} total")
    print(f"  - Train Fase 1:      {metadata['statistics']['splits']['train_fase_1']} (50% of pool)")
    print(f"  - Train Fase 2:      {metadata['statistics']['splits']['train_fase_2']} (50% of pool)")
    print(f"Subset do Futuro:      {metadata['statistics']['splits']['fase_1_mais_fase_2']} (Fase 1 + Fase 2)")
    print("\nFase 1 Cumulative Subsets:")
    for pct in cumulative_percentages:
        cnt = metadata['statistics']['splits']['subsets_fase_1'][f"pct_{pct}"]
        print(f"  - Subset {pct:3}%:          {cnt:4} images ({cnt // len(class_dirs)} per class)")
    print("="*50)


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        input_dir_str=args.input_dir,
        output_dir_str=args.output_dir,
        seed=args.seed,
        max_images_per_class=args.max_images_per_class,
        test_pct=args.test_pct,
        dry_run=args.dry_run,
    )
