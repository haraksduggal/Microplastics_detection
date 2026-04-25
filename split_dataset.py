"""
Dataset Splitter — Train / Val / Test
======================================
Splits microplastic (2000) and non-microplastic (2000) into:
  Train: 80%  →  1600 + 1600 = 3200
  Val:   10%  →   200 +  200 =  400
  Test:  10%  →   200 +  200 =  400

Usage:
    1. Set paths below
    2. Run: python split_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

# ============================================================
# CONFIG — UPDATE THESE PATHS
# ============================================================
MICRO_DIR = "/Users/haraksduggal/Downloads/microplastics/micro"
NON_MICRO_DIR = "/Users/haraksduggal/Downloads/microplastics/non-micro"
OUTPUT_DIR = "/Users/haraksduggal/Downloads/microplastics/dataset"

NON_MICRO_SAMPLE = 2000  # randomly pick 2000 from your 5000
SEED = 42

random.seed(SEED)

# ============================================================
# MAIN
# ============================================================

valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def get_images(directory):
    return sorted([f for f in Path(directory).rglob("*") if f.suffix.lower() in valid_ext])

# Get image lists
micro = get_images(MICRO_DIR)
non_micro = get_images(NON_MICRO_DIR)

print(f"Found {len(micro)} microplastic images")
print(f"Found {len(non_micro)} non-microplastic images")

# Sample non-micro down to 2000
if len(non_micro) > NON_MICRO_SAMPLE:
    non_micro = random.sample(non_micro, NON_MICRO_SAMPLE)
    print(f"Sampled {NON_MICRO_SAMPLE} non-microplastic images")

# Shuffle
random.shuffle(micro)
random.shuffle(non_micro)

# Split ratios
def split(files):
    n = len(files)
    t1 = int(n * 0.8)
    t2 = int(n * 0.9)
    return files[:t1], files[t1:t2], files[t2:]

micro_train, micro_val, micro_test = split(micro)
non_train, non_val, non_test = split(non_micro)

# Copy files
splits = {
    "train": {"microplastic": micro_train, "non_microplastic": non_train},
    "val":   {"microplastic": micro_val,   "non_microplastic": non_val},
    "test":  {"microplastic": micro_test,  "non_microplastic": non_test},
}

for split_name, classes in splits.items():
    for cls_name, files in classes.items():
        dest = Path(OUTPUT_DIR) / split_name / cls_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)

# Summary
print(f"\n{'='*45}")
print(f"  DATASET SUMMARY — saved to {OUTPUT_DIR}")
print(f"{'='*45}")
for split_name in ["train", "val", "test"]:
    mp = len(list((Path(OUTPUT_DIR) / split_name / "microplastic").glob("*")))
    nmp = len(list((Path(OUTPUT_DIR) / split_name / "non_microplastic").glob("*")))
    print(f"  {split_name:5s}:  {mp} micro  |  {nmp} non-micro  |  {mp+nmp} total")
print(f"{'='*45}")
print("Done! Ready for training.")
