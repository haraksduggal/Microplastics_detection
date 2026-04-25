"""
Simple Microplastic Image Augmentation
=======================================
Takes 781 images → produces exactly 2000 total
(781 originals kept as-is + 1219 new augmented images)

Usage:
    1. Set INPUT_DIR and OUTPUT_DIR below
    2. Run: python augment_mp.py
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ============================================================
# CONFIG — UPDATE THESE
# ============================================================
INPUT_DIR = "/Users/haraksduggal/Downloads/microplastics/micro"  # where your 781 original images are
OUTPUT_DIR = "/Users/haraksduggal/Downloads/microplastic_augmented_2000"
TARGET_TOTAL = 2000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def rotate(img):
    return img.rotate(random.choice([90, 180, 270]), fillcolor=(0, 0, 0))

def slight_rotate(img):
    return img.rotate(random.uniform(-25, 25), fillcolor=(0, 0, 0))

def h_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def v_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

def contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))

def saturation(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.4))

def blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

def noise(img):
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, random.uniform(5, 15), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def crop_resize(img):
    w, h = img.size
    f = random.uniform(0.75, 0.95)
    nw, nh = int(w * f), int(h * f)
    left, top = random.randint(0, w - nw), random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.LANCZOS)

def sharpen(img):
    return ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 2.0))

ALL_AUGS = [rotate, slight_rotate, h_flip, v_flip, brightness,
            contrast, saturation, blur, noise, crop_resize, sharpen]


def augment(img):
    """Apply 2-4 random augmentations"""
    for fn in random.sample(ALL_AUGS, random.randint(2, 4)):
        img = fn(img)
    return img


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    originals = [f for f in Path(INPUT_DIR).rglob("*") if f.suffix.lower() in valid_ext]
    print(f"Found {len(originals)} original images")

    needed = TARGET_TOTAL - len(originals)
    if needed <= 0:
        print("Already have enough images!")
        exit()

    print(f"Need to generate {needed} augmented images")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy originals
    print("Copying originals...")
    for i, fp in enumerate(originals):
        img = Image.open(fp).convert("RGB")
        img.save(out / f"original_{i:04d}{fp.suffix.lower()}")

    # Step 2: Generate exactly 'needed' augmented images
    print(f"Generating {needed} augmented images...")
    for i in range(needed):
        source = random.choice(originals)
        img = Image.open(source).convert("RGB")
        aug_img = augment(img)
        aug_img.save(out / f"aug_{i:04d}.png", quality=95)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{needed} done")

    total = len(list(out.glob("*")))
    print(f"\nDone! {total} total images saved to: {out.resolve()}")
