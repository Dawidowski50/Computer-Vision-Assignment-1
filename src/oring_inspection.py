#!/usr/bin/env python3
"""
Computer Vision Assignment 1

Commit 2:
- Manual histogram computation
- Manual Otsu thresholding
- Save binary threshold output
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


# ----------------------------
# Histogram + Otsu threshold
# ----------------------------

def histogram_u8(gray_u8: np.ndarray) -> np.ndarray:
    """Compute 256-bin histogram manually using NumPy."""
    return np.bincount(gray_u8.ravel(), minlength=256).astype(np.int64)


def otsu_threshold(hist: np.ndarray) -> int:
    """Compute Otsu threshold from histogram."""
    hist = hist.astype(np.float64)
    total = hist.sum()

    if total == 0:
        return 127

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan

    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    return int(np.nanargmax(sigma_b2))


def threshold_otsu(gray_u8: np.ndarray):
    hist = histogram_u8(gray_u8)
    t = otsu_threshold(hist)
    binary = (gray_u8 <= t).astype(np.uint8)
    return binary, t


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing O-ring images")
    parser.add_argument("--output", default="outputs", help="Output folder")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return 1

    print(f"Found {len(images)} images in {input_dir}")

    for img_path in images:
        gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Could not read {img_path.name}")
            continue

        binary01, t = threshold_otsu(gray)

        out_path = output_dir / f"{img_path.stem}_binary.png"
        cv.imwrite(str(out_path), (binary01 * 255).astype(np.uint8))

        print(f"{img_path.name}: Otsu T={t}")

    print("Done.") 
    return 0


if __name__ == "__main__":
    raise SystemExit(main())