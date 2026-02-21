#!/usr/bin/env python3
"""
Computer Vision Assignment 1

Commit 3:
- Manual histogram computation
- Manual Otsu thresholding
- Manual binary morphology (dilation/erosion + closing)
- Save binary + cleaned outputs
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
    return np.bincount(gray_u8.ravel(), minlength=256).astype(np.int64)


def otsu_threshold(hist: np.ndarray) -> int:
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    return int(np.nanargmax(sigma_b2))


def threshold_otsu(gray_u8: np.ndarray):
    t = otsu_threshold(histogram_u8(gray_u8))
    binary01 = (gray_u8 <= t).astype(np.uint8)  # ring assumed darker than background
    return binary01, t


# ----------------------------
# Binary morphology (from scratch)
# ----------------------------

def square_se(size: int = 3) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("SE size must be odd and >= 1")
    return np.ones((size, size), dtype=np.uint8)


def dilate01(binary01: np.ndarray, se: np.ndarray) -> np.ndarray:
    h, w = binary01.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(binary01, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(kh):
        for j in range(kw):
            if se[i, j] == 0:
                continue
            out = np.maximum(out, padded[i:i + h, j:j + w])

    return out


def erode01(binary01: np.ndarray, se: np.ndarray) -> np.ndarray:
    h, w = binary01.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(binary01, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.ones((h, w), dtype=np.uint8)

    for i in range(kh):
        for j in range(kw):
            if se[i, j] == 0:
                continue
            out = np.minimum(out, padded[i:i + h, j:j + w])

    return out


def close01(binary01: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    x = binary01
    for _ in range(iterations):
        x = dilate01(x, se)
    for _ in range(iterations):
        x = erode01(x, se)
    return x


def to_u8(binary01: np.ndarray) -> np.ndarray:
    return (binary01.astype(np.uint8) * 255)


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

    se = square_se(3)

    for img_path in images:
        gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Could not read {img_path.name}")
            continue

        binary01, t = threshold_otsu(gray)
        cleaned01 = close01(binary01, se, iterations=2)

        cv.imwrite(str(output_dir / f"{img_path.stem}_binary.png"), to_u8(binary01))
        cv.imwrite(str(output_dir / f"{img_path.stem}_cleaned.png"), to_u8(cleaned01))

        print(f"{img_path.name}: Otsu T={t}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())