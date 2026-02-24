#!/usr/bin/env python3
"""
Computer Vision Assignment 1

Commit 5:
- Manual histogram + Otsu threshold
- Manual binary morphology (closing)
- Manual connected component labelling (BFS)
- Extract largest component as O-ring
- Fill holes (flood-fill background from borders)
- Compute defect-hole pixels excluding the central hole (largest enclosed void)
- Save binary, cleaned, ring, filled masks
"""

import argparse
from collections import deque
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
    binary01 = (gray_u8 <= t).astype(np.uint8)
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
# Connected Component Labelling (BFS)
# ----------------------------

def ccl_labels(binary01: np.ndarray, connectivity: int = 8):
    h, w = binary01.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = []
    current = 0

    if connectivity == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
    elif connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    q = deque()

    for y in range(h):
        for x in range(w):
            if binary01[y, x] == 1 and labels[y, x] == 0:
                current += 1
                labels[y, x] = current
                q.append((y, x))
                count = 0

                while q:
                    cy, cx = q.popleft()
                    count += 1
                    for dy, dx in nbrs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary01[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                q.append((ny, nx))

                sizes.append(count)

    return labels, sizes


def largest_component(binary01: np.ndarray):
    labels, sizes = ccl_labels(binary01, connectivity=8)
    if not sizes:
        return np.zeros_like(binary01, dtype=np.uint8), 0
    k = int(np.argmax(np.asarray(sizes))) + 1
    mask01 = (labels == k).astype(np.uint8)
    return mask01, sizes[k - 1]


# ----------------------------
# Hole filling (from scratch)
# ----------------------------

def fill_holes01(binary01: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground mask by flood-filling background from borders.
    Background not reachable from border is a hole.
    """
    h, w = binary01.shape
    bg = (binary01 == 0)
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    # Seed border background pixels
    for x in range(w):
        if bg[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if bg[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if bg[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if bg[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        y, x = q.popleft()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if bg[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    holes = bg & (~visited)
    filled = binary01.copy()
    filled[holes] = 1
    return filled


# ----------------------------
# Defect holes excluding central hole
# ----------------------------

def defect_hole_area_excluding_central(ring01: np.ndarray, filled01: np.ndarray) -> int:
    """
    voidmask = filled - ring contains:
      - the expected central hole (largest void component)
      - any smaller enclosed voids (defect holes)
    Defect hole area = total void area - largest void area
    """
    voidmask = ((filled01 == 1) & (ring01 == 0)).astype(np.uint8)
    _, sizes = ccl_labels(voidmask, connectivity=8)
    if not sizes:
        return 0
    total_void = int(np.sum(sizes))
    largest_void = int(np.max(sizes))
    return max(0, total_void - largest_void)


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

        ring01, ring_size = largest_component(cleaned01)
        filled01 = fill_holes01(ring01)

        defect_holes = defect_hole_area_excluding_central(ring01, filled01)

        cv.imwrite(str(output_dir / f"{img_path.stem}_binary.png"), to_u8(binary01))
        cv.imwrite(str(output_dir / f"{img_path.stem}_cleaned.png"), to_u8(cleaned01))
        cv.imwrite(str(output_dir / f"{img_path.stem}_ring.png"), to_u8(ring01))
        cv.imwrite(str(output_dir / f"{img_path.stem}_filled.png"), to_u8(filled01))

        print(f"{img_path.name}: T={t}, ring_size={ring_size}, defect_holes={defect_holes}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())