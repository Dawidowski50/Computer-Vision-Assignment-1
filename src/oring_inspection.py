#!/usr/bin/env python3
"""
Computer Vision Assignment 1
O-ring inspection — project scaffold

Commit 1:
- Command-line interface
- Image loading
- Output directory creation
- Save input images to output folder
"""

import argparse
from pathlib import Path

import cv2 as cv


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


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
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {img_path.name}")
            continue

        out_path = output_dir / img_path.name
        cv.imwrite(str(out_path), img)
        print(f"Saved {out_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())