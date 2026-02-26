#!/usr/bin/env python3
"""
Computer Vision Assignment 1

Commit 7:
- Multi-radius angular gap sampling
- Robust break + nick detection
- Final PASS/FAIL logic
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np


# ----------------------------
# Parameters
# ----------------------------

SE_SIZE = 3
N_ANGLES = 720
GAP_BREAK_THR = 0.05
GAP_NICK_THR = 0.015
GAP_RADII_FRACTIONS = (0.15, 0.50, 0.85)
MIN_RING_AREA = 500


# ----------------------------
# Utility
# ----------------------------

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def to_u8(binary01: np.ndarray) -> np.ndarray:
    return (binary01.astype(np.uint8) * 255)


# ----------------------------
# Otsu
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

    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    return int(np.nanargmax(sigma_b2))


def threshold_otsu(gray_u8: np.ndarray):
    t = otsu_threshold(histogram_u8(gray_u8))
    binary01 = (gray_u8 <= t).astype(np.uint8)
    return binary01, t


# ----------------------------
# Morphology
# ----------------------------

def square_se(size: int = 3):
    return np.ones((size, size), dtype=np.uint8)


def dilate01(binary01: np.ndarray, se: np.ndarray):
    h, w = binary01.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary01, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(binary01)

    for i in range(kh):
        for j in range(kw):
            if se[i, j]:
                out = np.maximum(out, padded[i:i + h, j:j + w])
    return out


def erode01(binary01: np.ndarray, se: np.ndarray):
    h, w = binary01.shape
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary01, ((ph, ph), (pw, pw)), mode="constant")
    out = np.ones_like(binary01)

    for i in range(kh):
        for j in range(kw):
            if se[i, j]:
                out = np.minimum(out, padded[i:i + h, j:j + w])
    return out


def close01(binary01: np.ndarray, se: np.ndarray, iterations=2):
    x = binary01
    for _ in range(iterations):
        x = dilate01(x, se)
    for _ in range(iterations):
        x = erode01(x, se)
    return x


# ----------------------------
# Connected Components
# ----------------------------

def ccl_labels(binary01: np.ndarray):
    h, w = binary01.shape
    labels = np.zeros_like(binary01, dtype=np.int32)
    sizes = []
    current = 0
    q = deque()
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for y in range(h):
        for x in range(w):
            if binary01[y,x] and labels[y,x]==0:
                current += 1
                labels[y,x] = current
                q.append((y,x))
                count=0
                while q:
                    cy,cx = q.popleft()
                    count+=1
                    for dy,dx in nbrs:
                        ny,nx = cy+dy,cx+dx
                        if 0<=ny<h and 0<=nx<w:
                            if binary01[ny,nx] and labels[ny,nx]==0:
                                labels[ny,nx]=current
                                q.append((ny,nx))
                sizes.append(count)
    return labels,sizes


def largest_component(binary01):
    labels,sizes=ccl_labels(binary01)
    if not sizes:
        return np.zeros_like(binary01),0
    k=int(np.argmax(sizes))+1
    return (labels==k).astype(np.uint8),sizes[k-1]


# ----------------------------
# Gap Detection
# ----------------------------

def gap_at_radius(ring01, cy, cx, radius):
    h,w=ring01.shape
    angles=np.linspace(0,2*np.pi,N_ANGLES,endpoint=False)
    xs=np.round(cx+radius*np.cos(angles)).astype(int)
    ys=np.round(cy+radius*np.sin(angles)).astype(int)
    valid=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
    samples=ring01[ys[valid],xs[valid]]
    if samples.size==0:
        return 1.0
    missing=(samples==0).astype(int)
    doubled=np.concatenate([missing,missing])
    max_run=run=0
    for v in doubled:
        if v:
            run+=1
            max_run=max(max_run,run)
        else:
            run=0
    return min(max_run,len(missing))/len(missing)


def max_gap_multi_radius(ring01):
    ys,xs=np.nonzero(ring01)
    if xs.size==0:
        return 1.0
    cy,cx=float(ys.mean()),float(xs.mean())
    r=np.sqrt((xs-cx)**2+(ys-cy)**2)
    r_outer=float(np.percentile(r,99))
    r_inner=float(np.percentile(r,5))
    thickness=r_outer-r_inner
    worst=0
    for f in GAP_RADII_FRACTIONS:
        radius=r_inner+f*thickness
        worst=max(worst,gap_at_radius(ring01,cy,cx,radius))
    return worst


# ----------------------------
# Classification
# ----------------------------

@dataclass
class Features:
    ring_area:int
    max_gap_frac:float


def classify(feats:Features):
    if feats.ring_area<MIN_RING_AREA:
        return "FAIL"
    if feats.max_gap_frac>GAP_BREAK_THR:
        return "FAIL"
    if feats.max_gap_frac>GAP_NICK_THR:
        return "FAIL"
    return "PASS"


# ----------------------------
# Main
# ----------------------------

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",default="outputs")
    args=parser.parse_args()

    input_dir=Path(args.input)
    output_dir=Path(args.output)
    output_dir.mkdir(exist_ok=True)

    se=square_se(SE_SIZE)

    for img_path in list_images(input_dir):
        t0=time.perf_counter()
        gray=cv.imread(str(img_path),cv.IMREAD_GRAYSCALE)
        binary,_=threshold_otsu(gray)
        cleaned=close01(binary,se)
        ring,_=largest_component(cleaned)
        gap=max_gap_multi_radius(ring)
        feats=Features(int(ring.sum()),gap)
        decision=classify(feats)
        dt=(time.perf_counter()-t0)*1000

        overlay=np.stack([gray,gray,gray],axis=2)
        overlay[ring.astype(bool)]=[0,0,255]

        lines=[
            img_path.name,
            f"Result: {decision}",
            f"Max gap: {gap:.3f}",
            f"Time: {dt:.2f} ms"
        ]

        y=30
        for line in lines:
            cv.putText(overlay,line,(10,y),
                       cv.FONT_HERSHEY_SIMPLEX,0.6,
                       (255,255,255),2,cv.LINE_AA)
            cv.putText(overlay,line,(10,y),
                       cv.FONT_HERSHEY_SIMPLEX,0.6,
                       (0,0,0),1,cv.LINE_AA)
            y+=22

        cv.imwrite(str(output_dir/f"{img_path.stem}_annotated.png"),overlay)

        print(f"{img_path.name}: {decision} (gap={gap:.3f})")

    print("Done.")


if __name__=="__main__":
    main()