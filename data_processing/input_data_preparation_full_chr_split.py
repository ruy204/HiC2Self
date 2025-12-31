#!/usr/bin/env python3
"""
Preprocess Hi-C contact matrices into diagonally cropped, symmetric submatrices.

Input format (3 columns, tab-delimited):
    loc1    loc2    counts

Example (head):
    0.0     0.0     0.0
    0.0     10000.0 0.0
    0.0     20000.0 0.0

This script:
1) Reads a chromosome-wide sparse pairs/counts file (3 columns).
2) Builds a dense symmetric contact matrix for the chromosome (in two halves to reduce memory).
3) Crops diagonally along the chromosome into equally sized submatrices.
4) Saves output as a pickled numpy array of shape (N, mat_size, mat_size).

Recommended output extension: .pkl
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def load_chrom_sizes(chrom_sizes_path: Path) -> dict[str, int]:
    chrom_len = {}
    with chrom_sizes_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            chrom, length = line.split()[:2]
            chrom_len[chrom] = int(length)
    return chrom_len


def build_dense_half(
    df: pd.DataFrame,
    chrom_size_bins: int,
    half: int,
    resolution: int,
    mat_size: int,
) -> np.ndarray:
    """
    Build a dense symmetric matrix for either first or second half of the chromosome bins.

    half=0: bins [0 .. chrom_size_bins//2 + mat_size)
    half=1: bins [chrom_size_bins//2 .. chrom_size_bins) mapped to [0 .. chrom_size_bins//2)
    """
    pos1 = df.iloc[:, 0].to_numpy(dtype=np.int64)
    pos2 = df.iloc[:, 1].to_numpy(dtype=np.int64)
    counts = df.iloc[:, 2].to_numpy(dtype=np.float32)

    half_bins = chrom_size_bins // 2

    if half == 0:
        n = half_bins + mat_size
        mat = np.zeros((n, n), dtype=np.float32)
        for p1, p2, c in zip(pos1, pos2, counts):
            i = int(p1 // resolution)
            j = int(p2 // resolution)
            if 0 <= i < n and 0 <= j < n:
                mat[i, j] += c
    else:
        n = half_bins
        mat = np.zeros((n, n), dtype=np.float32)
        for p1, p2, c in zip(pos1, pos2, counts):
            i = int(p1 // resolution) - half_bins
            j = int(p2 // resolution) - half_bins
            if 0 <= i < n and 0 <= j < n:
                mat[i, j] += c

    # Symmetrize
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def crop_diagonal_submatrices(
    mat: np.ndarray,
    mat_size: int,
    crop_step_size: int,
    verbose_every: int = 1000,
) -> list[np.ndarray]:
    """
    Crop symmetric submatrices along the diagonal:
        mat[i:i+mat_size, i:i+mat_size]
    """
    n = mat.shape[0]
    out = []
    for i in range(0, n - mat_size, int(crop_step_size)):
        if verbose_every and i % verbose_every == 0:
            print(f"[crop] i={i} {datetime.now()}")
        sub = mat[i : i + mat_size, i : i + mat_size]
        out.append(sub)
    return out


def preprocess_chromosome(
    pairs_path: Path,
    chrom: str,
    chrom_len: dict[str, int],
    resolution: int,
    mat_size: int,
    crop_step_size: int,
) -> np.ndarray:
    """
    Returns:
        np.ndarray of shape (N, mat_size, mat_size)
    """
    if chrom not in chrom_len:
        raise ValueError(f"Chromosome '{chrom}' not found in chrom sizes file.")

    # Load sparse pairs/counts (3 columns)
    df = pd.read_csv(pairs_path, sep="\t", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Expected 3 columns (loc1, loc2, counts). Got {df.shape[1]} columns.")
    df = df.iloc[:, :3]
    df.columns = ["loc1", "loc2", "counts"]

    chrom_size_bins = chrom_len[chrom] // resolution + 1
    half_bins = chrom_size_bins // 2

    print(f"[info] chrom={chrom} length_bp={chrom_len[chrom]} resolution={resolution}")
    print(f"[info] chrom_size_bins={chrom_size_bins} half_bins={half_bins} mat_size={mat_size}")
    print(f"[info] pairs_path={pairs_path}")

    # (a) first half (+mat_size padding to allow diagonal crops near midpoint)
    print("[build] first half matrix...")
    mat1 = build_dense_half(df, chrom_size_bins, half=0, resolution=resolution, mat_size=mat_size)
    crops1 = crop_diagonal_submatrices(mat1, mat_size=mat_size, crop_step_size=crop_step_size)
    del mat1

    # (b) second half
    print("[build] second half matrix...")
    mat2 = build_dense_half(df, chrom_size_bins, half=1, resolution=resolution, mat_size=mat_size)
    crops2 = crop_diagonal_submatrices(mat2, mat_size=mat_size, crop_step_size=crop_step_size)
    del mat2

    # (c) merge; drop the first crop of second half to avoid duplicated midpoint region
    crops = crops1 + crops2[1:] if len(crops2) > 1 else crops1
    print(f"[done] total_submatrices={len(crops)}")

    # Stack to (N, H, W)
    out = np.stack(crops, axis=0).astype(np.float32)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare diagonally cropped Hi-C submatrices for HiC2Self training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the input sparse pairs file.")
    parser.add_argument("--subset_file", type=str, required=True,
                        help="Input filename (3-column tab-delimited: loc1, loc2, counts).")

    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the processed output.")
    parser.add_argument("--save_file", type=str, required=True,
                        help="Output filename (recommended extension: .pkl).")

    parser.add_argument("--assembly_dir", type=str, default="assembly",
                        help="Directory containing <assembly>.chrom.sizes (relative to repo root is OK).")
    parser.add_argument("--assembly", type=str, default="hg38",
                        help="Genome assembly name (expects <assembly>.chrom.sizes).")

    parser.add_argument("--chrom", type=str, required=True, help="Chromosome name, e.g. chr3.")
    parser.add_argument("--resolution", type=int, default=5000, help="Bin size in bp.")
    parser.add_argument("--mat_size", type=int, default=400, help="Submatrix size (H=W).")
    parser.add_argument("--crop_step_size", type=int, default=1,
                        help="Step size (in bins) to slide along the diagonal when cropping.")

    # kept for backward compatibility; currently not used
    parser.add_argument("--microc", type=str, default="hic",
                        help="Kept for backward compatibility (not used).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    save_dir = Path(args.save_dir)
    assembly_dir = Path(args.assembly_dir)

    pairs_path = input_dir / args.subset_file
    save_dir.mkdir(parents=True, exist_ok=True)

    chrom_sizes_path = assembly_dir / f"{args.assembly}.chrom.sizes"
    if not chrom_sizes_path.exists():
        raise FileNotFoundError(
            f"Could not find chrom sizes file: {chrom_sizes_path}\n"
            f"Set --assembly_dir to the folder containing {args.assembly}.chrom.sizes"
        )

    chrom_len = load_chrom_sizes(chrom_sizes_path)

    out = preprocess_chromosome(
        pairs_path=pairs_path,
        chrom=args.chrom,
        chrom_len=chrom_len,
        resolution=args.resolution,
        mat_size=args.mat_size,
        crop_step_size=args.crop_step_size,
    )

    out_path = save_dir / args.save_file
    with out_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[saved] {out_path}  shape={out.shape}  dtype={out.dtype}")


if __name__ == "__main__":
    main()
