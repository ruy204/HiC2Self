#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import math
import gzip
from pathlib import Path

'''
Script for preparing training data for HiC2Self from logFC calculated by HiC-DC+

Repo-friendly usage example:
python prediction_postprocessing/utils/data_preparation.py \
  --diff_loc data/hicdcplus_diff \
  --chrom_len_file data/assembly/hg19.chrom.sizes \
  --save_loc outputs/logFC_data \
  --save_file logFC_matrices.pkl \
  --res 5000
'''

#########################
#     Load functions    #
#########################

def load_chrom_len(chrom_len_file):
    return {item.split()[0]: int(item.strip().split()[1])
            for item in open(chrom_len_file).readlines()}

def df2mat(chrom, df_loc, chrom_len, res=5000, size=200,
           col_bin1=1, col_bin2=2, col_value=4, has_header=True):
    """
    Read HiC-DC+ diff file and convert to cropped [n_blocks, size, size] matrices.

    Default columns assume:
      col_bin1=1, col_bin2=2, col_value=4  (common HiC-DC+ formats)
    """
    file_name = f"diff_resWToverMUT_{chrom}.txt.gz"
    lr_hic_file = f"{df_loc}/{file_name}"

    mat_dim = int(math.ceil(chrom_len[chrom] * 1.0 / res))
    lr_contact_matrix = np.zeros((mat_dim, mat_dim), dtype=float)

    with gzip.open(lr_hic_file, "rt") as f:
        if has_header:
            _ = next(f, None)
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(col_bin1, col_bin2, col_value):
                continue

            try:
                idx1 = int(fields[col_bin1])
                idx2 = int(fields[col_bin2])
                value = float(fields[col_value])
            except ValueError:
                continue

            i = int(idx1 / res)
            j = int(idx2 / res)
            if i < 0 or j < 0 or i >= mat_dim or j >= mat_dim:
                continue

            lr_contact_matrix[i, j] = value
            lr_contact_matrix[j, i] = value  # enforce symmetry directly

    row_size, _ = lr_contact_matrix.shape
    crop_mats_lr = []
    for idx1 in range(0, row_size - size, size):
        lr_contact = lr_contact_matrix[idx1:idx1 + size, idx1:idx1 + size]
        crop_mats_lr.append(lr_contact)

    if len(crop_mats_lr) == 0:
        return np.zeros((0, size, size))

    crop_mats_lr = np.concatenate([item[np.newaxis, :] for item in crop_mats_lr], axis=0)
    return crop_mats_lr


def df2mat_allchr(df_loc, chrom_len, res=5000, size=200,
                  col_bin1=1, col_bin2=2, col_value=4, has_header=True):
    mat_list = []
    for chromosome in range(1, 23):
        chrom = f"chr{chromosome}"
        mat3d = df2mat(chrom, df_loc, chrom_len, res=res, size=size,
                       col_bin1=col_bin1, col_bin2=col_bin2, col_value=col_value,
                       has_header=has_header)
        mat_list.append(mat3d)
    return mat_list


#########################
#     Script running    #
#########################

def main():
    import argparse

    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR.parents[2]  # .../prediction_postprocessing/utils -> repo root

    ap = argparse.ArgumentParser()
    ap.add_argument("--diff_loc", default="data/hicdcplus_diff",
                    help="Directory containing diff_resWToverMUT_chr*.txt.gz (repo-relative by default).")
    ap.add_argument("--chrom_len_file", default="data/assembly/hg19.chrom.sizes",
                    help="Chrom sizes file (repo-relative by default).")
    ap.add_argument("--save_loc", default="outputs/logFC_data",
                    help="Where to save the pickle (repo-relative by default).")
    ap.add_argument("--save_file", default="logFC_matrices.pkl")
    ap.add_argument("--res", type=int, default=5000)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--col_bin1", type=int, default=1)
    ap.add_argument("--col_bin2", type=int, default=2)
    ap.add_argument("--col_value", type=int, default=4)
    ap.add_argument("--no_header", action="store_true")

    args = ap.parse_args()

    DIFF_LOC = str((REPO_ROOT / args.diff_loc).resolve())
    CHROM_LEN_FILE = str((REPO_ROOT / args.chrom_len_file).resolve())
    SAVE_LOC = (REPO_ROOT / args.save_loc).resolve()
    SAVE_LOC.mkdir(parents=True, exist_ok=True)
    SAVE_FILE = args.save_file

    chrom_len = load_chrom_len(CHROM_LEN_FILE)

    all_mats = df2mat_allchr(
        df_loc=DIFF_LOC,
        chrom_len=chrom_len,
        res=args.res,
        size=args.size,
        col_bin1=args.col_bin1,
        col_bin2=args.col_bin2,
        col_value=args.col_value,
        has_header=(not args.no_header),
    )

    with open(f"{SAVE_LOC}/{SAVE_FILE}", 'wb') as fp:
        pickle.dump(all_mats, fp)

    print(f"Saved: {SAVE_LOC}/{SAVE_FILE}")


if __name__ == '__main__':
    main()
