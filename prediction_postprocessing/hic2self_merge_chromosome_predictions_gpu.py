#!/usr/bin/env python3

'''Script for generating prediction for the entire chromosome. - on gpu
    - After HiC2Self is trained, use this script to generate prediction for the entire chromosome

Input low-coverage data: GM12878_hic001_noisy_mat_5kb_size400_chr3.txt
    - resolution: 5kb
    - data assembly: hg38

Example:
python prediction_postprocessing/hic2self_merge_chromosome_predictions_gpu.py \
--loss_selection NBloss \
--log_version 0.01 \
--log_path outputs/hic2self_logs \
--input_path data/GM12878/processed \
--input_file GM12878_hic001_noisy_mat_5kb_size400_chr3.pkl \
--chrom chr3 \
--resolution 5000 \
--mat_size 400 \
--assembly hg38 \
--chrom_sizes_file data/assembly/hg38.chrom.sizes \
--model_name bulk_retrain \
--layered_no no_layered \
--mask_width 3 \
--mask_type symmetric_grid \
--cell_type GM12878
'''

import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
import argparse, os, sys, random
from datetime import datetime
from pathlib import Path

torch.set_default_tensor_type(torch.DoubleTensor)

# ----------------------------
# Repo-relative imports
# ----------------------------
THIS_DIR = Path(__file__).resolve().parent                         # .../prediction_postprocessing
REPO_ROOT = THIS_DIR.parent                                        # repo root
UTILS_DIR = THIS_DIR / "utils"
TRAIN_UTILS_DIR = REPO_ROOT / "model_training" / "utils"

sys.path.append(str(UTILS_DIR))
sys.path.append(str(TRAIN_UTILS_DIR))

from data_preparation import *
from hic2self_model import *
from results_utils import *
from data_utils_hic2self import *

parser = argparse.ArgumentParser(
    description="Generate full-chromosome prediction and merge cropped windows",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--loss_selection", type=str, default="NBloss")
parser.add_argument("--log_version", type=str, default="0.01")
parser.add_argument("--log_path", type=str, default="outputs/hic2self_logs",
                    help="Base directory containing trained model checkpoints.")

parser.add_argument("--input_path", type=str, default="data/GM12878/processed",
                    help="Directory of the input cropped file (pickled list of matrices).")
parser.add_argument("--input_file", type=str, default="GM12878_hic001_noisy_mat_5kb_size400_chr3.pkl",
                    help="Name of the input cropped file (pickle).")

parser.add_argument("--chrom", type=str, default="chr3")
parser.add_argument("--resolution", type=int, default=5000)
parser.add_argument("--assembly", type=str, default="hg38")

parser.add_argument("--chrom_sizes_file", type=str, default="data/assembly/hg38.chrom.sizes",
                    help="Chrom sizes file path (repo-relative by default).")

parser.add_argument("--model_name", type=str, default="bulk_retrain")
parser.add_argument("--layered_no", type=str, default="no_layered")
parser.add_argument("--mask_width", type=int, default=3)
parser.add_argument("--mask_type", type=str, default="symmetric_grid")
parser.add_argument("--cell_type", type=str, default="GM12878")
parser.add_argument("--mat_size", type=int, default=400)


###################################
#      Helper functions           #
###################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_prediction(i, model, masker, cell_mat_list):
    mat = cell_mat_list[i]
    x = torch.tensor(mat).unsqueeze(0).unsqueeze(0).double().cuda()
    invariant_output = masker.infer_full_image(x, model)
    return invariant_output


def merge_average(start, end, model, masker, cell_mat_list):
    done_list = []
    for i in range(start, end):
        if i == start:
            a0 = generate_prediction(i, model, masker, cell_mat_list).detach().cpu().numpy()[0][0]
        a1 = generate_prediction(i + 1, model, masker, cell_mat_list).detach().cpu().numpy()[0][0]

        done_list.append(a0[0, :])

        avg_mat = (a0[1:, 1:] + a1[:-1, :-1]) / 2
        a1[:-1, :-1] = avg_mat
        a0 = a1

        if i % 100 == 0:
            print(i, datetime.now())
    return done_list


def format_preparation(start, end, model, masker, cell_mat_list,
                       save_loc, save_name_prefix, chrom_chr,
                       mat_size=400, resolution=10000):
    done_list = merge_average(start, end, model, masker, cell_mat_list)
    done_list = [i.tolist() for i in done_list]

    l1 = [[i] * mat_size for i in list(range(start, end, 1))]
    l2 = [[i + j for j in list(range(mat_size))] for i in list(range(start, end, 1))]

    done_long = [j for i in done_list for j in i]
    l1_long = [j for i in l1 for j in i]
    l2_long = [j for i in l2 for j in i]

    del done_list, l1, l2

    hiccupsdf = pd.DataFrame(
        np.array([
            [int(i) for i in l1_long],
            [int(i) for i in l2_long],
            [float(i) for i in done_long]
        ]).T
    )

    hiccupsdf.to_csv(f"{save_loc}/{save_name_prefix}_predictions.txt",
                     index=False, sep="\t", header=False)
    return hiccupsdf


def mat2hic(mat, chrom, save_pth, save_name_prefix, resolution):
    """
    Convert (bin1, bin2, count) to a HiCCUPS-like 9-column format.
    IMPORTANT: keep chrom columns as strings (do NOT cast to int).
    """
    mat.columns = ['bin1', 'bin2', 'count']
    mat_len = mat.shape[0]

    vec0 = [0] * mat_len
    vec1 = [1] * mat_len
    vec_chrom = [chrom] * mat_len

    vec_loc1 = [int(i * resolution) for i in mat['bin1'].to_list()]
    vec_loc2 = [int(i * resolution) for i in mat['bin2'].to_list()]
    vec_value = [float(i * 100) for i in mat['count'].to_list()]

    mergedf = pd.DataFrame(np.array([
        vec0, vec_chrom, vec_loc1, vec0, vec0, vec_chrom, vec_loc2, vec1, vec_value
    ]).T)

    # Cast only numeric columns; keep chrom columns as strings
    for col in [0, 2, 3, 4, 6, 7]:
        mergedf.iloc[:, col] = mergedf.iloc[:, col].astype(int)
    mergedf.iloc[:, 8] = mergedf.iloc[:, 8].astype(float)

    mergedf.to_csv(f'{save_pth}/{save_name_prefix}_predictions_for_hic.txt',
                   sep='\t', index=None, header=None)
    return mergedf


################################
#      Main function           #
################################

def main():
    set_seed(42)

    args = parser.parse_args()
    config = vars(args)
    print(config)

    LOSS_SELECTION = config['loss_selection']
    LOG_VERSION = config['log_version']
    LOG_PATH = str((REPO_ROOT / config['log_path']).resolve())

    PTH = str((REPO_ROOT / config['input_path']).resolve())
    FILE_NAME = config['input_file']

    RESOLUTION = config['resolution']
    CHROM = config['chrom']
    CHROM_CHR = CHROM.replace("chr", "")

    LAYERED = config['layered_no']
    MASK_WIDTH = config['mask_width']
    MASK_TYPE = config['mask_type']
    CELL_TYPE = config['cell_type']
    MAT_SIZE = config['mat_size']
    SAVE_NAME = config['model_name']

    # Chrom sizes (loaded but not used in this script currently; keep for future)
    chrom_sizes_file = (REPO_ROOT / config['chrom_sizes_file']).resolve()
    chrom_len = {item.split()[0]: int(item.strip().split()[1])
                 for item in open(chrom_sizes_file).readlines()}

    os.makedirs(f"{PTH}/../predictions", exist_ok=True)
    SAVE_PTH = f"{PTH}/../predictions"

    PREFIX1 = f"{CELL_TYPE}_predictions_{RESOLUTION}_{CHROM}"
    SAVE_NAME_PREFIX = f"{PREFIX1}_log{LOG_VERSION}_maskwidth{MASK_WIDTH}_{MASK_TYPE}"

    MODEL_SELECTION = "svd"
    MODEL_PATH = f"{LOG_PATH}/{LOG_VERSION}/{SAVE_NAME}_gpu_{LOSS_SELECTION}_{MODEL_SELECTION}_{LAYERED}.pt_model"
    print("Loading model:", MODEL_PATH)

    model = NBGPU().cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    masker = Masker3(width=MASK_WIDTH, mode='zero',
                     mask_type=MASK_TYPE, side_info=False)

    with open(f"{PTH}/{FILE_NAME}", 'rb') as fp:
        cell_mat_list = pickle.load(fp)

    end = len(cell_mat_list) - 100
    print("end:", end)

    hiccupsdf = format_preparation(
        0, end, model, masker, cell_mat_list,
        SAVE_PTH, SAVE_NAME_PREFIX, CHROM_CHR,
        resolution=RESOLUTION, mat_size=MAT_SIZE
    )

    _ = mat2hic(hiccupsdf, CHROM, SAVE_PTH, SAVE_NAME_PREFIX, resolution=RESOLUTION)


if __name__ == '__main__':
    main()
