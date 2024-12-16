#!/usr/bin/env python

'''Script for generating prediction for the entire chromosome. - on gpu
    - After HiC2Self is trained, use this script to generate prediction for the entire chromosome

Input low-coverage data: GM12878_hic001_noisy_mat_5kb_size400_chr3.txt
    - resolution: 5kb
    - data assembly: hg38

python /HiC2Self/prediction_postprocessing/hic2self_merge_chromosome_predictions_gpu.py \
--loss_selection NBloss \
--log_version 0.01 \
--input_path /data/GM12878/processed \
--input_file GM12878_hic001_noisy_mat_5kb_size400_chr3.txt \
--chrom chr3 \
--resolution 5000 \
--mat_size 400 \
--assembly hg38 \
--model_name 'bulk_retrain' \
--layered_no 'no_layered' \
--mask_width 3 \
--mask_type symmetric_grid \
--cell_type GM12878 
'''

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle
import pandas as pd
import argparse, os, sys, random
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
from datetime import datetime
from sklearn.decomposition import TruncatedSVD

sys.path.append('/HiC2Self/prediction_postprocessing/utils')
from data_preparation import *
from hic2self_model import *
from results_utils import *
sys.path.append('/HiC2Self/model_training/utils')
from data_utils_hic2self import *
chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('/home/yangr2/assembly/hg38.chrom.sizes').readlines()}

parser = argparse.ArgumentParser(description="Set-up data preparations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--loss_selection", help="Loss function used for training the model.", 
                    type=str, default = "NBloss")
parser.add_argument("--log_version", help="Index of train weight to use.", 
                    type=str, default = "0.01")
parser.add_argument("--input_path", help="Directory of the input cropped file.", 
                    type=str, default = "/data/GM12878/processed")
parser.add_argument("--input_file", help="Name of the input cropped file.", 
                    type=str, default = "GM12878_hic001_noisy_mat_5kb_size400_chr3.txt")
parser.add_argument("--chrom", help="Chromosome of the file to be predicted.", 
                    type=str, default = "chr3")
parser.add_argument("--resolution", help="Resolution of the Hi-C data.", 
                    type=int, default = 5000)
parser.add_argument("--assembly", help="Genome assembly.", 
                    type=str, default='hg38')
parser.add_argument("--model_name", help="Name prefix (save_name) of the model during training.", 
                    type=str, default='bulk_retrain')
parser.add_argument("--layered_no", help="End location for the predictions. (no_layered, layered)", 
                    type=str, default = 'no_layered')
parser.add_argument("--mask_width", help="Width of the mask.", 
                    type=int, default = 3)
parser.add_argument("--mask_type", help="Type of the mask.", 
                    type=str, default = 'symmetric_grid')
parser.add_argument("--cell_type", help="Cell type of the prediction.", 
                    type=str, default = 'GM12878')
parser.add_argument("--mat_size", help="Size of the input matrices.", 
                    type=int, default = 400)

args = parser.parse_args()
config = vars(args)
ASSEMBLY = config['assembly']
MAT_SIZE = config['mat_size']
chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in 
                open(f'/home/yangr2/assembly/{ASSEMBLY}.chrom.sizes').readlines()}

###################################
#      Load helper functions      #
###################################

def generate_prediction(i, model, masker, cell_mat_list):
    mat = cell_mat_list[i]
    x = torch.tensor(mat).unsqueeze(0).unsqueeze(0).double().cuda()
    invariant_output = masker.infer_full_image(x, model)
    return invariant_output

def merge_average(start, end, model, masker, cell_mat_list):
    done_list = []
    for i in range(start, end):
        if i == start:
            a0 = generate_prediction(i, model, masker, cell_mat_list).detach().numpy()[0][0]
        a1 = generate_prediction(i + 1, model, masker, cell_mat_list).detach().numpy()[0][0]
        done_list.append(a0[0,:])
        avg_mat = (a0[1:,1:] + a1[:(-1),:(-1)]) / 2
        a1[:(-1),:(-1)] = avg_mat
        a0 = a1
        del a1
        if i % 100 == 0:
            print(i, datetime.now())
    return done_list

def format_preparation(start, end, model, masker, cell_mat_list,
                       save_loc, save_name_prefix, chrom_chr, 
                       mat_size = 400, resolution = 10000):
    done_list = merge_average(start, end, model, masker, cell_mat_list)
    done_list = [i.tolist() for i in done_list]
    l1 = [[i]*mat_size for i in list(range(start,end,1))]
    l2 = [[i+j for j in list(range(mat_size))] for i in list(range(start,end,1))]
    done_long = [j for i in done_list for j in i]
    l1_long = [j for i in l1 for j in i]
    l2_long = [j for i in l2 for j in i]
    md_list = [i*resolution for i in list(set(l1_long))]
    del done_list, l1, l2
    hiccupsdf = pd.DataFrame(np.array([[int(i) for i in l1_long], 
                                      [int(i) for i in l2_long], 
                                      [float(i) for i in done_long]]).T)
    hiccupsdf.to_csv(f"{save_loc}/{save_name_prefix}_predictions.txt", index=False, sep="\t", header=False)
    return hiccupsdf

def mat2hic(mat,chrom,save_pth,save_name_prefix,resolution):
    mat.columns = ['bin1','bin2','count']
    mat_len = mat.shape[0]
    vec0 = [int(0)] * mat_len
    vec1 = [int(1)] * mat_len
    vec_chrom = [chrom] * mat_len
    vec_loc1 = [int(i*resolution) for i in mat.loc[:,'bin1'].to_list()]
    vec_loc2 = [int(i*resolution) for i in mat.loc[:,'bin2'].to_list()]
    vec_value = [float(i*100) for i in mat.loc[:,'count'].to_list()]
    mergedf = pd.DataFrame(np.array([vec0,vec_chrom,vec_loc1,vec0,vec0,vec_chrom,vec_loc2,vec1,vec_value]).T)
    mergedf.iloc[:,0] = [int(i) for i in mergedf.iloc[:,0].to_list()]
    mergedf.iloc[:,1] = [int(i) for i in mergedf.iloc[:,1].to_list()]
    mergedf.iloc[:,2] = [int(i) for i in mergedf.iloc[:,2].to_list()]
    mergedf.iloc[:,3] = [int(i) for i in mergedf.iloc[:,3].to_list()]
    mergedf.iloc[:,4] = [int(i) for i in mergedf.iloc[:,4].to_list()]
    mergedf.iloc[:,5] = [int(i) for i in mergedf.iloc[:,5].to_list()]
    mergedf.iloc[:,6] = [int(i) for i in mergedf.iloc[:,6].to_list()]
    mergedf.iloc[:,7] = [int(i) for i in mergedf.iloc[:,7].to_list()]
    mergedf.to_csv(f'{save_pth}/{save_name_prefix}_predictions_for_hic.txt', sep='\t', index=None, header=None)
    return mergedf

################################
#      Load main function      #
################################

def main():

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Make sure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disable auto-tuning for deterministic results
    
    # Set the seed
    set_seed(42)

    #0. Define arguments
    args = parser.parse_args()
    config = vars(args)
    print(config)

    LOSS_SELECTION = config['loss_selection']
    LOG_PATH = '/data/hic2self_logs'
    LOG_VERSION = config['log_version']
    PTH = config['input_path']
    FILE_NAME = config['input_file']
    RESOLUTION = config['resolution']
    CHROM = config['chrom']
    CHROM_CHR = str.replace(CHROM, "chr", "")
    LAYERED = config['layered_no']
    MASK_WIDTH = config['mask_width']
    MASK_TYPE = config['mask_type']
    CELL_TYPE = config['cell_type']
    MAT_SIZE = config['mat_size']
    SAVE_NAME = config['model_name']
    os.makedirs(f"{PTH}/../predictions", exist_ok=True)
    SAVE_PTH = f"{PTH}/../predictions"

    #1. Prepare save name
    PREFIX1 = f"{CELL_TYPE}_predictions_{RESOLUTION}_{CHROM}"
    SAVE_NAME_PREFIX = f"{PREFIX1}_log{LOG_VERSION}_maskwidth{MASK_WIDTH}_{MASK_TYPE}"

    #2. Load corresponding model
    MODEL_SELECTION = "svd"
    MODEL_PATH = f"{LOG_PATH}/{LOG_VERSION}/{SAVE_NAME}_gpu_{LOSS_SELECTION}_{MODEL_SELECTION}_{LAYERED}.pt_model"
    print(f"{LOG_VERSION}/{SAVE_NAME}_gpu_{LOSS_SELECTION}_{MODEL_SELECTION}_{LAYERED}.pt_model")
    model = NBGPU().cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
    masker = Masker3(width = MASK_WIDTH, mode = 'zero',
                     mask_type = MASK_TYPE, side_info = False)

    #3. Load input data 
    with open(f"{PTH}/{FILE_NAME}", 'rb') as fp:
        cell_mat_list = pickle.load(fp)
    end = len(cell_mat_list) - 100
    print(end)

    #4. Make prediction and save into certain format
    hiccupsdf = format_preparation(0, end, model, masker, cell_mat_list,
                           SAVE_PTH, SAVE_NAME_PREFIX, CHROM_CHR, 
                           resolution=RESOLUTION, mat_size = MAT_SIZE)
    mergedf = mat2hic(hiccupsdf, CHROM, SAVE_PTH, SAVE_NAME_PREFIX,resolution=RESOLUTION)

if __name__ == '__main__':
    main()













