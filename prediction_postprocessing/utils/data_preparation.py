#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle
import math
import gzip

'''
Script for preparing training data for HiC2Self from logFC calculated by HiC-DC+

Usage example: 
screen
bsub -n 2 -W 1:00 -R 'span[hosts=1] rusage[mem=128]' -Is /bin/bash
source /home/yangr2/dnabert_environment/bin/activate
python /lila/data/leslie/yangr2/setd2/train_hic2self/scripts/data_preparation.py
'''

DIFF_LOC = "/data/leslie/yangr2/setd2/hicdcplus/foldchange/diff_analysis_MUT_vs_WT_5000"
CHROM_LEN_FILE = "/home/yangr2/hg19.chrom.sizes"
chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f"{CHROM_LEN_FILE}").readlines()}

#########################
#     Load functions    #
#########################

def df2mat(chrom, df_loc = DIFF_LOC, chrom_len_file = chrom_len, res = 5000):
    size = 200
    file_name = f"diff_resWToverMUT_{chrom}.txt.gz"
    mat_dim = int(math.ceil(chrom_len[chrom]*1.0/res))
    lr_contact_matrix = np.zeros((mat_dim,mat_dim))
    lr_hic_file = f"{DIFF_LOC}/{file_name}"
    for line in gzip.open(lr_hic_file).readlines()[1:]:
        line = line.decode('utf8')
        if len(line.strip().split('\t')) == 3:
            idx1, idx2, value = int(line.strip().split('\t')[1]),
            int(line.strip().split('\t')[2]),
            float(line.strip().split('\t')[4])
            lr_contact_matrix[int(idx1/res)][int(idx2/res)] = value
    lr_contact_matrix += lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
    row_size, _ = lr_contact_matrix.shape
    crop_mats_lr = []
    for idx1 in range(0, row_size - size, size):
        lr_contact = lr_contact_matrix[idx1:idx1+size,idx1:idx1+size]
        crop_mats_lr.append(lr_contact)
    crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)
    return crop_mats_lr

def df2mat_allchr(df_loc = DIFF_LOC, chrom_len_file = chrom_len, res = 5000):
    mat_list = []
    for chromosome in range(1,23):
        chrom = f"chr{chromosome}"
        mat3d = df2mat(chrom, df_loc, chrom_len, res)
        mat_list.append(mat3d)
    return mat_list


#########################
#     Script running    #
#########################

def main():
    DIFF_LOC = "/data/leslie/yangr2/setd2/hicdcplus/foldchange/diff_analysis_MUT_vs_WT_5000"
    CHROM_LEN_FILE = "/home/yangr2/hg19.chrom.sizes"
    SAVE_LOC = "/data/leslie/yangr2/setd2/train_hic2self/logFC_data"
    SAVE_FILE = "logFC_matrices.pkl"

    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f"{CHROM_LEN_FILE}").readlines()}
    all_mats = df2mat_allchr(df_loc = DIFF_LOC, chrom_len_file = chrom_len, res = 5000)
    with open(f"{SAVE_LOC}/{SAVE_FILE}", 'wb') as fp:
        pickle.dump(all_mats, fp)
    
if __name__ == '__main__':
    main()