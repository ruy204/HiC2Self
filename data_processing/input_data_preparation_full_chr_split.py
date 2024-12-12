#!/usr/bin/env python

'''
Script for preprocessing Hi-C contact matrices 
(from 3-column pairs file or Juicer extracted .hic files)

head Extracted_5kb_chr3.txt
0.0     0.0     0.0
0.0     10000.0 0.0
0.0     20000.0 0.0
0.0     30000.0 0.0
0.0     40000.0 0.0

Usage example: 
screen 
chrom='chr3'

python /HiC2Self/data_processing/bulk_retrain_preparation_fullchr_split.py \
--input_dir /data/GM12878/hic_extraction \
--subset_file Extracted_5k_"${chrom}".txt \
--save_dir /data/GM12878/processed \
--save_file Extracted_noisy_mat_5kb_size400_"${chrom}".txt \
--chrom "${chrom}" \
--assembly hg38 \
--resolution 5000 \
--mat_size 400

'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Set-up data preparations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_dir", help="Directory of input data.", 
                    type=str, default = "/data/GM12878/hic_extraction")
parser.add_argument("--subset_file", help="Name of the subsetted pair file.", 
                    type=str, default="Extracted_5k_chr3.txt")
parser.add_argument("--save_dir", help="Directory to save processed file.", 
                    type=str, default="/data/GM12878/processed")
parser.add_argument("--save_file", help="Name of the subsetted pair file.", 
                    type=str, default="Extracted_noisy_mat_5kb_size400_chr3.txt")
parser.add_argument("--assembly_dir", help="Directory of the genome size file.", 
                    type=str, default="/data/assembly")
parser.add_argument("--chrom", help="Which chromasome to prepare the data in.", 
                    type=str, default="chr3")
parser.add_argument("--resolution", help="The targeted resolution for the ultrahigh-res matrix.", 
                    type=int, default=5000)
parser.add_argument("--mat_size", help="Size of each contact map.", 
                    type=int, default=400)
parser.add_argument("--assembly", help="Genome assembly.", 
                    type=str, default='hg38')
parser.add_argument("--crop_step_size", help="Step size to crop the matrices from chromosome.", 
                    type=int, default=1)
parser.add_argument("--mat_size_ratio", help="What proportion of the matrix to crop.", 
                    type=int, default=1)
parser.add_argument("--microc", help="Whether MicroC dataset.", 
                    type=str, default="hic")
args = parser.parse_args()
config = vars(args)
ASSEMBLY = config['assembly']
ASSEMBLY_DIR = config['assembly_dir']
CROP_STEP_SIZE = config['crop_step_size']
MAT_SIZE_RATIO = config['mat_size_ratio']
MICROC = config['microc']
chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in 
                open(f'/data/assembly/{ASSEMBLY}.chrom.sizes').readlines()}

#########################
#     Load Functions    #
#########################

def map_chroms(data_frame, chrom = "chr3", RESOLUTION = 500, size = 200, 
               chrom_len = chrom_len, crop_step_size = CROP_STEP_SIZE, microc = MICROC):
    CHROM_SIZE = chrom_len[chrom] // RESOLUTION + 1
    subset = data_frame #.iloc[:,[0,1,2,3,4]]
    subset.columns = ["loc1", "loc2", "counts"]
    pos1 = subset.loc[:,"loc1"].to_list()
    pos2 = subset.loc[:,"loc2"].to_list()
    counts_list = subset.loc[:,"counts"].to_list()
    
    #a) First half matrix
    cell_mat = np.zeros((int(CHROM_SIZE//2 + size), 
                         int(CHROM_SIZE//2 + size)))
    print("Matrix size fits memory.")
    cell_mat_size = cell_mat.shape[0]
    for row_i in range(subset.shape[0]):
        bin_position1 = int(int(pos1[row_i]) // RESOLUTION)
        bin_position2 = int(int(pos2[row_i]) // RESOLUTION)
        if bin_position1 <= cell_mat_size-1 and bin_position2 <= cell_mat_size-1:
            cell_mat[bin_position1, bin_position2] += counts_list[row_i]
    cell_mat = cell_mat + cell_mat.T - np.diag(np.diag(cell_mat))
    row_size, _ = cell_mat.shape
    crop_mats_lr = []
    # for idx1 in range(0, row_size - size, size):
    for idx1 in range(0, row_size - size, int(crop_step_size)):
        if idx1 % 1000 == 0:
            print(idx1, datetime.now())
        lr_contact = cell_mat[idx1:idx1+size,idx1:idx1+size]
        crop_mats_lr.append(lr_contact)
    del cell_mat
    
    #b) Second half matrix
    cell_mat = np.zeros((int(CHROM_SIZE//2), int(CHROM_SIZE//2)))
    cell_mat_size = cell_mat.shape[0]
    for row_i in range(subset.shape[0]):
        bin_position1 = int(pos1[row_i] // RESOLUTION - CHROM_SIZE//2)
        bin_position2 = int(pos2[row_i] // RESOLUTION - CHROM_SIZE//2)
        if bin_position1 <= cell_mat_size-1 and bin_position2 <= cell_mat_size-1:
            if bin_position1 >= 0 and bin_position2 >= 0:
                cell_mat[bin_position1, bin_position2] += counts_list[row_i]
    cell_mat = cell_mat + cell_mat.T - np.diag(np.diag(cell_mat))
    row_size, _ = cell_mat.shape
    crop_mats_lr2 = []
    # for idx1 in range(0, row_size - size, size):
    for idx1 in range(0, row_size - size, int(crop_step_size)):
        if idx1 % 1000 == 0:
            print(idx1, datetime.now())
        lr_contact = cell_mat[idx1:idx1+size,idx1:idx1+size]
        crop_mats_lr2.append(lr_contact)
    
    #c) Merge submatrices from both half
    crop_mats_lr_total = crop_mats_lr + crop_mats_lr2[1:]

    print("Finished cropping")
    crop_mats_lr_total = np.dstack(crop_mats_lr_total)
    crop_mats_lr_total = np.rollaxis(crop_mats_lr_total, -1)
    print("Finished stacking")
    return crop_mats_lr_total

#########################
#     Script running    #
#########################

def main():

    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt
    import pickle
    import argparse
    from datetime import datetime
    
    args = parser.parse_args()
    config = vars(args)
    print(config)
    PTH = config['input_dir']
    SUBSET_FILE = config['subset_file']
    SAVE_PTH = config['save_dir']
    SAVE_NAME = config['save_file']
    CHROM = config['chrom']
    RESOLUTION = config['resolution']
    MAT_SIZE = config['mat_size']
    ASSEMBLY = config['assembly']
    ASSEMBLY_DIR = config['assembly_dir']
    CROP_STEP_SIZE = config['crop_step_size']
    MICROC = config['microc']

    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in 
                 open(f'{ASSEMBLY_DIR}/{ASSEMBLY}.chrom.sizes').readlines()}

    #1. Load entire chromosome 1
    pair_file = pd.read_csv(f"{PTH}/{SUBSET_FILE}", sep="\t", header=None)
    
    #2. Load barcode 
    lr_list = map_chroms(pair_file, chrom=CHROM, RESOLUTION = RESOLUTION, size = MAT_SIZE,
                         chrom_len = chrom_len, crop_step_size=CROP_STEP_SIZE, microc = MICROC)

    with open(f"{SAVE_PTH}/{SAVE_NAME}", 'wb') as fp:
        pickle.dump(lr_list, fp)
    
if __name__ == '__main__':
    main()










