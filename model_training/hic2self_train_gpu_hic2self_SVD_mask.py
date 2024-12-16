#!/usr/bin/env python
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
import argparse, os
from hic2self_train_gpu_hic2self import *
from hic2self_model_SVD_mask import *
from data_utils_hic2self import *
import time

print(torch.__version__)

'''
Script for training HiC2Self on logFC of SETD2 mutation 

Usage example: 
source /home/yangr2/dnabert_environment/bin/activate
mkdir -p /data/hic2self_logs

python /HiC2Self/model_training/hic2self_train_gpu_hic2self_SVD_mask.py \
--input_file_name /data/GM12878/processed/GM12878_hic001_noisy_mat_5kb_size400_chr3.txt \
--log_dir /data/hic2self_logs \
--wandb \
--lr 1e-5 \
--v 0.01 \
--b 1 \
--e 10 \
--mask_width 3 \
--save_name "bulk_retrain" \
--model_selection "svd" \
--loss_selection "NBloss" \
--mask_type symmetric_grid

'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", help="File name of processed matrices", type=str)
    parser.add_argument('--wandb', action='store_true', help='Toggle wandb')
    parser.add_argument("--gpu", help="CUDA ID", default="0")
    parser.add_argument("--b", help="batch size", default=1, type=int)
    parser.add_argument("--e", help="number of epochs", default=15, type=int)
    parser.add_argument("--lr", help="initial learning rate", default="1e-4")
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--mask_width", help="Control mask width", default=3, type=int)
    parser.add_argument("--mask_type", help="Type pf mask", default="diagonal", type=str) #symmetric_grid
    parser.add_argument("--log_dir", help="Load processed input files.", type=str)
    parser.add_argument("--m", help="additional comments", default="")
    parser.add_argument("--loss_selection", help="Which loss function to use for training.",
                        default="NBloss", type=str)
    parser.add_argument("--SSIM_window_size", help="Window size in SSIM loss.",
                        default=11, type=int)
    parser.add_argument("--model_selection", help="Select base, SVD or Attn model.",
                        default="svd", type=str)
    parser.add_argument("--save_name", help="Prefix of input file name, used as output file name",
                        default="hic013", type=str)
    parser.add_argument("--zero_bin_threshold", help="Threshold to filter out very sparse cells",
                        default=0, type=int)
    parser.add_argument("--loss_weights", help="Weights for the low-resolution matrices",
                        default=1, type=float)

    args = parser.parse_args()
    config = vars(args)
    print(config)

    # if args.wandb:
    #     import wandb
    #     wandb.init(project="hic2self")
    if args.wandb:
        import wandb
        my_default_override = '/data/leslie/yangr2/wandb/'
        wandb.init(project="hic2self", dir=os.getenv("WANDB_DIR", my_default_override))

    INPUT_FILE_NAME = config['input_file_name']
    MASK_WIDTH = config['mask_width']
    MASK_TYPE = config['mask_type']
    LEARNING_RATE = float(config['lr'])
    EXPERIMENT_VERSION = config['v']
    LOG_PATH = f'{args.log_dir}/{EXPERIMENT_VERSION}/'
    BATCH_SIZE = config['b']
    NUM_EPOCH = config['e']
    LOSS_SELECTION = config['loss_selection']
    SSIM_WINDOW_SIZE = config['SSIM_window_size']
    MODEL_SELECTION = config['model_selection']
    SAVE_NAME = config['save_name']
    ZERO_BIN_THRESHOLD = config['zero_bin_threshold']
    LOSS_WEIGHTS = config['loss_weights']

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(0)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    # torch.set_default_dtype(torch.double)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = NBChannel(side=False, round=False, include_pi=False).cuda()
    if MODEL_SELECTION == "svd":
        model = NBGPU(side=False, round=False, include_pi=False).cuda()
    elif MODEL_SELECTION == "attn":
        model = NBAttn(side=False, round=False, include_pi=False).cuda()
    elif MODEL_SELECTION == "base":
        model = NBBase(side=False, round=False, include_pi=False).cuda()
    elif MODEL_SELECTION == "vplot":
        model = NBFilter(side=False, round=False, include_pi=False).cuda()
    masker = Masker2(width = MASK_WIDTH, mode='zero', mask_type = MASK_TYPE, side_info = False)

    if args.wandb:
        wandb.watch(model, log='all')
    if os.path.exists(LOG_PATH) == False:
        os.makedirs(LOG_PATH)
    # else:
    #     restore_latest(model, LOG_PATH, ext='.pt_model')
    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.v)
        f.write("\nBatch Size: " + str(args.b))
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nComments: " + args.m)

    if LOSS_SELECTION == "MSE":
        loss_function = nn.MSELoss()
    elif LOSS_SELECTION == "NBloss":
        loss_function = ZINB2()
    elif LOSS_SELECTION == "SSIM":
        loss_function = ZINB2()
    elif LOSS_SELECTION == "combined":
        loss_function = ZINB2()

    optimizer = Adam(model.parameters(), lr=0.0001)

    if SAVE_NAME in ["nagano", "GSE129029", "scHiCAR", "GSE211395",
                     "ultrahigh", "luo", "bulk_retrain", "CD4"]:
        all_image_counts = data_preparation_nagano(INPUT_FILE_NAME)
    else:
        all_image_counts = data_preparation(INPUT_FILE_NAME)

    m = nn.AvgPool2d(2, stride = 2)

    t0 = time.time()
    for epoch in range(int(NUM_EPOCH)):
        lr = np.maximum(LEARNING_RATE * np.power(0.5, (int(epoch / 16))), 1e-6) # learning rate decay
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

        data_loader = DataLoader(all_image_counts, batch_size=BATCH_SIZE,shuffle=True)
        model.train()
        print("Epoch: " + str(epoch) + "/" + str(NUM_EPOCH - 1))

        for i, batch in enumerate(data_loader):
            idx, train_image = batch
            # train_image = train_image.double()
            train_image = train_image.to(torch.float32)
            if np.count_nonzero(train_image) <= ZERO_BIN_THRESHOLD:
                continue
            net_input, mask = masker.mask(train_image, i)
            _, mask1 = masker.mask(m(train_image).clone(), i)
            _, mask2 = masker.mask(m(m(train_image)).clone(), i)
            _, mask3 = masker.mask(m(m(m(train_image))).clone(), i)
            train_image = torch.tensor(train_image).cuda()
            net_input = torch.tensor(net_input).cuda()
            mask = torch.tensor(mask).cuda()
            
            net_output = model(train_image, mask)
            net_pred = net_output[[0],[0],:,:].unsqueeze(0)
            net_theta = net_output[[0],[1],:,:].unsqueeze(0)
            
            if LOSS_SELECTION == "MSE":
                loss = torch.sum(loss_function(train_image*mask,net_pred*mask)).cuda()
            elif LOSS_SELECTION == "NBloss":
                loss = torch.sum(loss_function.nbloss(train_image*mask,net_pred*mask,net_theta*mask)).cuda()
            elif LOSS_SELECTION == "SSIM":
                loss = torch.sum(loss_function.ssim(train_image*mask, net_pred*mask, SSIM_WINDOW_SIZE)).cuda()
            elif LOSS_SELECTION == "combined":
                loss1 = torch.mean(loss_function.nbloss(train_image*mask,net_pred*mask,net_theta*mask)).cuda()
                lossa = loss1 # + 0.5*loss2*LOSS_WEIGHTS + 0.25*loss3*LOSS_WEIGHTS + 0.125*loss4*LOSS_WEIGHTS
                loss1 = torch.mean(loss_function.ssim(train_image*mask,net_pred*mask, window_size = SSIM_WINDOW_SIZE)).cuda()
                lossb = loss1 # + 0.5*loss2*LOSS_WEIGHTS + 0.25*loss3*LOSS_WEIGHTS + 0.125*loss4*LOSS_WEIGHTS
                loss = lossa + lossb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({LOSS_SELECTION: loss.item()})
                if LOSS_SELECTION == "combined":
                    wandb.log({"NBloss": lossa.item()})
                    wandb.log({"SSIM": lossb.item()})
            
            if i % 50 == 0:
                torch.save(model.state_dict(),f"{LOG_PATH}/{SAVE_NAME}_gpu_{LOSS_SELECTION}_{MODEL_SELECTION}_no_layered.pt_model")
    t1 = time.time()
    print(t1 - t0)

if __name__ == '__main__':
    main()