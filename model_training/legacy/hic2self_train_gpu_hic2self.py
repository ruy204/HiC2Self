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
from model_training.legacy.hic2self_train_gpu_hic2self import *
from hic2self_model_SVD_mask import *
from data_utils_hic2self import *
import time

print(torch.__version__)

'''
NOT MAINTAINED - just for reference

Usage example: 
screen
bsub -q gpuqueue -W 15:00 -gpu "num=1:mps=yes" -n 2 -R 'span[hosts=1] rusage[mem=128]' -Is /bin/bash
source /home/yangr2/dnabert_environment/bin/activate

python /data/leslie/yangr2/hic2self/scripts/hic2self_train_gpu_hic2self.py \
--input_file_name /data/leslie/yangr2/hic2self/data/hic013/processed_data/hic013_matrices_shape40.pkl \
--log_dir /data/leslie/yangr2/hic2self/hic2self_logs \
--wandb \
--lr 5e-6 \
--v 0.1 \
--b 1 \
--e 15 \
--mask_width 5 \
--model_selection "svd" \
--loss_selection "NBloss" \
--SSIM_window_size 20

python /data/leslie/yangr2/hic2self/scripts/hic2self_train_gpu_hic2self.py \
--input_file_name /data/leslie/yangr2/hic2self/data/hic013/processed_data/hic013_matrices_1kb_shape100.pkl \
--log_dir /data/leslie/yangr2/hic2self/hic2self_logs \
--wandb \
--lr 5e-6 \
--v 10.0 \
--b 1 \
--e 15 \
--mask_width 5 \
--model_selection "svd" \
--loss_selection "NBloss" \
--SSIM_window_size 20

----------- V-plot training -----------

python /data/leslie/yangr2/hic2self/scripts/hic2self_train_gpu_hic2self.py \
--input_file_name /data/leslie/yangr2/hic2self/data/vplot/processed_data/Jul22_GSE81806_200shape_ricc.pkl \
--log_dir /data/leslie/yangr2/hic2self/hic2self_logs \
--wandb \
--lr 5e-6 \
--v 7.1 \
--b 1 \
--e 15 \
--mask_width 5 \
--save_name "vplot" \
--model_selection "vplot" \
--loss_selection "SSIM" \
--SSIM_window_size 20
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

    args = parser.parse_args()
    config = vars(args)
    print(config)

    if args.wandb:
        import wandb
        wandb.init(project="hic2self")
    INPUT_FILE_NAME = config['input_file_name']
    MASK_WIDTH = config['mask_width']
    LEARNING_RATE = float(config['lr'])
    EXPERIMENT_VERSION = config['v']
    LOG_PATH = f'{args.log_dir}/{EXPERIMENT_VERSION}/'
    BATCH_SIZE = config['b']
    NUM_EPOCH = config['e']
    LOSS_SELECTION = config['loss_selection']
    SSIM_WINDOW_SIZE = config['SSIM_window_size']
    MODEL_SELECTION = config['model_selection']

    torch.cuda.set_device(int(args.gpu))
    torch.manual_seed(0)
    torch.set_default_tensor_type(torch.DoubleTensor)
    # model = NBChannel(side=False, round=False, include_pi=False).cuda()
    if MODEL_SELECTION == "svd":
        model = NBGPU(side=False, round=False, include_pi=False).cuda()
    elif MODEL_SELECTION == "attn":
        model = NBAttn(side=False, round=False, include_pi=False).cuda()
    elif MODEL_SELECTION == "base":
        model = NBBase(side=False, round=False, include_pi=False).cuda()
    elif MODEL_SELECTION == "vplot":
        model = NBFilter(side=False, round=False, include_pi=False).cuda()
    masker = Masker2(width = MASK_WIDTH, mode='zero', mask_type="diagonal", side_info=False)

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

    optimizer = Adam(model.parameters(), lr=0.0001)

    all_image_counts = data_preparation(INPUT_FILE_NAME)

    t0 = time.time()
    for epoch in range(int(NUM_EPOCH)):
        lr = np.maximum(LEARNING_RATE * np.power(0.5, (int(epoch / 16))), 1e-6) # learning rate decay
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

        data_loader = DataLoader(all_image_counts, batch_size=BATCH_SIZE,shuffle=True)
        model.train()
        print("Epoch: " + str(epoch) + "/" + str(NUM_EPOCH - 1))

        for i, batch in enumerate(data_loader):
            idx, train_image = batch
            if np.count_nonzero(train_image) == 0:
                continue
            net_input, mask = masker.mask(train_image, i)
            train_image = torch.tensor(train_image).cuda()
            net_input = torch.tensor(net_input).cuda()
            mask = torch.tensor(mask).cuda()

            net_output = model(net_input)
            net_pred = net_output[[0],[0],:,:].unsqueeze(0)
            net_theta = net_output[[0],[1],:,:].unsqueeze(0)

            if LOSS_SELECTION == "MSE":
                loss = torch.sum(loss_function(train_image*mask,net_pred*mask)).cuda()
            elif LOSS_SELECTION == "NBloss":
                loss = torch.sum(loss_function.nbloss(train_image*mask,net_pred*mask,net_theta*mask)).cuda()
            elif LOSS_SELECTION == "SSIM":
                loss = torch.sum(loss_function.ssim(train_image*mask, net_pred*mask, SSIM_WINDOW_SIZE)).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({LOSS_SELECTION: loss.item()})
            
            if i % 50 == 0:
            #     model.eval()
            #     net_input, mask = masker.mask(train_image, masker.n_masks - 1)
            #     train_image = torch.tensor(train_image).cuda()
            #     net_input, mask = torch.tensor(net_input).cuda(), torch.tensor(mask).cuda()
            #     net_output = model(net_input)
            #     val_loss = torch.sum(loss_function(train_image*mask,net_pred*mask)).cuda()
            #     # val_loss = loss_function.nbloss(net_output[[0],[0],:,:].unsqueeze(0), train_image,
            #     #                                 net_output[[0],[1],:,:].unsqueeze(0))
            #     # val_loss = torch.sum(val_loss)
            #     print("(", i, ") Loss: \t", round(loss.item(), 5), "\tVal Loss: \t", round(val_loss.item(), 5))
                # save(model, os.path.join(LOG_PATH, '%03d.pt_model' % epoch), num_to_keep=1)
                torch.save(model.state_dict(),f"{LOG_PATH}/hic013_gpu_{LOSS_SELECTION}_{MODEL_SELECTION}.pt_model")
    t1 = time.time()
    print(t1 - t0)

if __name__ == '__main__':
    main()