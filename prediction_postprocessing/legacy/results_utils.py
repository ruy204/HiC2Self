# Script for loading updated model and mask functions when checking results in the jupyter notebook
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
import argparse, os, sys
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
from datetime import datetime

sys.path.append('/data/leslie/yangr2/hic2self/scripts')
from data_preparation import *
# from hic2self_train_gpu import *
from hic2self_model import *
from data_utils import *
import time

sys.path.append('/data/leslie/yangr2/setd2/train_setd2/scripts/')
import pytorch_ssim
chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('/home/yangr2/assembly/hg38.chrom.sizes').readlines()}

import coolbox
from coolbox.api import *

from sklearn.decomposition import TruncatedSVD

class Masker3():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', mask_type="grid", side_info=False, 
                 infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width
        self.mask_type = mask_type
        if self.mask_type in ["grid", "symmetric_grid"]:
            self.n_masks = width ** 2
        elif self.mask_type in ["diagonal", "horizontal"]:
            self.n_masks = width
        self.side_info = side_info
        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, X, i):
        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        if self.mask_type == "grid":
            mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
            mask = mask.to(X.device)
        elif self.mask_type == "symmetric_grid":
            mask = symmetric_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
            mask = mask.to(X.device)
        elif self.mask_type == "diagonal":
            mask = diagonal_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
            mask = mask.to(X.device)
        elif self.mask_type == "horizontal":
          mask = symmetric_horizontal_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
          mask = mask.to(X.device)
        mask_inv = torch.ones(mask.shape).to(X.device) - mask.to(X.device)

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError
            
        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks
      
    def infer_full_image(self, X, model):

        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            # net_output = model(net_input)
            net_output = model(X, mask)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(X, mask)
                acc_tensor = acc_tensor + (net_output * mask).cpu()
            return acc_tensor

        
MASK_WIDTH = 5
# masker = Masker2(width = MASK_WIDTH, mode='zero', mask_type="diagonal", side_info=False)
masker = Masker3(width = MASK_WIDTH, mode='zero', mask_type="diagonal", side_info=False)

class NBGPU(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBGPU, self).__init__()
      self.include_pi = include_pi
      self.side = side
      self.round = round
      self.a = a
      self.b = b
      self.relu = nn.ReLU()
      self.conv1 = nn.Conv2d(5, 64, (5, 5), (1, 1), (2, 2)) #for not including side information
      self.conv2 = nn.Conv2d(8, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3a = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
      self.conv4 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv5 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv6 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
    
    def tn(self, tensor0):
      # return tensor0.detach().numpy()[0][0] # -- currently only support batch_size=1
      return tensor0[0][0].cpu()
    
    def minmax(self,mat):
      matmin = mat.min()
      matmax = mat.max()
      mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
      return mat2

    def forward(self, x, mask, side_mat=1):
    #   x0 = torch.log2(x+1)
      x0 = x
      X_noisy = self.tn(x0)
      svd_noisy = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
      svd_noisy.fit(X_noisy)
      x_np1 = np.dot(np.dot(svd_noisy.components_[self.a:1,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:1])),
                    svd_noisy.components_[self.a:1,:])
      x_np2 = np.dot(np.dot(svd_noisy.components_[self.a:2,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:2])),
                    svd_noisy.components_[self.a:2,:])
      x_np3 = np.dot(np.dot(svd_noisy.components_[self.a:3,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:3])),
                    svd_noisy.components_[self.a:3,:])
      x_np4 = np.dot(np.dot(svd_noisy.components_[self.a:4,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:4])),
                    svd_noisy.components_[self.a:4,:])
      mask_inv = torch.ones(mask.shape).to(x.device) - mask.to(x.device)
      x1 = torch.cat((x0 * mask_inv,
                      torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda() * mask_inv,
                      torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0).cuda() * mask_inv,
                      torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0).cuda() * mask_inv,
                      torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0).cuda() * mask_inv),1)
      if self.side==False:
        x = self.relu(self.conv1(x1).clone())
      elif self.side==True:
        x1 = torch.cat((x1,torch.log2(side_mat+1)),dim=1)
        x = self.relu(self.conv2(x1).clone())
      x = self.relu(self.conv3a(x).clone())
      x = self.relu(self.conv3(x).clone())

      #mu layer
      x2 = self.conv4(x)
      if self.round == True:
        x_mu = torch.round(torch.pow((x2+1),2))
        # x_mu = torch.round(x2)
      else:
        x_mu = torch.pow((x2+1),2)
        # x_mu = x2
      
      #theta layer
      x3 = self.conv5(x)
      x_theta = torch.pow((x3+1),2)
      # x_theta = x3
      x_mu_theta = torch.cat((x_mu,x_theta),dim=1)

      #pi layer for zero-inflated model
      if self.include_pi == True:
        x_pi = torch.clamp(self.conv6(x),0,1)
        x_out = torch.cat((x_mu_theta,x_pi),dim=1)
      else:
        x_out = x_mu_theta
      return x_out

class NBnoGPU(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBnoGPU, self).__init__()
      self.include_pi = include_pi
      self.side = side
      self.round = round
      self.a = a
      self.b = b
      self.relu = nn.ReLU()
      self.conv1 = nn.Conv2d(5, 64, (5, 5), (1, 1), (2, 2)) #for not including side information
      self.conv2 = nn.Conv2d(8, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3a = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
      self.conv4 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv5 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv6 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
    
    def tn(self, tensor0):
      # return tensor0.detach().numpy()[0][0] # -- currently only support batch_size=1
      return tensor0[0][0].cpu()
    
    def minmax(self,mat):
      matmin = mat.min()
      matmax = mat.max()
      mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
      return mat2

    def forward(self, x, mask, side_mat=1):
    #   x0 = torch.log2(x+1)
      x0 = x
      X_noisy = self.tn(x0)
      svd_noisy = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
      svd_noisy.fit(X_noisy)
      x_np1 = np.dot(np.dot(svd_noisy.components_[self.a:1,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:1])),
                    svd_noisy.components_[self.a:1,:])
      x_np2 = np.dot(np.dot(svd_noisy.components_[self.a:2,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:2])),
                    svd_noisy.components_[self.a:2,:])
      x_np3 = np.dot(np.dot(svd_noisy.components_[self.a:3,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:3])),
                    svd_noisy.components_[self.a:3,:])
      x_np4 = np.dot(np.dot(svd_noisy.components_[self.a:4,:].T,
                                               np.diag(svd_noisy.singular_values_[self.a:4])),
                    svd_noisy.components_[self.a:4,:])
      # mask_inv = torch.ones(mask.shape).to(x.device) - mask.to(x.device)
      mask_inv = torch.ones(mask.shape) - mask
      x1 = torch.cat((x0 * mask_inv,
                      torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0) * mask_inv,
                      torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0) * mask_inv,
                      torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0) * mask_inv,
                      torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0) * mask_inv),1)
      if self.side==False:
        x = self.relu(self.conv1(x1).clone())
      elif self.side==True:
        x1 = torch.cat((x1,torch.log2(side_mat+1)),dim=1)
        x = self.relu(self.conv2(x1).clone())
      x = self.relu(self.conv3a(x).clone())
      x = self.relu(self.conv3(x).clone())

      #mu layer
      x2 = self.conv4(x)
      if self.round == True:
        x_mu = torch.round(torch.pow((x2+1),2))
        # x_mu = torch.round(x2)
      else:
        x_mu = torch.pow((x2+1),2)
        # x_mu = x2
      
      #theta layer
      x3 = self.conv5(x)
      x_theta = torch.pow((x3+1),2)
      # x_theta = x3
      x_mu_theta = torch.cat((x_mu,x_theta),dim=1)

      #pi layer for zero-inflated model
      if self.include_pi == True:
        x_pi = torch.clamp(self.conv6(x),0,1)
        x_out = torch.cat((x_mu_theta,x_pi),dim=1)
      else:
        x_out = x_mu_theta
      return x_out


class NBPoolUp(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBPoolUp, self).__init__()
      self.include_pi = include_pi
      self.side = side
      self.round = round
      self.a = a
      self.b = b
      self.m1 = nn.AvgPool2d(2, stride = 2)
      self.m2 = nn.AvgPool2d(3, stride = 3)
      self.m3 = nn.AvgPool2d(4, stride = 4)
      self.m4 = nn.AvgPool2d(5, stride = 5)
      self.relu = nn.ReLU()
      self.conv1 = nn.Conv2d(5, 64, (5, 5), (1, 1), (2, 2)) #for not including side information
      self.conv2 = nn.Conv2d(8, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3a = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
      self.conv4 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv5 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv6 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
    
    def tn(self, tensor0):
      # return tensor0.detach().numpy()[0][0] # -- currently only support batch_size=1
      return tensor0[0][0].cpu()
    
    def minmax(self,mat):
      matmin = mat.min()
      matmax = mat.max()
      mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
      return mat2

    def forward(self, x, mask, side_mat=1):
    #   x0 = torch.log2(x+1)
      x0 = x
      X_noisy = self.tn(x0)
      svd_noisy = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
      svd_noisy.fit(X_noisy)
      x_np1 = self.m1(x0)
      x_np1 = F.interpolate(x_np1, size=(100, 100), mode='bilinear', align_corners=False)
      x_np2 = self.m2(x0)
      x_np2 = F.interpolate(x_np2, size=(100, 100), mode='bilinear', align_corners=False)
      x_np3 = self.m3(x0)
      x_np3 = F.interpolate(x_np3, size=(100, 100), mode='bilinear', align_corners=False)
      x_np4 = self.m4(x0)
      x_np4 = F.interpolate(x_np4, size=(100, 100), mode='bilinear', align_corners=False)
      mask_inv = torch.ones(mask.shape).to(x.device) - mask.to(x.device)
      x1 = torch.cat((x0 * mask_inv,x_np1.to(x.device) * mask_inv,
                      x_np2.to(x.device) * mask_inv,
                      x_np3.to(x.device) * mask_inv,
                      x_np4.to(x.device) * mask_inv),1)
      x = self.relu(self.conv1(x1).clone())
      x = self.relu(self.conv3a(x).clone())
      x = self.relu(self.conv3(x).clone())

      #mu layer
      x2 = self.conv4(x)
      if self.round == True:
        x_mu = torch.round(torch.pow((x2+1),2))
        # x_mu = torch.round(x2)
      else:
        x_mu = torch.pow((x2+1),2)
        # x_mu = x2
      
      #theta layer
      x3 = self.conv5(x)
      x_theta = torch.pow((x3+1),2)
      # x_theta = x3
      x_mu_theta = torch.cat((x_mu,x_theta),dim=1)

      #pi layer for zero-inflated model
      if self.include_pi == True:
        x_pi = torch.clamp(self.conv6(x),0,1)
        x_out = torch.cat((x_mu_theta,x_pi),dim=1)
      else:
        x_out = x_mu_theta
      return x_out

def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    return torch.Tensor(A)

def symmetric_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x and j % patch_size == phase_y):
                A[i, j] = 1
    A = np.triu(A, k=0)
    A = A + A.T - np.diag(np.diag(A))
    return torch.Tensor(A)

def diagonal_grid_mask(shape, patch_size, phase_x, phase_y):
  A = torch.zeros(shape[-2:])
  for i in range(shape[-2]):
      for j in range(shape[-1]):
          if ((i+j) % patch_size == phase_x): #and j % patch_size == phase_x):
              A[i, j] = 1
  return torch.Tensor(A)

def symmetric_horizontal_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if (i % patch_size == phase_x):
                A[i, j] = 1
    A = np.triu(A, k=0)
    A = A + A.T - np.diag(np.diag(A))
    return torch.Tensor(A)

def interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv