#!/usr/bin/env python
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from skimage import filters

class NBChannel(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBChannel, self).__init__()
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
      x1 = torch.cat((x0 * mask_inv,torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0) * mask_inv,
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
      x1 = torch.cat((x0 * mask_inv,torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda() * mask_inv,
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
