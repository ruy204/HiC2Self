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
      return tensor0.detach().numpy()[0][0] # -- currently only support batch_size=1
    
    def minmax(self,mat):
      matmin = mat.min()
      matmax = mat.max()
      mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
      return mat2

    def forward(self, x, side_mat=1):
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
      x1 = torch.cat((x0,torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0),
                      torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0),
                      torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0),
                      torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0)),1)
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

    def forward(self, x, side_mat=1):
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
      x1 = torch.cat((x0,torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda(),
                      torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0).cuda(),
                      torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0).cuda(),
                      torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0).cuda()),1)
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

class NBBase(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBBase, self).__init__()
      self.include_pi = include_pi
      self.side = side
      self.round = round
      self.a = a
      self.b = b
      self.relu = nn.ReLU()
      self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)) #for not including side information
      self.conv2 = nn.Conv2d(8, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3a = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
      self.conv4 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv5 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
      self.conv6 = nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1))
    
    def tn(self, tensor0):
      # return tensor0.detach().numpy()[0][0] # -- currently only support batch_size=1
      return tensor0[0][0].cpu()
    
    def forward(self, x, side_mat=1):
      x1 = torch.log2(x+1)
      # X_noisy = self.tn(x0)
      # svd_noisy = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
      # svd_noisy.fit(X_noisy)
      # x_np1 = np.dot(np.dot(svd_noisy.components_[self.a:1,:].T,
      #                                          np.diag(svd_noisy.singular_values_[self.a:1])),
      #               svd_noisy.components_[self.a:1,:])
      # x_np2 = np.dot(np.dot(svd_noisy.components_[self.a:2,:].T,
      #                                          np.diag(svd_noisy.singular_values_[self.a:2])),
      #               svd_noisy.components_[self.a:2,:])
      # x_np3 = np.dot(np.dot(svd_noisy.components_[self.a:3,:].T,
      #                                          np.diag(svd_noisy.singular_values_[self.a:3])),
      #               svd_noisy.components_[self.a:3,:])
      # x_np4 = np.dot(np.dot(svd_noisy.components_[self.a:4,:].T,
      #                                          np.diag(svd_noisy.singular_values_[self.a:4])),
      #               svd_noisy.components_[self.a:4,:])
      # x1 = torch.cat((x0,torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda(),
      #                 torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0).cuda(),
      #                 torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0).cuda(),
      #                 torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0).cuda()),1)
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

class NBAttn(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBAttn, self).__init__()
      self.include_pi = include_pi
      self.side = side
      self.round = round
      self.a = a
      self.b = b
      self.relu = nn.ReLU()
      self.conv1a = nn.Conv2d(5, 16, (5, 5), (1, 1), (2, 2)) #for not including side information
      self.conv1b = nn.Conv2d(4, 16, (5, 5), (1, 1), (2, 2))
      self.conv1c = nn.Conv2d(4, 16, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3a = nn.Conv2d(48, 48, (5, 5), (1, 1), (2, 2)) 
      self.conv3 = nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1))
      self.conv4 = nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1))
      self.conv5 = nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1))
      self.conv6 = nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1))
    
    def tn(self, tensor0):
      return tensor0[0][0].cpu()
      # return tensor0.cpu().detach().numpy()[0][0] # -- currently only support batch_size=1
    
    def minmax(self,mat):
      matmin = mat.min()
      matmax = mat.max()
      mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
      return mat2

    def forward(self, x, side_mat=1):
      x0 = torch.log2(x+1)
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
      
      #svd channels
      x1 = torch.cat((x0,
                torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda(),
                torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0).cuda(),
                torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0).cuda(),
                torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0).cuda()),1)

      #product attention
      input_size = x0[0][0].shape[0]
      x0_cat = torch.cat((x0, x0, x0, x0),1)[0] #[4,300,300]
      x_np = torch.cat((torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda(),
                        torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0).cuda(),
                        torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0).cuda(),
                        torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0).cuda()),1)[0] #[4,300,300]
      qk_product = torch.bmm(x0_cat,x_np.transpose(1,2)) #[4,300,300]
      attn = F.softmax(qk_product.view(-1, input_size), dim=1).view(4, -1, input_size) #[4,300,300]
      context = torch.bmm(attn, x0_cat) #[4,300,300]

      x1 = self.relu(self.conv1a(x1).clone())
      x2 = self.relu(self.conv1b(attn.unsqueeze(0)).clone())
      x3 = self.relu(self.conv1c(context.unsqueeze(0)).clone())
      x = torch.cat((x1,x2,x3),1)
      x = self.relu(self.conv3a(x).clone())
      x = self.relu(self.conv3(x).clone())

      #mu layer
      x2 = self.conv4(x)
      if self.round == True:
        x_mu = torch.round(torch.pow((x2+1),2))
      else:
        x_mu = torch.pow((x2+1),2)
      
      #theta layer
      x3 = self.conv5(x)
      x_theta = torch.pow((x3+1),2)
      x_mu_theta = torch.cat((x_mu,x_theta),dim=1)

      #pi layer for zero-inflated model
      if self.include_pi == True:
        x_pi = torch.clamp(self.conv6(x),0,1)
        x_out = torch.cat((x_mu_theta,x_pi),dim=1)
      else:
        x_out = x_mu_theta
      return x_out
    

class NBAttnCPU(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
      super(NBAttnCPU, self).__init__()
      self.include_pi = include_pi
      self.side = side
      self.round = round
      self.a = a
      self.b = b
      self.relu = nn.ReLU()
      self.conv1a = nn.Conv2d(5, 16, (5, 5), (1, 1), (2, 2)) #for not including side information
      self.conv1b = nn.Conv2d(4, 16, (5, 5), (1, 1), (2, 2))
      self.conv1c = nn.Conv2d(4, 16, (5, 5), (1, 1), (2, 2)) #for including side information
      self.conv3a = nn.Conv2d(48, 48, (5, 5), (1, 1), (2, 2)) 
      self.conv3 = nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1))
      self.conv4 = nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1))
      self.conv5 = nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1))
      self.conv6 = nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1))
    
    def tn(self, tensor0):
      return tensor0[0][0].cpu()
      # return tensor0.cpu().detach().numpy()[0][0] # -- currently only support batch_size=1
    
    def minmax(self,mat):
      matmin = mat.min()
      matmax = mat.max()
      mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
      return mat2

    def forward(self, x, side_mat=1):
      x0 = torch.log2(x+1)
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
      
      #svd channels
      x1 = torch.cat((x0,
                torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0)),1)

      #product attention
      input_size = x0[0][0].shape[0]
      x0_cat = torch.cat((x0, x0, x0, x0),1)[0] #[4,300,300]
      x_np = torch.cat((torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0),
                        torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0),
                        torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0),
                        torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0)),1)[0] #[4,300,300]
      qk_product = torch.bmm(x0_cat,x_np.transpose(1,2)) #[4,300,300]
      attn = F.softmax(qk_product.view(-1, input_size), dim=1).view(4, -1, input_size) #[4,300,300]
      context = torch.bmm(attn, x0_cat) #[4,300,300]

      x1 = self.relu(self.conv1a(x1).clone())
      x2 = self.relu(self.conv1b(attn.unsqueeze(0)).clone())
      x3 = self.relu(self.conv1c(context.unsqueeze(0)).clone())
      x = torch.cat((x1,x2,x3),1)
      x = self.relu(self.conv3a(x).clone())
      x = self.relu(self.conv3(x).clone())

      #mu layer
      x2 = self.conv4(x)
      if self.round == True:
        x_mu = torch.round(torch.pow((x2+1),2))
      else:
        x_mu = torch.pow((x2+1),2)
      
      #theta layer
      x3 = self.conv5(x)
      x_theta = torch.pow((x3+1),2)
      x_mu_theta = torch.cat((x_mu,x_theta),dim=1)

      #pi layer for zero-inflated model
      if self.include_pi == True:
        x_pi = torch.clamp(self.conv6(x),0,1)
        x_out = torch.cat((x_mu_theta,x_pi),dim=1)
      else:
        x_out = x_mu_theta
      return x_out

class NBFilter(nn.Module):
    def __init__(self,side=False,round=False,include_pi=False,a=0,b=5):
        super(NBFilter, self).__init__()
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
        return tensor0.cpu().detach().numpy()[0][0] # -- currently only support batch_size=1
    
    def minmax(self,mat):
        matmin = mat.min()
        matmax = mat.max()
        mat2 = ((mat - matmin)/(matmax - matmin))*2 - 1
        return mat2

    def forward(self, x, side_mat=1):
    #   x0 = torch.log2(x+1)
        x0 = x
        X_noisy = self.tn(x0)
        x_np1 = filters.sato(X_noisy,sigmas = range(0,10,2))
        x_np2 = filters.sato(X_noisy,sigmas = range(10,20,2))
        x_np3 = filters.sato(X_noisy,sigmas = range(20,30,2))
        x_np4 = filters.sato(X_noisy,sigmas = range(30,40,2))
        x1 = torch.cat((x0,torch.from_numpy(x_np1).unsqueeze(0).unsqueeze(0).cuda(),
                      torch.from_numpy(x_np2).unsqueeze(0).unsqueeze(0).cuda(),
                      torch.from_numpy(x_np3).unsqueeze(0).unsqueeze(0).cuda(),
                      torch.from_numpy(x_np4).unsqueeze(0).unsqueeze(0).cuda()),1)
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