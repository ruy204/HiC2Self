## Data util functions for training HiC2Self
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
import os, glob
import sys
sys.path.append('/data/leslie/yangr2/setd2/train_setd2/scripts/')
import pytorch_ssim

################################
#    DataLoader Preparation    #
################################

class JointDataset():
    def __init__(self, data, mode='train'):
        self.mode = mode
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        return img


def data_preparation(data_file_name):
    with open(data_file_name, 'rb') as fp:
        all_mats = pickle.load(fp)
    all_mats2 = np.concatenate(all_mats, 0)
    mat_list = np.split(all_mats2, all_mats2.shape[0], axis=0)
    print(mat_list[0].shape)
    torchtransform = transforms.Compose([transforms.ToTensor()])
    mat_train = [torchtransform(i[0,:,:][:,:,np.newaxis]) for i in mat_list]
    idx_train = list(range(len(mat_train)))
    all_image = JointDataset([[i,j] for i,j in zip(idx_train,mat_train)])
    return all_image

def data_preparation_nagano(data_file_name):
    with open(data_file_name, 'rb') as fp:
        all_mats = pickle.load(fp)
    torchtransform = transforms.Compose([transforms.ToTensor()])
    mat_train = [torchtransform(i[:,:,np.newaxis]) for i in all_mats]
    idx_train = list(range(len(mat_train)))
    all_image = JointDataset([[i,j] for i,j in zip(idx_train,mat_train)])
    return all_image

###################################
#    Training helper functions    #
###################################

class Masker2():
    """Object for masking and demasking"""

    def __init__(self, width=3, mode='zero', mask_type="grid", side_info=False, infer_single_pass=False, include_mask_as_input=False):
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
      
    def infer_full_image1(self, X, model):

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

    def infer_full_image2(self, X, model,side_info):

      net_input, mask = self.mask(X, 0)
      net_output = model(X,side_info)
      acc_tensor = torch.zeros(net_output.shape).cpu()
      for i in range(self.n_masks):
        net_input, mask = self.mask(X,i)
        net_output = model(X, side_info)
        acc_tensor = acc_tensor + (net_output * mask).cpu()
      return acc_tensor
    
    def infer_full_image(self, X, model, side_info=1):
      if self.side_info == True:
        return self.infer_full_image2(X,model,side_info)
      elif self.side_info == False:
        return self.infer_full_image1(X,model)

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

###############################
#    Define loss functions    #
###############################

class ZINB2(nn.Module):
  def __init__(self, scale_factor = 1.0, ridge_lambda=0.0):
    super(ZINB2, self).__init__()
    self.eps = 1e-10 #for numerical stability
    self.scale_factor = scale_factor
    self.ridge_lambda = ridge_lambda
#   def nbloss(self, y_true, y_pred, theta, mean=True): # Archived Nov 23, 2023
#     self.theta = theta
#     scale_factor = self.scale_factor
#     eps = self.eps
#     theta = torch.clamp(self.theta,0,1e6) / (scale_factor**2)
#     y_pred = y_pred / scale_factor
#     t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
#     t2 = (theta+y_true)*torch.log(1.0 + (y_pred/(theta+eps))) + y_true*(torch.log(theta+eps)-torch.log(y_pred+eps))
#     final = t1+t2
#     return final
  def nbloss(self, y_true, y_pred, theta):
    self.theta = theta
    scale_factor = self.scale_factor
    eps = self.eps
    theta = torch.clamp(self.theta,0,1e6) / (scale_factor**2)
    y_pred = y_pred / scale_factor
    t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
    t2 = (theta+y_true)*torch.log(1.0 + (y_pred/(theta+eps))) + y_true*(torch.log(theta+eps)-torch.log(y_pred+eps))
    final = t1+t2
    # final = t2
    return final
  def loss(self, y_true, y_pred, pi, theta,mean=True):
    # self.pi = torch.clamp(pi,0,1)
    self.pi = pi
    self.theta = theta
    scale_factor = self.scale_factor
    eps = self.eps
    theta = torch.clamp(self.theta,0,1e6) / (scale_factor**2)
    y_pred = y_pred / scale_factor
    pi = self.pi / scale_factor
    nb_case = self.nbloss(y_true,y_pred,self.theta) - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(theta/(theta+y_pred+eps),theta)
    zero_case = - torch.log(pi+((1.0-pi)*zero_nb)+eps)
    result = torch.where((y_true>=1e-8),nb_case,zero_case)
    ridge = self.ridge_lambda*torch.square(pi)
    result += ridge
    return result
  def ssim(self, y_true, y_pred, window_size = 11):
     ssim_fn = pytorch_ssim.SSIM(window_size)
     ssim_loss = ssim_fn(y_pred, y_true)
     return ssim_loss

#####################
#    Save models    #
#####################

def save(net, file_name, num_to_keep=1):
    """Saves the net to file, creating folder paths if necessary.
    Args:
        net(torch.nn.module): The network to save
        file_name(str): the path to save the file.
        num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
            Defaults to 1. Specifying < 0 will not remove any previous saves.
    """

    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), file_name)
    extension = os.path.splitext(file_name)[1]
    checkpoints = sorted(glob.glob(folder + '/*' + extension), key=os.path.getmtime)
    print('Saved %s\n' % file_name)
    if num_to_keep > 0:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)