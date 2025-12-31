"""
Data utility functions for training HiC2Self.

This module provides:
- Dataset wrappers for preprocessed Hi-C submatrices
- Mask generation utilities (grid / symmetric_grid / diagonal / horizontal)
- Masking/inference helper (Masker2)
- Loss functions (ZINB2 + optional SSIM)
- Model checkpoint saving
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Optional dependency: pytorch-ssim
try:
    import pytorch_ssim  # pip install pytorch-ssim
except Exception:
    pytorch_ssim = None


################################
#    DataLoader Preparation    #
################################

class JointDataset(Dataset):
    """
    Dataset that returns (index, tensor) pairs.
    This matches your original behavior.
    """
    def __init__(self, data: List[Tuple[int, torch.Tensor]], mode: str = "train"):
        self.mode = mode
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        return self.data[index]


def _to_tensor_list_from_concat(all_mats: Any) -> List[torch.Tensor]:
    """
    Support your original pickled format where `all_mats` is often a list of
    arrays that can be concatenated along axis=0.
    Each element becomes a 2D matrix after indexing [0,:,:] in the original code.
    """
    all_mats2 = np.concatenate(all_mats, axis=0)  # (N, H, W)
    mat_list = np.split(all_mats2, all_mats2.shape[0], axis=0)  # list of (1,H,W)

    torchtransform = transforms.Compose([transforms.ToTensor()])
    # original: i[0,:,:][:,:,np.newaxis]
    mats = [torchtransform(i[0, :, :][:, :, np.newaxis]) for i in mat_list]  # (1,H,W)
    return mats


def data_preparation(data_file_name: str) -> JointDataset:
    """
    Load pickled matrices and return a Dataset of (index, tensor) pairs.
    Expected pickle format: list-like where elements are concat-able along axis=0.
    """
    with open(data_file_name, "rb") as fp:
        all_mats = pickle.load(fp)

    mats = _to_tensor_list_from_concat(all_mats)
    idx = list(range(len(mats)))
    return JointDataset(list(zip(idx, mats)))


def data_preparation_nagano(data_file_name: str) -> JointDataset:
    """
    Alternative loader for pickles where `all_mats` is already a list/array
    of shape (N, H, W).
    """
    with open(data_file_name, "rb") as fp:
        all_mats = pickle.load(fp)

    torchtransform = transforms.Compose([transforms.ToTensor()])
    mats = [torchtransform(i[:, :, np.newaxis]) for i in all_mats]  # (1,H,W)
    idx = list(range(len(mats)))
    return JointDataset(list(zip(idx, mats)))


###################################
#    Training helper functions    #
###################################

@dataclass
class Masker2:
    """
    Object for masking and demasking.

    mask_type:
      - "grid": standard grid mask
      - "symmetric_grid": symmetric version (upper-tri reflected)
      - "diagonal": mask based on (i+j) mod grid_size
      - "horizontal": symmetric horizontal stripes
    """
    width: int = 3
    mode: str = "zero"
    mask_type: str = "grid"
    side_info: bool = False
    infer_single_pass: bool = False
    include_mask_as_input: bool = False

    def __post_init__(self):
        self.grid_size = self.width
        if self.mask_type in ["grid", "symmetric_grid"]:
            self.n_masks = self.grid_size ** 2
        elif self.mask_type in ["diagonal", "horizontal"]:
            self.n_masks = self.grid_size
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def __len__(self) -> int:
        return self.n_masks

    def mask(self, X: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: (B, C, H, W)
        Returns:
          net_input: masked input (optionally concat mask as extra channel)
          mask: (H, W) float tensor with 1 where masked/predicted
        """
        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size

        shape_hw = X[0, 0].shape  # (H,W)

        if self.mask_type == "grid":
            mask = pixel_grid_mask(shape_hw, self.grid_size, phasex, phasey)
        elif self.mask_type == "symmetric_grid":
            mask = symmetric_grid_mask(shape_hw, self.grid_size, phasex, phasey)
        elif self.mask_type == "diagonal":
            mask = diagonal_grid_mask(shape_hw, self.grid_size, phasex)
        elif self.mask_type == "horizontal":
            mask = symmetric_horizontal_mask(shape_hw, self.grid_size, phasex)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        mask = mask.to(X.device)
        mask_inv = (1.0 - mask).to(X.device)

        if self.mode == "interpolate":
            masked = interpolate_mask(X, mask, mask_inv)
        elif self.mode == "zero":
            masked = X * mask_inv
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")

        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask[None, None, :, :].repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def infer_full_image1(self, X: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Inference without side_info.
        Assumes model signature supports either:
          - model(net_input)  (single pass)
          - model(X, mask)    (masked pass)
        """
        if self.infer_single_pass:
            if self.include_mask_as_input:
                zero_mask = torch.zeros_like(X[:, 0:1])
                net_input = torch.cat((X, zero_mask), dim=1)
            else:
                net_input = X
            return model(net_input)

        # masked accumulation
        _, mask0 = self.mask(X, 0)
        net_output0 = model(X, mask0)
        acc = torch.zeros_like(net_output0).cpu()

        for i in range(self.n_masks):
            _, mask = self.mask(X, i)
            net_output = model(X, mask)
            acc += (net_output * mask).cpu()

        return acc

    def infer_full_image2(self, X: torch.Tensor, model: nn.Module, side_info: Any) -> torch.Tensor:
        """
        Inference with side_info (assumes model(X, side_info)).
        """
        net_output0 = model(X, side_info)
        acc = torch.zeros_like(net_output0).cpu()

        for i in range(self.n_masks):
            _, mask = self.mask(X, i)
            net_output = model(X, side_info)
            acc += (net_output * mask).cpu()

        return acc

    def infer_full_image(self, X: torch.Tensor, model: nn.Module, side_info: Any = None) -> torch.Tensor:
        if self.side_info:
            return self.infer_full_image2(X, model, side_info)
        return self.infer_full_image1(X, model)


def pixel_grid_mask(shape_hw: Tuple[int, int], patch_size: int, phase_x: int, phase_y: int) -> torch.Tensor:
    H, W = shape_hw
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = ((yy % patch_size == phase_x) & (xx % patch_size == phase_y)).float()
    return mask


def symmetric_grid_mask(shape_hw: Tuple[int, int], patch_size: int, phase_x: int, phase_y: int) -> torch.Tensor:
    mask = pixel_grid_mask(shape_hw, patch_size, phase_x, phase_y)
    mask = torch.triu(mask, diagonal=0)
    mask = mask + mask.T - torch.diag(torch.diag(mask))
    return mask


def diagonal_grid_mask(shape_hw: Tuple[int, int], patch_size: int, phase_x: int) -> torch.Tensor:
    H, W = shape_hw
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = (((yy + xx) % patch_size) == phase_x).float()
    return mask


def symmetric_horizontal_mask(shape_hw: Tuple[int, int], patch_size: int, phase_x: int) -> torch.Tensor:
    H, W = shape_hw
    yy = torch.arange(H)[:, None].repeat(1, W)
    mask = ((yy % patch_size) == phase_x).float()
    mask = torch.triu(mask, diagonal=0)
    mask = mask + mask.T - torch.diag(torch.diag(mask))
    return mask


def interpolate_mask(tensor: torch.Tensor, mask: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
    """
    Interpolate masked pixels using a simple smoothing kernel.
    tensor: (B, C, H, W)
    mask/mask_inv: (H, W)
    """
    device = tensor.device
    mask = mask.to(device)
    mask_inv = mask_inv.to(device)

    kernel = np.array(
        [[0.5, 1.0, 0.5],
         [1.0, 0.0, 1.0],
         [0.5, 1.0, 0.5]],
        dtype=np.float32
    )
    kernel = torch.tensor(kernel, device=device)[None, None, :, :]
    kernel = kernel / kernel.sum()

    filtered = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)
    return filtered * mask + tensor * mask_inv


###############################
#    Define loss functions    #
###############################

class ZINB2(nn.Module):
    def __init__(self, scale_factor: float = 1.0, ridge_lambda: float = 0.0):
        super().__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.ridge_lambda = ridge_lambda

    def nbloss(self, y_true: torch.Tensor, y_pred: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        sf = self.scale_factor

        theta = torch.clamp(theta, 0, 1e6) / (sf ** 2)
        y_pred = y_pred / sf

        t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        return t1 + t2

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, pi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        sf = self.scale_factor

        theta = torch.clamp(theta, 0, 1e6) / (sf ** 2)
        y_pred = y_pred / sf
        pi = pi / sf  # keeping your original scaling behavior

        nb_case = self.nbloss(y_true, y_pred, theta) - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        result = torch.where((y_true >= 1e-8), nb_case, zero_case)
        result = result + self.ridge_lambda * torch.square(pi)
        return result

    def ssim(self, y_true: torch.Tensor, y_pred: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        if pytorch_ssim is None:
            raise ImportError("pytorch-ssim is not installed. Install it with: pip install pytorch-ssim")
        ssim_fn = pytorch_ssim.SSIM(window_size)
        return ssim_fn(y_pred, y_true)


#####################
#    Save models    #
#####################

def save(net: nn.Module, file_name: str, num_to_keep: int = 1) -> None:
    """
    Save model state dict to file, creating folder paths if necessary.

    Args:
        net: torch module
        file_name: full output path (including filename)
        num_to_keep: number of latest checkpoints to keep in the same folder
                     (<=0 keeps all)
    """
    folder = os.path.dirname(file_name)
    os.makedirs(folder, exist_ok=True)

    torch.save(net.state_dict(), file_name)

    ext = os.path.splitext(file_name)[1]
    checkpoints = sorted(glob.glob(os.path.join(folder, "*" + ext)), key=os.path.getmtime)

    print(f"Saved {file_name}")
    if num_to_keep > 0 and len(checkpoints) > num_to_keep:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)
