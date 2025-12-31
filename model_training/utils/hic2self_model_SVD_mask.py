#!/usr/bin/env python
"""
HiC2Self model with SVD-derived input channels and masking.

This model constructs additional channels from low-rank SVD reconstructions
computed from the (masked) input matrix, and predicts Negative Binomial
parameters (mu, theta) and optionally pi (for ZINB).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD


class HiC2SelfSVDMaskNet(nn.Module):
    """
    A CNN that takes masked Hi-C submatrices plus SVD-derived channels as input.

    Forward signature matches your training code:
        forward(x, mask, side_mat=...)

    Args:
        side: include side information (expects side_mat to be provided)
        round_mu: whether to round mu output (legacy behavior)
        include_pi: include pi head for ZINB
        svd_start: starting component index (a in your original code)
        svd_n_components: number of components to compute with TruncatedSVD (>=4 recommended)
        log_side: apply log2(side_mat+1) when concatenating side info (matches your code)
    """
    def __init__(
        self,
        side: bool = False,
        round_mu: bool = False,
        include_pi: bool = False,
        svd_start: int = 0,
        svd_n_components: int = 10,
        log_side: bool = True,
    ):
        super().__init__()
        self.include_pi = include_pi
        self.side = side
        self.round_mu = round_mu

        self.svd_start = svd_start
        self.svd_n_components = svd_n_components
        self.log_side = log_side

        self.relu = nn.ReLU()

        # input channels:
        #   5 = (masked X) + (rank1..rank4 svd recon)  => 5
        #   if side=True, concatenate 3 extra channels from side_mat? (your code used conv2 expecting 8)
        #   your current code concatenates torch.log2(side_mat+1) which presumably has 3 channels.
        self.conv1 = nn.Conv2d(5, 64, kernel_size=5, stride=1, padding=2)  # no side info
        self.conv2 = nn.Conv2d(8, 64, kernel_size=5, stride=1, padding=2)  # with side info

        self.conv3a = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        # heads
        self.conv_mu = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_theta = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_pi = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _safe_pow2plus1(x: torch.Tensor) -> torch.Tensor:
        # legacy transformation: (x+1)^2 ensures non-negativity
        return torch.pow(x + 1.0, 2)

    def _svd_recon_channels(self, x2d: np.ndarray) -> np.ndarray:
        """
        Build 4 SVD reconstruction matrices:
            using components [a:a+1], [a:a+2], [a:a+3], [a:a+4]
        Returns array of shape (4, H, W) in float32.
        """
        svd = TruncatedSVD(n_components=self.svd_n_components, n_iter=7, random_state=42)
        svd.fit(x2d)

        # V = components_  shape: (k, W) if X is (H,W)
        # reconstruction: V.T @ diag(S) @ V   (your original formulation)
        V = svd.components_.astype(np.float32)  # (k, W)
        S = svd.singular_values_.astype(np.float32)  # (k,)

        chans = []
        for r in [1, 2, 3, 4]:
            sl = slice(self.svd_start, self.svd_start + r)
            Vsl = V[sl, :]          # (r, W)
            Ssl = S[sl]             # (r,)
            recon = (Vsl.T @ np.diag(Ssl) @ Vsl).astype(np.float32)  # (W, W) if square
            chans.append(recon)

        return np.stack(chans, axis=0)  # (4, H, W) assuming square

    def forward(self, x: torch.Tensor, mask: torch.Tensor, side_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, 1, H, W) typically
        mask: (H, W) with 1 where to predict, 0 where observed (as in your masker)
        side_mat: optional side information tensor, expected shape (B, 3, H, W) if side=True
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got shape {tuple(x.shape)}")
        if x.size(0) < 1:
            raise ValueError("Empty batch.")
        if mask.dim() != 2:
            raise ValueError(f"Expected mask to be 2D (H,W), got shape {tuple(mask.shape)}")

        device = x.device
        mask = mask.to(device)
        mask_inv = (1.0 - mask).to(device)  # (H,W)

        # We will build SVD channels per-sample (safe for any batch size, albeit slower for >1)
        B, C, H, W = x.shape
        if C != 1:
            # If you ever pass multi-channel x, we still only SVD the first channel.
            x_base = x[:, 0:1, :, :]
        else:
            x_base = x

        svd_chan_list = []
        for b in range(B):
            # Convert to numpy (CPU) for sklearn SVD
            x2d = x_base[b, 0].detach().cpu().numpy().astype(np.float32)  # (H,W)
            svd_chans = self._svd_recon_channels(x2d)  # (4,H,W)
            svd_chan_list.append(torch.from_numpy(svd_chans))

        svd_chan = torch.stack(svd_chan_list, dim=0).to(device)  # (B,4,H,W)

        # Apply mask_inv to everything (matches your original: multiply channels by mask_inv)
        # x_in channels: [masked original] + [masked svd1..4]  => 5 channels
        x_masked = x_base * mask_inv
        svd_masked = svd_chan * mask_inv
        x1 = torch.cat([x_masked, svd_masked], dim=1)  # (B,5,H,W)

        if not self.side:
            h = self.relu(self.conv1(x1))
        else:
            if side_mat is None:
                raise ValueError("side_mat must be provided when side=True")
            side_in = side_mat
            if self.log_side:
                side_in = torch.log2(side_in + 1.0)
            h_in = torch.cat([x1, side_in], dim=1)  # expects (B,8,H,W)
            h = self.relu(self.conv2(h_in))

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3(h))

        # mu head
        mu_raw = self.conv_mu(h)
        mu = self._safe_pow2plus1(mu_raw)
        if self.round_mu:
            mu = torch.round(mu)

        # theta head
        theta_raw = self.conv_theta(h)
        theta = self._safe_pow2plus1(theta_raw)

        out = torch.cat([mu, theta], dim=1)  # (B,2,H,W)

        # optional pi head
        if self.include_pi:
            pi = torch.clamp(self.conv_pi(h), 0.0, 1.0)
            out = torch.cat([out, pi], dim=1)  # (B,3,H,W)

        return out
