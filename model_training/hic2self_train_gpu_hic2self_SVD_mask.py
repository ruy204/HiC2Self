#!/usr/bin/env python
"""
Train HiC2Self (SVD-mask model).

Example:
python model_training/hic2self_train_gpu_hic2self_SVD_mask.py \
  --input_file_name /path/to/processed.pkl \
  --log_dir ./logs \
  --lr 1e-5 \
  --b 1 \
  --e 10 \
  --mask_width 3 \
  --save_name bulk_retrain \
  --model_selection svd \
  --loss_selection NBloss \
  --mask_type symmetric_grid
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_utils_hic2self import Masker2, ZINB2, data_preparation, data_preparation_nagano
from hic2self_model_SVD_mask import HiC2SelfSVDMaskNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file_name", type=str, required=True,
                        help="Path to processed matrices (pickle).")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Output directory for logs and checkpoints.")

    parser.add_argument("--wandb", action="store_true", help="Toggle wandb logging.")
    parser.add_argument("--gpu", default="0",
                        help="CUDA ID (kept for backward compatibility; can be ignored if CUDA_VISIBLE_DEVICES is set).")

    parser.add_argument("--b", default=1, type=int, help="Batch size.")
    parser.add_argument("--e", default=15, type=int, help="Number of epochs.")
    parser.add_argument("--lr", default="1e-4", help="Initial learning rate.")
    parser.add_argument("--v", default="0.1", help="Experiment version string.")

    parser.add_argument("--mask_width", default=3, type=int, help="Mask grid width.")
    parser.add_argument("--mask_type", default="diagonal", type=str,
                        help="Mask type: grid, symmetric_grid, diagonal, horizontal.")

    parser.add_argument("--m", default="", help="Additional comments.")

    parser.add_argument("--loss_selection", default="NBloss", type=str,
                        help="Loss: MSE, NBloss, SSIM, combined.")
    parser.add_argument("--SSIM_window_size", default=11, type=int, help="SSIM window size.")
    parser.add_argument("--model_selection", default="svd", type=str,
                        help="Model: svd (currently supported in this script).")

    parser.add_argument("--save_name", default="hic013", type=str,
                        help="Checkpoint name prefix.")
    parser.add_argument("--zero_bin_threshold", default=0, type=int,
                        help="Skip samples with <= this many nonzero entries.")
    parser.add_argument("--loss_weights", default=1.0, type=float,
                        help="(Reserved) weights for multi-res losses (if used).")

    # New (optional) SVD params
    parser.add_argument("--svd_start", default=0, type=int, help="SVD start component index.")
    parser.add_argument("--svd_n_components", default=10, type=int, help="Number of SVD components computed.")

    # New (optional) checkpointing
    parser.add_argument("--save_every", default=50, type=int, help="Save checkpoint every N steps.")

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    print(config)
    print("PyTorch:", torch.__version__)

    # Respect CUDA_VISIBLE_DEVICES if set; otherwise allow selecting GPU by index.
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(args.gpu))
        except Exception:
            # Not fatal; sometimes CUDA_VISIBLE_DEVICES remaps indices
            pass

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_default_dtype(torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # wandb
    if args.wandb:
        import wandb
        # Default to local ./wandb if WANDB_DIR not set
        wandb_dir = os.getenv("WANDB_DIR", str(Path(args.log_dir) / "wandb"))
        wandb.init(project="hic2self", dir=wandb_dir)
        wandb.config.update(config)

    # Paths
    log_path = Path(args.log_dir) / str(args.v)
    log_path.mkdir(parents=True, exist_ok=True)

    # Save setup info
    with open(log_path / "setup.txt", "a+", encoding="utf-8") as f:
        f.write(f"\nVersion: {args.v}")
        f.write(f"\nBatch Size: {args.b}")
        f.write(f"\nInitial Learning Rate: {args.lr}")
        f.write(f"\nMask width/type: {args.mask_width} / {args.mask_type}")
        f.write(f"\nLoss: {args.loss_selection}")
        f.write(f"\nModel: {args.model_selection}")
        f.write(f"\nComments: {args.m}\n")

    # Model
    if args.model_selection.lower() != "svd":
        raise ValueError(
            f"This updated script currently supports model_selection='svd' only. "
            f"Got: {args.model_selection}"
        )

    model = HiC2SelfSVDMaskNet(
        side=False,
        round_mu=False,
        include_pi=False,
        svd_start=args.svd_start,
        svd_n_components=args.svd_n_components,
    ).to(device)

    masker = Masker2(width=args.mask_width, mode="zero", mask_type=args.mask_type,
                     side_info=False, infer_single_pass=False, include_mask_as_input=False)

    if args.wandb:
        wandb.watch(model, log="all")

    # Loss
    if args.loss_selection == "MSE":
        loss_function = nn.MSELoss(reduction="sum")
    else:
        loss_function = ZINB2()

    # Data
    if args.save_name in ["nagano", "GSE129029", "scHiCAR", "GSE211395",
                          "ultrahigh", "luo", "bulk_retrain", "CD4"]:
        dataset = data_preparation_nagano(args.input_file_name)
    else:
        dataset = data_preparation(args.input_file_name)

    data_loader = DataLoader(dataset, batch_size=args.b, shuffle=True)

    # Training loop
    t0 = time.time()
    global_step = 0

    for epoch in range(int(args.e)):
        lr = max(float(args.lr) * (0.5 ** (epoch // 16)), 1e-6)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        model.train()
        print(f"Epoch {epoch}/{args.e - 1} | lr={lr:g}")

        for i, batch in enumerate(data_loader):
            idx, train_image = batch  # train_image: (B,1,H,W)

            train_image = train_image.to(torch.float32).to(device)

            # Skip extremely sparse samples (keep your original behavior)
            # NOTE: for B>1 this checks total nonzeros across the batch
            if torch.count_nonzero(train_image).item() <= args.zero_bin_threshold:
                continue

            # IMPORTANT: use i % n_masks, not raw i
            mask_i = i % len(masker)

            _, mask = masker.mask(train_image, mask_i)
            mask = mask.to(device)

            # Forward
            net_output = model(train_image, mask)  # (B,2,H,W) => mu, theta
            mu = net_output[:, 0:1, :, :]
            theta = net_output[:, 1:2, :, :]

            # Compute loss on masked pixels
            if args.loss_selection == "MSE":
                loss = loss_function(train_image * mask, mu * mask)
            elif args.loss_selection == "NBloss":
                loss = torch.sum(loss_function.nbloss(train_image * mask, mu * mask, theta * mask))
            elif args.loss_selection == "SSIM":
                loss = torch.sum(loss_function.ssim(train_image * mask, mu * mask, window_size=args.SSIM_window_size))
            elif args.loss_selection == "combined":
                loss_nb = torch.mean(loss_function.nbloss(train_image * mask, mu * mask, theta * mask))
                loss_ssim = torch.mean(loss_function.ssim(train_image * mask, mu * mask, window_size=args.SSIM_window_size))
                loss = loss_nb + loss_ssim
            else:
                raise ValueError(f"Unknown loss_selection: {args.loss_selection}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"loss": float(loss.item()), "lr": lr, "epoch": epoch}, step=global_step)
                if args.loss_selection == "combined":
                    wandb.log({"NBloss": float(loss_nb.item()), "SSIM": float(loss_ssim.item())}, step=global_step)

            # Save checkpoint
            if global_step % int(args.save_every) == 0:
                ckpt_name = f"{args.save_name}_{args.loss_selection}_{args.model_selection}_step{global_step}.pt"
                ckpt_path = log_path / ckpt_name
                torch.save(model.state_dict(), ckpt_path.as_posix())
                print(f"[step {global_step}] saved: {ckpt_path}")

            global_step += 1

    t1 = time.time()
    print("Training time (s):", t1 - t0)


if __name__ == "__main__":
    main()
