# Training HiC2Self

This folder contains scripts for training the HiC2Self model using masked self-supervision on Hi-C contact maps.

---

## Step 1. Data preparation

Before training, prepare the input data using the scripts in: `data_processing`. This step generates diagonally cropped Hi-C submatrices that will be used as training samples.

## Step 2. Train HiC2Self

### Required inputs

- Processed data from Step 1  
  (e.g. `GM12878_hic001_noisy_mat_5kb_size400_chr3.pkl`)
- Training scripts in this folder

### Example training command

```bash
python model_training/hic2self_train_gpu_hic2self_SVD_mask.py \
  --input_file_name data/GM12878/processed/GM12878_hic001_noisy_mat_5kb_size400_chr3.pkl \
  --log_dir logs/hic2self \
  --lr 1e-5 \
  --v 0.01 \
  --b 1 \
  --e 10 \
  --mask_width 3 \
  --save_name bulk_retrain \
  --model_selection svd \
  --loss_selection NBloss \
  --mask_type symmetric_grid

```

Arguments 
```
1. Required
--input_file_name  
  Path to the processed input data generated in Step 1.
--log_dir  
  Directory where training logs and model checkpoints will be saved.

2. Training configuration
--wandb  
  Enable Weights & Biases logging (optional).
--lr
  Learning rate.
--v  
  Experiment version identifier (used to organize output folders).
--b  
  Batch size.  
  **Note:** Must be set to `1` when using SVD-based training.
--e  
  Total number of training epochs.

3. Model configuration
--mask_width  
  Width of the masking grid. Larger values result in sparser masking.
--model_selection  
  Model architecture to use.  
  Currently supported:
  - `svd` — SVD-enhanced HiC2Self model.
--mask_type  
  Masking strategy applied during training:
  - grid
  - symmetric_grid
  - diagonal
  - horizontal
--loss_selection  
  Loss function used for training:
  - `NBloss` — Negative binomial loss (default)
  - `SSIM` — Structural similarity loss

4. Save
--save_name  
  Prefix used for naming saved model checkpoints.

```

After training, model checkpoints and logs will be written to:

```
logs/hic2self/<version>/
```

