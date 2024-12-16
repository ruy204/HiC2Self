# Training script for HiC2Self 

## Step 1. Data preparation

- For data preparation, please follow /data_processing

## Step 2. Train HiC2Self 

#### Start materials 
- processed data from Step 1, using chromosome 3 as an example (Extracted_noisy_mat_5kb_size400_chr3.txt)
- additional functional scripts from this folder

#### Training script

```
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

```

Arguments 
```
--input_file_name: name and path of processed input data from Step 1
-- log_dir: location to save the training logs
-- wandb: if you have wandb installed, you could use wandb to track the training
-- lr: learning rate
-- v: log version of the training 
-- b: batch size, please note that we need to set batch size to 1 in order to conduct SVD for every matrix 
-- e: total training epochs 
-- mask_width: width between masked pixels. Higher mask width gives more sparse masks. 
-- save_name: name prefix of the trained model 
-- model_selection:  (svd) model with SVD enhanced low-rank signals
-- loss_selection: (NBloss) negative log-likelihood of the negative-binomial loss 
-- mask_type: symmetric_grid mask 
```

After training, the trained model can be found at 

```
/data/hic2self_logs/0.01
```

