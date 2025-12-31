# HiC2Self Post-processing

This folder contains utilities to (1) run inference with a trained HiC2Self model and generate **chromosome-scale predictions**, and (2) call **significant interactions** from predicted contact maps using **HiC-DC+**.

## Step 1 — Generate chromosome-scale predictions (GPU)

This step loads:
- a trained HiC2Self checkpoint (`.pt_model`)
- a list of cropped input matrices (`.pkl`)
and outputs:
- `*_predictions.txt` (bin1, bin2, predicted value)
- `*_predictions_for_hic.txt` (HiCCUPS-like 9-column format)

### Example

Run from the **repo root**:

```bash
python prediction_postprocessing/hic2self_merge_chromosome_predictions_gpu.py \
  --loss_selection NBloss \
  --log_version 0.01 \
  --log_path outputs/hic2self_logs \
  --input_path data/GM12878/processed \
  --input_file GM12878_hic001_noisy_mat_5kb_size400_chr3.pkl \
  --chrom chr3 \
  --resolution 5000 \
  --mat_size 400 \
  --assembly hg38 \
  --chrom_sizes_file data/assembly/hg38.chrom.sizes \
  --model_name bulk_retrain \
  --layered_no no_layered \
  --mask_width 3 \
  --mask_type symmetric_grid \
  --cell_type GM12878

```

Arguments 
```
1. Model checkpoint selection
--log_path: base folder for trained models (default: outputs/hic2self_logs)
--log_version: subfolder/version under --log_path (e.g. 0.01)
--model_name: model prefix used during training (e.g. bulk_retrain)
--loss_selection: loss function name used during training (e.g. NBloss)
--layered_no: training framework tag used in checkpoint naming (no_layered or layered)

2. Input
--input_path: folder containing the input cropped matrices
--input_file: pickle file containing a list of cropped matrices (one per window)
--chrom: chromosome being predicted (e.g. chr3)
--resolution: bin size in bp (e.g. 5000)
--mat_size: matrix size per window (e.g. 400)

3. Masking / inference
--mask_width: mask grid width (e.g. 3)
--mask_type: symmetric_grid, grid, diagonal, or horizontal

4. Genome assembly
--assembly: e.g. hg38 (label only; used for naming)
--chrom_sizes_file: chrom sizes file (e.g. data/assembly/hg38.chrom.sizes)

5. Naming
--cell_type: prefix used in output filenames (e.g. GM12878)
```

### Output location
```
<INPUT_PATH>/../predictions/
```
For example, 
```
data/GM12878/predictions/
  GM12878_predictions_5000_chr3_log0.01_maskwidth3_symmetric_grid_predictions.txt
  GM12878_predictions_5000_chr3_log0.01_maskwidth3_symmetric_grid_predictions_for_hic.txt
```

## b) Step 2. Call significant interactions using HiC-DC+

### Install HiC-DC+

Install HiC-DC+ following the official instructions:

HiC-DC+ GitHub: https://github.com/mervesa/HiCDCPlus

### Run HiC-DC+ to identify significant interactions 

This repo includes an example R pipeline:

prediction_postprocessing/hicdcplus_significant_interactions_5kb.R
(adjust paths/parameters as needed)

Example invocation (replace placeholders):

```
Rscript prediction_postprocessing/hicdcplus_significant_interactions_5kb.R \
  10000 \
  outputs/predictions/GM12878_predictions_5000_chr3_log0.01_maskwidth3_symmetric_grid_predictions.txt \
  outputs/hicdcplus/features/ \
  outputs/hicdcplus/intermediate/ \
  hg38 \
  3
```








