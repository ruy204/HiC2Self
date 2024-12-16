# Post-processing with trained HiC2Self model 

## a) Step 1. Generate predictions for the entire chromosome and assembly into a .hic file

```
python /HiC2Self/prediction_postprocessing/hic2self_merge_chromosome_predictions_gpu.py \
--loss_selection NBloss \
--log_version 0.01 \
--input_path /data/GM12878/processed \
--input_file GM12878_hic001_noisy_mat_5kb_size400_chr3.txt \
--chrom chr3 \
--resolution 5000 \
--mat_size 400 \
--assembly hg38 \
--model_name 'bulk_retrain' \
--layered_no 'no_layered' \
--mask_width 3 \
--mask_type symmetric_grid \
--cell_type GM12878 
```

Arguments 
```
#a) Load prefix for model names
--loss_function: NBloss function used to train the model 
--log_version: version of model 
--model_name: model prefix name when training
--layered_no: whether to use hierarchical training framework to train the model (default is no in this example)
--mask_width: width of the mask during training 
--mask_type: default is 'symmetric_grid' in this example

#b) Load prefix for input and output data
--input_path
--input_file
--chrom
--resolution
--assembly hg38
--cell_type: GM12878 in this example

#c) Load additional information 
--mat_size: size of the input matrices (400 in this example)

```

## b) Step 2. Call significant interactions using HiC-DC+

### <i> Prepare R environment 

Please install HiC-DC+ following the instructions here: [HiC-DC+ github](https://github.com/mervesa/HiCDCPlus)

### <2> Run HiC-DC+ to identify significant interactions 

```
Rscript /data/leslie/yangr2/hic2self/scripts/hicdcplus/hicdcplus_normalization_ppl.R \
10000 \
/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/predictions/predictions.txt
'/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/features/' \
'/data/leslie/yangr2/hic2self/data/deeploop/GM12878/predictions/intermediate/' \
'hg19' \
3
```








