# HiC2Self

HiC2Self: Self-supervised denoising for bulk and single-cell Hi-C contact maps 

<div align="center">
  <img src="figures/HiC2Self_figure1-01.jpg" width="500"><br>
  <em>Overview of the HiC2Self framework.</em>
</div>

- Manuscript: [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.21.624767v1)
- Poster: HiC2Self_poster.pdf

## Requirements

We recomment creating the enrivonment using the provided `environment.yml` file:
```
mamba env create -f environment.yml
```

* Core dependencies
    * python=3.10
    * PyTorch, NumPy, SciPy, scikit-learn, h5py, anndata, UMAP-learn
* Visualization
    * matplotlib, seaborn
    * coolbox
* Other
    * torchvision 
    * pytorch-ssim

A complete walkthrough for setting up the environment can be found in `create_hic2self_env.sh`.

## Data Processing

HiC2Self does __NOT__ require Hi-C data normalization. The HiC2Self framework consists of three main steps:

1. Data preprocessing

Raw Hi-C contact maps are first processed by cropping them along the diagonal into equally sized, diagonally symmetric submatrices. These submatrices serve as the basic training and inference units for the model.
This step is implemented in: `data_processing/input_data_preparation_full_chr_split.py`

2. Model training

HiC2Self is trained using the set of diagonally symmetric submatrices generated in the preprocessing step. The model learns to denoise low-coverage contact maps by predicting high-quality contact patterns from masked corrupted inputs.

3. Inference

During inference, HiC2Self predicts each diagonal submatrix independently. Overlapping predictions along the diagonal are then averaged to reconstruct the full chromosome-scale contact map.
Inference using trained models is implemented in the `prediction_postprocessing` directory.

For detailed implementation and usage instructions, please refer to the `data_processing` and `prediction_postprocessing` directories.

## Keep connected 

- Email: [ruiyang204@gmail.com](ruiyang204@gmail.com)
- others: [@RuiYang53922541](https://twitter.com/RuiYang53922541), [@ruiyang.bsky.social](https://bsky.app/profile/ruiyang.bsky.social)

