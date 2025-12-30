#!/bin/bash

# Clean up environment for HiC2Self -- Dec 4, 2025

ENV_PTH="/data1/lesliec/yangr/lilac_yangr2/environments"

#1. Download and install conda and mamba 
cd "${ENV_PTH}" # cd into the lab folder, create an environment folder to store all the environments 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# eval "$(/data1/lesliec/yangr/lilac_yangr2/environments/miniconda3/bin/conda shell.YOUR_SHELL_NAME hook)"

# add miniconda3 to PATH
export PATH="${ENV_PTH}/miniconda3/bin:$PATH"
source ~/.bashrc
# test whether conda was added to PATH
which conda

#2. Install mamba and create mamba environment 
# Install mamba in the base environment 
conda install mamba -n base -c conda-forge

#3. Create environment and install the packages 
mamba env create \
    --prefix /data1/lesliec/yangr/lilac_yangr2/environments/hic2self_env \
    --file /data1/lesliec/yangr/lilac_yangr2/hic2self/environment.yml

#4. Activate the environment 
eval "$(mamba shell hook --shell bash)"
mamba activate /data1/lesliec/yangr/lilac_yangr2/environments/hic2self_env

# echo 'eval "$(mamba shell hook --shell bash)"' >> ~/.bashrc
# alias hic2env="mamba activate /data1/lesliec/yangr/lilac_yangr2/environments/hic2self_env"

#5. Install additional packages 
# pip install wandb --upgrade
# wandb login 
# wandb profile appended: /home/BTC_yangr/.netrc

#6. Create jupyter environment
export JUPYTER_DATA_DIR=/data1/lesliec/yangr/lilac_yangr2/hic2self/notebooks
python -m ipykernel install --user --name=pytorch_env --display-name "Python (pytorch_env)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:Trues


