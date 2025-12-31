#!/bin/bash
set -e

# ======================================
# HiC2Self environment setup script
# ======================================

# Get the directory of this script (repo root)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define environment paths relative to repo
ENV_DIR="${REPO_DIR}/envs"
ENV_NAME="hic2self_env"
ENV_PATH="${ENV_DIR}/${ENV_NAME}"

echo "Repo directory: ${REPO_DIR}"
echo "Environment path: ${ENV_PATH}"

# --------------------------------------
# 1. Install Miniconda (if not exists)
# --------------------------------------
if [ ! -d "${ENV_DIR}/miniconda3" ]; then
    echo "Installing Miniconda..."
    mkdir -p "${ENV_DIR}"
    cd "${ENV_DIR}"

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "${ENV_DIR}/miniconda3"
fi

# Activate conda
export PATH="${ENV_DIR}/miniconda3/bin:$PATH"
source "${ENV_DIR}/miniconda3/etc/profile.d/conda.sh"

# --------------------------------------
# 2. Install mamba
# --------------------------------------
conda install -y -n base -c conda-forge mamba

# --------------------------------------
# 3. Create environment from YAML
# --------------------------------------
mamba env create \
    --prefix "${ENV_PATH}" \
    --file "${REPO_DIR}/environment.yml"

# --------------------------------------
# 4. Activate environment
# --------------------------------------
mamba activate "${ENV_PATH}"

# --------------------------------------
# 5. (Optional) Jupyter kernel
# --------------------------------------
python -m ipykernel install --user --name hic2self --display-name "Python (hic2self)"

# --------------------------------------
# 6. Environment variables (optional)
# --------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================"
echo "HiC2Self environment setup complete!"
echo "Activate with:"
echo "  mamba activate ${ENV_PATH}"
echo "======================================"
