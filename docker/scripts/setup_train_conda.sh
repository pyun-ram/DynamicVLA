#!/usr/bin/env bash
# Create conda env "dynamicvla-train" (PyTorch 2.7.1 + project requirements).
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/opt/conda}"
ENV_NAME="${TRAIN_CONDA_ENV:-dynamicvla-train}"
ENV_YAML="${ENV_YAML:-/tmp/env/environment_train.yaml}"
REQUIREMENTS="${REQUIREMENTS:-/tmp/env/requirements.txt}"

# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env create -f "${ENV_YAML}"
fi

conda activate "${ENV_NAME}"
pip install --no-cache-dir -r "${REQUIREMENTS}"

python -c "import torch, transformers, lerobot; print('train env OK', torch.__version__, torch.cuda.is_available())"
