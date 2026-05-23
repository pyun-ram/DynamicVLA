#!/usr/bin/env bash
# Create conda env "dynamicvla-train" (PyTorch 2.7.1 + project requirements).
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/opt/conda}"
ENV_NAME="${TRAIN_CONDA_ENV:-dynamicvla-train}"
ENV_YAML="${ENV_YAML:-/tmp/env/environment_train.yaml}"
REQUIREMENTS="${REQUIREMENTS:-/tmp/env/requirements.txt}"

# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env create -f "${ENV_YAML}"
fi

conda activate "${ENV_NAME}"
# Pin cu126 wheels to match audited host env (torch 2.7.1+cu126, cudnn 9.5.x via nvidia-cudnn-cu12)
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
# requirements.txt also pins torch; install the rest only to keep cu126 wheels
grep -vE '^(torch|torchvision)([=<> ]|$)' "${REQUIREMENTS}" > /tmp/requirements-no-torch.txt
pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

python -c "import torch, transformers, lerobot; print('train env OK', torch.__version__, torch.cuda.is_available())"
