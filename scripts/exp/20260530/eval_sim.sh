#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Online eval — SIMULATOR side (DynamicVLA / IsaacLab), runs in conda `isaaclab`.
#
# This launches simulations/evaluate.py, which is the ZMQ *server*: it BINDS the
# obs (PUB) and action (PULL) sockets and streams observations to the policy
# client (3d_diffuser_actor/scripts/exp/20260530/eval_client.sh).
#
# Pairs with the 20260530 model (scripts/exp/20260530/train.sh):
#   diffuser_actor, pick, C120 / H3 / DT100 / rectified_flow / 6D / 360x480.
#
# Usage (run on the HOST, in the DynamicVLA repo root):
#   conda activate isaaclab          # or let the script do it
#   bash scripts/exp/20260530/eval_sim.sh [NUM_EPISODES] [ENV_CFG] [GPU_ID]
#
# Start THIS first, then start the client in the docker container.
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: no `set -u`. The isaaclab conda activation sources
# IsaacLab/_isaac_sim/setup_conda_env.sh, which references $ZSH_VERSION unbound;
# under nounset that aborts activation and the script exits silently.
set -eo pipefail

PROJECT_ROOT=$(pwd)

NUM_EPISODES=${1:-10}
# env_cfg is an IsaacLab env-config JSON (scene + object + instruction).
# NOTE: the 20260530 model is a *pick* model, but tests/ currently only ships
# place / long-horizon configs. Point this at a pick env_cfg that matches the
# trained task; the default below is a placeholder so the script is runnable.
ENV_CFG=${2:-tests/1-1_place_franka_apple13d_O02_01048988_09b0.json}
GPU_ID=${3:-0}

# ZMQ: bind on 0.0.0.0 so the policy client inside the docker container can
# reach these ports via the docker0 gateway (172.17.0.1). 127.0.0.1 would only
# be reachable from the host itself.
HOST=0.0.0.0
IMG_PORT=3186
ACT_PORT=3188

export CUDA_VISIBLE_DEVICES=${GPU_ID}

cd "${PROJECT_ROOT}"
# `conda activate` needs conda.sh sourced in a non-interactive shell
# (`conda init` only patches interactive .bashrc, not `bash script.sh`).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaaclab

python simulations/evaluate.py \
    --headless --enable_cameras \
    --env_cfg "${ENV_CFG}" \
    --scene_dir "${PROJECT_ROOT}/scenes" \
    --object_dir "${PROJECT_ROOT}/objects" \
    --host ${HOST} \
    --img_port ${IMG_PORT} \
    --act_port ${ACT_PORT} \
    -n "${NUM_EPISODES}" \
    --save \
    --execution_mode streaming \
    --output_dir "${PROJECT_ROOT}/output/online_eval/20260530"
