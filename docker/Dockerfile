# DynamicVLA — public pulls only: nvidia/cuda (Docker Hub) + Isaac Sim zip + dual conda
# No nvcr.io / docker login. See docker/README.md and docker/CUDA-VERSIONS.md
#   isaaclab          — Isaac Lab v2.2.1 + requirements-isaac.txt
#   dynamicvla-train  — PyTorch 2.7.1 (cu126) + requirements.txt
#
# Base CUDA/cuDNN chosen from host env audit (see docker/CUDA-VERSIONS.md):
#   dynamicvla-train → torch 2.7.1+cu126, nvidia-cudnn-cu12 9.5.x
#   isaaclab         → torch 2.7.0+cu128 (via Isaac Lab deps); Isaac Sim bundles its own runtime
#
# Build (repo root):
#   docker build -f docker/Dockerfile -t dynamicvla:latest .
#
# Optional: use a local Isaac Sim zip (skip download):
#   docker build -f docker/Dockerfile -t dynamicvla:latest \
#     --build-arg ISAACSIM_ZIP_CONTEXT=docker/vendor \
#   (place isaac-sim-standalone-4.5.0-linux-x86_64.zip under docker/vendor/)

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG ISAACSIM_ZIP_URL=https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip
# Set to a path under build context to COPY a local zip instead of downloading (e.g. docker/vendor)
ARG ISAACSIM_ZIP_CONTEXT=

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    ISAACSIM_PATH=/opt/isaacsim \
    ISAACLAB_PATH=/opt/IsaacLab \
    ISAACLAB_CLONE_TAG=v2.2.1 \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git wget unzip bzip2 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libvulkan1 \
    build-essential libmemcached-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Isaac Sim 4.5.0 standalone (same layout as bare-metal _isaac_sim)
RUN mkdir -p /tmp/isaac_dl
COPY docker/scripts/install_isaacsim.sh /opt/install_isaacsim.sh
RUN chmod +x /opt/install_isaacsim.sh
# Optional local zip: docker/vendor/isaac-sim-standalone-4.5.0-linux-x86_64.zip
COPY ${ISAACSIM_ZIP_CONTEXT:-docker/empty-vendor} /tmp/isaac_vendor/
RUN ISAACSIM_ZIP_URL="${ISAACSIM_ZIP_URL}" \
    ISAACSIM_PATH="${ISAACSIM_PATH}" \
    ISAAC_VENDOR_DIR=/tmp/isaac_vendor \
    /opt/install_isaacsim.sh

# Miniconda (shared by both envs)
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --set auto_activate_base false \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN git clone --depth 1 --branch "${ISAACLAB_CLONE_TAG}" \
      https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" \
    || (git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" \
        && cd "${ISAACLAB_PATH}" && git checkout "${ISAACLAB_CLONE_TAG}")

WORKDIR /workspace/DynamicVLA

COPY environment_train.yaml requirements.txt requirements-isaac.txt /tmp/env/
COPY docker/scripts/setup_isaaclab_conda.sh docker/scripts/setup_train_conda.sh /opt/
RUN chmod +x /opt/setup_isaaclab_conda.sh /opt/setup_train_conda.sh

ENV REQUIREMENTS_ISAAC=/tmp/env/requirements-isaac.txt
RUN REQUIREMENTS_ISAAC=/tmp/env/requirements-isaac.txt /opt/setup_isaaclab_conda.sh

RUN ENV_YAML=/tmp/env/environment_train.yaml REQUIREMENTS=/tmp/env/requirements.txt /opt/setup_train_conda.sh

RUN "${CONDA_DIR}/bin/conda" clean -afy \
    && rm -rf /tmp/isaac_dl /tmp/isaac_vendor /root/.cache/pip

RUN cat >> /etc/bash.bashrc <<'BASHRC'

source /opt/conda/etc/profile.d/conda.sh
echo "DynamicVLA: conda activate isaaclab | conda activate dynamicvla-train"
echo "Base image CUDA: 12.6 + cuDNN (devel); see docker/CUDA-VERSIONS.md"
BASHRC

SHELL ["/bin/bash", "-c"]
RUN source "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && conda activate isaaclab \
    && python -c "import isaaclab; print('isaaclab', isaaclab.__version__)" \
    && conda activate dynamicvla-train \
    && python -c "import torch; print('dynamicvla-train', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /workspace/DynamicVLA
CMD ["/bin/bash"]
