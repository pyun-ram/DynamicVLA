# DynamicVLA — both conda envs in one image (matches bare-metal setup):
#   - dynamicvla-train  (training / inference, torch==2.7.1)
#   - isaaclab          (Isaac Sim 4.5 + Isaac Lab v2.2.1 + requirements-isaac.txt)
#
# Prerequisite: docker login nvcr.io
#
# Build:
#   docker build -f docker/Dockerfile -t dynamicvla:latest .
#
# Run:
#   docker run --gpus all -it --rm \
#     -e OMNI_KIT_ACCEPT_EULA=YES \
#     -v "$(pwd)":/workspace/DynamicVLA \
#     --shm-size=16g \
#     dynamicvla:latest bash
#
# Terminal A (simulation):
#   conda activate isaaclab
#   cd /workspace/DynamicVLA/simulations && python evaluate.py ... --headless --enable_cameras
#
# Terminal B (inference, same container second shell or host network):
#   conda activate dynamicvla-train
#   cd /workspace/DynamicVLA && python scripts/inference.py -p pretrained_weights/dynamic-vla-DOM -r euler -d -s

ARG ISAACSIM_IMAGE=nvcr.io/nvidia/isaac-sim:4.5.0
FROM ${ISAACSIM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    ISAACSIM_PATH=/isaac-sim \
    ISAACLAB_PATH=/opt/IsaacLab \
    ISAACLAB_CLONE_TAG=v2.2.1 \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git wget bzip2 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential libmemcached-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (shared by both envs)
RUN if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then \
      curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
      && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
      && rm /tmp/miniconda.sh; \
    fi \
    && "${CONDA_DIR}/bin/conda" config --set auto_activate_base false \
    && "${CONDA_DIR}/bin/conda" clean -afy

# Isaac Lab v2.2.1
RUN git clone --depth 1 --branch "${ISAACLAB_CLONE_TAG}" \
      https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" \
    || (git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}" \
        && cd "${ISAACLAB_PATH}" && git checkout "${ISAACLAB_CLONE_TAG}")

RUN if [[ ! -d "${ISAACSIM_PATH}" ]]; then \
      if [[ -d /isaac-sim ]]; then ln -sfn /isaac-sim "${ISAACSIM_PATH}"; \
      elif [[ -d /home/ubuntu/isaac-sim ]]; then ln -sfn /home/ubuntu/isaac-sim "${ISAACSIM_PATH}"; \
      fi; \
    fi

WORKDIR /workspace/DynamicVLA

COPY environment_train.yaml requirements.txt requirements-isaac.txt /tmp/env/
COPY docker/scripts/setup_isaaclab_conda.sh docker/scripts/setup_train_conda.sh /opt/
RUN chmod +x /opt/setup_isaaclab_conda.sh /opt/setup_train_conda.sh

# 1) isaaclab env (Isaac Lab + simulation extras)
ENV REQUIREMENTS_ISAAC=/tmp/env/requirements-isaac.txt
RUN REQUIREMENTS_ISAAC=/tmp/env/requirements-isaac.txt /opt/setup_isaaclab_conda.sh

# 2) dynamicvla-train env (VLA training / inference)
RUN ENV_YAML=/tmp/env/environment_train.yaml REQUIREMENTS=/tmp/env/requirements.txt /opt/setup_train_conda.sh

RUN "${CONDA_DIR}/bin/conda" clean -afy

# Shell hints (no default env — pick explicitly like on bare metal)
RUN cat >> /etc/bash.bashrc <<'BASHRC'

source /opt/conda/etc/profile.d/conda.sh
echo "DynamicVLA conda envs:  conda activate isaaclab | conda activate dynamicvla-train"
BASHRC

SHELL ["/bin/bash", "-c"]
RUN source "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && conda activate isaaclab \
    && python -c "import isaaclab; print('isaaclab', isaaclab.__version__)" \
    && conda activate dynamicvla-train \
    && python -c "import torch; print('dynamicvla-train', torch.__version__)"

WORKDIR /workspace/DynamicVLA
CMD ["/bin/bash"]
