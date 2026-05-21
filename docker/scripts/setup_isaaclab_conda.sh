#!/usr/bin/env bash
# Create conda env "isaaclab" and install Isaac Lab + DynamicVLA simulation extras.
# Expects: ISAACLAB_PATH, ISAACSIM_PATH (or _isaac_sim symlink already under ISAACLAB_PATH).
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-/opt/IsaacLab}"
ISAACSIM_PATH="${ISAACSIM_PATH:-/isaac-sim}"
CONDA_DIR="${CONDA_DIR:-/opt/conda}"
ENV_NAME="${ISAAC_CONDA_ENV:-isaaclab}"
REQUIREMENTS_ISAAC="${REQUIREMENTS_ISAAC:-/workspace/DynamicVLA/requirements-isaac.txt}"
# When building the unified Docker image, REQUIREMENTS_ISAAC is set to /tmp/env/requirements-isaac.txt

if [[ ! -d "${ISAACLAB_PATH}" ]]; then
  echo "ERROR: ISAACLAB_PATH not found: ${ISAACLAB_PATH}" >&2
  exit 1
fi

if [[ ! -d "${ISAACSIM_PATH}" ]]; then
  echo "ERROR: ISAACSIM_PATH not found: ${ISAACSIM_PATH}" >&2
  exit 1
fi

# Link Isaac Sim into Isaac Lab tree (same as bare-metal install).
ln -sfn "${ISAACSIM_PATH}" "${ISAACLAB_PATH}/_isaac_sim"

# shellcheck source=/dev/null
source "${CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 pip -y
fi

conda activate "${ENV_NAME}"

# Isaac Lab 2.2.x extensions (matches manual pip install -e flow when isaaclab.sh -i is unavailable).
pip install --upgrade "pip<26" "setuptools<81" wheel
for pkg in isaaclab isaaclab_assets isaaclab_tasks; do
  pip install -e "${ISAACLAB_PATH}/source/${pkg}"
done
pip install -e "${ISAACLAB_PATH}/source/isaaclab_rl[none]"
pip install -e "${ISAACLAB_PATH}/source/isaaclab_mimic[none]"

if [[ -f "${REQUIREMENTS_ISAAC}" ]]; then
  pip install -r "${REQUIREMENTS_ISAAC}"
fi

# conda activate hooks (paths used by evaluate.py / isaaclab.sh)
ACTIVATE_D="${CONDA_DIR}/envs/${ENV_NAME}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_D}"
cat > "${ACTIVATE_D}/dynamicvla_isaaclab.sh" <<EOF
export ISAACLAB_PATH="${ISAACLAB_PATH}"
export ISAAC_PATH="${ISAACLAB_PATH}/_isaac_sim"
export OMNI_KIT_ACCEPT_EULA=YES
source "\${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" 2>/dev/null || true
EOF

echo "isaaclab env ready: ${ENV_NAME}"
python -c "import isaaclab; print('isaaclab', isaaclab.__version__)"
