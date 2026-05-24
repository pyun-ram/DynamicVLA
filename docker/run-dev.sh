#!/usr/bin/env bash
# Start or attach a long-lived dev container (docker run, no compose).
# Usage (from repo root):
#   ./docker/run-dev.sh          # start if missing, else exec bash
#   ./docker/run-dev.sh --new    # remove old container and start fresh
set -euo pipefail

IMAGE="${DYNAMICVLA_IMAGE:-dynamicvla:latest}"
NAME="${DYNAMICVLA_CONTAINER:-dynamicvla-dev}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DYNAMICVLA_DATA_ROOT:-$(dirname "${REPO_ROOT}")}"

start_new=false
for arg in "$@"; do
  case "$arg" in
    --new) start_new=true ;;
    -h|--help)
      echo "Usage: $0 [--new]"
      echo "  IMAGE=${IMAGE}  NAME=${NAME}"
      exit 0
      ;;
  esac
done

if $start_new; then
  docker rm -f "${NAME}" 2>/dev/null || true
fi

if ! docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Starting ${NAME} from ${IMAGE} ..."
  docker run --gpus all -d \
    --name "${NAME}" \
    -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
    -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}" \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e ACCEPT_EULA=Y \
    -v "${DATA_ROOT}:/workspace" \
    -v "${REPO_ROOT}:/workspace/DynamicVLA" \
    -w /workspace/DynamicVLA \
    --shm-size="${DYNAMICVLA_SHM:-16g}" \
    "${IMAGE}" \
    sleep infinity
elif ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Starting stopped container ${NAME} ..."
  docker start "${NAME}" >/dev/null
fi

exec docker exec -it "${NAME}" bash
