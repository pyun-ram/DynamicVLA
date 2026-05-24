#!/usr/bin/env bash
# Build dynamicvla image — public pulls only (no nvcr.io login).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

IMAGE="${DYNAMICVLA_IMAGE:-dynamicvla:latest}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04}"

echo "Pre-pull base (Docker Hub): ${CUDA_IMAGE}"
docker pull "${CUDA_IMAGE}"

VENDOR_ZIP="docker/vendor/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
if [[ ! -f "${VENDOR_ZIP}" ]] && ! compgen -G "docker/vendor/*.zip" >/dev/null; then
  echo "ERROR: Isaac Sim zip not found in docker/vendor/" >&2
  echo "Run:  bash setup/download_assets.sh" >&2
  exit 1
fi
echo "Using bind-mount docker/vendor/ (no COPY of zip into build context layers)"

export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile -t "${IMAGE}" --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" .
