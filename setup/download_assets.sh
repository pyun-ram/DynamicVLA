#!/usr/bin/env bash
# Download assets for DynamicVLA on a fresh machine (docker + nvidia-smi only).
#
# Default (no flags): Isaac Sim zip only.
# With --train / --test / ... alone: only those assets (no Isaac unless --isaac).
#
# Examples:
#   ./download_assets.sh
#   ./download_assets.sh --train
#   ./download_assets.sh --train --isaac
#   ./download_assets.sh --test --objects
#   ./download_assets.sh --all
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=setup/urls.sh
source "${SCRIPT_DIR}/urls.sh"
# shellcheck source=setup/lib.sh
source "${SCRIPT_DIR}/lib.sh"

DATA_ROOT="${DYNAMICVLA_DATA_ROOT:-$(dirname "${REPO_ROOT}")}"
VENDOR_DIR="${REPO_ROOT}/docker/vendor"
CACHE_DIR="${DATA_ROOT}/.download_cache"

DO_ISAAC=0
DO_TEST=0
DO_OBJECTS=0
DO_SCENES=0
DO_TRAIN=0
DO_MODEL=0
FORCE=0
WANT_OPTIONAL=0
EXPLICIT_ISAAC=0
EXPLICIT_NO_ISAAC=0

usage() {
  cat <<'EOF'
Usage: ./download_assets.sh [OPTIONS]

Downloads assets for DynamicVLA. Run from repo root after git clone.

Options:
  --data-root PATH   Project data root (default: parent of repo clone)
                     Layout: objects/ scenes/ tests/ test-envs.txt datasets/
  --isaac            Download Isaac Sim zip -> docker/vendor/
  --no-isaac         Skip Isaac Sim zip (default when using --train / --test / ...)
  --test             DOM Testing Set (Google Drive gdown) -> tests/ + test-envs.txt
  --objects          DOM 3D Objects (Google Drive gdown) -> objects/
  --scenes           DOM 3D Scenes -> scenes/
  --train            DOM training set (Hugging Face hzxie/DOM) -> datasets/DOM
  --model            Pretrained weights (Hugging Face hzxie/dynamic-vla-DOM)
                     -> pretrained_weights/dynamic-vla-DOM
  --all              Enable --test --objects --scenes --train (not --model)
  --force            Re-download HF assets; resume if .incomplete files exist
  -h, --help         Show this help

Then build the image:
  ./build_docker.sh

Environment:
  DYNAMICVLA_DATA_ROOT        Same as --data-root
  ISAACSIM_ZIP_URL            Override Isaac Sim download URL
  HF_ENDPOINT                 e.g. https://hf-mirror.com (faster in CN)
  HF_HUB_ENABLE_HF_TRANSFER=1 + pip install hf_transfer (multi-connection)
  HF_MAX_WORKERS              Parallel files (default 8)

URLs are defined in setup/urls.sh (from project README).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
    --isaac) DO_ISAAC=1; EXPLICIT_ISAAC=1; shift ;;
    --no-isaac) DO_ISAAC=0; EXPLICIT_NO_ISAAC=1; shift ;;
    --test) DO_TEST=1; WANT_OPTIONAL=1; shift ;;
    --objects) DO_OBJECTS=1; WANT_OPTIONAL=1; shift ;;
    --scenes) DO_SCENES=1; WANT_OPTIONAL=1; shift ;;
    --train) DO_TRAIN=1; WANT_OPTIONAL=1; shift ;;
    --model) DO_MODEL=1; WANT_OPTIONAL=1; shift ;;
    --force) FORCE=1; shift ;;
    --all)
      DO_TEST=1
      DO_OBJECTS=1
      DO_SCENES=1
      DO_TRAIN=1
      WANT_OPTIONAL=1
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (try --help)" ;;
  esac
done

# No flags -> Isaac only. Any of --train/--test/... -> skip Isaac unless --isaac.
if [[ "${WANT_OPTIONAL}" -eq 0 && "${EXPLICIT_ISAAC}" -eq 0 && "${EXPLICIT_NO_ISAAC}" -eq 0 ]]; then
  DO_ISAAC=1
elif [[ "${WANT_OPTIONAL}" -eq 1 && "${EXPLICIT_ISAAC}" -eq 0 ]]; then
  DO_ISAAC=0
fi

if (( DO_ISAAC + DO_TEST + DO_OBJECTS + DO_SCENES + DO_TRAIN + DO_MODEL == 0 )); then
  die "Nothing to download. Use --isaac, --train, --test, ... or run without flags."
fi

need_cmd file unzip
if [[ "${DO_SCENES}" -eq 1 || "${DO_ISAAC}" -eq 1 ]]; then
  need_cmd curl
fi
if [[ "${DO_TEST}" -eq 1 || "${DO_OBJECTS}" -eq 1 || "${DO_TRAIN}" -eq 1 || "${DO_MODEL}" -eq 1 ]]; then
  need_cmd python3
fi

ensure_dir "${VENDOR_DIR}" "${CACHE_DIR}" "${DATA_ROOT}"
ensure_dir "${DATA_ROOT}/objects" "${DATA_ROOT}/scenes" "${DATA_ROOT}/tests" "${DATA_ROOT}/datasets"
ensure_dir "${REPO_ROOT}/pretrained_weights"

log "Repo:      ${REPO_ROOT}"
log "Data root: ${DATA_ROOT}"

if [[ "${DO_ISAAC}" -eq 1 ]]; then
  isaac_out="${VENDOR_DIR}/${ISAACSIM_ZIP_NAME}"
  download_file "${ISAACSIM_ZIP_URL}" "${isaac_out}"
  log "Isaac Sim zip ready for Docker build: ${isaac_out}"
fi

if [[ "${DO_TEST}" -eq 1 ]]; then
  archive="${CACHE_DIR}/${DOM_TEST_ARCHIVE_NAME}"
  download_gdrive_file "${DOM_TEST_GDRIVE_ID}" "${archive}"
  extract_archive "${archive}" "${DATA_ROOT}"
  if [[ -f "${DATA_ROOT}/test-envs.txt" ]]; then
    log "Found test-envs.txt"
  elif [[ -f "${DATA_ROOT}/tests/test-envs.txt" ]]; then
    cp -f "${DATA_ROOT}/tests/test-envs.txt" "${DATA_ROOT}/test-envs.txt"
  else
    warn "test-envs.txt not found after DOM-Test extract; check ${DATA_ROOT}"
  fi
fi

if [[ "${DO_OBJECTS}" -eq 1 ]]; then
  archive="${CACHE_DIR}/${DOM_OBJECTS_ARCHIVE_NAME}"
  download_gdrive_file "${DOM_OBJECTS_GDRIVE_ID}" "${archive}"
  extract_archive "${archive}" "${DATA_ROOT}/objects"
fi

if [[ "${DO_SCENES}" -eq 1 ]]; then
  archive="${CACHE_DIR}/DOM-3D-Scenes.archive"
  download_file "${DOM_SCENES_URL}" "${archive}"
  extract_archive "${archive}" "${DATA_ROOT}/scenes"
fi

if [[ "${DO_TRAIN}" -eq 1 ]]; then
  if [[ "${FORCE}" -eq 1 && -d "${DATA_ROOT}/datasets/DOM" ]]; then
    log "Removing incomplete/cached DOM dataset for --force..."
    rm -rf "${DATA_ROOT}/datasets/DOM"
  fi
  download_hf_repo "${HF_DOM_DATASET}" "${DATA_ROOT}/datasets/DOM" dataset "${FORCE}"
fi

if [[ "${DO_MODEL}" -eq 1 ]]; then
  if [[ "${FORCE}" -eq 1 && -d "${REPO_ROOT}/pretrained_weights/dynamic-vla-DOM" ]]; then
    rm -rf "${REPO_ROOT}/pretrained_weights/dynamic-vla-DOM"
  fi
  download_hf_repo "${HF_DOM_MODEL}" "${REPO_ROOT}/pretrained_weights/dynamic-vla-DOM" model "${FORCE}"
fi

cat <<EOF

Done.

Next:
  ./build_docker.sh
  ./docker/run-dev.sh

Data layout (evaluate.py paths are relative to data root parent):
  ${DATA_ROOT}/objects/
  ${DATA_ROOT}/scenes/
  ${DATA_ROOT}/tests/
  ${DATA_ROOT}/test-envs.txt
  ${DATA_ROOT}/datasets/DOM/     (if --train)

Inside container (example):
  python simulations/evaluate.py \\
    --scene_dir ../scenes --env_cfg ../test-envs.txt ...

Set DYNAMICVLA_DATA_ROOT=${DATA_ROOT} when starting docker if data is not repo parent.
EOF
