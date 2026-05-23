#!/usr/bin/env bash
# Install Isaac Sim 4.5.0 standalone into ISAACSIM_PATH (for nvidia/cuda base images).
set -euo pipefail

ISAACSIM_PATH="${ISAACSIM_PATH:-/opt/isaacsim}"
ISAACSIM_ZIP_URL="${ISAACSIM_ZIP_URL:-https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip}"
ISAAC_VENDOR_DIR="${ISAAC_VENDOR_DIR:-/tmp/isaac_vendor}"
ZIP_NAME="isaac-sim-standalone-4.5.0-linux-x86_64.zip"

is_isaacsim_root() {
  local d=$1
  [[ -f "${d}/VERSION" && -d "${d}/kit" && -f "${d}/isaac-sim.sh" ]]
}

if [[ -d "${ISAACSIM_PATH}" ]] && is_isaacsim_root "${ISAACSIM_PATH}"; then
  echo "Isaac Sim already present at ${ISAACSIM_PATH}"
  cat "${ISAACSIM_PATH}/VERSION"
  exit 0
fi

mkdir -p /tmp/isaac_dl
local_zip=""
if [[ -f "${ISAAC_VENDOR_DIR}/${ZIP_NAME}" ]]; then
  local_zip="${ISAAC_VENDOR_DIR}/${ZIP_NAME}"
  echo "Using local zip: ${local_zip}"
elif compgen -G "${ISAAC_VENDOR_DIR}/*.zip" >/dev/null 2>&1; then
  local_zip="$(ls -1 "${ISAAC_VENDOR_DIR}"/*.zip | head -1)"
  echo "Using local zip: ${local_zip}"
fi

if [[ -n "${local_zip}" ]]; then
  cp -f "${local_zip}" /tmp/isaac_dl/isaac-sim.zip
else
  echo "Downloading Isaac Sim from ${ISAACSIM_ZIP_URL} ..."
  curl -fL -C - -o /tmp/isaac_dl/isaac-sim.zip "${ISAACSIM_ZIP_URL}"
fi

extract_dir="$(mktemp -d)"
unzip -q /tmp/isaac_dl/isaac-sim.zip -d "${extract_dir}"
rm -f /tmp/isaac_dl/isaac-sim.zip

src=""
if is_isaacsim_root "${extract_dir}"; then
  # Zip root is the Isaac Sim tree (no wrapper directory).
  src="${extract_dir}"
elif compgen -G "${extract_dir}/isaac-sim*" >/dev/null 2>&1; then
  src="$(find "${extract_dir}" -maxdepth 1 -type d -name 'isaac-sim*' | head -1)"
else
  # Single top-level directory (e.g. isaac-sim-4.5.0).
  entries=("${extract_dir}"/*)
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]] && is_isaacsim_root "${entries[0]}"; then
    src="${entries[0]}"
  fi
fi

if [[ -z "${src}" ]] || ! is_isaacsim_root "${src}"; then
  echo "ERROR: Could not locate Isaac Sim root inside zip." >&2
  echo "Extract dir contents:" >&2
  ls -la "${extract_dir}" >&2
  exit 1
fi

rm -rf "${ISAACSIM_PATH}"
mkdir -p "$(dirname "${ISAACSIM_PATH}")"
mv "${src}" "${ISAACSIM_PATH}"
rm -rf "${extract_dir}"

echo "Installed Isaac Sim:"
cat "${ISAACSIM_PATH}/VERSION"
