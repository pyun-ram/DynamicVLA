#!/usr/bin/env bash
# Create NVIDIA Vulkan ICD JSON when GL libs are mounted but icd.d is empty.
# Usage: ./docker/scripts/fix-vulkan-icd.sh
set -euo pipefail

icd_dir=/usr/share/vulkan/icd.d
icd_file="${icd_dir}/nvidia_icd.json"

has_glx=false
if ldconfig -p 2>/dev/null | grep -q 'libGLX_nvidia\.so\.0'; then
  has_glx=true
elif [[ -e /lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]] || \
     [[ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]]; then
  has_glx=true
fi

if [[ -f "${icd_file}" ]]; then
  echo "[OK]   ${icd_file} already exists"
elif ! $has_glx; then
  echo "[FAIL] libGLX_nvidia.so.0 not found (need graphics capability / driver mount)"
  exit 1
else
  api_version=1.3.0
  for host_icd in /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.x86_64.json; do
    if [[ -f "${host_icd}" ]]; then
      if ver=$(sed -n 's/.*"api_version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${host_icd}" | head -1) && \
         [[ -n "${ver}" ]]; then
        api_version="${ver}"
        break
      fi
    fi
  done

  mkdir -p "${icd_dir}"
  cat >"${icd_file}" <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "${api_version}"
    }
}
EOF
  echo "[OK]   wrote ${icd_file} (api_version=${api_version})"
  echo "[HINT] export VK_ICD_FILENAMES=${icd_file}"
  export VK_ICD_FILENAMES="${icd_file}"
fi

if [[ -z "${VK_ICD_FILENAMES:-}" && -f "${icd_file}" ]]; then
  export VK_ICD_FILENAMES="${icd_file}"
fi

if command -v vulkaninfo >/dev/null 2>&1; then
  if vulkaninfo --summary; then
    echo "[OK]   vulkaninfo --summary"
    exit 0
  fi
  echo "[FAIL] vulkaninfo --summary failed"
  exit 1
fi

echo "[WARN] vulkaninfo not installed; ICD fix applied (apt install vulkan-tools to verify)"
exit 0
