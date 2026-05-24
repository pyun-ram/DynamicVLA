#!/usr/bin/env bash
# GPU / Vulkan diagnostics inside the dev container.
# Usage: ./docker/scripts/verify-gpu.sh
set -uo pipefail

fail=0
ok() { echo "[OK]   $*"; }
warn() { echo "[WARN] $*"; fail=1; }
bad() { echo "[FAIL] $*"; fail=1; }

section() { echo; echo "=== $* ==="; }

section "nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi; then
    ok "nvidia-smi"
  else
    bad "nvidia-smi exited non-zero"
  fi
else
  bad "nvidia-smi not found"
fi

section "NVIDIA_* environment"
for v in NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES; do
  if [[ -n "${!v:-}" ]]; then
    echo "  ${v}=${!v}"
  else
    warn "${v} is unset (run-dev.sh sets compute,utility,graphics)"
  fi
done
caps="${NVIDIA_DRIVER_CAPABILITIES:-}"
if [[ -n "${caps}" ]] && [[ "${caps}" == *graphics* ]]; then
  ok "NVIDIA_DRIVER_CAPABILITIES includes graphics"
else
  bad "NVIDIA_DRIVER_CAPABILITIES must include graphics for Vulkan (got: ${caps:-<unset>})"
fi

section "/dev/nvidia*"
if compgen -G '/dev/nvidia*' >/dev/null 2>&1; then
  ls -la /dev/nvidia* 2>/dev/null || true
  ok "NVIDIA device nodes present"
else
  bad "no /dev/nvidia* (container likely missing --gpus all)"
fi

section "Vulkan ICD (/usr/share/vulkan/icd.d)"
icd_dir=/usr/share/vulkan/icd.d
has_nvidia_icd() {
  compgen -G "${icd_dir}/*nvidia*" >/dev/null 2>&1 || \
    grep -lqi nvidia "${icd_dir}"/*.json 2>/dev/null
}
has_glx_lib() {
  ldconfig -p 2>/dev/null | grep -q 'libGLX_nvidia\.so\.0' || \
    [[ -e /lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]]
}
if [[ -d "${icd_dir}" ]]; then
  ls -la "${icd_dir}" || true
  if has_nvidia_icd; then
    ok "NVIDIA Vulkan ICD JSON found"
  elif has_glx_lib; then
    fix_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fix-vulkan-icd.sh"
    if [[ -x "${fix_script}" ]]; then
      "${fix_script}" || true
    fi
    ls -la "${icd_dir}" || true
    if has_nvidia_icd; then
      ok "NVIDIA Vulkan ICD JSON (auto-fixed via fix-vulkan-icd.sh)"
    else
      bad "no NVIDIA ICD in ${icd_dir}; run: ./docker/scripts/fix-vulkan-icd.sh"
    fi
  else
    bad "no NVIDIA ICD in ${icd_dir} (graphics capability not mounted?)"
  fi
else
  bad "${icd_dir} missing"
fi

section "libnvidia-gl (ldconfig)"
if ldconfig -p 2>/dev/null | grep -q 'libnvidia-glcore'; then
  ldconfig -p 2>/dev/null | grep -E 'libnvidia-gl|libGLX_nvidia' || true
  ok "libnvidia-gl* in ldconfig cache"
else
  bad "libnvidia-glcore not in ldconfig (graphics drivers not injected)"
fi

section "vulkaninfo --summary"
if command -v vulkaninfo >/dev/null 2>&1; then
  if [[ -f "${icd_dir}/nvidia_icd.json" ]]; then
    export VK_ICD_FILENAMES="${icd_dir}/nvidia_icd.json"
  fi
  if vulkaninfo --summary 2>&1; then
    ok "vulkaninfo --summary"
  else
    bad "vulkaninfo failed (often ERROR_INCOMPATIBLE_DRIVER without graphics cap)"
  fi
else
  warn "vulkaninfo not installed (apt install vulkan-tools)"
fi

section "summary"
if [[ "${fail}" -eq 0 ]]; then
  echo "[OK]   GPU/Vulkan checks passed"
  exit 0
fi
echo "[FAIL] Fix: recreate dev container with graphics capability (./docker/run-dev.sh --new), not apt alone"
exit 1
