#!/usr/bin/env bash
# Shared helpers for setup scripts.
set -euo pipefail

log() { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARN: %s\n' "$*" >&2; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

need_cmd() {
  local c
  for c in "$@"; do
    command -v "${c}" >/dev/null 2>&1 || die "Missing command: ${c}"
  done
}

ensure_dir() {
  mkdir -p "$@"
}

is_html_response() {
  local f=$1
  [[ -f "${f}" ]] || return 1
  file -b "${f}" | grep -qi 'HTML'
}

download_with_curl() {
  local url=$1 out=$2
  curl -fL --retry 5 --retry-delay 10 -C - --progress-bar -o "${out}" "${url}"
}

absolute_url_from_page() {
  # Usage: absolute_url_from_page CANDIDATE_URL PAGE_URL
  local candidate=$1 page_url=$2
  local origin
  if [[ "${candidate}" =~ ^https?:// ]]; then
    echo "${candidate}"
    return 0
  fi
  origin="$(printf '%s' "${page_url}" | sed -E 's#^(https?://[^/]+).*$#\1#')"
  if [[ "${candidate}" =~ ^// ]]; then
    echo "https:${candidate}"
  elif [[ "${candidate}" =~ ^/ ]]; then
    echo "${origin}${candidate}"
  else
    echo "${origin}/${candidate#./}"
  fi
}

extract_archive_url_from_html() {
  # Usage: extract_archive_url_from_html HTML_FILE PAGE_URL
  local html_file=$1 page_url=$2
  local candidate resolved line

  # First pass: direct archive-like links.
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    candidate="${line%\"}"
    candidate="${candidate#\"}"
    candidate="${candidate%\'}"
    candidate="${candidate#\'}"
    resolved="$(absolute_url_from_page "${candidate}" "${page_url}")"
    if [[ "${resolved}" =~ \.(zip|tar|tgz|tar\.gz|tar\.xz|tar\.bz2)(\?.*)?$ ]]; then
      echo "${resolved}"
      return 0
    fi
  done < <(
    {
      grep -Eoi 'https?://[^"'"'"'<>()[:space:]]+' "${html_file}" || true
      grep -Eoi '(href|src)=["'"'"'][^"'"'"']+["'"'"']' "${html_file}" \
        | sed -E 's/^(href|src)=["'"'"']([^"'"'"']+)["'"'"']$/\2/i' || true
    } | awk '!seen[$0]++'
  )

  # Second pass: signed links that may not expose archive extension.
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    candidate="${line%\"}"
    candidate="${candidate#\"}"
    candidate="${candidate%\'}"
    candidate="${candidate#\'}"
    resolved="$(absolute_url_from_page "${candidate}" "${page_url}")"
    if [[ "${resolved}" =~ (token=|signature=|expires=|X-Amz-|download=|filename=) ]]; then
      echo "${resolved}"
      return 0
    fi
  done < <(
    {
      grep -Eoi 'https?://[^"'"'"'<>()[:space:]]+' "${html_file}" || true
      grep -Eoi '(href|src)=["'"'"'][^"'"'"']+["'"'"']' "${html_file}" \
        | sed -E 's/^(href|src)=["'"'"']([^"'"'"']+)["'"'"']$/\2/i' || true
    } | awk '!seen[$0]++'
  )

  return 1
}

die_with_manual_download_hint() {
  # Usage: die_with_manual_download_hint SOURCE_URL DEST_FILE [HTML_FILE]
  local source_url=$1 dest_file=$2 html_file=${3:-}
  [[ -n "${html_file}" && -f "${html_file}" ]] && mv -f "${html_file}" "${dest_file}.html"
  printf '[setup] ERROR: Downloaded HTML page instead of archive from: %s\n' "${source_url}" >&2
  printf '[setup] ERROR: This gateway may require browser interaction or returns a landing page.\n' >&2
  printf '[setup] ERROR: Manual fallback:\n' >&2
  printf '[setup] ERROR:   1) Open the URL in browser and download the real archive (.zip/.tar/.tar.gz)\n' >&2
  printf '[setup] ERROR:   2) Move/rename it to: %s\n' "${dest_file}" >&2
  printf '[setup] ERROR:   3) Re-run your previous download command\n' >&2
  printf '[setup] ERROR: Saved HTML response for debugging at: %s.html\n' "${dest_file}" >&2
  exit 1
}

ensure_gdown() {
  if command -v gdown >/dev/null 2>&1; then
    return 0
  fi
  if python3 -c "import gdown" 2>/dev/null; then
    return 0
  fi
  need_cmd python3
  log "Installing gdown (user) for Google Drive downloads..."
  python3 -m pip install --user -q gdown
}

run_gdown() {
  # Usage: run_gdown FILE_ID OUTPUT_PATH
  local file_id=$1 out=$2
  local url="https://drive.google.com/file/d/${file_id}/view"
  if command -v gdown >/dev/null 2>&1; then
    gdown "${url}" --fuzzy -O "${out}"
  else
    python3 -m gdown "${url}" --fuzzy -O "${out}"
  fi
}

download_gdrive_file() {
  # Usage: download_gdrive_file GOOGLE_DRIVE_FILE_ID OUTPUT_PATH
  local file_id=$1 out=$2
  local partial
  ensure_gdown
  ensure_dir "$(dirname "${out}")"
  partial="${out}.partial"
  if [[ -f "${out}" ]]; then
    if is_html_response "${out}"; then
      warn "Cached file is HTML, re-downloading: ${out}"
      rm -f "${out}"
    else
      log "Already exists, skip download: ${out}"
      return 0
    fi
  fi
  log "Downloading Google Drive file ${file_id}"
  log "         -> ${out}"
  rm -f "${partial}"
  run_gdown "${file_id}" "${partial}"
  if [[ ! -s "${partial}" ]]; then
    die "gdown produced empty file for id=${file_id}"
  fi
  if is_html_response "${partial}"; then
    die "gdown returned HTML (check file id or Drive permissions): ${file_id}"
  fi
  mv -f "${partial}" "${out}"
}

download_file() {
  # Usage: download_file URL OUTPUT_PATH
  local url=$1 out=$2
  local partial retry_url
  ensure_dir "$(dirname "${out}")"
  partial="${out}.partial"
  if [[ -f "${out}" ]]; then
    if is_html_response "${out}"; then
      warn "Cached file is HTML (not archive), re-downloading: ${out}"
      rm -f "${out}"
    else
      log "Already exists, skip download: ${out}"
      return 0
    fi
  fi
  log "Downloading: ${url}"
  log "         -> ${out}"
  download_with_curl "${url}" "${partial}"
  if is_html_response "${partial}"; then
    warn "Received HTML response; trying to resolve archive URL from page..."
    if retry_url="$(extract_archive_url_from_html "${partial}" "${url}")"; then
      rm -f "${partial}"
      log "Resolved fallback URL: ${retry_url}"
      download_with_curl "${retry_url}" "${partial}"
      if is_html_response "${partial}"; then
        die_with_manual_download_hint "${retry_url}" "${out}" "${partial}"
      fi
    else
      die_with_manual_download_hint "${url}" "${out}" "${partial}"
    fi
  fi
  mv -f "${partial}" "${out}"
}

detect_archive_type() {
  local f=$1
  local mime
  mime="$(file -b "${f}")"
  case "${mime}" in
    *Zip*) echo zip ;;
    *gzip*) echo tgz ;;
    *tar*) echo tar ;;
    *) echo unknown ;;
  esac
}

extract_archive() {
  # Usage: extract_archive ARCHIVE DEST_DIR
  local archive=$1 dest=$2
  local kind tmp inner n
  kind="$(detect_archive_type "${archive}")"
  tmp="$(mktemp -d)"
  case "${kind}" in
    zip) unzip -q -o "${archive}" -d "${tmp}" ;;
    tgz|tar) tar -xf "${archive}" -C "${tmp}" ;;
    *) die "Unsupported archive type for ${archive}: ${kind} ($(file -b "${archive}"))" ;;
  esac
  ensure_dir "${dest}"
  inner="$(find "${tmp}" -mindepth 1 -maxdepth 1 | head -1)"
  n="$(find "${tmp}" -mindepth 1 -maxdepth 1 | wc -l)"
  if [[ "${n}" -eq 1 && -d "${inner}" ]]; then
    # Single top-level directory → merge contents into dest
    shopt -s dotglob
    for item in "${inner}"/*; do
      base="$(basename "${item}")"
      if [[ -e "${dest}/${base}" ]]; then
        warn "Exists, skip: ${dest}/${base}"
      else
        mv "${item}" "${dest}/"
      fi
    done
    shopt -u dotglob
  else
    shopt -s dotglob
    for item in "${tmp}"/*; do
      base="$(basename "${item}")"
      if [[ -e "${dest}/${base}" ]]; then
        warn "Exists, skip: ${dest}/${base}"
      else
        mv "${item}" "${dest}/"
      fi
    done
    shopt -u dotglob
  fi
  rm -rf "${tmp}"
  log "Extracted ${archive} -> ${dest}"
}

hf_download_incomplete() {
  local root=$1
  [[ -d "${root}" ]] || return 1
  find "${root}" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

hf_repo_ready() {
  local local_dir=$1 force=${2:-0}
  [[ "${force}" -eq 1 ]] && return 1
  [[ -d "${local_dir}" ]] || return 1
  [[ -n "$(ls -A "${local_dir}" 2>/dev/null)" ]] || return 1
  if hf_download_incomplete "${local_dir}"; then
    return 1
  fi
  return 0
}

hf_clear_stale_locks() {
  local root=$1
  [[ -d "${root}" ]] || return 0
  local n
  n="$(find "${root}" -name '*.lock' 2>/dev/null | wc -l)"
  if [[ "${n}" -gt 0 ]]; then
    warn "Removing ${n} stale Hugging Face lock file(s) under ${root}"
    find "${root}" -name '*.lock' -delete
  fi
}

download_hf_repo() {
  # Usage: download_hf_repo REPO_ID LOCAL_DIR [dataset|model] [force]
  # Speed tips (export before ./download_assets.sh --train):
  #   export HF_ENDPOINT=https://hf-mirror.com          # mirror (CN)
  #   export HF_HUB_ENABLE_HF_TRANSFER=1                # needs: pip install hf_transfer
  #   export HF_MAX_WORKERS=8
  local repo_id=$1 local_dir=$2 repo_type=${3:-dataset} force=${4:-0}
  if hf_repo_ready "${local_dir}" "${force}"; then
    log "HF ${repo_type} already complete: ${local_dir} ($(du -sh "${local_dir}" 2>/dev/null | cut -f1))"
    log "Use --force to re-download from scratch."
    return 0
  fi
  if hf_download_incomplete "${local_dir}"; then
    warn "Incomplete Hugging Face download detected; resuming..."
    hf_clear_stale_locks "${local_dir}"
  fi
  need_cmd python3
  if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    log "Installing huggingface_hub (user) for download..."
    python3 -m pip install --user -q "huggingface_hub>=0.26"
  fi
  # hf_transfer + mirror often hangs at 0.00B; disable unless user forces it.
  if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
    if [[ "${HF_ENDPOINT:-}" == *hf-mirror* ]] && [[ "${HF_FORCE_HF_TRANSFER:-0}" != "1" ]]; then
      warn "Disabling HF_HUB_ENABLE_HF_TRANSFER (hf_transfer + hf-mirror often stalls)."
      warn "Use official endpoint, or: export HF_FORCE_HF_TRANSFER=1 to keep it."
      unset HF_HUB_ENABLE_HF_TRANSFER
    else
      if ! python3 -c "import hf_transfer" 2>/dev/null; then
        log "Installing hf_transfer (user) for faster downloads..."
        python3 -m pip install --user -q hf_transfer
      fi
      export HF_HUB_ENABLE_HF_TRANSFER=1
    fi
  fi
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    log "HF_ENDPOINT=${HF_ENDPOINT}"
  else
    log "Tip: slow from huggingface.co? try: export HF_ENDPOINT=https://hf-mirror.com"
  fi
  export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-info}"
  log "Downloading Hugging Face ${repo_type} ${repo_id} -> ${local_dir}"
  log "Progress bar is per-file; listing repo may take a minute..."
  HF_REPO_ID="${repo_id}" \
  HF_LOCAL_DIR="${local_dir}" \
  HF_REPO_TYPE="${repo_type}" \
  HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}" \
  python3 -u - <<'PY'
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from huggingface_hub import HfApi, snapshot_download

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ["HF_REPO_TYPE"]
local_dir = os.environ["HF_LOCAL_DIR"]
max_workers = int(os.environ.get("HF_MAX_WORKERS", "8"))
endpoint = os.environ.get("HF_ENDPOINT") or None
api_timeout_s = int(os.environ.get("HF_API_TIMEOUT", "20"))

print("Connecting to Hugging Face...", flush=True)
api = HfApi(endpoint=endpoint)
try:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(api.list_repo_files, repo_id, repo_type=repo_type)
        files = fut.result(timeout=api_timeout_s)
    print(f"Repo file count: {len(files)}", flush=True)
except FuturesTimeoutError:
    print(
        f"WARN: list_repo_files timeout after {api_timeout_s}s; continuing with snapshot_download",
        flush=True,
    )
except Exception as e:
    print(f"WARN: list_repo_files failed ({e}); continuing with snapshot_download", flush=True)

print("Starting snapshot_download (resume enabled by default)...", flush=True)
path = snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=local_dir,
    max_workers=max_workers,
    endpoint=endpoint,
)
print("snapshot:", path, flush=True)
PY
}
