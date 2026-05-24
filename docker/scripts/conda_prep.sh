#!/usr/bin/env bash
# Accept Anaconda channel ToS (required by recent Miniconda in non-interactive builds).
conda_accept_tos() {
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if ! conda tos --help >/dev/null 2>&1; then
    return 0
  fi
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
}
