#!/usr/bin/env bash
# Batch-run evaluate.py on every tests/*.json config.
# Run from repo:  cd simulations && ./batch_evaluate.sh

set -euo pipefail
cd "$(dirname "$0")"

shopt -s nullglob
tests=(../tests/*.json)
count=${#tests[@]}

echo "Found ${count} test config(s) in ../tests/:"
ls -1 ../tests/*.json

for f in "${tests[@]}"; do
  NAME=$(basename "$f")
  echo "=== ${NAME} ==="
  python3 evaluate.py --scene_dir ../scenes --object_dir ../objects \
    --env_cfg "../tests/${NAME}" --enable_cameras -n 1 \
    --output_dir ../output/evaluation/ --headless --save
done
