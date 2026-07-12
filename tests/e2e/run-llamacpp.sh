#!/usr/bin/env bash
# End-to-end pipeline test against real quantized Chandra on CPU.
# Slow by design; expect several minutes for the OCR step alone.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
PY="$REPO_ROOT/.venv/bin/python"

./download-model.sh

avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "$avail_gb" -lt 5 ]; then
  echo "WARNING: only ${avail_gb} GB RAM available; the 5B Q4 model needs ~4 GB." >&2
  echo "Stop other containers/services first or expect OOM." >&2
fi

cleanup() { docker compose -f docker-compose.llamacpp.yml down; }
trap cleanup EXIT

rm -rf consume && mkdir -p consume
docker compose -f docker-compose.llamacpp.yml up -d --build

"$PY" make_test_page.py consume/e2e-test.png
"$PY" assert_e2e.py --expect "RECHNUNG" --expect "CHANDRA" --timeout 2400
echo "== llama.cpp e2e passed =="
