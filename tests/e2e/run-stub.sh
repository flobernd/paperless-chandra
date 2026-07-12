#!/usr/bin/env bash
# End-to-end pipeline test against the stub Chandra server.
# Usage: tests/e2e/run-stub.sh   (from anywhere; needs docker + repo venv)
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
PY="$REPO_ROOT/.venv/bin/python"

cleanup() { docker compose -f docker-compose.stub.yml down -v; }
trap cleanup EXIT

rm -rf consume && mkdir -p consume
docker compose -f docker-compose.stub.yml up -d --build

"$PY" make_test_page.py consume/e2e-test.png
"$PY" make_test_page.py consume/e2e-rotated.png 180
"$PY" assert_e2e.py \
  --expect "PAPERLESS CHANDRA STUB OK" --expect "Stub invoice 2026-0042" \
  --expect-docs 2 --assert-upright e2e-rotated --timeout 600
echo "== stub e2e passed =="
