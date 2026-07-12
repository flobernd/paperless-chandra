#!/bin/bash
# Bootstrap paperless-chandra inside a paperless-ngx Docker container.
#
# Mount this script (plus the matching paperless-chandra wheel/sdist)
# under /custom-cont-init.d/ - the paperless-ngx base image runs every
# executable in that directory before starting paperless itself.
#
# OCR recognition is delegated to a separate OpenAI-compatible inference
# server (vLLM, llama.cpp, Ollama); see examples/docker-compose.vllm.yml
# for a matching runtime config. This script installs no native
# libraries and no inference runtime of its own.
#
# A pre-built artifact is REQUIRED next to this script - paperless-chandra
# is not published to PyPI. The script supports both shapes:
#   * paperless-chandra.tar.gz   (sdist)
#   * paperless_chandra-*.whl    (wheel - any version, first match wins)
#
# Obtain one with either of:
#   pip wheel --no-deps "git+https://github.com/flobernd/paperless-chandra.git@v0.1.0"
#   # or build docker/builder.Dockerfile and copy its /dist output
#
# The install is idempotent: repeated container restarts skip work that
# is already done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${SCRIPT_DIR}/paperless-chandra.tar.gz"
# First whl matching the pattern wins. shopt avoids a literal glob string
# in the unmatched case.
shopt -s nullglob
WHEELS=("${SCRIPT_DIR}"/paperless_chandra-*.whl)
shopt -u nullglob

# ---------------------------------------------------------------------------
# Python package
# ---------------------------------------------------------------------------
PACKAGE_NAME="paperless-chandra"

is_installed() {
    pip show "${PACKAGE_NAME}" >/dev/null 2>&1
}

if is_installed; then
    echo "${PACKAGE_NAME} already installed - skipping pip install"
else
    if [ -f "${TARBALL}" ]; then
        echo "=== Installing ${PACKAGE_NAME} from ${TARBALL} ==="
        pip install --no-cache-dir "${TARBALL}"
    elif [ ${#WHEELS[@]} -gt 0 ]; then
        echo "=== Installing ${PACKAGE_NAME} from ${WHEELS[0]} ==="
        pip install --no-cache-dir "${WHEELS[0]}"
    else
        echo "ERROR: no paperless-chandra artifact found next to setup.sh." >&2
        echo "       Expected one of:" >&2
        echo "         ${TARBALL}" >&2
        echo "         ${SCRIPT_DIR}/paperless_chandra-*.whl" >&2
        echo "       See docker/builder.Dockerfile for how to build one." >&2
        exit 1
    fi
fi

echo "=== paperless-chandra bootstrap complete ==="
