#!/usr/bin/env bash
# Download the quantized Chandra GGUF + mmproj for the llama.cpp e2e stack.
# ~4 GB total; files land in tests/e2e/models/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

# Chandra-OCR-2 has no official datalab-to GGUF repo; prithivMLmods/chandra-ocr-2-GGUF
# is the community quantization linked from datalab-to/chandra-ocr-2's "Quantizations"
# panel and carries the base_model:quantized tag back to the official checkpoint.
MODEL_URL="https://huggingface.co/prithivMLmods/chandra-ocr-2-GGUF/resolve/main/chandra-ocr-2.Q4_K_M.gguf"
MMPROJ_URL="https://huggingface.co/prithivMLmods/chandra-ocr-2-GGUF/resolve/main/chandra-ocr-2.mmproj-f16.gguf"

[ -f models/chandra-q4_k_m.gguf ] || curl -L --fail -o models/chandra-q4_k_m.gguf "$MODEL_URL"
[ -f models/mmproj.gguf ] || curl -L --fail -o models/mmproj.gguf "$MMPROJ_URL"
ls -lh models/
