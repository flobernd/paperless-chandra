# syntax=docker/dockerfile:1.7
#
# paperless-ngx + paperless-chandra.
# OCR recognition is delegated to a separate OpenAI-compatible inference
# server (vLLM, llama.cpp, Ollama) - see docker-compose.vllm.yml for the
# matching runtime config.
#
# Build:
#   docker build \
#     --build-arg PLUGIN_REF=v0.1.0 \
#     -f examples/Dockerfile \
#     -t paperless-chandra:latest .

ARG PLUGIN_REF=master
ARG PAPERLESS_TAG=latest

# --- Stage 1: build the plugin wheel from source --------------------------
FROM python:3.12-slim AS plugin
ARG PLUGIN_REF
ADD https://github.com/flobernd/paperless-chandra.git#${PLUGIN_REF} /src
WORKDIR /src
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

# --- Stage 2: layer the plugin onto paperless-ngx -------------------------
FROM ghcr.io/paperless-ngx/paperless-ngx:${PAPERLESS_TAG}

COPY --from=plugin /dist/*.whl /tmp/plugin/
RUN pip install --no-cache-dir /tmp/plugin/*.whl \
 && rm -rf /tmp/plugin
