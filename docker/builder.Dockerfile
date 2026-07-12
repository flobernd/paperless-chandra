# Builder image: produces the paperless-chandra wheel.
#
# The version-controlled build recipe for the plugin. The wheel is
# extracted from /dist (docker create + docker cp, see README) and staged
# next to setup.sh, so the paperless container installs a prebuilt wheel
# and needs no build tools of its own.
#
# Build context is the repo root.
FROM python:3.12-slim

WORKDIR /src

# Build inputs: the package sources plus the files pyproject.toml's metadata
# references (readme + license). Copied explicitly so unrelated repo content
# does not invalidate the layer cache.
COPY pyproject.toml README.md LICENSE ./
COPY paperless_chandra ./paperless_chandra

# Produce the wheel under /dist.
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist
