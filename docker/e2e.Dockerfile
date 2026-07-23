# paperless-ngx + the locally built plugin, for the e2e compose stacks.
# Build context is the repo root.
FROM python:3.12-slim AS plugin
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY paperless_chandra ./paperless_chandra
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

FROM ghcr.io/paperless-ngx/paperless-ngx:latest
COPY --from=plugin /dist/*.whl /tmp/plugin/
RUN pip install --no-cache-dir /tmp/plugin/*.whl \
 && rm -rf /tmp/plugin
