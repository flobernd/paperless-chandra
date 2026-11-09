"""ocrmypdf plugin: registers the Chandra engine and its CLI options.

This module is the one loaded by ``ocrmypdf.ocr(plugins=[...])`` when the
paperless parser invokes ocrmypdf. The engine lives inside this package
rather than being published as a standalone ocrmypdf plugin, so ocrmypdf's
entry-point auto-discovery never registers it: calls from paperless's
built-in Tesseract parser are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import ocrmypdf
from ocrmypdf.exceptions import MissingDependencyError

from paperless_chandra import (
    CONTENT_FORMATS,
    DEFAULT_CONTENT_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
)
from paperless_chandra.engine.client import ChandraClientError, normalize_server_url
from paperless_chandra.engine.engine import ChandraEngine, content_format

log = logging.getLogger("paperless.chandra.plugin")

#: Successful probes per (url, api_key); check_options runs once per
#: document, and one probe per worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_server(server_url: str, api_key: str) -> None:
    """Fail fast when the Chandra server is unreachable or rejects the key.

    Without this, a wrong URL or key surfaces only at first inference,
    deep inside a celery task.
    """
    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        # A vLLM server under load answers /models slowly, and the probe runs
        # once per worker process, so a generous timeout is cheap.
        with urlopen(request, timeout=30):  # noqa: S310
            pass
    except HTTPError as e:
        if e.code in (401, 403):
            raise MissingDependencyError(
                f"The Chandra server at {base} rejected the API key (HTTP {e.code}). "
                "Check PAPERLESS_CHANDRA_API_KEY against the server configuration."
            ) from e
        log.warning(
            "Chandra server preflight got HTTP %d from %s/models; continuing.",
            e.code,
            base,
        )
    except OSError as e:
        raise MissingDependencyError(
            f"The Chandra server at {base} is not reachable ({getattr(e, 'reason', e)}). "
            "Check PAPERLESS_CHANDRA_SERVER_URL and that the inference "
            "server container is running."
        ) from e
    _PROBED_SERVERS.add(cache_key)


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Register ``--chandra-*`` CLI args so they're accepted as kwargs.

    ocrmypdf validates kwargs against its argparse parser; these must be
    registered before ``ocrmypdf.ocr(chandra_server_url=...)`` is called
    from the paperless parser.
    """
    chandra = parser.add_argument_group("Chandra", "Options for the Chandra OCR engine")
    chandra.add_argument(
        "--chandra-server-url",
        default="",
        dest="chandra_server_url",
        metavar="URL",
        help=(
            "URL of the OpenAI-compatible inference server hosting Chandra "
            "(e.g. http://gpu-box:8000). The /v1 suffix is appended when missing."
        ),
    )
    chandra.add_argument(
        "--chandra-model-name",
        default=DEFAULT_MODEL_NAME,
        dest="chandra_model_name",
        metavar="NAME",
        help=f"Served model name advertised by the server (default: {DEFAULT_MODEL_NAME}).",
    )
    chandra.add_argument(
        "--chandra-api-key",
        default="",
        dest="chandra_api_key",
        metavar="KEY",
        help="Bearer token for the server. Leave blank if the server needs no auth.",
    )
    chandra.add_argument(
        "--chandra-max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        dest="chandra_max_output_tokens",
        metavar="N",
        help=f"Per-page output token budget (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    chandra.add_argument(
        "--chandra-content-format",
        choices=list(CONTENT_FORMATS),
        default=DEFAULT_CONTENT_FORMAT,
        dest="chandra_content_format",
        help="Document content stored by paperless: plain text (default) or markdown.",
    )


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate configuration and probe the inference server."""
    fmt = content_format(options)
    if fmt not in CONTENT_FORMATS:
        # argparse validates the CLI path via `choices=`; the Python API
        # (ocrmypdf.ocr(chandra_content_format=...)) skips that check.
        raise ValueError(
            f"Unknown chandra_content_format={fmt!r}. Valid choices: {list(CONTENT_FORMATS)}."
        )

    # argparse's `type=int` has no lower bound, and the Python API skips
    # argparse entirely, so a non-positive value would otherwise only
    # surface per-page inside generate_vllm.
    tokens = getattr(options, "chandra_max_output_tokens", None)
    if tokens is not None and tokens < 1:
        raise ValueError(f"chandra_max_output_tokens must be at least 1, got {tokens}.")

    server_url = (getattr(options, "chandra_server_url", "") or "").strip()
    try:
        _probe_server(server_url, (getattr(options, "chandra_api_key", "") or "").strip())
    except ChandraClientError as e:
        raise MissingDependencyError(str(e)) from e


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine() -> Any:
    """Return the Chandra engine.

    ``tryfirst=True`` makes this hookimpl run before the built-in
    Tesseract plugin's ``get_ocr_engine``; ocrmypdf's ``firstresult=True``
    policy then short-circuits, so Tesseract never claims the engine.
    """
    return ChandraEngine()
