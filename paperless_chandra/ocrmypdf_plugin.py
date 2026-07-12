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
from urllib.request import Request, urlopen

import ocrmypdf

from paperless_chandra.engine.client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    normalize_server_url,
)
from paperless_chandra.engine.engine import ChandraEngine

log = logging.getLogger("paperless.chandra.plugin")

_CONTENT_FORMATS = ("text", "markdown")

#: Successful probes per (url, api_key); check_options runs once per
#: document, and one probe per worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_server(server_url: str, api_key: str) -> None:
    """Fail fast when the Chandra server is unreachable or rejects the key.

    Without this, a wrong URL or key surfaces only at first inference,
    deep inside a celery task.
    """
    from urllib.error import HTTPError, URLError

    from ocrmypdf.exceptions import MissingDependencyError

    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=5):  # noqa: S310
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
    except URLError as e:
        raise MissingDependencyError(
            f"The Chandra server at {base} is not reachable ({e.reason}). "
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
        choices=list(_CONTENT_FORMATS),
        default="text",
        dest="chandra_content_format",
        help="Document content stored by paperless: plain text (default) or markdown.",
    )


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate configuration and probe the inference server."""
    from ocrmypdf.exceptions import MissingDependencyError

    fmt = (getattr(options, "chandra_content_format", "") or "text").strip()
    if fmt not in _CONTENT_FORMATS:
        # argparse validates the CLI path via `choices=`; the Python API
        # (ocrmypdf.ocr(chandra_content_format=...)) skips that check.
        raise ValueError(
            f"Unknown chandra_content_format={fmt!r}. Valid choices: {list(_CONTENT_FORMATS)}."
        )

    try:
        import chandra.model.vllm  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "chandra-ocr is not installed. Install it with: pip install chandra-ocr"
        ) from e

    server_url = (getattr(options, "chandra_server_url", "") or "").strip()
    if not server_url:
        raise MissingDependencyError(
            "paperless-chandra requires --chandra-server-url (or the "
            "PAPERLESS_CHANDRA_SERVER_URL env var) to point at an "
            "OpenAI-compatible server hosting Chandra."
        )
    _probe_server(server_url, (getattr(options, "chandra_api_key", "") or "").strip())


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine() -> Any:
    """Return the Chandra engine.

    ``tryfirst=True`` makes this hookimpl run before the built-in
    Tesseract plugin's ``get_ocr_engine``; ocrmypdf's ``firstresult=True``
    policy then short-circuits, so Tesseract never claims the engine.
    """
    return ChandraEngine()
