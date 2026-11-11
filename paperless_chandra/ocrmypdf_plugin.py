"""ocrmypdf plugin: registers the Chandra engine and its CLI options.

This module is the one loaded by ``ocrmypdf.ocr(plugins=[...])`` when the
paperless parser invokes ocrmypdf. The engine lives inside this package
rather than being published as a standalone ocrmypdf plugin, so ocrmypdf's
entry-point auto-discovery never registers it: calls from paperless's
built-in Tesseract parser are unaffected.
"""

from __future__ import annotations

import functools
from typing import Any

import ocrmypdf
from ocrmypdf.exceptions import MissingDependencyError

from paperless_chandra import (
    CONTENT_FORMATS,
    DEFAULT_CONTENT_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
)
from paperless_chandra.engine.client import ChandraClientError, probe_server, request_settings
from paperless_chandra.engine.engine import ChandraEngine, content_format


# check_options runs once per document, and one successful probe per worker
# process is enough. functools.cache keeps nothing for a probe that raises, so
# the next document probes again.
@functools.cache
def _probe_server(url: str, api_key: str) -> None:
    probe_server(url, api_key)


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Register the ``--chandra-*`` arguments of the ocrmypdf command line.

    The Python API stores an unknown keyword such as ``chandra_server_url``
    without checking it against these arguments.
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
        "--chandra-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        dest="chandra_timeout",
        metavar="SECONDS",
        help=f"Seconds each wait for data from the server may last before the document fails, "
        f"and no limit for the whole request (default: {DEFAULT_TIMEOUT_SECONDS}).",
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
    """Validate the options and probe the inference server unless the run skips pages with text."""
    fmt = content_format(options)
    if fmt not in CONTENT_FORMATS:
        # argparse validates the CLI path via `choices=`; the Python API
        # (ocrmypdf.ocr(chandra_content_format=...)) skips that check.
        raise ValueError(
            f"Unknown chandra_content_format={fmt!r}. Valid choices: {list(CONTENT_FORMATS)}."
        )

    try:
        settings = request_settings(options)
        # A run with skip_text may send no page to the server, and a page that
        # does go there reports a server that is down.
        if not getattr(options, "skip_text", False):
            _probe_server(settings.url, settings.api_key)
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
