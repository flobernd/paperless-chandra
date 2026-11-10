"""Paperless-ngx parser that runs ocrmypdf with the Chandra engine.

``PaperlessChandraParser`` subclasses the upstream Tesseract parser,
``paperless.parsers.tesseract.RasterisedDocumentParser``, and inherits its
pipeline: the OCR modes, the image handling, the PDF/A conversion and the
safe-fallback retry. The subclass adds its identity, the remote-service
flag, its score and one override of ``construct_ocrmypdf_parameters``. The
override loads the Chandra ocrmypdf plugin and passes it these options, each
read from its ``PAPERLESS_CHANDRA_*`` variable unless the OCR user arguments
set it:

* ``chandra_content_format``
* ``chandra_max_output_tokens``
* ``chandra_model_name``
* ``chandra_server_url``
* ``chandra_timeout``

The parent writes the options to the debug log, so the API key stays out of
them and the engine reads ``PAPERLESS_CHANDRA_API_KEY`` from the environment.

The plugin relies on three contracts of the parent's ``parse``:

* It calls the builder for each ocrmypdf run, the safe-fallback run
  included, so each run loads the Chandra engine.
* It takes the document text from the sidecar file, which ocrmypdf
  assembles from the page text the engine writes.
* Its exception handling decides what fails the consumption: an error the
  engine raises inside ``ocrmypdf.ocr`` leaves ``parse`` as a ``ParseError``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final

from paperless.parsers.tesseract import RasterisedDocumentParser
from paperless.version import __version__ as _host_version

from paperless_chandra import (
    CONTENT_FORMATS,
    DEFAULT_CONTENT_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
    __version__,
)

logger = logging.getLogger("paperless.parsing.chandra")

_BUILDER_HOOK_MISSING: Final[str] = (
    "paperless-ngx's RasterisedDocumentParser no longer defines "
    "construct_ocrmypdf_parameters, so paperless-chandra cannot attach its engine"
)

# The registry logs a module that fails to import, skips it and lets
# Tesseract take every document. The module stays importable so that the
# parser fails each consumption with this message instead.
_BUILDER_HOOK_PRESENT: Final[bool] = "construct_ocrmypdf_parameters" in vars(
    RasterisedDocumentParser
)
if not _BUILDER_HOOK_PRESENT:
    logger.error(_BUILDER_HOOK_MISSING)

_TESTED_MAJOR: Final[int] = 3
if _host_version[0] > _TESTED_MAJOR:
    logger.warning(
        "paperless-chandra is tested with paperless-ngx 3.0 to 3.4 and this host runs %s.",
        ".".join(map(str, _host_version)),
    )

_OCRMYPDF_PLUGIN_MODULE: Final[str] = "paperless_chandra.ocrmypdf_plugin"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer value for %s=%r; using default %d.",
            name,
            raw,
            default,
        )
        return default


def _env_choice(name: str, choices: tuple[str, ...], default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw not in choices:
        logger.warning(
            "Invalid value for %s=%r; valid choices are %s. Using default %r.",
            name,
            raw,
            choices,
            default,
        )
        return default
    return raw


def _user_plugins(value: Any) -> list[str | Path]:
    """The ``plugins`` entry of the OCR user arguments, as a list.

    ocrmypdf accepts a single plugin as a bare string, which ``list()``
    would split into characters. The Chandra module itself is dropped
    because pluggy refuses to register a module twice.
    """
    if not value:
        return []
    items = [value] if isinstance(value, str | Path) else list(value)
    return [item for item in items if item != _OCRMYPDF_PLUGIN_MODULE]


class PaperlessChandraParser(RasterisedDocumentParser):
    """OCR parser using Chandra via ocrmypdf (drop-in for the Tesseract parser)."""

    name: str = "Paperless-ngx Chandra Parser"
    version: str = __version__
    author: str = "Florian Bernd"
    url: str = "https://github.com/flobernd/paperless-chandra"
    uses_remote_service: bool = True

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        if mime_type not in cls.supported_mime_types():
            return None
        # Default 15 beats Tesseract's 10 - once installed, Chandra wins.
        return _env_int("PAPERLESS_CHANDRA_SCORE", default=15)

    def __init__(self, logging_group: object | None = None) -> None:
        # Raised before the parent creates its scratch directory, which only
        # __exit__ removes.
        if not _BUILDER_HOOK_PRESENT:
            raise RuntimeError(_BUILDER_HOOK_MISSING)
        super().__init__(logging_group)
        # The parent points self.log at the Tesseract logger, which would
        # file the inherited methods' messages under Tesseract.
        self.log = logger

        self._server_url: str = os.environ.get("PAPERLESS_CHANDRA_SERVER_URL", "").strip()
        self._model_name: str = (
            os.environ.get("PAPERLESS_CHANDRA_MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME
        )
        self._max_output_tokens: int = _env_int(
            "PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS",
            default=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        self._timeout: int = _env_int("PAPERLESS_CHANDRA_TIMEOUT", default=DEFAULT_TIMEOUT_SECONDS)
        self._content_format: str = _env_choice(
            "PAPERLESS_CHANDRA_CONTENT_FORMAT",
            choices=CONTENT_FORMATS,
            default=DEFAULT_CONTENT_FORMAT,
        )

    def construct_ocrmypdf_parameters(
        self,
        input_file: Path,
        mime_type: str,
        output_file: Path,
        sidecar_file: Path,
        *,
        safe_fallback: bool = False,
        skip_text: bool = False,
    ) -> dict[str, Any]:
        # The result already holds the OCR user arguments, which the parent
        # merges last.
        args: dict[str, Any] = super().construct_ocrmypdf_parameters(
            input_file,
            mime_type,
            output_file,
            sidecar_file,
            safe_fallback=safe_fallback,
            skip_text=skip_text,
        )
        args["plugins"] = [_OCRMYPDF_PLUGIN_MODULE, *_user_plugins(args.get("plugins"))]
        # parse() writes these arguments to the debug log, and paperless
        # keeps debug messages in its log file by default.
        if "chandra_api_key" in args:
            del args["chandra_api_key"]
            self.log.warning(
                "Ignoring chandra_api_key in the OCR user arguments, because paperless "
                "logs them. Set PAPERLESS_CHANDRA_API_KEY instead.",
            )
        for key, value in (
            ("chandra_content_format", self._content_format),
            ("chandra_max_output_tokens", self._max_output_tokens),
            ("chandra_model_name", self._model_name),
            ("chandra_server_url", self._server_url),
            ("chandra_timeout", self._timeout),
        ):
            args.setdefault(key, value)
        return args
