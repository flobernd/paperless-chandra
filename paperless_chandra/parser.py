"""Paperless-ngx parser that runs ocrmypdf with the Chandra engine.

``PaperlessChandraParser`` subclasses the upstream Tesseract parser,
``paperless.parsers.tesseract.RasterisedDocumentParser``, and inherits its
pipeline: the OCR modes, the image handling, the PDF/A conversion and the
safe-fallback retry. The subclass adds its identity, the remote-service
flag, its score and two overrides, ``construct_ocrmypdf_parameters`` and
``extract_text``. The first override loads the Chandra ocrmypdf plugin and
passes it these options, each read from its ``PAPERLESS_CHANDRA_*`` variable
unless the OCR user arguments set it:

* ``chandra_content_format``
* ``chandra_max_output_tokens``
* ``chandra_model_name``
* ``chandra_server_url``
* ``chandra_timeout``

The parent writes the options to the debug log, so the API key stays out of
them and the engine reads ``PAPERLESS_CHANDRA_API_KEY`` from the environment.

In redo mode the parent takes the document text from the archive, because
ocrmypdf blanks the visible text of a page before the engine reads it. One
override of ``extract_text`` returns the sidecar of a redo installation
when the markdown format is set and the archive keeps no text of the input.
The ocrmypdf plugin reports the pages that keep such text for that purpose,
in a file beside the sidecar.

The plugin relies on four contracts of the parent's ``parse``:

* It calls the builder for each ocrmypdf run, the safe-fallback run
  included, so each run loads the Chandra engine.
* It calls ``extract_text`` with the sidecar file of the run that just
  ended, in a scratch directory that no other document uses.
* It takes the document text from the sidecar file, which ocrmypdf
  assembles from the page text the engine writes.
* Its exception handling decides what fails the consumption: an error the
  engine raises inside ``ocrmypdf.ocr`` leaves ``parse`` as a ``ParseError``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Final

from paperless.models import ModeChoices
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
_BUILDER_HOOK_PRESENT: Final[bool] = hasattr(
    RasterisedDocumentParser, "construct_ocrmypdf_parameters"
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

_SKIP_MARKER: Final[str] = "[OCR skipped on page"


def _mask_report_path(sidecar_file: Path) -> Path:
    return sidecar_file.with_name(f"{sidecar_file.name}.mask.json")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer value for %s=%r. Using default %d.",
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
            "Invalid value for %s=%r. Valid choices are %s. Using default %r.",
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
        # The default 15 beats Tesseract's 10, but not the 20 of paperless-ngx's
        # own remote engine when that engine is configured.
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

        self._chandra_options: dict[str, int | str] = {
            "chandra_content_format": _env_choice(
                "PAPERLESS_CHANDRA_CONTENT_FORMAT",
                choices=CONTENT_FORMATS,
                default=DEFAULT_CONTENT_FORMAT,
            ),
            "chandra_max_output_tokens": _env_int(
                "PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS",
                default=DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            "chandra_model_name": (
                os.environ.get("PAPERLESS_CHANDRA_MODEL_NAME", "").strip() or DEFAULT_MODEL_NAME
            ),
            "chandra_server_url": os.environ.get("PAPERLESS_CHANDRA_SERVER_URL", "").strip(),
            "chandra_timeout": _env_int(
                "PAPERLESS_CHANDRA_TIMEOUT",
                default=DEFAULT_TIMEOUT_SECONDS,
            ),
        }

    def construct_ocrmypdf_parameters(
        self,
        input_file: Path,
        mime_type: str,
        output_file: Path,
        sidecar_file: Path,
        # The keywords pass through unread, so a keyword that a later
        # paperless-ngx release adds needs no change here.
        **kwargs: Any,
    ) -> dict[str, Any]:
        # The result already holds the OCR user arguments, which the parent
        # merges last.
        args: dict[str, Any] = super().construct_ocrmypdf_parameters(
            input_file,
            mime_type,
            output_file,
            sidecar_file,
            **kwargs,
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
        for key, value in self._chandra_options.items():
            args.setdefault(key, value)
        # Set after the user arguments, so that the report lies where
        # extract_text looks for it.
        if self.settings.mode == ModeChoices.REDO:
            report = _mask_report_path(sidecar_file)
            # A report that an earlier run left at this path must not speak
            # for this one, in which the hook can end without a report.
            report.unlink(missing_ok=True)
            args["chandra_mask_report"] = str(report)
        return args

    def extract_text(self, sidecar_file: Path | None, pdf_file: Path) -> str | None:
        # The parent reads no sidecar in redo mode, because a page with blanked
        # text has only the rest of its text there.
        if sidecar_file is not None and self.settings.mode == ModeChoices.REDO:
            text = self._whole_sidecar(sidecar_file)
            if text:
                self.log.debug("Using text from sidecar file")
                return text
        return super().extract_text(sidecar_file, pdf_file)

    def _whole_sidecar(self, sidecar_file: Path) -> str | None:
        """The post-processed sidecar text when the archive holds no other text, else ``None``."""
        try:
            report = json.loads(_mask_report_path(sidecar_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if report != {"masked": []} or not sidecar_file.is_file():
            return None
        try:
            # paperless-ngx moved post_process_text to another module in 3.0.5
            # and kept the name importable from the tesseract module. A release
            # that drops the name fails this import, which costs the markdown
            # content of redo mode. At the head of this module the same failure
            # would cost the parser, because the registry passes it over.
            from paperless.parsers.tesseract import (
                post_process_text,
                read_file_handle_unicode_errors,
            )
        except ImportError:
            return None
        text = read_file_handle_unicode_errors(sidecar_file)
        # ocrmypdf writes the marker for a page that got no OCR. Such a page keeps
        # its old text layer, which the report takes for stripped.
        if _SKIP_MARKER in text:
            return None
        return post_process_text(text)
