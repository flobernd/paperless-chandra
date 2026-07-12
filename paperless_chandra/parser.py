"""Paperless-ngx parser plugin that runs ocrmypdf with the Chandra engine.

This parser mirrors ``paperless.parsers.tesseract.RasterisedDocumentParser``
almost exactly. The structural differences are:

* ``plugins=["paperless_chandra.ocrmypdf_plugin"]`` is added so ocrmypdf
  loads the Chandra engine instead of Tesseract.
* No language mapping: Chandra is language-agnostic and reads mixed
  scripts in a single pass, so ``PAPERLESS_OCR_LANGUAGE`` is passed
  through verbatim (it still drives paperless's own search stemming, and
  its first code labels the hOCR).
* The ``chandra_*`` family (server URL, model name, API key, output token
  budget, content format) is passed through as ocrmypdf kwargs
  (registered as CLI args by the ``add_options`` hookimpl).

Everything else - image alpha removal, DPI handling, PDF/A conversion,
``OCR_MODE`` semantics, the safe-fallback retry - is identical to the
Tesseract parser. The implementation is intentionally kept close to
``tesseract.py`` so behaviour stays consistent for users switching
engines.
"""

from __future__ import annotations

import datetime
import importlib.resources
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn, Self

from django.conf import settings
from documents.parsers import ParseError, make_thumbnail_from_pdf
from documents.utils import copy_file_with_basic_stats, maybe_override_pixel_limit, run_subprocess
from paperless.config import OcrConfig
from paperless.models import CleanChoices, ModeChoices, OutputTypeChoices
from paperless.parsers.tesseract import post_process_text
from paperless.parsers.utils import (
    PDF_TEXT_MIN_LENGTH,
    extract_pdf_text,
    is_tagged_pdf,
    read_file_handle_unicode_errors,
)
from PIL import Image

from paperless_chandra import __version__

if TYPE_CHECKING:
    from types import TracebackType

    from paperless.parsers import MetadataEntry, ParserContext

logger = logging.getLogger("paperless.parsing.chandra")

_SRGB_ICC_DATA: Final[bytes] = (
    importlib.resources.files("ocrmypdf.data").joinpath("sRGB.icc").read_bytes()
)

# Identical to tesseract.py - paperless-chandra is intended as a drop-in
# replacement, so the supported-MIME set must match exactly.
_SUPPORTED_MIME_TYPES: Final[dict[str, str]] = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/heic": ".heic",
}

_OCRMYPDF_PLUGIN_MODULE: Final[str] = "paperless_chandra.ocrmypdf_plugin"


class _NoTextFoundError(Exception):
    """Internal signal that ocrmypdf produced an empty sidecar."""


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


class PaperlessChandraParser:
    """OCR parser using Chandra via ocrmypdf (drop-in for the Tesseract parser)."""

    name: str = "Paperless-ngx Chandra Parser"
    version: str = __version__
    author: str = "Florian Bernd"
    url: str = "https://github.com/flobernd/paperless-chandra"

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def supported_mime_types(cls) -> dict[str, str]:
        return _SUPPORTED_MIME_TYPES

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        if mime_type not in _SUPPORTED_MIME_TYPES:
            return None
        # Default 15 beats Tesseract's 10 - once installed, Chandra wins.
        return _env_int("PAPERLESS_CHANDRA_SCORE", default=15)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def can_produce_archive(self) -> bool:
        return True

    @property
    def requires_pdf_rendition(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, logging_group: object | None = None) -> None:
        settings.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        self.tempdir = Path(
            tempfile.mkdtemp(prefix="paperless-chandra-", dir=settings.SCRATCH_DIR),
        )
        self.settings = OcrConfig()
        self.archive_path: Path | None = None
        self.text: str | None = None
        self.date: datetime.datetime | None = None
        self.log = logger

        # Plugin-specific env vars, read once per parser instance.
        self._server_url: str = (os.environ.get("PAPERLESS_CHANDRA_SERVER_URL", "") or "").strip()
        self._model_name: str = (
            os.environ.get("PAPERLESS_CHANDRA_MODEL_NAME", "chandra") or ""
        ).strip() or "chandra"
        self._api_key: str = (os.environ.get("PAPERLESS_CHANDRA_API_KEY", "") or "").strip()
        self._max_output_tokens: int = _env_int(
            "PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS",
            default=12384,
        )
        self._content_format: str = _env_choice(
            "PAPERLESS_CHANDRA_CONTENT_FORMAT",
            choices=("text", "markdown"),
            default="text",
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.log.debug("Cleaning up temporary directory %s", self.tempdir)
        shutil.rmtree(self.tempdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Core parsing interface
    # ------------------------------------------------------------------

    def configure(self, context: ParserContext) -> None:
        pass

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_text(self) -> str | None:
        return self.text

    def get_date(self) -> datetime.datetime | None:
        return self.date

    def get_archive_path(self) -> Path | None:
        return self.archive_path

    # ------------------------------------------------------------------
    # Thumbnail, page count, metadata
    # ------------------------------------------------------------------

    def get_thumbnail(self, document_path: Path, mime_type: str) -> Path:
        return make_thumbnail_from_pdf(
            self.archive_path or Path(document_path),
            self.tempdir,
        )

    def get_page_count(self, document_path: Path, mime_type: str) -> int | None:
        if mime_type == "application/pdf":
            from paperless.parsers.utils import get_page_count_for_pdf

            return get_page_count_for_pdf(Path(document_path), log=self.log)
        return None

    def extract_metadata(
        self,
        document_path: Path,
        mime_type: str,
    ) -> list[MetadataEntry]:
        if mime_type != "application/pdf":
            return []

        from paperless.parsers.utils import extract_pdf_metadata

        return extract_pdf_metadata(Path(document_path), log=self.log)

    # ------------------------------------------------------------------
    # Image helpers (engine-agnostic - same as tesseract.py)
    # ------------------------------------------------------------------

    def is_image(self, mime_type: str) -> bool:
        return mime_type in {
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/bmp",
            "image/gif",
            "image/webp",
            "image/heic",
        }

    def has_alpha(self, image: Path) -> bool:
        with Image.open(image) as im:
            return im.mode in ("RGBA", "LA")

    def remove_alpha(self, image_path: Path) -> Path:
        no_alpha_image = Path(self.tempdir) / "image-no-alpha"
        run_subprocess(
            [
                settings.CONVERT_BINARY,
                "-alpha",
                "off",
                str(image_path),
                str(no_alpha_image),
            ],
            logger=self.log,
        )
        return no_alpha_image

    def get_dpi(self, image: Path) -> int | None:
        try:
            with Image.open(image) as im:
                x, _ = im.info["dpi"]
                return round(x)
        except Exception as e:
            self.log.warning(
                "Error while getting DPI from image %s: %s",
                image,
                e,
            )
            return None

    def calculate_a4_dpi(self, image: Path) -> int | None:
        try:
            with Image.open(image) as im:
                width, _ = im.size
                dpi = int(width / (21 / 2.54))
                self.log.debug(
                    "Estimated DPI %d based on image width %d",
                    dpi,
                    width,
                )
                return dpi
        except Exception as e:
            self.log.warning(
                "Error while calculating DPI for image %s: %s",
                image,
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Text extraction (identical to tesseract.py)
    # ------------------------------------------------------------------

    def extract_text(
        self,
        sidecar_file: Path | None,
        pdf_file: Path,
    ) -> str | None:
        text: str | None = None
        if (
            sidecar_file is not None
            and sidecar_file.is_file()
            and self.settings.mode != ModeChoices.REDO
        ):
            text = read_file_handle_unicode_errors(sidecar_file)
            if "[OCR skipped on page" not in text:
                self.log.debug("Using text from sidecar file")
                return post_process_text(text)
            self.log.debug("Incomplete sidecar file: discarding.")

        if not Path(pdf_file).is_file():
            return None

        return post_process_text(extract_pdf_text(Path(pdf_file), log=self.log))

    # ------------------------------------------------------------------
    # ocrmypdf argument construction
    # ------------------------------------------------------------------

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
        ocrmypdf_args: dict[str, Any] = {
            "input_file_or_options": input_file,
            "output_file": output_file,
            "use_threads": True,
            "jobs": settings.THREADS_PER_WORKER,
            "language": self.settings.language or "eng",
            "output_type": self.settings.output_type,
            "progress_bar": False,
            # ─ Chandra engine wiring ────────────────────────────────────
            "plugins": [_OCRMYPDF_PLUGIN_MODULE],
            # ocrmypdf 17.x removed the 'hocr' renderer and routes everything
            # through the fpdf2 renderer, which still calls our generate_hocr
            # and renders the invisible text layer itself. No renderer override
            # is required (and 'hocr' would be silently ignored).
            # Custom kwargs registered by our add_options hookimpl.
            "chandra_server_url": self._server_url,
            "chandra_model_name": self._model_name,
            "chandra_api_key": self._api_key,
            "chandra_max_output_tokens": self._max_output_tokens,
            "chandra_content_format": self._content_format,
        }

        if "pdfa" in ocrmypdf_args["output_type"]:
            ocrmypdf_args["color_conversion_strategy"] = self.settings.color_conversion_strategy

        # OCR-mode flags (mutually exclusive). See tesseract.py:293-302.
        if safe_fallback or self.settings.mode == ModeChoices.FORCE:
            ocrmypdf_args["force_ocr"] = True
        elif self.settings.mode == ModeChoices.REDO:
            ocrmypdf_args["redo_ocr"] = True
        elif skip_text or self.settings.mode == ModeChoices.OFF:
            ocrmypdf_args["skip_text"] = True
        elif self.settings.mode == ModeChoices.AUTO:
            pass
        else:  # pragma: no cover
            raise ParseError(f"Invalid ocr mode: {self.settings.mode}")

        if self.settings.clean == CleanChoices.CLEAN:
            ocrmypdf_args["clean"] = True
        elif self.settings.clean == CleanChoices.FINAL:
            if self.settings.mode == ModeChoices.REDO:
                ocrmypdf_args["clean"] = True
            else:
                ocrmypdf_args["clean_final"] = True

        if self.settings.deskew and self.settings.mode != ModeChoices.REDO:
            ocrmypdf_args["deskew"] = True

        if self.settings.rotate:
            ocrmypdf_args["rotate_pages"] = True
            ocrmypdf_args["rotate_pages_threshold"] = self.settings.rotate_threshold

        if self.settings.pages is not None and self.settings.pages > 0:
            ocrmypdf_args["pages"] = f"1-{self.settings.pages}"
        else:
            ocrmypdf_args["sidecar"] = sidecar_file

        if self.is_image(mime_type):
            maybe_override_pixel_limit()

            dpi = self.get_dpi(input_file)
            a4_dpi = self.calculate_a4_dpi(input_file)

            if self.has_alpha(input_file):
                self.log.info(
                    "Removing alpha layer from %s for compatibility with img2pdf",
                    input_file,
                )
                ocrmypdf_args["input_file_or_options"] = self.remove_alpha(input_file)

            if dpi:
                self.log.debug(
                    "Detected DPI for image %s: %d",
                    input_file,
                    dpi,
                )
                ocrmypdf_args["image_dpi"] = dpi
            elif self.settings.image_dpi is not None:
                ocrmypdf_args["image_dpi"] = self.settings.image_dpi
            elif a4_dpi:
                ocrmypdf_args["image_dpi"] = a4_dpi
            else:
                raise ParseError(
                    f"Cannot produce archive PDF for image {input_file}, "
                    f"no DPI information is present in this image and "
                    f"OCR_IMAGE_DPI is not set.",
                )
            if ocrmypdf_args["image_dpi"] < 70:  # pragma: no cover
                self.log.warning(
                    "Image DPI of %d is low, OCR may fail",
                    ocrmypdf_args["image_dpi"],
                )

        if self.settings.user_args is not None:
            try:
                ocrmypdf_args = {**ocrmypdf_args, **self.settings.user_args}
            except Exception as e:
                self.log.warning(
                    "There is an issue with PAPERLESS_OCR_USER_ARGS, so "
                    "they will not be used. %s: %s",
                    type(e).__name__,
                    e,
                )

        if self.settings.max_image_pixel is not None and self.settings.max_image_pixel >= 0:
            max_pixels_mpixels = self.settings.max_image_pixel / 1_000_000.0
            msg = (
                "OCR pixel limit is disabled!"
                if max_pixels_mpixels == 0
                else f"Calculated {max_pixels_mpixels} megapixels for OCR"
            )
            self.log.debug(msg)
            ocrmypdf_args["max_image_mpixels"] = max_pixels_mpixels

        return ocrmypdf_args

    # ------------------------------------------------------------------
    # PDF/A conversion without OCR (mode=off path)
    # ------------------------------------------------------------------

    def _convert_image_to_pdfa(self, document_path: Path) -> Path:
        """Wrap an image as PDF/A-2b without invoking any OCR engine."""
        import img2pdf
        import pikepdf

        plain_pdf_path = Path(self.tempdir) / "image_plain.pdf"
        try:
            convert_kwargs: dict = {}
            if self.settings.image_dpi is not None:
                convert_kwargs["layout_fun"] = img2pdf.get_fixed_dpi_layout_fun(
                    (self.settings.image_dpi, self.settings.image_dpi),
                )
            plain_pdf_path.write_bytes(
                img2pdf.convert(str(document_path), **convert_kwargs),
            )
        except Exception as e:
            raise ParseError(
                f"img2pdf conversion failed for {document_path}: {e!s}",
            ) from e

        pdfa_path = Path(self.tempdir) / "archive.pdf"
        try:
            with pikepdf.open(plain_pdf_path) as pdf:
                cs = pdf.make_stream(_SRGB_ICC_DATA)
                cs["/N"] = 3
                output_intent = pikepdf.Dictionary(
                    Type=pikepdf.Name("/OutputIntent"),
                    S=pikepdf.Name("/GTS_PDFA1"),
                    OutputConditionIdentifier=pikepdf.String("sRGB"),
                    DestOutputProfile=cs,
                )
                pdf.Root["/OutputIntents"] = pdf.make_indirect(
                    pikepdf.Array([output_intent]),
                )
                meta = pdf.open_metadata(set_pikepdf_as_editor=False)
                meta["pdfaid:part"] = "2"
                meta["pdfaid:conformance"] = "B"
                pdf.save(pdfa_path)
        except Exception as e:
            self.log.warning(
                f"PDF/A metadata stamping failed ({e!s}); falling back to plain PDF.",
            )
            pdfa_path.write_bytes(plain_pdf_path.read_bytes())

        return pdfa_path

    def _convert_pdf_to_pdfa(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Convert a PDF to PDF/A via Ghostscript without OCR."""
        from ocrmypdf._exec.ghostscript import generate_pdfa
        from ocrmypdf.pdfa import generate_pdfa_ps

        output_type = self.settings.output_type
        if output_type == OutputTypeChoices.PDF:
            copy_file_with_basic_stats(input_path, output_path)
            return

        pdfa_part = "2" if output_type == "pdfa" else output_type.split("-")[-1]

        pdfmark = Path(self.tempdir) / "pdfa.ps"
        generate_pdfa_ps(pdfmark)

        color_strategy = self.settings.color_conversion_strategy or "RGB"

        self.log.debug(
            "Converting PDF to PDF/A-%s via Ghostscript (no OCR): %s",
            pdfa_part,
            input_path,
        )

        generate_pdfa(
            pdf_pages=[pdfmark, input_path],
            output_file=output_path,
            compression="auto",
            color_conversion_strategy=color_strategy,
            pdfa_part=pdfa_part,
        )

    def _handle_subprocess_output_error(self, e: Exception) -> NoReturn:
        if "Ghostscript PDF/A rendering" in str(e):
            self.log.warning(
                "Ghostscript PDF/A rendering failed, consider setting "
                "PAPERLESS_OCR_USER_ARGS: "
                "'{\"continue_on_soft_render_error\": true}'",
            )
        raise ParseError(
            f"SubprocessOutputError: {e!s}. See logs for more information.",
        ) from e

    # ------------------------------------------------------------------
    # The actual parse() - same flow as tesseract.py:parse()
    # ------------------------------------------------------------------

    def parse(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool = True,
    ) -> None:
        import ocrmypdf
        from ocrmypdf import EncryptedPdfError, InputFileError, SubprocessOutputError
        from ocrmypdf.exceptions import DigitalSignatureError, PriorOcrFoundError

        if mime_type == "application/pdf":
            text_original = self.extract_text(None, document_path)
            original_has_text = is_tagged_pdf(document_path, log=self.log) or (
                text_original is not None and len(text_original) > PDF_TEXT_MIN_LENGTH
            )
        else:
            text_original = None
            original_has_text = False

        self.log.debug(
            "Text detection: original_has_text=%s (text_length=%d, mode=%s, produce_archive=%s)",
            original_has_text,
            len(text_original) if text_original else 0,
            self.settings.mode,
            produce_archive,
        )

        # --- OCR_MODE=off: never invoke OCR engine ---
        if self.settings.mode == ModeChoices.OFF:
            if not produce_archive:
                self.log.debug(
                    "OCR: skipped - OCR_MODE=off, no archive requested;"
                    " returning pdftotext content only",
                )
                self.text = text_original or ""
                return
            if self.is_image(mime_type):
                self.log.debug(
                    "OCR: skipped - OCR_MODE=off, image input; converting to PDF/A without OCR",
                )
                try:
                    self.archive_path = self._convert_image_to_pdfa(document_path)
                    self.text = ""
                except Exception as e:
                    raise ParseError(
                        f"Image to PDF/A conversion failed: {e!s}",
                    ) from e
                return
            archive_path = Path(self.tempdir) / "archive.pdf"
            try:
                self._convert_pdf_to_pdfa(document_path, archive_path)
                self.archive_path = archive_path
                self.text = text_original or ""
            except SubprocessOutputError as e:
                self._handle_subprocess_output_error(e)
            except Exception as e:
                raise ParseError(f"{e.__class__.__name__}: {e!s}") from e
            return

        # --- OCR_MODE=auto: skip ocrmypdf entirely if text exists, no archive ---
        if self.settings.mode == ModeChoices.AUTO and original_has_text and not produce_archive:
            self.log.debug(
                "Document has text and no archive requested; skipping OCRmyPDF entirely.",
            )
            self.text = text_original
            return

        # --- All other paths: run ocrmypdf with Chandra plugin ---
        archive_path = Path(self.tempdir) / "archive.pdf"
        sidecar_file = Path(self.tempdir) / "sidecar.txt"

        skip_text = self.settings.mode == ModeChoices.AUTO and original_has_text

        # skip_text=True means ocrmypdf re-uses existing text; the sidecar
        # is still produced because archive PDF/A output extraction needs it.
        if skip_text:
            self.log.debug(
                "OCR strategy: PDF/A conversion only (skip_text)"
                " - OCR_MODE=auto, document already has text",
            )
        else:
            self.log.debug(
                "OCR strategy: full OCR via Chandra - OCR_MODE=%s",
                self.settings.mode,
            )

        args = self.construct_ocrmypdf_parameters(
            document_path,
            mime_type,
            archive_path,
            sidecar_file,
            skip_text=skip_text,
        )

        try:
            log_args = {k: ("***" if k == "chandra_api_key" else v) for k, v in args.items()}
            self.log.debug("Calling OCRmyPDF with args: %s", log_args)
            ocrmypdf.ocr(**args)

            if produce_archive:
                self.archive_path = archive_path

            self.text = self.extract_text(sidecar_file, archive_path)

            if not self.text:
                raise _NoTextFoundError(
                    "No text was found in the original document",
                )
        except (DigitalSignatureError, EncryptedPdfError):
            self.log.warning(
                "This file is encrypted and/or signed, OCR is impossible. Using "
                "any text present in the original file.",
            )
            if original_has_text:
                self.text = text_original
        except SubprocessOutputError as e:
            self._handle_subprocess_output_error(e)
        except (_NoTextFoundError, InputFileError, PriorOcrFoundError) as e:
            self.log.warning(
                "Encountered an error while running OCR: %s. Attempting force OCR to get the text.",
                e,
            )

            archive_path_fallback = Path(self.tempdir) / "archive-fallback.pdf"
            sidecar_file_fallback = Path(self.tempdir) / "sidecar-fallback.txt"

            args = self.construct_ocrmypdf_parameters(
                document_path,
                mime_type,
                archive_path_fallback,
                sidecar_file_fallback,
                safe_fallback=True,
            )

            try:
                log_args = {k: ("***" if k == "chandra_api_key" else v) for k, v in args.items()}
                self.log.debug(
                    "Fallback: Calling OCRmyPDF with args: %s",
                    log_args,
                )
                ocrmypdf.ocr(**args)
                self.text = self.extract_text(
                    sidecar_file_fallback,
                    archive_path_fallback,
                )
                if produce_archive:
                    self.archive_path = archive_path_fallback
            except Exception as e:
                raise ParseError(f"{e.__class__.__name__}: {e!s}") from e

        except Exception as e:
            raise ParseError(f"{e.__class__.__name__}: {e!s}") from e

        if not self.text:
            if original_has_text:
                self.text = text_original
            else:
                self.log.warning(
                    f"No text was found in {document_path}, the content will be empty.",
                )
                self.text = ""
