"""The ocrmypdf ``OcrEngine`` backed by a self-hosted Chandra server."""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Any

from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from PIL import Image

from paperless_chandra.engine import blocks, client, deskew, pdf
from paperless_chandra.engine.hocr import write_document

log = logging.getLogger("paperless.chandra.engine")

_rotate_warned = False


def _chandra_version() -> str:
    try:
        return importlib.metadata.version("chandra-ocr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class ChandraEngine(OcrEngine):
    """Chandra implementation of ocrmypdf's ``OcrEngine``."""

    @staticmethod
    def version() -> str:
        return _chandra_version()

    @staticmethod
    def creator_tag(options: Any) -> str:
        return f"Chandra {ChandraEngine.version()}"

    def __str__(self) -> str:
        return f"Chandra {ChandraEngine.version()}"

    @staticmethod
    def languages(options: Any) -> set[str]:
        # Chandra is language-agnostic (90+ languages in one pass); report
        # whatever was requested so ocrmypdf preflight never aborts.
        return set(getattr(options, "languages", None) or []) | {"eng"}

    @staticmethod
    def get_orientation(input_file: Path, options: Any) -> OrientationConfidence:
        global _rotate_warned
        if not _rotate_warned:
            log.warning(
                "PAPERLESS_OCR_ROTATE_PAGES is not supported by paperless-chandra; "
                "pages are OCR'd in their stored orientation."
            )
            _rotate_warned = True
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options: Any) -> float:
        try:
            return deskew.estimate_skew(input_file)
        except Exception:
            log.exception("Deskew estimation failed for %s; skipping deskew.", input_file)
            return 0.0

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        from chandra.output import parse_chunks

        with Image.open(input_file) as img:
            image = img.convert("RGB")

        raw = client.ocr_image(image, options)
        chunks = parse_chunks(raw, image) if raw else []

        langs = list(getattr(options, "languages", None) or [])
        page = blocks.page_from_chunks(
            chunks,
            image.width,
            image.height,
            langs[0] if langs else "eng",
        )
        content_format = (getattr(options, "chandra_content_format", "") or "text").strip()
        if content_format == "markdown" and raw:
            sidecar = blocks.markdown_sidecar(raw)
        else:
            sidecar = blocks.page_sidecar(page)
        write_document(page, output_hocr, output_text, sidecar=sidecar)

    @classmethod
    def generate_pdf(
        cls,
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        """Render OCR output as a text-only PDF.

        ``cls.generate_hocr`` (not the base method) so a subclass still
        runs if a user forces the ``generate_pdf`` renderer path.
        """
        output_hocr = output_pdf.with_suffix(".hocr")
        cls.generate_hocr(input_file, output_hocr, output_text, options)
        pdf.render_textonly(input_file, output_hocr, output_pdf)
