"""Render an hOCR document to a text-only, overlay-ready PDF.

Per ocrmypdf's ``OcrEngine.generate_pdf`` contract the page must carry the
invisible text *only* - no raster. ocrmypdf's sandwich pipeline grafts the
original scan onto it afterwards, so drawing the image here would
double-render it. ocrmypdf 17 replaced the old monolithic transform with a
parser plus an fpdf2 renderer; this uses that surface directly.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def render_textonly(input_file: Path, hocr_file: Path, output_pdf: Path) -> None:
    """Parse ``hocr_file`` and write an invisible-text-only ``output_pdf``.

    The page is sized from the hOCR ``ocr_page`` box (input pixel
    dimensions). hOCR carries no scan resolution, so DPI falls back to the
    source image's own metadata and finally to 300.
    """
    import ocrmypdf
    from ocrmypdf.font import MultiFontManager
    from ocrmypdf.fpdf_renderer import Fpdf2PdfRenderer
    from ocrmypdf.hocrtransform import HocrParser

    ocr_page = HocrParser(hocr_file).parse()

    dpi = ocr_page.dpi
    if not dpi:
        with Image.open(input_file) as img:
            dpi = img.info.get("dpi", (300, 300))[0]

    font_dir = Path(ocrmypdf.__file__).parent / "data"
    renderer = Fpdf2PdfRenderer(
        page=ocr_page,
        dpi=float(dpi or 300),
        multi_font_manager=MultiFontManager(font_dir),
        invisible_text=True,
        image=None,
    )
    renderer.render(output_pdf)
