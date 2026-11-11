"""render_textonly turns hOCR into a one-page invisible-text PDF."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest
from PIL import Image

from paperless_chandra.engine.hocr import Block, Line, Page, Word, render_hocr
from paperless_chandra.engine.pdf import render_textonly


def _render(tmp_path) -> Path:
    img_path = tmp_path / "page.tif"
    # A PNG stores its resolution in whole pixels per meter, which turns 72 dpi into 72.009.
    Image.new("RGB", (400, 300), "white").save(img_path, dpi=(72, 72))

    line = Line(
        box=(10, 10, 200, 40),
        words=[
            Word("hello", (10, 10, 100, 40)),
            Word("world", (110, 10, 200, 40)),
        ],
    )
    page = Page(
        width=400,
        height=300,
        lang="eng",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )
    hocr_path = tmp_path / "page.hocr"
    hocr_path.write_text(render_hocr(page), encoding="utf-8")

    out_pdf = tmp_path / "text.pdf"
    render_textonly(img_path, hocr_path, out_pdf)
    return out_pdf


def test_render_textonly_sizes_the_page_from_the_image_dpi(tmp_path):
    with pikepdf.open(_render(tmp_path)) as pdf_doc:
        # 400 by 300 pixels at 72 dpi are 400 by 300 points.
        assert [float(v) for v in pdf_doc.pages[0].mediabox] == pytest.approx([0, 0, 400, 300])


def test_render_textonly_writes_invisible_text(tmp_path):
    with pikepdf.open(_render(tmp_path)) as pdf_doc:
        modes = [
            int(operands[0])
            for operands, operator in pikepdf.parse_content_stream(pdf_doc.pages[0])
            if str(operator) == "Tr"
        ]
    # Text rendering mode 3 of the PDF specification draws nothing.
    assert set(modes) == {3}
