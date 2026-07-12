"""render_textonly turns hOCR into a one-page invisible-text PDF."""

from __future__ import annotations

import pikepdf
from PIL import Image

from paperless_chandra.engine.hocr import Block, Line, Page, Word, render_hocr
from paperless_chandra.engine.pdf import render_textonly


def test_render_textonly_produces_a_pdf(tmp_path):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(img_path, dpi=(72, 72))

    line = Line(
        box=(10, 10, 200, 40),
        confidence=90,
        text="hello world",
        words=[
            Word("hello", (10, 10, 100, 40), 90),
            Word("world", (110, 10, 200, 40), 90),
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

    with pikepdf.open(out_pdf) as pdf_doc:
        assert len(pdf_doc.pages) == 1
