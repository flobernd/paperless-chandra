"""hOCR serialisation and the BCP-47 lang sanitiser."""

from __future__ import annotations

from paperless_chandra.engine.hocr import (
    Block,
    Line,
    Page,
    Word,
    render_hocr,
    to_hocr_lang,
    write_document,
)


def _page(line: Line) -> Page:
    return Page(
        width=400,
        height=300,
        lang="eng",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )


def _line() -> Line:
    return Line(
        box=(10, 10, 90, 40),
        confidence=88,
        text="hi",
        words=[Word("hi", (10, 10, 90, 40), 88)],
    )


def test_to_hocr_lang_passes_iso_codes_through():
    assert to_hocr_lang("eng") == "eng"
    assert to_hocr_lang(" DE ") == "de"


def test_to_hocr_lang_rejects_unexpected_values():
    assert to_hocr_lang("") == "und"
    assert to_hocr_lang(None) == "und"
    assert to_hocr_lang("german") == "und"
    assert to_hocr_lang('en" onload="x') == "und"


def test_render_hocr_emits_page_word_and_conf():
    html = render_hocr(_page(_line()))
    assert 'class="ocr_page"' in html
    assert "bbox 0 0 400 300" in html
    assert 'class="ocrx_word"' in html
    assert "x_wconf 88" in html
    assert 'lang="eng"' in html


def test_render_hocr_escapes_text():
    line = Line(
        box=(0, 0, 50, 10),
        confidence=90,
        text="<b>&",
        words=[Word("<b>&", (0, 0, 50, 10), 90)],
    )
    assert "&lt;b&gt;&amp;" in render_hocr(_page(line))


def test_write_document_sidecar_override(tmp_path):
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(_line()), hocr_path, text_path, sidecar="custom order")
    assert text_path.read_text(encoding="utf-8") == "custom order"
    assert "ocrx_word" in hocr_path.read_text(encoding="utf-8")


def test_write_document_default_sidecar_is_line_text(tmp_path):
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(_line()), hocr_path, text_path)
    assert text_path.read_text(encoding="utf-8") == "hi"
