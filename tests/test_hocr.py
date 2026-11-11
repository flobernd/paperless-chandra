"""hOCR serialisation and the BCP-47 lang sanitiser."""

from __future__ import annotations

from xml.etree import ElementTree

from paperless_chandra.engine.geometry import BBox
from paperless_chandra.engine.hocr import (
    Block,
    Line,
    Page,
    Word,
    clean_text,
    render_hocr,
    to_hocr_lang,
    write_document,
)


def _page(line: Line, *, width: int = 400, height: int = 300) -> Page:
    return Page(
        width=width,
        height=height,
        lang="eng",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )


def _line(*, text: str = "hi", box: BBox = (10, 10, 90, 40)) -> Line:
    return Line(box=box, words=[Word(text, box)])


def test_to_hocr_lang_passes_iso_codes_through():
    assert to_hocr_lang("eng") == "eng"
    assert to_hocr_lang(" DE ") == "de"


def test_to_hocr_lang_rejects_unexpected_values():
    assert to_hocr_lang("") == "und"
    assert to_hocr_lang(None) == "und"
    assert to_hocr_lang("german") == "und"
    assert to_hocr_lang('en" onload="x') == "und"


def test_render_hocr_emits_page_and_word():
    html = render_hocr(_page(_line()))
    assert 'class="ocr_page"' in html
    assert "bbox 0 0 400 300" in html
    assert 'class="ocrx_word"' in html
    assert 'lang="eng"' in html


def test_render_hocr_escapes_text():
    line = Line(
        box=(0, 0, 50, 10),
        words=[Word("<b>&", (0, 0, 50, 10))],
    )
    assert "&lt;b&gt;&amp;" in render_hocr(_page(line))


def test_render_drops_xml_invalid_characters_from_words():
    # ocrmypdf's hOCR parser rejects a NUL and U+FFFE, the code point right after the
    # last one XML allows below U+10000. A lone surrogate cannot be encoded as UTF-8.
    hocr = render_hocr(_page(_line(text="in\x00voice\ud800\ufffe")))
    ElementTree.fromstring(hocr.encode("utf-8"))
    assert "invoice" in hocr and "\x00" not in hocr
    assert "\ufffe" not in hocr


def test_clean_text_keeps_whitespace_and_non_ascii_characters():
    # The sidecar separates its lines and blocks with newlines.
    text = "a\tb\r\nc 日 \ufb01 \ufffd \U0001f600"
    assert clean_text(text) == text


def test_write_document_cleans_the_sidecar(tmp_path):
    write_document(_page(_line()), tmp_path / "p.hocr", tmp_path / "p.txt", sidecar="a\x00b")
    assert (tmp_path / "p.txt").read_text(encoding="utf-8") == "ab"


def test_render_clamps_a_box_outside_the_page():
    hocr = render_hocr(_page(_line(box=(-5, -5, 1300, 40)), width=1200, height=1600))
    assert 'title="bbox 0 0 1200 40' in hocr
    assert "1300" not in hocr


def test_render_gives_a_zero_width_box_one_pixel():
    page = _page(_line(box=(50, 10, 50, 30)))
    assert 'title="bbox 50 10 51 30' in render_hocr(page)


def test_render_keeps_one_pixel_for_a_box_past_the_right_edge():
    # The box is inverted as well, and ocrmypdf's hOCR parser drops an inverted box.
    page = _page(_line(box=(1202, 10, 1200, 30)), width=1200, height=1600)
    assert 'title="bbox 1199 10 1200 30' in render_hocr(page)


def test_render_keeps_one_pixel_for_a_box_below_the_bottom_edge():
    page = _page(_line(box=(10, 1601, 20, 1700)), width=1200, height=1600)
    assert 'title="bbox 10 1599 20 1600' in render_hocr(page)
