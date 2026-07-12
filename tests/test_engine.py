"""ChandraEngine: hOCR generation, sidecar formats, hook behaviour."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from paperless_chandra.engine import client as client_module
from paperless_chandra.engine import osd as osd_module
from paperless_chandra.engine.engine import ChandraEngine

#: bbox values are normalised 0-1000; with a 1000x1000 page they map 1:1.
FIXTURE_HTML = (
    '<div data-label="Section-Header" data-bbox="100 50 900 120"><h1>Invoice 42</h1></div>'
    '<div data-label="Text" data-bbox="100 150 900 400">'
    "<p>Hello world<br>Second line</p></div>"
    '<div data-label="Image" data-bbox="0 500 500 900"><img alt="a cat"/></div>'
)


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
        "chandra_content_format": "text",
        "languages": ["eng"],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _page_png(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(path, dpi=(72, 72))
    return path


def _run_generate_hocr(tmp_path, monkeypatch, options, raw=FIXTURE_HTML):
    monkeypatch.setattr(client_module, "ocr_image", lambda image, opts: raw)
    hocr = tmp_path / "out.hocr"
    text = tmp_path / "out.txt"
    ChandraEngine.generate_hocr(_page_png(tmp_path), hocr, text, options)
    return hocr, text


def test_generate_hocr_writes_words_and_sidecar(tmp_path, monkeypatch):
    hocr, text = _run_generate_hocr(tmp_path, monkeypatch, _options())
    hocr_html = hocr.read_text(encoding="utf-8")
    assert "Invoice" in hocr_html
    assert 'class="ocrx_word"' in hocr_html
    sidecar = text.read_text(encoding="utf-8")
    assert "Invoice 42" in sidecar
    assert "Hello world\nSecond line" in sidecar
    assert "cat" not in sidecar


def test_generate_hocr_markdown_sidecar(tmp_path, monkeypatch):
    _, text = _run_generate_hocr(tmp_path, monkeypatch, _options(chandra_content_format="markdown"))
    assert "# Invoice 42" in text.read_text(encoding="utf-8")


def test_generate_hocr_empty_response_writes_empty_sidecar(tmp_path, monkeypatch):
    hocr, text = _run_generate_hocr(tmp_path, monkeypatch, _options(), raw="")
    assert text.read_text(encoding="utf-8") == ""
    assert 'class="ocr_page"' in hocr.read_text(encoding="utf-8")


def test_generate_pdf_dispatches_through_cls(tmp_path, monkeypatch):
    calls = []

    class Recorder(ChandraEngine):
        @staticmethod
        def generate_hocr(input_file, output_hocr, output_text, options):
            calls.append("hocr")
            ChandraEngine.generate_hocr(input_file, output_hocr, output_text, options)

    monkeypatch.setattr(client_module, "ocr_image", lambda image, opts: FIXTURE_HTML)
    out_pdf = tmp_path / "text.pdf"
    Recorder.generate_pdf(_page_png(tmp_path), out_pdf, tmp_path / "t.txt", _options())
    assert calls == ["hocr"]
    assert out_pdf.stat().st_size > 0


def test_languages_reports_requested_codes():
    assert ChandraEngine.languages(_options(languages=["eng", "deu"])) >= {"eng", "deu"}


def test_get_orientation_maps_osd_result(tmp_path, monkeypatch):
    monkeypatch.setattr(osd_module, "detect_orientation", lambda input_file: (180, 9.95))
    oc = ChandraEngine.get_orientation(_page_png(tmp_path), _options())
    assert oc.angle == 180
    assert oc.confidence == 9.95


def test_creator_tag_mentions_chandra():
    assert "Chandra" in ChandraEngine.creator_tag(_options())
