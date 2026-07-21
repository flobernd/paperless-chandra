"""Client wrapper: URL normalisation, settings application, error mapping."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from PIL import Image

from paperless_chandra.engine import client
from paperless_chandra.engine.client import (
    ChandraClientError,
    normalize_server_url,
    ocr_image,
)


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _image():
    return Image.new("RGB", (100, 100), "white")


class _Result:
    def __init__(self, raw="<div></div>", error=False, token_count=7):
        self.raw = raw
        self.error = error
        self.token_count = token_count


@pytest.fixture(autouse=True)
def _reset_think_warned():
    client._think_warned = False
    yield
    client._think_warned = False


def test_normalize_appends_v1():
    assert normalize_server_url("http://host:8000") == "http://host:8000/v1"


def test_normalize_keeps_existing_version_suffix():
    assert normalize_server_url("http://host:8000/v1/") == "http://host:8000/v1"


def test_normalize_rejects_empty():
    with pytest.raises(ChandraClientError):
        normalize_server_url("   ")


def test_ocr_image_sends_layout_prompt_and_returns_raw(monkeypatch):
    captured = {}

    def fake_generate(batch, max_output_tokens=None, vllm_api_base=None, **kwargs):
        captured["batch"] = batch
        captured["max_output_tokens"] = max_output_tokens
        captured["vllm_api_base"] = vllm_api_base
        return [_Result(raw="<div>ok</div>")]

    monkeypatch.setattr(client, "generate_vllm", fake_generate)
    raw = ocr_image(_image(), _options())
    assert raw == "<div>ok</div>"
    assert captured["batch"][0].prompt_type == "ocr_layout"
    assert captured["max_output_tokens"] == 12384
    assert captured["vllm_api_base"] == "http://ocr-host:8000/v1"


def test_ocr_image_applies_model_and_key_to_chandra_settings(monkeypatch):
    from chandra.settings import settings as chandra_settings

    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result()])
    ocr_image(_image(), _options(chandra_model_name="custom", chandra_api_key="sekrit"))
    assert chandra_settings.VLLM_MODEL_NAME == "custom"
    assert chandra_settings.VLLM_API_KEY == "sekrit"


def test_ocr_image_defaults_api_key_to_vllm_convention(monkeypatch):
    from chandra.settings import settings as chandra_settings

    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result()])
    ocr_image(_image(), _options(chandra_api_key=""))
    assert chandra_settings.VLLM_API_KEY == "EMPTY"


def test_ocr_image_raises_on_error_flag(monkeypatch):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(error=True)])
    with pytest.raises(ChandraClientError):
        ocr_image(_image(), _options())


def test_ocr_image_raises_on_empty_result_list(monkeypatch):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [])
    with pytest.raises(ChandraClientError):
        ocr_image(_image(), _options())


def test_ocr_image_returns_empty_string_for_empty_raw(monkeypatch):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw="")])
    assert ocr_image(_image(), _options()) == ""


def _warning_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_unclosed_think_block_warns_with_enable_thinking_hint(monkeypatch, caplog):
    raw = "<think>\nThe image shows an invoice.\n<div data-label='Text'>hi</div>"
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw=raw)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == raw
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    assert "enable_thinking" in warnings[0].getMessage()


def test_empty_content_with_generated_tokens_warns(monkeypatch, caplog):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw="", token_count=534)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == ""
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    assert "enable_thinking" in warnings[0].getMessage()


def test_empty_content_without_generated_tokens_is_silent(monkeypatch, caplog):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw="", token_count=0)])
    with caplog.at_level(logging.WARNING):
        ocr_image(_image(), _options())
    assert not _warning_records(caplog)


def test_closed_think_block_is_silent(monkeypatch, caplog):
    raw = "<think>\nplan\n</think>\n<div data-label='Text'>hi</div>"
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw=raw)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == raw
    assert not _warning_records(caplog)


def test_normal_output_is_silent(monkeypatch, caplog):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw="<div>ok</div>")])
    with caplog.at_level(logging.WARNING):
        ocr_image(_image(), _options())
    assert not _warning_records(caplog)


def test_think_warning_fires_once_across_both_kinds(monkeypatch, caplog):
    results = iter(
        [
            [_Result(raw="<think>\n<div>hi</div>")],
            [_Result(raw="", token_count=99)],
        ]
    )
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: next(results))
    with caplog.at_level(logging.WARNING):
        ocr_image(_image(), _options())
        ocr_image(_image(), _options())
    assert len(_warning_records(caplog)) == 1
