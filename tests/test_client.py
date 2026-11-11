"""Page request: SDK wiring, retry policy, error mapping."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from chandra.prompts import PROMPT_MAPPING
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image

from paperless_chandra.engine import client
from paperless_chandra.engine.client import (
    ChandraClientError,
    normalize_server_url,
    ocr_image,
    resolve_api_key,
)

# Ends in a closing tag, so only the library's cut_from_end check flags it.
_LOOPED = "<p>" + "la " * 400 + "</p>"


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


def _completion(content, finish_reason="stop", tokens=5, usage=True, **message_fields):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, **message_fields),
            )
        ],
        usage=SimpleNamespace(completion_tokens=tokens) if usage else None,
    )


def _request():
    return httpx.Request("POST", "http://gpu:8000/v1/chat/completions")


def _status_error(status, message="error"):
    return APIStatusError(message, response=httpx.Response(status, request=_request()), body=None)


class _FakeOpenAI:
    """Stands in for the OpenAI client class: records the request, replays a script."""

    def __init__(self):
        self.init_kwargs = {}
        self.calls = []
        self._script = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def script(self, items):
        self._script = list(items)

    def __call__(self, **kwargs):
        self.init_kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake(monkeypatch):
    sdk = _FakeOpenAI()
    monkeypatch.setattr(client, "OpenAI", sdk)
    return sdk


@pytest.fixture(autouse=True)
def _reset_think_warned():
    client._think_warned = False
    yield
    client._think_warned = False


def test_server_url_gains_the_v1_suffix(fake):
    fake.script([_completion("x")])
    ocr_image(_image(), _options(chandra_server_url="http://ocr-host:8000"))
    assert fake.init_kwargs["base_url"] == "http://ocr-host:8000/v1"


def test_normalize_keeps_existing_version_suffix():
    assert normalize_server_url("http://host:8000/v1/") == "http://host:8000/v1"


def test_normalize_rejects_empty():
    with pytest.raises(ChandraClientError):
        normalize_server_url("   ")


def test_sends_the_layout_prompt_with_the_page_image(fake):
    fake.script([_completion("<div>ok</div>")])
    assert ocr_image(_image(), _options()) == "<div>ok</div>"
    content = fake.calls[0]["messages"][0]["content"]
    assert content[1]["text"] == PROMPT_MAPPING["ocr_layout"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_request_carries_the_configured_model_and_budget(fake):
    fake.script([_completion("x")])
    ocr_image(_image(), _options(chandra_model_name="chandra-2", chandra_max_output_tokens=2048))
    assert fake.calls[0]["model"] == "chandra-2"
    assert fake.calls[0]["max_tokens"] == 2048


def test_an_option_set_to_none_keeps_its_default(fake):
    fake.script([_completion("x")])
    ocr_image(_image(), _options(chandra_model_name=None, chandra_timeout=None))
    assert fake.calls[0]["model"] == "chandra"
    assert fake.init_kwargs["timeout"] == 600


def test_client_carries_the_timeout_and_no_sdk_retries(fake):
    fake.script([_completion("x")])
    ocr_image(_image(), _options(chandra_timeout=42))
    assert fake.init_kwargs["timeout"] == 42 and fake.init_kwargs["max_retries"] == 0


def test_timeout_fails_once_and_names_the_variable(fake):
    fake.script([APITimeoutError(request=_request())])
    with pytest.raises(ChandraClientError, match="PAPERLESS_CHANDRA_TIMEOUT"):
        ocr_image(_image(), _options())
    assert len(fake.calls) == 1


def test_connection_retries_span_the_whole_page(fake, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake.script(
        [
            APIConnectionError(request=_request()),
            APIConnectionError(request=_request()),
            _completion(_LOOPED),
            APIConnectionError(request=_request()),
        ]
    )
    with pytest.raises(ChandraClientError, match="not reachable"):
        ocr_image(_image(), _options())
    assert len(fake.calls) == 4


@pytest.mark.parametrize("content", [None, "<think>\nThe image shows an invoice."])
def test_thinking_output_at_the_budget_carries_the_template_hint(fake, content):
    fake.script([_completion(content, finish_reason="length", tokens=12384)])
    with pytest.raises(ChandraClientError, match="enable_thinking"):
        ocr_image(_image(), _options())


def test_plain_truncation_carries_no_template_hint(fake):
    fake.script([_completion("<div>partial", finish_reason="length", tokens=12384)])
    with pytest.raises(ChandraClientError) as excinfo:
        ocr_image(_image(), _options())
    assert "enable_thinking" not in str(excinfo.value)


def test_unauthorized_names_the_api_key_variable(fake):
    fake.script([_status_error(401)])
    with pytest.raises(ChandraClientError, match="PAPERLESS_CHANDRA_API_KEY"):
        ocr_image(_image(), _options())


def test_forbidden_names_the_api_key_variable(fake):
    fake.script([_status_error(403)])
    with pytest.raises(ChandraClientError, match="PAPERLESS_CHANDRA_API_KEY"):
        ocr_image(_image(), _options())


def test_not_found_names_the_model_variable(fake):
    fake.script([_status_error(404)])
    with pytest.raises(ChandraClientError, match="PAPERLESS_CHANDRA_MODEL_NAME"):
        ocr_image(_image(), _options())


def test_bad_request_carries_the_server_message(fake):
    fake.script([_status_error(400, "max_tokens is too large")])
    with pytest.raises(ChandraClientError) as excinfo:
        ocr_image(_image(), _options())
    assert "max_tokens is too large" in str(excinfo.value)
    assert "PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS" in str(excinfo.value)


def test_other_status_carries_the_status_and_the_server_message(fake):
    fake.script([_status_error(503, "engine overloaded")])
    with pytest.raises(ChandraClientError, match="503.*engine overloaded"):
        ocr_image(_image(), _options())


# What the SDK returns for a 200 answer whose body is no chat completion.
@pytest.mark.parametrize(
    "answer",
    [
        SimpleNamespace(choices=[], usage=SimpleNamespace(completion_tokens=0)),
        SimpleNamespace(choices=None, usage=None),
        "<html><body>welcome</body></html>",
    ],
    ids=["empty-choices", "null-choices", "html-body"],
)
def test_answer_without_a_choice_names_the_server_url(fake, answer):
    fake.script([answer])
    with pytest.raises(ChandraClientError, match="PAPERLESS_CHANDRA_SERVER_URL"):
        ocr_image(_image(), _options())


# A check keyed on the content lets null content through. Output that repeats
# itself reaches the loop retry unless the filter check comes first.
@pytest.mark.parametrize("content", [None, _LOOPED], ids=["null-content", "looping-output"])
def test_filtered_page_fails_without_a_retry(fake, content):
    fake.script([_completion(content, finish_reason="content_filter")] * 3)
    with pytest.raises(ChandraClientError, match="content_filter"):
        ocr_image(_image(), _options())
    assert len(fake.calls) == 1


def test_refusal_fails_and_carries_its_text(fake):
    fake.script([_completion(None, refusal="I cannot transcribe this document.")])
    with pytest.raises(ChandraClientError, match="I cannot transcribe this document"):
        ocr_image(_image(), _options())


def test_filtered_page_with_a_refusal_carries_both(fake):
    refusal = "This page shows a passport."
    fake.script([_completion(None, finish_reason="content_filter", refusal=refusal)])
    with pytest.raises(ChandraClientError) as excinfo:
        ocr_image(_image(), _options())
    assert "content_filter" in str(excinfo.value)
    assert refusal in str(excinfo.value)


def test_null_content_with_tokens_is_empty_and_warns(fake, caplog):
    fake.script([_completion(None, tokens=534)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == ""
    assert "enable_thinking" in caplog.text


def test_missing_usage_block_counts_zero_tokens(fake, caplog):
    fake.script([_completion("", usage=False)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == ""
    assert caplog.text == ""


def test_repeat_loop_retries_with_a_higher_temperature(fake):
    fake.script([_completion(_LOOPED), _completion("<div>ok</div>")])
    assert ocr_image(_image(), _options()) == "<div>ok</div>"
    assert fake.calls[1]["temperature"] > fake.calls[0]["temperature"]


def test_loop_at_the_budget_after_two_retries_fails(fake):
    fake.script([_completion(_LOOPED, finish_reason="length")] * 3)
    with pytest.raises(ChandraClientError, match="PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS"):
        ocr_image(_image(), _options())
    assert len(fake.calls) == 3


def test_loop_after_two_retries_returns_the_last_page(fake):
    last = "<p>" + "lo " * 400 + "</p>"
    fake.script([_completion(_LOOPED), _completion(_LOOPED), _completion(last)])
    assert ocr_image(_image(), _options()) == last


def test_api_key_option_wins_over_the_environment(fake, monkeypatch):
    monkeypatch.setenv("PAPERLESS_CHANDRA_API_KEY", "from-env")
    fake.script([_completion("x")])
    ocr_image(_image(), _options(chandra_api_key=" sekrit "))
    assert fake.init_kwargs["api_key"] == "sekrit"


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("PAPERLESS_CHANDRA_API_KEY", " sekrit ")
    assert resolve_api_key(_options(chandra_api_key="")) == "sekrit"


def test_api_key_defaults_to_the_vllm_convention(monkeypatch):
    monkeypatch.delenv("PAPERLESS_CHANDRA_API_KEY", raising=False)
    assert resolve_api_key(_options(chandra_api_key="")) == "EMPTY"


def test_api_key_that_is_no_string_is_left_out_of_the_error():
    with pytest.raises(ValueError, match="chandra_api_key") as raised:
        resolve_api_key(_options(chandra_api_key=b"sk-73914286"))
    assert "73914286" not in str(raised.value)


def _warning_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_unclosed_think_block_warns_with_enable_thinking_hint(fake, caplog):
    raw = "<think>\nThe image shows an invoice.\n<div data-label='Text'>hi</div>"
    fake.script([_completion(raw)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == raw
    warnings = _warning_records(caplog)
    assert len(warnings) == 1
    assert "enable_thinking" in warnings[0].getMessage()


def test_closed_think_block_is_silent(fake, caplog):
    raw = "<think>\nplan\n</think>\n<div data-label='Text'>hi</div>"
    fake.script([_completion(raw)])
    with caplog.at_level(logging.WARNING):
        assert ocr_image(_image(), _options()) == raw
    assert not _warning_records(caplog)


def test_think_warning_fires_once_across_both_kinds(fake, caplog):
    fake.script([_completion("<think>\n<div>hi</div>"), _completion("", tokens=99)])
    with caplog.at_level(logging.WARNING):
        ocr_image(_image(), _options())
        ocr_image(_image(), _options())
    assert len(_warning_records(caplog)) == 1
