"""check_options preflight."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from ocrmypdf.exceptions import MissingDependencyError
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from paperless_chandra import ocrmypdf_plugin


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
        "chandra_content_format": "text",
        "chandra_timeout": 600,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


_MODEL_LIST = SimpleNamespace(data=[SimpleNamespace(id="chandra")])


def _request():
    return httpx.Request("GET", "http://ocr-host:8000/v1/models")


def _status_error(status):
    return APIStatusError("error", response=httpx.Response(status, request=_request()), body=None)


class _FakeOpenAI:
    """Stands in for the OpenAI client class: records the client's arguments, replays one answer."""

    def __init__(self):
        self.init_kwargs = {}
        self.requests = 0
        self.answer = _MODEL_LIST
        self.models = SimpleNamespace(list=self._list)

    def __call__(self, **kwargs):
        self.init_kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def _list(self):
        self.requests += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    fake = _FakeOpenAI()
    monkeypatch.setattr(ocrmypdf_plugin, "OpenAI", fake)
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    return fake


def test_check_options_accepts_the_lowest_limits(monkeypatch):
    probed = []
    monkeypatch.setattr(ocrmypdf_plugin, "_probe_server", lambda url, key: probed.append(url))
    ocrmypdf_plugin.check_options(_options(chandra_max_output_tokens=1, chandra_timeout=1))
    assert probed == ["http://ocr-host:8000/v1"]


def test_check_options_requires_server_url():
    with pytest.raises(MissingDependencyError, match="PAPERLESS_CHANDRA_SERVER_URL"):
        ocrmypdf_plugin.check_options(_options(chandra_server_url="  "))


def test_skip_text_run_does_not_probe_the_server(sdk):
    ocrmypdf_plugin.check_options(_options(skip_text=True))
    assert sdk.requests == 0


def test_skip_text_run_still_requires_a_server_url():
    with pytest.raises(MissingDependencyError, match="PAPERLESS_CHANDRA_SERVER_URL"):
        ocrmypdf_plugin.check_options(_options(skip_text=True, chandra_server_url="  "))


def test_preflight_sends_the_environment_key_when_the_option_is_empty(sdk, monkeypatch):
    monkeypatch.setenv("PAPERLESS_CHANDRA_API_KEY", "from-env")
    ocrmypdf_plugin.check_options(_options(chandra_api_key=""))
    assert sdk.init_kwargs["api_key"] == "from-env"


def test_preflight_keeps_its_own_limit_and_no_sdk_retries(sdk):
    ocrmypdf_plugin.check_options(_options(chandra_timeout=5))
    assert sdk.init_kwargs["timeout"] == 30 and sdk.init_kwargs["max_retries"] == 0


def test_check_options_rejects_unknown_content_format():
    with pytest.raises(ValueError):
        ocrmypdf_plugin.check_options(_options(chandra_content_format="html"))


def test_check_options_rejects_zero_max_output_tokens():
    with pytest.raises(ValueError, match="chandra_max_output_tokens"):
        ocrmypdf_plugin.check_options(_options(chandra_max_output_tokens=0))


def test_check_options_rejects_zero_timeout():
    with pytest.raises(ValueError, match="chandra_timeout"):
        ocrmypdf_plugin.check_options(_options(chandra_timeout=0))


def test_check_options_rejects_a_string_for_an_integer_option():
    with pytest.raises(ValueError, match="chandra_timeout"):
        ocrmypdf_plugin.check_options(_options(chandra_timeout="900"))


def test_check_options_rejects_a_non_finite_number():
    with pytest.raises(ValueError, match="chandra_timeout"):
        ocrmypdf_plugin.check_options(_options(chandra_timeout=float("nan")))


def test_check_options_rejects_a_number_for_a_text_option():
    with pytest.raises(ValueError, match="chandra_model_name"):
        ocrmypdf_plugin.check_options(_options(chandra_model_name=5))


def test_probe_maps_auth_failure(sdk):
    sdk.answer = _status_error(401)
    with pytest.raises(MissingDependencyError, match="rejected the API key"):
        ocrmypdf_plugin.check_options(_options())


def test_probe_maps_unreachable_server(sdk):
    sdk.answer = APIConnectionError(request=_request())
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin.check_options(_options())
    assert sdk.requests == 1


def test_probe_fails_on_a_server_error(sdk):
    sdk.answer = _status_error(503)
    with pytest.raises(
        MissingDependencyError, match="HTTP 503 at /models.*PAPERLESS_CHANDRA_SERVER_URL"
    ):
        ocrmypdf_plugin.check_options(_options())
    assert sdk.requests == 1


def test_probe_maps_a_timeout_to_not_reachable(sdk):
    sdk.answer = APITimeoutError(request=_request())
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin.check_options(_options())


@pytest.mark.parametrize(
    "answer",
    [
        SimpleNamespace(data=None),
        AttributeError("'str' object has no attribute '_set_private_attributes'"),
    ],
    ids=["json-without-data", "html-page"],
)
def test_probe_fails_on_an_answer_without_a_model_list(sdk, answer):
    sdk.answer = answer
    with pytest.raises(
        MissingDependencyError, match="without a model list.*PAPERLESS_CHANDRA_SERVER_URL"
    ):
        ocrmypdf_plugin.check_options(_options())


def test_probe_maps_a_url_without_a_scheme(monkeypatch):
    # The real SDK refuses the URL before it opens a connection.
    monkeypatch.setattr(ocrmypdf_plugin, "OpenAI", OpenAI)
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin.check_options(_options(chandra_server_url="gpu-host"))
