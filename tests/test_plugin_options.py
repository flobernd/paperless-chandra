"""check_options preflight."""

from __future__ import annotations

import contextlib
import logging
import urllib.error
from types import SimpleNamespace

import pytest
from ocrmypdf.exceptions import MissingDependencyError

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


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    original = ocrmypdf_plugin._probe_server
    stub = lambda url, key: None  # noqa: E731
    stub.__wrapped__ = original  # type: ignore[attr-defined]
    monkeypatch.setattr(ocrmypdf_plugin, "_probe_server", stub)
    ocrmypdf_plugin._PROBED_SERVERS.clear()


def test_check_options_accepts_the_lowest_limits(monkeypatch):
    probed = []
    monkeypatch.setattr(ocrmypdf_plugin, "_probe_server", lambda url, key: probed.append(url))
    ocrmypdf_plugin.check_options(_options(chandra_max_output_tokens=1, chandra_timeout=1))
    assert probed == ["http://ocr-host:8000/v1"]


def test_check_options_requires_server_url():
    with pytest.raises(MissingDependencyError, match="PAPERLESS_CHANDRA_SERVER_URL"):
        ocrmypdf_plugin.check_options(_options(chandra_server_url="  "))


def test_preflight_sends_the_environment_key_when_the_option_is_empty(monkeypatch):
    monkeypatch.setattr(ocrmypdf_plugin, "_probe_server", ocrmypdf_plugin._probe_server.__wrapped__)  # type: ignore[attr-defined]
    monkeypatch.setenv("PAPERLESS_CHANDRA_API_KEY", "from-env")
    sent = []

    def capture(request, timeout):
        sent.append(request)
        return contextlib.nullcontext()

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", capture)
    ocrmypdf_plugin.check_options(_options(chandra_api_key=""))
    assert sent[0].get_header("Authorization") == "Bearer from-env"


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


def test_probe_maps_auth_failure(monkeypatch):
    def raise_401(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", raise_401)
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    with pytest.raises(MissingDependencyError, match="rejected the API key"):
        ocrmypdf_plugin._probe_server.__wrapped__("http://host:1", "bad")  # type: ignore[attr-defined]


def test_probe_maps_unreachable_server(monkeypatch):
    def raise_conn(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", raise_conn)
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin._probe_server.__wrapped__("http://host:1", "")  # type: ignore[attr-defined]


def test_probe_maps_a_socket_error_without_reason(monkeypatch):
    def raise_timeout(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", raise_timeout)
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin._probe_server.__wrapped__("http://gpu:8000", "")  # type: ignore[attr-defined]


def test_probe_continues_on_a_server_error(monkeypatch, caplog):
    def raise_500(request, timeout):
        raise urllib.error.HTTPError("http://gpu:8000/v1/models", 500, "boom", {}, None)

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", raise_500)
    with caplog.at_level(logging.WARNING):
        ocrmypdf_plugin._probe_server.__wrapped__("http://gpu:8000", "")  # type: ignore[attr-defined]
    assert "HTTP 500" in caplog.text
