"""check_options preflight and engine binding."""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace

import pytest
from ocrmypdf.exceptions import MissingDependencyError

from paperless_chandra import ocrmypdf_plugin
from paperless_chandra.engine.engine import ChandraEngine


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
        "chandra_content_format": "text",
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


def test_get_ocr_engine_returns_chandra_engine():
    assert isinstance(ocrmypdf_plugin.get_ocr_engine(), ChandraEngine)


def test_check_options_accepts_valid_configuration():
    ocrmypdf_plugin.check_options(_options())


def test_check_options_requires_server_url():
    with pytest.raises(MissingDependencyError):
        ocrmypdf_plugin.check_options(_options(chandra_server_url="  "))


def test_check_options_rejects_unknown_content_format():
    with pytest.raises(ValueError):
        ocrmypdf_plugin.check_options(_options(chandra_content_format="html"))


@pytest.mark.parametrize("tokens", [0, -1])
def test_check_options_rejects_non_positive_max_output_tokens(tokens):
    with pytest.raises(ValueError):
        ocrmypdf_plugin.check_options(_options(chandra_max_output_tokens=tokens))


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
