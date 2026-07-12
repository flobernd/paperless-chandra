"""Tesseract OSD orientation detection: output parsing and failure handling."""

from __future__ import annotations

import logging
import subprocess

import pytest

from paperless_chandra.engine import osd

CANNED_OSD_OUTPUT = b"""Page number: 0
Orientation in degrees: 180
Rotate: 180
Orientation confidence: 9.95
Script: Latin
Script confidence: 5.35
"""


@pytest.fixture(autouse=True)
def _reset_warn_once():
    osd._osd_warned = False
    yield
    osd._osd_warned = False


def _run(monkeypatch, *, returncode=0, stdout=b"", side_effect=None):
    def fake_run(*args, **kwargs):
        if side_effect is not None:
            raise side_effect
        if returncode:
            raise subprocess.CalledProcessError(returncode, args, output=stdout)
        return subprocess.CompletedProcess(args, returncode, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_parses_canned_osd_output(tmp_path, monkeypatch):
    _run(monkeypatch, stdout=CANNED_OSD_OUTPUT)
    assert osd.detect_orientation(tmp_path / "page.png") == (180, 9.95)


def test_missing_orientation_keys_defaults_to_zero(tmp_path, monkeypatch):
    _run(monkeypatch, stdout=b"Script: Latin\nScript confidence: 5.35\n")
    assert osd.detect_orientation(tmp_path / "page.png") == (0, 0.0)


def test_too_few_characters_is_benign_and_silent(tmp_path, monkeypatch, caplog):
    _run(
        monkeypatch,
        returncode=1,
        stdout=b"Too few characters. Skipping this page\n",
    )
    with caplog.at_level(logging.WARNING):
        result = osd.detect_orientation(tmp_path / "page.png")
    assert result == (0, 0.0)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_image_too_large_is_benign_and_silent(tmp_path, monkeypatch, caplog):
    _run(monkeypatch, returncode=1, stdout=b"Image too large\n")
    with caplog.at_level(logging.WARNING):
        result = osd.detect_orientation(tmp_path / "page.png")
    assert result == (0, 0.0)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_missing_binary_warns_once_across_two_calls(tmp_path, monkeypatch, caplog):
    _run(monkeypatch, side_effect=FileNotFoundError())
    with caplog.at_level(logging.WARNING):
        first = osd.detect_orientation(tmp_path / "page.png")
        second = osd.detect_orientation(tmp_path / "page.png")
    assert first == (0, 0.0)
    assert second == (0, 0.0)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1


def test_other_called_process_error_warns_once_with_excerpt(tmp_path, monkeypatch, caplog):
    _run(
        monkeypatch,
        returncode=1,
        stdout=b"Error, unable to open data file osd.traineddata\n",
    )
    with caplog.at_level(logging.WARNING):
        result = osd.detect_orientation(tmp_path / "page.png")
    assert result == (0, 0.0)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "osd.traineddata" in warnings[0].getMessage()


def test_timeout_returns_zero(tmp_path, monkeypatch):
    _run(monkeypatch, side_effect=subprocess.TimeoutExpired(cmd="tesseract", timeout=60))
    assert osd.detect_orientation(tmp_path / "page.png") == (0, 0.0)
