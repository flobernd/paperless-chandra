"""Projection-profile skew estimation on synthetic line patterns."""

from __future__ import annotations

from PIL import Image, ImageDraw

from paperless_chandra.engine.deskew import estimate_skew


def _bar_page(tmp_path, angle: float):
    """White page with black text-like bars, rotated by ``angle`` degrees."""
    img = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(img)
    for y in range(80, 560, 40):
        draw.rectangle([60, y, 740, y + 12], fill=0)
    if angle:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
    path = tmp_path / "page.png"
    img.save(path)
    return path


def test_straight_page_reports_near_zero(tmp_path):
    assert abs(estimate_skew(_bar_page(tmp_path, 0.0))) <= 0.2


def test_skewed_page_reports_correction_angle(tmp_path):
    # PIL rotates counterclockwise for positive angles; the estimator
    # returns the counterclockwise correction, so the sign inverts.
    est = estimate_skew(_bar_page(tmp_path, 2.0))
    assert abs(est + 2.0) <= 0.5
