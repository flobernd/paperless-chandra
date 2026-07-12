"""Projection-profile skew estimation for ocrmypdf's deskew hook.

The classic Postl method: rotate a downscaled binarised page through
candidate angles and score how sharply ink concentrates into rows (sum of
squared differences of adjacent row-ink counts). Text lines give a strong
peak at the deskewed angle. Pure PIL + numpy so the estimator is exact to
test in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

#: Scans are rarely skewed beyond a few degrees; a wider search would start
#: locking onto rotated tables and figures instead of the text body.
MAX_SKEW_DEGREES = 5.0
_COARSE_STEP = 0.5
_FINE_STEP = 0.1
#: Skew is a global property; full resolution adds cost, not signal.
_TARGET_WIDTH = 1200


def _score(img: Image.Image, angle: float) -> float:
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
    ink = 255 - np.asarray(rotated, dtype=np.int64)
    profile = ink.sum(axis=1)
    diff = profile[1:] - profile[:-1]
    return float((diff * diff).sum())


def estimate_skew(input_file: Path) -> float:
    """Degrees to rotate counterclockwise so text lines run horizontally."""
    with Image.open(input_file) as raw:
        gray = raw.convert("L")
        if gray.width > _TARGET_WIDTH:
            height = max(1, round(gray.height * _TARGET_WIDTH / gray.width))
            gray = gray.resize((_TARGET_WIDTH, height))
        # A global mean threshold is enough: the score only needs ink rows to
        # dominate background rows, not a clean segmentation.
        arr = np.asarray(gray)
        binary = Image.fromarray(np.where(arr < arr.mean(), 0, 255).astype(np.uint8))

    best_angle = 0.0
    best_score = _score(binary, 0.0)
    steps = int(MAX_SKEW_DEGREES / _COARSE_STEP)
    for i in range(-steps, steps + 1):
        angle = i * _COARSE_STEP
        s = _score(binary, angle)
        if s > best_score:
            best_angle, best_score = angle, s

    fine_span = int(_COARSE_STEP / _FINE_STEP)
    for i in range(-fine_span, fine_span + 1):
        angle = best_angle + i * _FINE_STEP
        if abs(angle) > MAX_SKEW_DEGREES:
            continue
        s = _score(binary, angle)
        if s > best_score:
            best_angle, best_score = angle, s

    # A best fit at the search boundary means the true angle is outside the
    # range; rotating by the clamped value would make things worse.
    if abs(best_angle) >= MAX_SKEW_DEGREES:
        return 0.0
    return best_angle
