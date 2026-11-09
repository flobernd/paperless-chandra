"""Projection-profile skew estimation for ocrmypdf's deskew hook.

The classic Postl method: rotate a downscaled binarised page through
candidate angles and score how sharply ink concentrates into rows (sum of
squared differences of adjacent row-ink counts). Text lines give a strong
peak at the deskewed angle. Pure PIL + numpy so the estimator is exact to
test in CI.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    # Row sums of brightness differ from the ink sums only in sign, so the
    # squared differences are the same without inverting the page.
    profile = np.asarray(rotated).sum(axis=1, dtype=np.int64)
    diff = profile[1:] - profile[:-1]
    return float((diff * diff).sum())


def _best(
    img: Image.Image, angles: Iterable[float], best_angle: float, best_score: float
) -> tuple[float, float]:
    # A blank page scores the same at every angle, and the strict > keeps its
    # seed of 0.0 instead of drifting to the last candidate.
    for angle in angles:
        s = _score(img, angle)
        if s > best_score:
            best_angle, best_score = angle, s
    return best_angle, best_score


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
        binary = Image.fromarray(np.where(arr < arr.mean(), np.uint8(0), np.uint8(255)))

    # The fine window is centered on the coarse winner, so each tenth inside
    # its cell is scored once. i == 0 is the incumbent in both lists, and its
    # score is already known.
    steps = int(MAX_SKEW_DEGREES / _COARSE_STEP)
    coarse = [i * _COARSE_STEP for i in range(-steps, steps + 1) if i != 0]
    best_angle, best_score = _best(binary, coarse, 0.0, _score(binary, 0.0))

    fine_span = int(_COARSE_STEP / _FINE_STEP)
    fine = [
        round(best_angle + i * _FINE_STEP, 1) for i in range(-fine_span + 1, fine_span) if i != 0
    ]
    fine = [angle for angle in fine if abs(angle) <= MAX_SKEW_DEGREES]
    best_angle, _ = _best(binary, fine, best_angle, best_score)

    # A best fit at the search boundary means the true angle is outside the
    # range; rotating by the clamped value would make things worse.
    if abs(best_angle) >= MAX_SKEW_DEGREES:
        return 0.0
    return best_angle
