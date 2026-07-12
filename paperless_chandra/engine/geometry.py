"""Pure geometry for synthesising hOCR word boxes.

Chandra reports block-level bounding boxes only; these helpers spread a
line's words across the line box so the invisible text layer has a
position per word.
"""

from __future__ import annotations

from collections.abc import Sequence

#: ``(x0, y0, x1, y1)`` axis-aligned box in image pixels.
BBox = tuple[int, int, int, int]

#: Weight of one inter-word gap relative to a single glyph when
#: :func:`estimate_word_boxes` partitions a line - a space is narrower
#: than an average glyph.
_GAP_WEIGHT = 0.5


def estimate_word_boxes(words: Sequence[str], box: BBox) -> list[BBox]:
    """Spread ``words`` across a line ``box`` by a length-weighted partition.

    * a word's weight is ``max(1, len(word))``;
    * each inter-word gap weighs ``0.5`` (a space is narrower than a glyph);
    * each width is proportional to its weight over the total.

    The last word's right edge is snapped to the ``box`` right edge so
    rounding never leaves a gap. Every word inherits the line ``box``'s
    ``y0`` / ``y1``. A single word fills the whole box; empty input returns
    ``[]``. Rough by design - it backs the invisible layer, not layout
    analysis.
    """
    if not words:
        return []
    x0, y0, x1, y1 = box
    if len(words) == 1:
        return [(x0, y0, x1, y1)]

    line_width = x1 - x0
    weights = [max(1, len(w)) for w in words]
    total = sum(weights) + _GAP_WEIGHT * (len(words) - 1)
    gap_width = round(line_width * _GAP_WEIGHT / total) if total > 0 else 0

    boxes: list[BBox] = []
    cursor = x0
    last = len(words) - 1
    for i, weight in enumerate(weights):
        width = round(line_width * weight / total) if total > 0 else 0
        right = x1 if i == last else cursor + width
        boxes.append((cursor, y0, right, y1))
        cursor = right + gap_width
    return boxes
