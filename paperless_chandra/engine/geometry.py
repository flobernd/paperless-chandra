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

    Each edge is the rounded share of the weight to its left, so no
    rounding error adds up and the last word ends at the ``box`` right edge.
    Every word inherits the line ``box``'s
    ``y0`` / ``y1``. A single word fills the whole box; empty input returns
    ``[]``. Rough by design - it backs the invisible layer, not layout
    analysis.
    """
    if not words:
        return []
    x0, y0, x1, y1 = box

    line_width = x1 - x0
    weights = [max(1, len(w)) for w in words]
    total = sum(weights) + _GAP_WEIGHT * (len(words) - 1)

    boxes: list[BBox] = []
    left = 0.0
    for weight in weights:
        boxes.append(
            (
                x0 + round(line_width * left / total),
                y0,
                x0 + round(line_width * (left + weight) / total),
                y1,
            )
        )
        left += weight + _GAP_WEIGHT
    return boxes
