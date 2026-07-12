"""estimate_word_boxes partitions a line box without gaps or overlaps."""

from __future__ import annotations

from paperless_chandra.engine.geometry import estimate_word_boxes


def test_empty_input_returns_empty():
    assert estimate_word_boxes([], (0, 0, 100, 20)) == []


def test_single_word_fills_the_box():
    assert estimate_word_boxes(["hello"], (10, 5, 90, 25)) == [(10, 5, 90, 25)]


def test_last_word_snaps_to_right_edge():
    boxes = estimate_word_boxes(["a", "bb", "ccc"], (0, 0, 300, 20))
    assert boxes[-1][2] == 300


def test_boxes_are_monotone_and_non_overlapping():
    boxes = estimate_word_boxes(["alpha", "beta", "gamma"], (0, 0, 500, 20))
    for prev, cur in zip(boxes, boxes[1:], strict=False):
        assert prev[2] <= cur[0]
    for x0, y0, x1, y1 in boxes:
        assert x1 > x0
        assert (y0, y1) == (0, 20)


def test_longer_words_get_wider_boxes():
    short, long = estimate_word_boxes(["ab", "abcdefgh"], (0, 0, 400, 20))
    assert (long[2] - long[0]) > (short[2] - short[0])
