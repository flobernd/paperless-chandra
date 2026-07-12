"""Chandra layout chunks to a typed OCR page, plus sidecar rendering.

Chandra returns one HTML document per page whose top-level ``<div>``s are
layout blocks with pixel bounding boxes (``chandra.output.parse_chunks``).
Blocks carry no word or line geometry, so this module synthesises it:
block content is split into text lines at HTML boundaries, each line gets
an equal vertical slice of the block box, and word boxes are estimated
proportionally within the line. Rough by design - the boxes back the
invisible PDF text layer, not layout analysis.
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

from paperless_chandra.engine.geometry import BBox, estimate_word_boxes
from paperless_chandra.engine.hocr import Block, Line, Page, Word

log = logging.getLogger("paperless.chandra.blocks")

OCR_SYSTEM = "Chandra via paperless-chandra"

#: Chandra reports no per-word confidence; the text layer still needs an
#: ``x_wconf``, so a fixed high value is used.
CONFIDENCE = 95

#: Labels whose content is not document text. ``Image`` / ``Figure`` alt
#: descriptions and ``Diagram`` mermaid code are model narration and must
#: not pollute the searchable layer. Unknown or missing labels are kept:
#: losing real text is worse than indexing an unexpected block.
NON_TEXT_LABELS: frozenset[str] = frozenset(
    {"Image", "Figure", "Blank-Page", "Diagram", "Chemical-Block"}
)

#: Tags that terminate a text line. ``tr`` keeps one table row per line;
#: cells are space-joined separately.
_LINE_BREAK_TAGS = ("p", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "div")

_WS_RE = re.compile(r"[^\S\n]+")


def block_lines(content_html: str) -> list[str]:
    """Split a block's inner HTML into plain-text reading lines."""
    soup = BeautifulSoup(content_html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for widget in soup.find_all("input"):
        if (widget.get("type") or "").lower() == "checkbox":
            widget.replace_with("[x]" if widget.has_attr("checked") else "[ ]")
        else:
            widget.replace_with(str(widget.get("value") or ""))
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")
    for tag in soup.find_all(_LINE_BREAK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = (_WS_RE.sub(" ", line).strip() for line in soup.get_text().split("\n"))
    return [line for line in lines if line]


def _chunk_box(chunk: dict, width: int, height: int) -> BBox:
    """Chunk bbox clamped to the page; degenerate boxes widen to full page.

    Upstream ``parse_layout`` falls back to ``[0, 0, 1, 1]`` when the model
    emits a malformed ``data-bbox``; widening keeps that block's text in
    the layer instead of stacking it into one pixel.
    """
    bbox = list(chunk.get("bbox") or [])[:4]
    if len(bbox) < 4:
        return 0, 0, width, height
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(x1, width), min(y1, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        log.debug("Degenerate chunk bbox %s widened to full page.", bbox)
        return 0, 0, width, height
    return x0, y0, x1, y1


def page_from_chunks(chunks: list[dict], width: int, height: int, lang: str) -> Page:
    """Build a typed :class:`Page` from ``chandra.output.parse_chunks`` output."""
    page = Page(width=width, height=height, lang=lang, ocr_system=OCR_SYSTEM)
    for chunk in chunks:
        label = str(chunk.get("label") or "")
        if label in NON_TEXT_LABELS:
            continue
        lines_text = block_lines(str(chunk.get("content") or ""))
        if not lines_text:
            continue
        x0, y0, x1, y1 = _chunk_box(chunk, width, height)
        block_h = max(y1 - y0, 1)
        n_lines = len(lines_text)
        lines: list[Line] = []
        for i, text in enumerate(lines_text):
            # Per-line boundary arithmetic instead of one truncated line
            # height: rounding never accumulates and a block shorter in
            # pixels than its line count still yields non-degenerate boxes.
            ly0 = y0 + (i * block_h) // n_lines
            ly1 = y0 + ((i + 1) * block_h) // n_lines
            if ly1 <= ly0:
                ly1 = ly0 + 1
            line_box = (x0, ly0, x1, ly1)
            tokens = text.split()
            words = [
                Word(tok, box, CONFIDENCE)
                for tok, box in zip(tokens, estimate_word_boxes(tokens, line_box), strict=False)
            ]
            if not words:
                continue
            lines.append(Line(box=line_box, confidence=CONFIDENCE, text=text, words=words))
        if lines:
            page.blocks.append(Block(box=(x0, y0, x1, y1), lines=lines))
    return page


def page_sidecar(page: Page) -> str:
    """Plain-text sidecar: lines per block, blank line between blocks."""
    return "\n\n".join(
        "\n".join(line.text for line in block.lines if line.text) for block in page.blocks
    )


def markdown_sidecar(raw_html: str) -> str:
    """Markdown sidecar via Chandra's own converter.

    Headers and footers are kept (invoice numbers and dates live there);
    images are dropped - their alt-text descriptions are model narration
    and the referenced image files are never extracted here.
    """
    from chandra.output import parse_markdown

    return parse_markdown(raw_html, include_headers_footers=True, include_images=False)
