"""Chandra layout chunks to a typed OCR page, plus sidecar rendering.

Chandra returns one HTML document per page whose top-level ``<div>``s are
layout blocks with pixel bounding boxes (``chandra.output.parse_chunks``).
Blocks carry no word or line geometry, so this module synthesises it:
block content is split into reading lines at HTML boundaries, a reading
line that fills several lines of the page is split into that many lines,
each line gets an equal vertical slice of the block box, and word boxes
are estimated proportionally within the line. Rough by design - the boxes
back the invisible PDF text layer, not layout analysis.
"""

from __future__ import annotations

import logging
import math
import re

from bs4 import BeautifulSoup

from paperless_chandra.engine.geometry import BBox, estimate_word_boxes
from paperless_chandra.engine.hocr import Block, Line, Page, Word, clamp_box, clean_text

log = logging.getLogger("paperless.chandra.blocks")

OCR_SYSTEM = "Chandra via paperless-chandra"

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

#: A glyph is about half as wide as its font size, and the next line starts
#: about 1.2 font sizes lower. The two give the number of lines that a text
#: of a known length fills in a box of a known shape.
_GLYPH_WIDTH = 0.5
_LINE_PITCH = 1.2


def _wrap_lines(lines: list[str], width: int, height: int) -> list[tuple[list[str], float, bool]]:
    """Break each reading line into the lines it most likely fills in the block.

    Chandra returns a paragraph as one line with the box of the whole block.
    A text layer line of that shape holds glyphs that are far narrower than
    they are high, and a text extractor then finds no gap between the words.
    Each entry is a line of the text layer: its words, the share of the
    block width that it fills, and whether it continues the reading line
    before it.
    """
    total = sum(len(line) for line in lines)
    # n lines of total / n characters each: a glyph is width * n / total wide,
    # so the lines are n * _LINE_PITCH * (width * n / total) / _GLYPH_WIDTH high.
    fitted = math.sqrt(total * height * _GLYPH_WIDTH / (_LINE_PITCH * width))
    per_line = total / max(fitted, 1.0)
    rows: list[tuple[list[str], float, bool]] = []
    for line in lines:
        words = line.split()
        count = min(len(words), max(1, round(len(line) / per_line)))
        target = len(line) / count
        parts: list[list[str]] = [[]]
        used = 0
        for word in words:
            # A word moves to the next line when more than half of it is past the target.
            if parts[-1] and len(parts) < count and used + len(word) / 2 > target:
                parts.append([])
                used = 0
            parts[-1].append(word)
            used += len(word) + 1
        lengths = [sum(len(word) for word in part) + len(part) - 1 for part in parts]
        longest = max(lengths)
        rows.extend(
            (part, length / longest, i > 0)
            for i, (part, length) in enumerate(zip(parts, lengths, strict=True))
        )
    return rows


def block_lines(content_html: str) -> list[str]:
    """Split a block's inner HTML into plain-text reading lines."""
    # chandra-ocr decodes a character reference such as ``&#1;`` in the block,
    # so an XML-invalid code point can arrive here although the engine cleaned
    # the raw answer. Left in, it becomes an empty word with a box of its own.
    soup = BeautifulSoup(clean_text(content_html), "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    _write_out_widgets(soup)
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")
    for tag in soup.find_all(_LINE_BREAK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = (_WS_RE.sub(" ", line).strip() for line in soup.get_text().split("\n"))
    return [line for line in lines if line]


def _write_out_widgets(soup: BeautifulSoup) -> None:
    """Replace each form widget with the text that it shows on the page.

    A widget holds its state in attributes, which neither ``get_text`` nor
    the markdown converter reads.
    """
    for widget in soup.find_all("input"):
        if str(widget.get("type") or "").lower() == "checkbox":
            widget.replace_with("[x]" if widget.has_attr("checked") else "[ ]")
        else:
            widget.replace_with(str(widget.get("value") or ""))


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
    x0, y0, x1, y1 = clamp_box((x0, y0, x1, y1), width, height)
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
        rows = _wrap_lines(lines_text, x1 - x0, block_h)
        n_lines = len(rows)
        lines: list[Line] = []
        for i, (tokens, share, continues) in enumerate(rows):
            # Per-line boundary arithmetic instead of one truncated line
            # height: rounding never accumulates and a block shorter in
            # pixels than its line count still yields non-degenerate boxes.
            ly0 = y0 + (i * block_h) // n_lines
            ly1 = y0 + ((i + 1) * block_h) // n_lines
            line_box = clamp_box((x0, ly0, x0 + round((x1 - x0) * share), ly1), width, height)
            words = [
                Word(tok, box)
                for tok, box in zip(tokens, estimate_word_boxes(tokens, line_box), strict=False)
            ]
            lines.append(Line(box=line_box, words=words, continues=continues))
        page.blocks.append(Block(box=(x0, y0, x1, y1), lines=lines))
    return page


def page_sidecar(page: Page) -> str:
    """Plain-text sidecar: reading lines per block, blank line between blocks."""
    blocks: list[str] = []
    for block in page.blocks:
        lines: list[str] = []
        for line in block.lines:
            text = " ".join(word.text for word in line.words)
            if line.continues and lines:
                lines[-1] = f"{lines[-1]} {text}"
            else:
                lines.append(text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def markdown_sidecar(raw_html: str) -> str:
    """Markdown sidecar via Chandra's own converter.

    Headers and footers are kept (invoice numbers and dates live there);
    images are dropped - their alt-text descriptions are model narration
    and the referenced image files are never extracted here.
    """
    from chandra.output import parse_markdown

    soup = BeautifulSoup(raw_html, "html.parser")
    _write_out_widgets(soup)
    # chandra-ocr converts the content of the layout blocks without the block
    # elements around it, so two blocks that hold inline elements only run
    # together. A block element inside each layout block keeps them apart.
    for block in soup.find_all("div", recursive=False):
        inner = soup.new_tag("div")
        inner.extend(list(block.contents))
        block.append(inner)
    return parse_markdown(str(soup), include_headers_footers=True, include_images=False)
