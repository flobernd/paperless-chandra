"""Typed OCR document model and hOCR / sidecar serialisation.

The engine populates a :class:`Page` and hands it here; this module is the
single place that knows the hOCR wire format. The output is plain
hOCR/XHTML so ocrmypdf's ``HocrParser`` can render the invisible text
layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from paperless_chandra.engine.geometry import BBox

_LANG_RE = re.compile(r"^[a-z]{2,3}$")
_XML_INVALID_RE = re.compile(r"[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def to_hocr_lang(code: str | None) -> str:
    """BCP-47 primary subtag for the hOCR ``lang`` attribute.

    Two- and three-letter ISO 639 codes (paperless passes Tesseract codes
    like ``eng``) are valid BCP-47 subtags and pass through lowercased;
    anything else becomes ``und`` (undetermined) so an unexpected value
    never leaks into the attribute.
    """
    cleaned = (code or "").strip().lower()
    return cleaned if _LANG_RE.fullmatch(cleaned) else "und"


#: Chandra reports no confidence for a word, so each line and each word
#: carries this one fixed high ``x_wconf``.
CONFIDENCE = 95


@dataclass(frozen=True)
class Word:
    """One recognised word and its pixel box."""

    text: str
    box: BBox


@dataclass
class Line:
    """A reading line: its box and its words.

    The words carry the text. A Chandra reading line has one space between
    its words, so the sidecar gets the line back by joining them with one
    space.
    """

    box: BBox
    words: list[Word] = field(default_factory=list)


@dataclass
class Block:
    """A layout region (column area / paragraph) containing reading lines."""

    box: BBox
    lines: list[Line] = field(default_factory=list)


@dataclass
class Page:
    """A single OCR'd page, ready to serialise."""

    width: int
    height: int
    lang: str  # Tesseract code from paperless; normalised to BCP-47 on serialise
    ocr_system: str
    blocks: list[Block] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Drop the code points XML 1.0 forbids, lone surrogates included.

    ocrmypdf's hOCR parser rejects a document that holds one, and a lone
    surrogate cannot be written as UTF-8 at all.
    """
    return _XML_INVALID_RE.sub("", text)


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    """Clamp ``box`` inside a ``width`` by ``height`` page, keeping at least one pixel.

    A word box estimated for a narrow line can come out zero-width, and
    ocrmypdf's hOCR parser drops an element whose box is inverted.
    """
    x0 = min(max(box[0], 0), width - 1)
    y0 = min(max(box[1], 0), height - 1)
    x1 = min(width, max(box[2], x0 + 1))
    y1 = min(height, max(box[3], y0 + 1))
    return (x0, y0, x1, y1)


def _attr(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def _bbox(box: BBox, page: Page) -> str:
    x0, y0, x1, y1 = clamp_box(box, page.width, page.height)
    return f"bbox {x0} {y0} {x1} {y1}"


def render_hocr(page: Page) -> str:
    """Serialise a :class:`Page` to an hOCR/XHTML string.

    The language is normalised to a BCP-47 subtag and attribute-escaped so
    an unexpected engine code can never break out of ``lang="…"``.
    """
    lang = _attr(to_hocr_lang(page.lang))
    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
        '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
        "<head>",
        "<title></title>",
        '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
        f'<meta name="ocr-system" content="{_attr(page.ocr_system)}" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
        "</head>",
        "<body>",
        f'<div class="ocr_page" id="page_1" title="bbox 0 0 {page.width} {page.height}">',
    ]

    word_no = 0
    line_no = 0
    for block_no, block in enumerate(page.blocks, start=1):
        block_bbox = _bbox(block.box, page)
        out.append(
            f'<div class="ocr_carea" id="carea_{block_no}" title="{block_bbox}">',
        )
        out.append(
            f'<p class="ocr_par" id="par_{block_no}" lang="{lang}" title="{block_bbox}">',
        )
        for line in block.lines:
            line_no += 1
            # The baseline sits flat on the line box bottom because Chandra
            # reports no geometry finer than a block box.
            out.append(
                f'<span class="ocr_line" id="line_{line_no}" '
                f'title="{_bbox(line.box, page)}; baseline 0.000000 0; '
                f'x_wconf {CONFIDENCE}">',
            )
            last = len(line.words) - 1
            for i, word in enumerate(line.words):
                word_no += 1
                out.append(
                    f'<span class="ocrx_word" id="word_{word_no}" '
                    f'title="{_bbox(word.box, page)}; x_wconf {CONFIDENCE}">'
                    f"{escape(clean_text(word.text))}</span>",
                )
                if i < last:
                    out.append(" ")
            out.append("</span>")
        out.append("</p>")
        out.append("</div>")

    out.extend(["</div>", "</body>", "</html>"])
    return "\n".join(out)


def write_document(page: Page, hocr_path: Path, text_path: Path, *, sidecar: str) -> None:
    """Write the hOCR document and its plain-text sidecar to disk."""
    hocr_path.write_text(render_hocr(page), encoding="utf-8")
    text_path.write_text(clean_text(sidecar), encoding="utf-8")
