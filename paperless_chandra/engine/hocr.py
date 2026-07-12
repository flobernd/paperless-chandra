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


def to_hocr_lang(code: str | None) -> str:
    """BCP-47 primary subtag for the hOCR ``lang`` attribute.

    Two- and three-letter ISO 639 codes (paperless passes Tesseract codes
    like ``eng``) are valid BCP-47 subtags and pass through lowercased;
    anything else becomes ``und`` (undetermined) so an unexpected value
    never leaks into the attribute.
    """
    cleaned = (code or "").strip().lower()
    return cleaned if _LANG_RE.fullmatch(cleaned) else "und"


@dataclass(frozen=True)
class Word:
    """One recognised word and its pixel box."""

    text: str
    box: BBox
    confidence: int  # 0-100, serialised as x_wconf


@dataclass
class Line:
    """A reading line: its box, its words, and its fitted baseline.

    ``text`` is the canonical recognised text for the sidecar. It is kept
    separate from ``words`` because some engines return an authoritative
    line transcription whose spacing differs from a naive word join.

    ``baseline`` is ``(slope, constant)`` in hOCR-relative form: ``slope``
    is the baseline gradient and ``constant`` is its y at the line box's
    left edge, measured up from the line box bottom. The default
    ``(0.0, 0.0)`` is a flat baseline on the box bottom.
    """

    box: BBox
    confidence: int
    text: str
    words: list[Word] = field(default_factory=list)
    baseline: tuple[float, float] = (0.0, 0.0)


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


def _attr(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def _bbox(b: BBox) -> str:
    return f"bbox {b[0]} {b[1]} {b[2]} {b[3]}"


def _baseline(line: Line) -> str:
    slope, constant = line.baseline
    return f"baseline {slope:.6f} {constant:.0f}"


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
        out.append(
            f'<div class="ocr_carea" id="carea_{block_no}" title="{_bbox(block.box)}">',
        )
        out.append(
            f'<p class="ocr_par" id="par_{block_no}" lang="{lang}" title="{_bbox(block.box)}">',
        )
        for line in block.lines:
            line_no += 1
            out.append(
                f'<span class="ocr_line" id="line_{line_no}" '
                f'title="{_bbox(line.box)}; {_baseline(line)}; '
                f'x_wconf {line.confidence}">',
            )
            last = len(line.words) - 1
            for i, word in enumerate(line.words):
                word_no += 1
                out.append(
                    f'<span class="ocrx_word" id="word_{word_no}" '
                    f'title="{_bbox(word.box)}; x_wconf {word.confidence}">'
                    f"{escape(word.text)}</span>",
                )
                if i < last:
                    out.append(" ")
            out.append("</span>")
        out.append("</p>")
        out.append("</div>")

    out.extend(["</div>", "</body>", "</html>"])
    return "\n".join(out)


def sidecar_text(page: Page) -> str:
    """Plain-text rendering: one line of recognised text per OCR line."""
    return "\n".join(line.text for block in page.blocks for line in block.lines if line.text)


def write_document(
    page: Page,
    hocr_path: Path,
    text_path: Path,
    *,
    sidecar: str | None = None,
) -> None:
    """Write the hOCR document and its plain-text sidecar to disk.

    ``sidecar`` overrides the default per-line rendering when the caller has
    a better (reading-ordered) text for the same page.
    """
    hocr_path.write_text(render_hocr(page), encoding="utf-8")
    text_path.write_text(sidecar if sidecar is not None else sidecar_text(page), encoding="utf-8")
