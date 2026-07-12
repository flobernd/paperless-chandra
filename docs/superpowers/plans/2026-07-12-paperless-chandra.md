# paperless-chandra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A paperless-ngx 3.x (beta) parser plugin that replaces Tesseract with Chandra OCR served by a self-hosted OpenAI-compatible inference server.

**Architecture:** A `paperless_ngx.parsers` entry point (`PaperlessChandraParser`) drives `ocrmypdf` with a custom `OcrEngine` (`ChandraEngine`). The engine sends each rasterised page to a remote server via the upstream `chandra-ocr` client, converts the returned layout-block HTML into synthesized line/word boxes, and emits hOCR that ocrmypdf renders as the invisible PDF text layer. Spec: `docs/superpowers/specs/2026-07-12-paperless-chandra-design.md`.

**Tech Stack:** Python 3.12+, ocrmypdf >= 17.4, chandra-ocr >= 0.2, BeautifulSoup4, numpy, PIL, pytest, docker (paperless-ngx:beta, llama.cpp for e2e).

## Global Constraints

- Package `paperless_chandra`, distribution `paperless-chandra`, version `0.1.0`, license MIT, author `Florian Bernd <git@flobernd.de>`, Python `>=3.12` (host has 3.13.5).
- Reference implementation (read-only source for copies): `/home/administrator/workspace/paperless-paddleocr`. Chandra source for reference: `/tmp/claude-1000/-home-administrator-workspace-paperless-chandra/5da9259f-8ac1-46f6-85c1-3b36d9c5c34d/scratchpad/chandra` (if missing: `git clone --depth 1 https://github.com/datalab-to/chandra.git <scratchpad>/chandra`).
- Import chandra internals ONLY from submodules `chandra.model.schema`, `chandra.model.vllm`, `chandra.settings`, `chandra.output`, `chandra.prompts`. Never `from chandra.model import ...` (its `__init__` pulls the HF backend).
- Prose style (docs, comments, commit messages): never use em-dashes (`—`) or en-dashes (`–`); use ` - ` or rephrase. Comments explain WHY only, never narrate WHAT or the edit itself. Never mention AI assistance anywhere.
- Commit messages: imperative subject, capitalized, no trailing period, no `feat:`/`fix:` prefixes. One task = one commit at the end of the task unless a step says otherwise.
- Commits are SSH-signed. Every commit command must run in the same Bash call as: `export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774`. If signing fails, rediscover the socket: probe `$XDG_RUNTIME_DIR/openssh_agent` and `/tmp/ssh-*/agent.*` for one whose `SSH_AUTH_SOCK=<sock> ssh-add -L` output contains the key in `git config user.signingkey`.
- All work happens on branch `master` in `/home/administrator/workspace/paperless-chandra`.
- Dev commands use the repo venv: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy` (created in Task 1).
- Every task ends with `.venv/bin/ruff check . && .venv/bin/ruff format --check .` passing on the files it touched (run `.venv/bin/ruff format <files>` to fix).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `README.md` (stub), `.gitignore`, `.editorconfig`, `.gitattributes`, `.yamllint`, `.markdownlint.yaml`
- Create: `paperless_chandra/__init__.py`, `paperless_chandra/engine/__init__.py`, `tests/__init__.py` (empty)
- Create: `.venv/` (not committed)

**Interfaces:**
- Produces: `paperless_chandra.__version__: str` (later tasks import it); entry point `chandra = paperless_chandra.parser:PaperlessChandraParser`; dev venv with pytest/ruff/mypy/build/pypdf.

- [ ] **Step 1: Copy meta files from the reference repo**

```bash
cd /home/administrator/workspace/paperless-chandra
REF=/home/administrator/workspace/paperless-paddleocr
cp "$REF/.editorconfig" "$REF/.gitattributes" "$REF/.yamllint" "$REF/.markdownlint.yaml" "$REF/.gitignore" "$REF/LICENSE" .
grep -rn "paddle" .editorconfig .gitattributes .yamllint .markdownlint.yaml .gitignore LICENSE || true
```

Remove any paddle-specific lines the grep finds (e.g. `.paddlex` cache entries in `.gitignore`); keep everything generic.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "paperless-chandra"
version = "0.1.0"
description = "Chandra OCR provider for paperless-ngx"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
license-files = ["LICENSE"]
authors = [
    {name = "Florian Bernd", email = "git@flobernd.de"},
]
keywords = ["paperless", "paperless-ngx", "chandra", "ocr", "ocrmypdf"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: End Users/Desktop",
    # NB: no "License ::" classifiers - PEP 639 / setuptools>=77 require the
    # SPDX `license` expression above to be the single source of truth and
    # reject the two from coexisting.
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Multimedia :: Graphics :: Capture :: Scanners",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Text Processing :: Linguistic",
]
dependencies = [
    "ocrmypdf>=17.4",
    "chandra-ocr>=0.2",
    "beautifulsoup4>=4.12",
    "numpy>=1.26",
]

[project.urls]
Homepage = "https://github.com/flobernd/paperless-chandra"
Issues = "https://github.com/flobernd/paperless-chandra/issues"

[project.entry-points."paperless_ngx.parsers"]
chandra = "paperless_chandra.parser:PaperlessChandraParser"

[tool.setuptools.packages.find]
include = ["paperless_chandra*"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = [
    "E501", # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.12"
check_untyped_defs = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
strict_equality = true
```

- [ ] **Step 3: Write the package skeleton and README stub**

`paperless_chandra/__init__.py`:

```python
"""Chandra OCR provider for paperless-ngx."""

__version__ = "0.1.0"
```

`paperless_chandra/engine/__init__.py`:

```python
"""Chandra OCR engine for ocrmypdf: client, page model, hOCR serialisation."""
```

`tests/__init__.py`: empty file.

`README.md` (stub; the full README is Task 14 - the stub exists because `pyproject.toml` references it and docker builds copy it):

```markdown
# paperless-chandra

A [Chandra OCR](https://github.com/datalab-to/chandra) provider for
[paperless-ngx](https://github.com/paperless-ngx/paperless-ngx), delivered as a
parser plugin that replaces the built-in Tesseract OCR pipeline. Requires
paperless-ngx 3.x (beta) and a self-hosted OpenAI-compatible inference server
running Chandra.

Documentation is being written; see
`docs/superpowers/specs/2026-07-12-paperless-chandra-design.md` for the design.
```

- [ ] **Step 4: Create the venv and verify the package builds**

```bash
cd /home/administrator/workspace/paperless-chandra
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e . pytest ruff mypy build pypdf
.venv/bin/python -c "import paperless_chandra; print(paperless_chandra.__version__)"
.venv/bin/python -c "import importlib.metadata as m; eps = m.entry_points(group='paperless_ngx.parsers'); print([e.name for e in eps])"
```

Expected: `0.1.0` and a list containing `'chandra'`. Note the entry point resolves to `paperless_chandra.parser`, which does not exist yet - only discovery is checked here, not resolution.

Confirm `.gitignore` covers `.venv/` (add a `.venv/` line if the copied file lacks it).

- [ ] **Step 5: Verify ruff passes and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add pyproject.toml LICENSE README.md .gitignore .editorconfig .gitattributes .yamllint .markdownlint.yaml paperless_chandra tests
git commit -m "Add project scaffolding and package skeleton"
```

---

### Task 2: Geometry and hOCR modules (adapted copies)

**Files:**
- Create: `paperless_chandra/engine/geometry.py`
- Create: `paperless_chandra/engine/hocr.py` (copy of `/home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/paddle_engine/hocr.py` with edits)
- Test: `tests/test_geometry.py`, `tests/test_hocr.py`

**Interfaces:**
- Produces: `geometry.BBox = tuple[int, int, int, int]`; `geometry.estimate_word_boxes(words: Sequence[str], box: BBox) -> list[BBox]`; `hocr.Word(text: str, box: BBox, confidence: int)`; `hocr.Line(box, confidence, text, words=[], baseline=(0.0, 0.0))`; `hocr.Block(box, lines=[])`; `hocr.Page(width, height, lang, ocr_system, blocks=[])`; `hocr.render_hocr(page) -> str`; `hocr.write_document(page, hocr_path, text_path, *, sidecar=None) -> None`; `hocr.to_hocr_lang(code: str | None) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_geometry.py`:

```python
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
```

`tests/test_hocr.py`:

```python
"""hOCR serialisation and the BCP-47 lang sanitiser."""

from __future__ import annotations

from paperless_chandra.engine.hocr import (
    Block,
    Line,
    Page,
    Word,
    render_hocr,
    to_hocr_lang,
    write_document,
)


def _page(line: Line) -> Page:
    return Page(
        width=400,
        height=300,
        lang="eng",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )


def _line() -> Line:
    return Line(
        box=(10, 10, 90, 40),
        confidence=88,
        text="hi",
        words=[Word("hi", (10, 10, 90, 40), 88)],
    )


def test_to_hocr_lang_passes_iso_codes_through():
    assert to_hocr_lang("eng") == "eng"
    assert to_hocr_lang(" DE ") == "de"


def test_to_hocr_lang_rejects_unexpected_values():
    assert to_hocr_lang("") == "und"
    assert to_hocr_lang(None) == "und"
    assert to_hocr_lang("german") == "und"
    assert to_hocr_lang('en" onload="x') == "und"


def test_render_hocr_emits_page_word_and_conf():
    html = render_hocr(_page(_line()))
    assert 'class="ocr_page"' in html
    assert "bbox 0 0 400 300" in html
    assert 'class="ocrx_word"' in html
    assert "x_wconf 88" in html
    assert 'lang="eng"' in html


def test_render_hocr_escapes_text():
    line = Line(
        box=(0, 0, 50, 10),
        confidence=90,
        text="<b>&",
        words=[Word("<b>&", (0, 0, 50, 10), 90)],
    )
    assert "&lt;b&gt;&amp;" in render_hocr(_page(line))


def test_write_document_sidecar_override(tmp_path):
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(_line()), hocr_path, text_path, sidecar="custom order")
    assert text_path.read_text(encoding="utf-8") == "custom order"
    assert "ocrx_word" in hocr_path.read_text(encoding="utf-8")


def test_write_document_default_sidecar_is_line_text(tmp_path):
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(_line()), hocr_path, text_path)
    assert text_path.read_text(encoding="utf-8") == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_geometry.py tests/test_hocr.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperless_chandra.engine.geometry'`

- [ ] **Step 3: Write `paperless_chandra/engine/geometry.py`**

This is the paddle module trimmed to what Chandra needs (Chandra has no detection polygons, so `poly_to_bbox` is dropped):

```python
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
```

- [ ] **Step 4: Copy and adapt `hocr.py`**

```bash
cp /home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/paddle_engine/hocr.py \
   /home/administrator/workspace/paperless-chandra/paperless_chandra/engine/hocr.py
```

Then apply exactly these edits:

1. Replace the module docstring's last sentence block. Old:

```python
Both engine paths (classic and VL) populate a :class:`Page` and hand it
here; this module is the single place that knows the hOCR wire format. The
output is plain hOCR/XHTML so ocrmypdf's ``HocrParser`` can render the
invisible text layer and the multi-language merge in
:mod:`paperless_paddleocr.ocrmypdf_plugin` can parse the ``ocrx_word`` spans
back out (it keys off ``bbox`` and ``x_wconf`` in each span ``title``).
```

New:

```python
The engine populates a :class:`Page` and hands it here; this module is the
single place that knows the hOCR wire format. The output is plain
hOCR/XHTML so ocrmypdf's ``HocrParser`` can render the invisible text
layer.
```

2. Replace the imports. Old:

```python
from paperless_paddleocr.languages import to_hocr_lang
from paperless_paddleocr.paddle_engine.geometry import BBox
```

New:

```python
import re

from paperless_chandra.engine.geometry import BBox
```

(keep the existing `dataclass`/`pathlib`/`saxutils` imports; `import re` goes in the stdlib group.)

3. Add the sanitiser right after the imports (the paddle repo maps engine codes through a table; paperless hands this plugin Tesseract codes, and 2-3 letter ISO 639 codes are already valid BCP-47 primary subtags):

```python
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
```

4. In the `Page` dataclass, change the `lang` field comment from `# PaddleOCR code; normalised to BCP-47 on serialise` to `# Tesseract code from paperless; normalised to BCP-47 on serialise`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_geometry.py tests/test_hocr.py -q`
Expected: PASS (all tests)

- [ ] **Step 6: Lint and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/engine/geometry.py paperless_chandra/engine/hocr.py tests/test_geometry.py tests/test_hocr.py
git commit -m "Add geometry and hOCR serialisation modules"
```

---

### Task 3: Deskew estimator and text-only PDF renderer (copies)

**Files:**
- Create: `paperless_chandra/engine/deskew.py` (copy of `/home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/paddle_engine/deskew.py`)
- Create: `paperless_chandra/engine/pdf.py` (copy of `/home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/paddle_engine/pdf.py`)
- Test: `tests/test_deskew.py`, `tests/test_pdf_render.py`

**Interfaces:**
- Consumes: `hocr.render_hocr`, `hocr.Page/Block/Line/Word` (Task 2).
- Produces: `deskew.estimate_skew(input_file: Path) -> float` (degrees counterclockwise); `pdf.render_textonly(input_file: Path, hocr_file: Path, output_pdf: Path) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_deskew.py`:

```python
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
```

`tests/test_pdf_render.py`:

```python
"""render_textonly turns hOCR into a one-page invisible-text PDF."""

from __future__ import annotations

import pikepdf
from PIL import Image

from paperless_chandra.engine.hocr import Block, Line, Page, Word, render_hocr
from paperless_chandra.engine.pdf import render_textonly


def test_render_textonly_produces_a_pdf(tmp_path):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(img_path, dpi=(72, 72))

    line = Line(
        box=(10, 10, 200, 40),
        confidence=90,
        text="hello world",
        words=[
            Word("hello", (10, 10, 100, 40), 90),
            Word("world", (110, 10, 200, 40), 90),
        ],
    )
    page = Page(
        width=400,
        height=300,
        lang="eng",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )
    hocr_path = tmp_path / "page.hocr"
    hocr_path.write_text(render_hocr(page), encoding="utf-8")

    out_pdf = tmp_path / "text.pdf"
    render_textonly(img_path, hocr_path, out_pdf)

    with pikepdf.open(out_pdf) as pdf_doc:
        assert len(pdf_doc.pages) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_deskew.py tests/test_pdf_render.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperless_chandra.engine.deskew'`

- [ ] **Step 3: Copy both modules and adapt docstrings**

```bash
cd /home/administrator/workspace/paperless-chandra
REF=/home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/paddle_engine
cp "$REF/deskew.py" paperless_chandra/engine/deskew.py
cp "$REF/pdf.py" paperless_chandra/engine/pdf.py
grep -rn "paddle" paperless_chandra/engine/deskew.py paperless_chandra/engine/pdf.py || true
```

In `deskew.py`, change the docstring sentence `Pure PIL + numpy so the estimator is exact to test in CI without paddle installed.` to `Pure PIL + numpy so the estimator is exact to test in CI.` No other changes (neither module imports anything package-internal). If the grep shows any other paddle mention, rephrase it the same way.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_deskew.py tests/test_pdf_render.py -q`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/engine/deskew.py paperless_chandra/engine/pdf.py tests/test_deskew.py tests/test_pdf_render.py
git commit -m "Add deskew estimator and text-only PDF renderer"
```

---

### Task 4: Layout blocks to typed page (blocks.py)

**Files:**
- Create: `paperless_chandra/engine/blocks.py`
- Test: `tests/test_blocks.py`

**Interfaces:**
- Consumes: `geometry.estimate_word_boxes`, `hocr.Page/Block/Line/Word` (Task 2); `chandra.output.parse_markdown` (upstream).
- Produces: `blocks.OCR_SYSTEM = "Chandra via paperless-chandra"`; `blocks.CONFIDENCE = 95`; `blocks.NON_TEXT_LABELS: frozenset[str]`; `blocks.block_lines(content_html: str) -> list[str]`; `blocks.page_from_chunks(chunks: list[dict], width: int, height: int, lang: str) -> Page`; `blocks.page_sidecar(page: Page) -> str`; `blocks.markdown_sidecar(raw_html: str) -> str`. Chunk dicts have the upstream `parse_chunks` shape: `{"bbox": [x0, y0, x1, y1] (pixels), "label": str, "content": str (inner HTML)}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_blocks.py`:

```python
"""Chandra layout chunks: HTML line splitting and page synthesis."""

from __future__ import annotations

from paperless_chandra.engine.blocks import (
    block_lines,
    markdown_sidecar,
    page_from_chunks,
    page_sidecar,
)


# ---------------------------------------------------------------- block_lines


def test_paragraphs_become_separate_lines():
    assert block_lines("<p>first</p><p>second</p>") == ["first", "second"]


def test_br_splits_lines_inside_a_paragraph():
    assert block_lines("<p>first<br>second</p>") == ["first", "second"]


def test_list_items_become_lines():
    assert block_lines("<ul><li>alpha</li><li>beta</li></ul>") == ["alpha", "beta"]


def test_table_rows_one_line_cells_space_joined():
    html = "<table><tr><td>a1</td><td>a2</td></tr><tr><td>b1</td><td>b2</td></tr></table>"
    assert block_lines(html) == ["a1 a2", "b1 b2"]


def test_math_keeps_latex_source():
    assert block_lines("<p>area <math>x^2 + y_1</math> end</p>") == ["area x^2 + y_1 end"]


def test_checkboxes_render_as_brackets():
    html = '<p><input type="checkbox" checked/> yes <input type="checkbox"/> no</p>'
    assert block_lines(html) == ["[x] yes [ ] no"]


def test_plain_text_without_tags_is_one_line():
    assert block_lines("just words") == ["just words"]


def test_whitespace_is_collapsed_and_empties_dropped():
    assert block_lines("<p>  a   b  </p><p>   </p>") == ["a b"]


# ---------------------------------------------------------- page_from_chunks


def _chunk(label="Text", bbox=(100, 100, 500, 200), content="<p>hello world</p>"):
    return {"label": label, "bbox": list(bbox), "content": content}


def test_page_synthesises_lines_and_words():
    page = page_from_chunks([_chunk()], 1000, 1000, "eng")
    assert len(page.blocks) == 1
    line = page.blocks[0].lines[0]
    assert line.text == "hello world"
    assert [w.text for w in line.words] == ["hello", "world"]
    assert line.box == (100, 100, 500, 200)
    assert all(w.confidence == 95 for w in line.words)


def test_multi_line_block_splits_box_vertically():
    page = page_from_chunks(
        [_chunk(content="<p>one</p><p>two</p>", bbox=(0, 0, 400, 100))], 1000, 1000, "eng"
    )
    first, second = page.blocks[0].lines
    assert first.box == (0, 0, 400, 50)
    assert second.box == (0, 50, 400, 100)


def test_non_text_labels_are_skipped():
    chunks = [
        _chunk(label="Image", content='<img alt="a cat"/>'),
        _chunk(label="Figure", content="<p>figure narration</p>"),
        _chunk(label="Blank-Page", content=""),
        _chunk(label="Text", content="<p>kept</p>"),
    ]
    page = page_from_chunks(chunks, 1000, 1000, "eng")
    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "kept"


def test_unknown_labels_are_kept():
    page = page_from_chunks([_chunk(label="block")], 1000, 1000, "eng")
    assert len(page.blocks) == 1


def test_degenerate_bbox_widens_to_full_page():
    page = page_from_chunks([_chunk(bbox=(0, 0, 1, 1))], 800, 600, "eng")
    assert page.blocks[0].box == (0, 0, 800, 600)


def test_bbox_is_clamped_to_page():
    page = page_from_chunks([_chunk(bbox=(500, 500, 2000, 2000))], 800, 600, "eng")
    assert page.blocks[0].box == (500, 500, 800, 600)


def test_empty_content_produces_no_block():
    page = page_from_chunks([_chunk(content="  ")], 1000, 1000, "eng")
    assert page.blocks == []


# ----------------------------------------------------------------- sidecars


def test_page_sidecar_joins_blocks_with_blank_lines():
    chunks = [
        _chunk(content="<p>one</p><p>two</p>", bbox=(0, 0, 400, 100)),
        _chunk(content="<p>three</p>", bbox=(0, 200, 400, 300)),
    ]
    page = page_from_chunks(chunks, 1000, 1000, "eng")
    assert page_sidecar(page) == "one\ntwo\n\nthree"


def test_markdown_sidecar_converts_headings_and_keeps_footers():
    raw = (
        '<div data-label="Section-Header" data-bbox="0 0 500 50"><h1>Title</h1></div>'
        '<div data-label="Page-Footer" data-bbox="0 900 500 950"><p>page 1 of 2</p></div>'
    )
    md = markdown_sidecar(raw)
    assert "# Title" in md
    assert "page 1 of 2" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_blocks.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperless_chandra.engine.blocks'`

- [ ] **Step 3: Write `paperless_chandra/engine/blocks.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_blocks.py -q`
Expected: PASS. If `test_table_rows_one_line_cells_space_joined` fails on cell joining, the `cell.append(" ")` call needs to move the space outside the cell: use `cell.insert_after(" ")` instead and re-run.

- [ ] **Step 5: Lint and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/engine/blocks.py tests/test_blocks.py
git commit -m "Add layout chunk to typed page conversion"
```

---

### Task 5: Remote client wrapper (client.py)

**Files:**
- Create: `paperless_chandra/engine/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `chandra.model.schema.BatchInputItem`, `chandra.model.vllm.generate_vllm`, `chandra.settings.settings` (upstream).
- Produces: `client.ChandraClientError(RuntimeError)`; `client.normalize_server_url(raw: str) -> str`; `client.ocr_image(image: PIL.Image.Image, options: Any) -> str` (raw HTML); `client.DEFAULT_MODEL_NAME = "chandra"`; `client.DEFAULT_MAX_OUTPUT_TOKENS = 12384`. Options attributes read: `chandra_server_url`, `chandra_model_name`, `chandra_api_key`, `chandra_max_output_tokens`.

- [ ] **Step 1: Write the failing tests**

`tests/test_client.py`:

```python
"""Client wrapper: URL normalisation, settings application, error mapping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from paperless_chandra.engine import client
from paperless_chandra.engine.client import (
    ChandraClientError,
    normalize_server_url,
    ocr_image,
)


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _image():
    return Image.new("RGB", (100, 100), "white")


class _Result:
    def __init__(self, raw="<div></div>", error=False):
        self.raw = raw
        self.error = error
        self.token_count = 7


def test_normalize_appends_v1():
    assert normalize_server_url("http://host:8000") == "http://host:8000/v1"


def test_normalize_keeps_existing_version_suffix():
    assert normalize_server_url("http://host:8000/v1/") == "http://host:8000/v1"


def test_normalize_rejects_empty():
    with pytest.raises(ChandraClientError):
        normalize_server_url("   ")


def test_ocr_image_sends_layout_prompt_and_returns_raw(monkeypatch):
    captured = {}

    def fake_generate(batch, max_output_tokens=None, vllm_api_base=None, **kwargs):
        captured["batch"] = batch
        captured["max_output_tokens"] = max_output_tokens
        captured["vllm_api_base"] = vllm_api_base
        return [_Result(raw="<div>ok</div>")]

    monkeypatch.setattr(client, "generate_vllm", fake_generate)
    raw = ocr_image(_image(), _options())
    assert raw == "<div>ok</div>"
    assert captured["batch"][0].prompt_type == "ocr_layout"
    assert captured["max_output_tokens"] == 12384
    assert captured["vllm_api_base"] == "http://ocr-host:8000/v1"


def test_ocr_image_applies_model_and_key_to_chandra_settings(monkeypatch):
    from chandra.settings import settings as chandra_settings

    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result()])
    ocr_image(_image(), _options(chandra_model_name="custom", chandra_api_key="sekrit"))
    assert chandra_settings.VLLM_MODEL_NAME == "custom"
    assert chandra_settings.VLLM_API_KEY == "sekrit"


def test_ocr_image_defaults_api_key_to_vllm_convention(monkeypatch):
    from chandra.settings import settings as chandra_settings

    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result()])
    ocr_image(_image(), _options(chandra_api_key=""))
    assert chandra_settings.VLLM_API_KEY == "EMPTY"


def test_ocr_image_raises_on_error_flag(monkeypatch):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(error=True)])
    with pytest.raises(ChandraClientError):
        ocr_image(_image(), _options())


def test_ocr_image_raises_on_empty_result_list(monkeypatch):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [])
    with pytest.raises(ChandraClientError):
        ocr_image(_image(), _options())


def test_ocr_image_returns_empty_string_for_empty_raw(monkeypatch):
    monkeypatch.setattr(client, "generate_vllm", lambda *a, **k: [_Result(raw="")])
    assert ocr_image(_image(), _options()) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_client.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperless_chandra.engine.client'`

- [ ] **Step 3: Write `paperless_chandra/engine/client.py`**

```python
"""Thin wrapper around the chandra-ocr remote client.

The only module that touches the upstream package's configuration.
``chandra.settings.settings`` is a module-global pydantic object; the
model name and API key can only be passed through it (``generate_vllm``
accepts the base URL as a kwarg but reads the rest from settings at call
time). The values derive from env vars and are constant per worker
process, so re-applying them before every call is idempotent; the lock
only guards the theoretical mixed-write window between threads.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from chandra.model.schema import BatchInputItem
from chandra.model.vllm import generate_vllm
from chandra.settings import settings as chandra_settings
from PIL import Image

log = logging.getLogger("paperless.chandra.client")

DEFAULT_MODEL_NAME = "chandra"
DEFAULT_MAX_OUTPUT_TOKENS = 12384

_SETTINGS_LOCK = threading.Lock()


class ChandraClientError(RuntimeError):
    """Chandra inference failed or is misconfigured."""


def normalize_server_url(raw: str) -> str:
    """Normalised OpenAI-compatible base URL.

    chandra hands the URL to the OpenAI SDK, which appends
    ``/chat/completions``; servers route under ``/v1``. Accept the URL
    with or without the version suffix.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ChandraClientError(
            "Chandra requires PAPERLESS_CHANDRA_SERVER_URL (or "
            "--chandra-server-url) to point at an OpenAI-compatible "
            "inference server."
        )
    url = cleaned.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url = f"{url}/v1"
    return url


def ocr_image(image: Image.Image, options: Any) -> str:
    """OCR one page image via the remote server; return Chandra's raw HTML."""
    url = normalize_server_url(getattr(options, "chandra_server_url", "") or "")
    model = (getattr(options, "chandra_model_name", "") or "").strip() or DEFAULT_MODEL_NAME
    api_key = (getattr(options, "chandra_api_key", "") or "").strip() or "EMPTY"
    max_tokens = (
        int(getattr(options, "chandra_max_output_tokens", 0) or 0) or DEFAULT_MAX_OUTPUT_TOKENS
    )

    with _SETTINGS_LOCK:
        chandra_settings.VLLM_MODEL_NAME = model
        chandra_settings.VLLM_API_KEY = api_key

    results = generate_vllm(
        [BatchInputItem(image=image, prompt_type="ocr_layout")],
        max_output_tokens=max_tokens,
        vllm_api_base=url,
    )
    if not results:
        raise ChandraClientError("Chandra returned no result for the page.")
    result = results[0]
    if result.error:
        raise ChandraClientError(
            f"Chandra inference failed after retries against {url}. "
            "Check the inference server logs and the PAPERLESS_CHANDRA_* settings."
        )
    log.debug("Chandra returned %d output tokens for one page.", result.token_count)
    return result.raw or ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_client.py -q`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/engine/client.py tests/test_client.py
git commit -m "Add remote Chandra client wrapper"
```

---

### Task 6: The ocrmypdf engine (engine.py)

**Files:**
- Create: `paperless_chandra/engine/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `client.ocr_image` (Task 5), `blocks.*` (Task 4), `deskew.estimate_skew`, `pdf.render_textonly` (Task 3), `hocr.write_document` (Task 2), `chandra.output.parse_chunks` (upstream).
- Produces: `engine.ChandraEngine(ocrmypdf.pluginspec.OcrEngine)` with staticmethods `version()`, `creator_tag(options)`, `languages(options)`, `get_orientation(input_file, options)`, `get_deskew(input_file, options)`, `generate_hocr(input_file, output_hocr, output_text, options)` and classmethod `generate_pdf(input_file, output_pdf, output_text, options)`. Options attribute read (besides the client's): `chandra_content_format` (`"text"` or `"markdown"`).

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
"""ChandraEngine: hOCR generation, sidecar formats, hook behaviour."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from paperless_chandra.engine import client as client_module
from paperless_chandra.engine.engine import ChandraEngine

#: bbox values are normalised 0-1000; with a 1000x1000 page they map 1:1.
FIXTURE_HTML = (
    '<div data-label="Section-Header" data-bbox="100 50 900 120"><h1>Invoice 42</h1></div>'
    '<div data-label="Text" data-bbox="100 150 900 400">'
    "<p>Hello world<br>Second line</p></div>"
    '<div data-label="Image" data-bbox="0 500 500 900"><img alt="a cat"/></div>'
)


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
        "chandra_content_format": "text",
        "languages": ["eng"],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _page_png(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(path, dpi=(72, 72))
    return path


def _run_generate_hocr(tmp_path, monkeypatch, options, raw=FIXTURE_HTML):
    monkeypatch.setattr(client_module, "ocr_image", lambda image, opts: raw)
    hocr = tmp_path / "out.hocr"
    text = tmp_path / "out.txt"
    ChandraEngine.generate_hocr(_page_png(tmp_path), hocr, text, options)
    return hocr, text


def test_generate_hocr_writes_words_and_sidecar(tmp_path, monkeypatch):
    hocr, text = _run_generate_hocr(tmp_path, monkeypatch, _options())
    hocr_html = hocr.read_text(encoding="utf-8")
    assert "Invoice" in hocr_html
    assert 'class="ocrx_word"' in hocr_html
    sidecar = text.read_text(encoding="utf-8")
    assert "Invoice 42" in sidecar
    assert "Hello world\nSecond line" in sidecar
    assert "cat" not in sidecar


def test_generate_hocr_markdown_sidecar(tmp_path, monkeypatch):
    _, text = _run_generate_hocr(
        tmp_path, monkeypatch, _options(chandra_content_format="markdown")
    )
    assert "# Invoice 42" in text.read_text(encoding="utf-8")


def test_generate_hocr_empty_response_writes_empty_sidecar(tmp_path, monkeypatch):
    hocr, text = _run_generate_hocr(tmp_path, monkeypatch, _options(), raw="")
    assert text.read_text(encoding="utf-8") == ""
    assert 'class="ocr_page"' in hocr.read_text(encoding="utf-8")


def test_generate_pdf_dispatches_through_cls(tmp_path, monkeypatch):
    calls = []

    class Recorder(ChandraEngine):
        @staticmethod
        def generate_hocr(input_file, output_hocr, output_text, options):
            calls.append("hocr")
            ChandraEngine.generate_hocr(input_file, output_hocr, output_text, options)

    monkeypatch.setattr(client_module, "ocr_image", lambda image, opts: FIXTURE_HTML)
    out_pdf = tmp_path / "text.pdf"
    Recorder.generate_pdf(_page_png(tmp_path), out_pdf, tmp_path / "t.txt", _options())
    assert calls == ["hocr"]
    assert out_pdf.stat().st_size > 0


def test_languages_reports_requested_codes():
    assert ChandraEngine.languages(_options(languages=["eng", "deu"])) >= {"eng", "deu"}


def test_get_orientation_is_a_noop(tmp_path):
    oc = ChandraEngine.get_orientation(_page_png(tmp_path), _options())
    assert oc.angle == 0
    assert oc.confidence == 0.0


def test_creator_tag_mentions_chandra():
    assert "Chandra" in ChandraEngine.creator_tag(_options())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_engine.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperless_chandra.engine.engine'`

- [ ] **Step 3: Write `paperless_chandra/engine/engine.py`**

```python
"""The ocrmypdf ``OcrEngine`` backed by a self-hosted Chandra server."""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Any

from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from PIL import Image

from paperless_chandra.engine import blocks, client, deskew, pdf
from paperless_chandra.engine.hocr import write_document

log = logging.getLogger("paperless.chandra.engine")

_rotate_warned = False


def _chandra_version() -> str:
    try:
        return importlib.metadata.version("chandra-ocr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class ChandraEngine(OcrEngine):
    """Chandra implementation of ocrmypdf's ``OcrEngine``."""

    @staticmethod
    def version() -> str:
        return _chandra_version()

    @staticmethod
    def creator_tag(options: Any) -> str:
        return f"Chandra {ChandraEngine.version()}"

    def __str__(self) -> str:
        return f"Chandra {ChandraEngine.version()}"

    @staticmethod
    def languages(options: Any) -> set[str]:
        # Chandra is language-agnostic (90+ languages in one pass); report
        # whatever was requested so ocrmypdf preflight never aborts.
        return set(getattr(options, "languages", None) or []) | {"eng"}

    @staticmethod
    def get_orientation(input_file: Path, options: Any) -> OrientationConfidence:
        global _rotate_warned
        if not _rotate_warned:
            log.warning(
                "PAPERLESS_OCR_ROTATE_PAGES is not supported by paperless-chandra; "
                "pages are OCR'd in their stored orientation."
            )
            _rotate_warned = True
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options: Any) -> float:
        try:
            return deskew.estimate_skew(input_file)
        except Exception:
            log.exception("Deskew estimation failed for %s; skipping deskew.", input_file)
            return 0.0

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        from chandra.output import parse_chunks

        with Image.open(input_file) as img:
            image = img.convert("RGB")

        raw = client.ocr_image(image, options)
        chunks = parse_chunks(raw, image) if raw else []

        langs = list(getattr(options, "languages", None) or [])
        page = blocks.page_from_chunks(
            chunks,
            image.width,
            image.height,
            langs[0] if langs else "eng",
        )
        content_format = (getattr(options, "chandra_content_format", "") or "text").strip()
        if content_format == "markdown" and raw:
            sidecar = blocks.markdown_sidecar(raw)
        else:
            sidecar = blocks.page_sidecar(page)
        write_document(page, output_hocr, output_text, sidecar=sidecar)

    @classmethod
    def generate_pdf(
        cls,
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        """Render OCR output as a text-only PDF.

        ``cls.generate_hocr`` (not the base method) so a subclass still
        runs if a user forces the ``generate_pdf`` renderer path.
        """
        output_hocr = output_pdf.with_suffix(".hocr")
        cls.generate_hocr(input_file, output_hocr, output_text, options)
        pdf.render_textonly(input_file, output_hocr, output_pdf)
```

Note: `generate_hocr` calls `client.ocr_image` through the module attribute (`from paperless_chandra.engine import client`), which is what lets the tests monkeypatch `client_module.ocr_image`. Keep it that way.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_engine.py -q`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/engine/engine.py tests/test_engine.py
git commit -m "Add Chandra ocrmypdf engine"
```

---

### Task 7: ocrmypdf plugin hookimpls (ocrmypdf_plugin.py)

**Files:**
- Create: `paperless_chandra/ocrmypdf_plugin.py`
- Test: `tests/test_plugin_options.py`

**Interfaces:**
- Consumes: `ChandraEngine` (Task 6), `client.normalize_server_url`, `client.DEFAULT_MODEL_NAME`, `client.DEFAULT_MAX_OUTPUT_TOKENS` (Task 5).
- Produces: hookimpls `add_options(parser)`, `check_options(options)`, `get_ocr_engine()`. Registered ocrmypdf kwargs: `chandra_server_url` (str), `chandra_model_name` (str), `chandra_api_key` (str), `chandra_max_output_tokens` (int), `chandra_content_format` (`text|markdown`). Task 8's parser passes exactly these.

- [ ] **Step 1: Write the failing tests**

`tests/test_plugin_options.py`:

```python
"""check_options preflight and engine binding."""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace

import pytest
from ocrmypdf.exceptions import MissingDependencyError

from paperless_chandra import ocrmypdf_plugin
from paperless_chandra.engine.engine import ChandraEngine


def _options(**overrides):
    defaults = {
        "chandra_server_url": "http://ocr-host:8000",
        "chandra_model_name": "chandra",
        "chandra_api_key": "",
        "chandra_max_output_tokens": 12384,
        "chandra_content_format": "text",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(ocrmypdf_plugin, "_probe_server", lambda url, key: None)
    ocrmypdf_plugin._PROBED_SERVERS.clear()


def test_get_ocr_engine_returns_chandra_engine():
    assert isinstance(ocrmypdf_plugin.get_ocr_engine(), ChandraEngine)


def test_check_options_accepts_valid_configuration():
    ocrmypdf_plugin.check_options(_options())


def test_check_options_requires_server_url():
    with pytest.raises(MissingDependencyError):
        ocrmypdf_plugin.check_options(_options(chandra_server_url="  "))


def test_check_options_rejects_unknown_content_format():
    with pytest.raises(ValueError):
        ocrmypdf_plugin.check_options(_options(chandra_content_format="html"))


def test_probe_maps_auth_failure(monkeypatch):
    def raise_401(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", raise_401)
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    with pytest.raises(MissingDependencyError, match="rejected the API key"):
        ocrmypdf_plugin._probe_server.__wrapped__("http://host:1", "bad")  # type: ignore[attr-defined]


def test_probe_maps_unreachable_server(monkeypatch):
    def raise_conn(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ocrmypdf_plugin, "urlopen", raise_conn)
    ocrmypdf_plugin._PROBED_SERVERS.clear()
    with pytest.raises(MissingDependencyError, match="not reachable"):
        ocrmypdf_plugin._probe_server.__wrapped__("http://host:1", "")  # type: ignore[attr-defined]
```

Note on `__wrapped__`: the autouse fixture replaces `_probe_server` with a lambda; the two probe tests need the real function. Implement the module so the original is reachable: assign `_probe_server.__wrapped__ = _probe_server` is NOT valid Python on a plain function object created later - instead, the fixture must save and expose the original. Simplify: change the autouse fixture to store the original on the monkeypatch target before replacing:

```python
@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    original = ocrmypdf_plugin._probe_server
    stub = lambda url, key: None  # noqa: E731
    stub.__wrapped__ = original  # type: ignore[attr-defined]
    monkeypatch.setattr(ocrmypdf_plugin, "_probe_server", stub)
    ocrmypdf_plugin._PROBED_SERVERS.clear()
```

With that fixture, `ocrmypdf_plugin._probe_server.__wrapped__` in the two probe tests resolves to the real implementation. Use this fixture version in the test file (replace the first `_no_network` shown above).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_plugin_options.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperless_chandra.ocrmypdf_plugin'`

- [ ] **Step 3: Write `paperless_chandra/ocrmypdf_plugin.py`**

```python
"""ocrmypdf plugin: registers the Chandra engine and its CLI options.

This module is the one loaded by ``ocrmypdf.ocr(plugins=[...])`` when the
paperless parser invokes ocrmypdf. The engine lives inside this package
rather than being published as a standalone ocrmypdf plugin, so ocrmypdf's
entry-point auto-discovery never registers it: calls from paperless's
built-in Tesseract parser are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.request import Request, urlopen

import ocrmypdf

from paperless_chandra.engine.client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    normalize_server_url,
)
from paperless_chandra.engine.engine import ChandraEngine

log = logging.getLogger("paperless.chandra.plugin")

_CONTENT_FORMATS = ("text", "markdown")

#: Successful probes per (url, api_key); check_options runs once per
#: document, and one probe per worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_server(server_url: str, api_key: str) -> None:
    """Fail fast when the Chandra server is unreachable or rejects the key.

    Without this, a wrong URL or key surfaces only at first inference,
    deep inside a celery task.
    """
    from urllib.error import HTTPError, URLError

    from ocrmypdf.exceptions import MissingDependencyError

    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=5):  # noqa: S310
            pass
    except HTTPError as e:
        if e.code in (401, 403):
            raise MissingDependencyError(
                f"The Chandra server at {base} rejected the API key (HTTP {e.code}). "
                "Check PAPERLESS_CHANDRA_API_KEY against the server configuration."
            ) from e
        log.warning(
            "Chandra server preflight got HTTP %d from %s/models; continuing.",
            e.code,
            base,
        )
    except URLError as e:
        raise MissingDependencyError(
            f"The Chandra server at {base} is not reachable ({e.reason}). "
            "Check PAPERLESS_CHANDRA_SERVER_URL and that the inference "
            "server container is running."
        ) from e
    _PROBED_SERVERS.add(cache_key)


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Register ``--chandra-*`` CLI args so they're accepted as kwargs.

    ocrmypdf validates kwargs against its argparse parser; these must be
    registered before ``ocrmypdf.ocr(chandra_server_url=...)`` is called
    from the paperless parser.
    """
    chandra = parser.add_argument_group("Chandra", "Options for the Chandra OCR engine")
    chandra.add_argument(
        "--chandra-server-url",
        default="",
        dest="chandra_server_url",
        metavar="URL",
        help=(
            "URL of the OpenAI-compatible inference server hosting Chandra "
            "(e.g. http://gpu-box:8000). The /v1 suffix is appended when missing."
        ),
    )
    chandra.add_argument(
        "--chandra-model-name",
        default=DEFAULT_MODEL_NAME,
        dest="chandra_model_name",
        metavar="NAME",
        help=f"Served model name advertised by the server (default: {DEFAULT_MODEL_NAME}).",
    )
    chandra.add_argument(
        "--chandra-api-key",
        default="",
        dest="chandra_api_key",
        metavar="KEY",
        help="Bearer token for the server. Leave blank if the server needs no auth.",
    )
    chandra.add_argument(
        "--chandra-max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        dest="chandra_max_output_tokens",
        metavar="N",
        help=f"Per-page output token budget (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    chandra.add_argument(
        "--chandra-content-format",
        choices=list(_CONTENT_FORMATS),
        default="text",
        dest="chandra_content_format",
        help="Document content stored by paperless: plain text (default) or markdown.",
    )


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate configuration and probe the inference server."""
    from ocrmypdf.exceptions import MissingDependencyError

    fmt = (getattr(options, "chandra_content_format", "") or "text").strip()
    if fmt not in _CONTENT_FORMATS:
        # argparse validates the CLI path via `choices=`; the Python API
        # (ocrmypdf.ocr(chandra_content_format=...)) skips that check.
        raise ValueError(
            f"Unknown chandra_content_format={fmt!r}. Valid choices: {list(_CONTENT_FORMATS)}."
        )

    try:
        import chandra.model.vllm  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "chandra-ocr is not installed. Install it with: pip install chandra-ocr"
        ) from e

    server_url = (getattr(options, "chandra_server_url", "") or "").strip()
    if not server_url:
        raise MissingDependencyError(
            "paperless-chandra requires --chandra-server-url (or the "
            "PAPERLESS_CHANDRA_SERVER_URL env var) to point at an "
            "OpenAI-compatible server hosting Chandra."
        )
    _probe_server(server_url, (getattr(options, "chandra_api_key", "") or "").strip())


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine() -> Any:
    """Return the Chandra engine.

    ``tryfirst=True`` makes this hookimpl run before the built-in
    Tesseract plugin's ``get_ocr_engine``; ocrmypdf's ``firstresult=True``
    policy then short-circuits, so Tesseract never claims the engine.
    """
    return ChandraEngine()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_plugin_options.py -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite, lint, and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/pytest tests/ -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/ocrmypdf_plugin.py tests/test_plugin_options.py
git commit -m "Add ocrmypdf plugin hookimpls with server preflight"
```

---

### Task 8: The paperless parser (parser.py)

**Files:**
- Create: `paperless_chandra/parser.py` (copy of `/home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/parser.py` with the edits below)

**Interfaces:**
- Consumes: entry point target declared in Task 1; ocrmypdf kwargs registered in Task 7.
- Produces: `PaperlessChandraParser` with the paperless parser protocol (`supported_mime_types`, `score`, `parse`, `get_text`, `get_archive_path`, ...). Imports django/paperless modules, so it is only importable inside a paperless-ngx runtime (verified by Task 9's smoke test; here only `py_compile` runs).

- [ ] **Step 1: Copy the reference parser**

```bash
cp /home/administrator/workspace/paperless-paddleocr/paperless_paddleocr/parser.py \
   /home/administrator/workspace/paperless-chandra/paperless_chandra/parser.py
```

- [ ] **Step 2: Apply the edits**

Work through the file top to bottom. Every occurrence not listed here stays byte-identical to the reference so behaviour matches the Tesseract/paddle parsers exactly.

1. Replace the module docstring (lines 1-28) with:

```python
"""Paperless-ngx parser plugin that runs ocrmypdf with the Chandra engine.

This parser mirrors ``paperless.parsers.tesseract.RasterisedDocumentParser``
almost exactly. The structural differences are:

* ``plugins=["paperless_chandra.ocrmypdf_plugin"]`` is added so ocrmypdf
  loads the Chandra engine instead of Tesseract.
* No language mapping: Chandra is language-agnostic and reads mixed
  scripts in a single pass, so ``PAPERLESS_OCR_LANGUAGE`` is passed
  through verbatim (it still drives paperless's own search stemming, and
  its first code labels the hOCR).
* The ``chandra_*`` family (server URL, model name, API key, output token
  budget, content format) is passed through as ocrmypdf kwargs
  (registered as CLI args by the ``add_options`` hookimpl).

Everything else - image alpha removal, DPI handling, PDF/A conversion,
``OCR_MODE`` semantics, the safe-fallback retry - is identical to the
Tesseract parser. The implementation is intentionally kept close to
``tesseract.py`` so behaviour stays consistent for users switching
engines.
"""
```

2. Change the two package imports:

```python
from paperless_paddleocr import __version__
from paperless_paddleocr.languages import resolve_paddle_languages
```

to:

```python
from paperless_chandra import __version__
```

3. Change the logger name from `"paperless.parsing.paddleocr"` to `"paperless.parsing.chandra"`.

4. Change `_OCRMYPDF_PLUGIN_MODULE` from `"paperless_paddleocr.ocrmypdf_plugin"` to `"paperless_chandra.ocrmypdf_plugin"`.

5. Change the tempdir prefix `"paperless-paddleocr-"` to `"paperless-chandra-"`.

6. Rename the class and its metadata:

```python
class PaperlessChandraParser:
    """OCR parser using Chandra via ocrmypdf (drop-in for the Tesseract parser)."""

    name: str = "Paperless-ngx Chandra Parser"
    version: str = __version__
    author: str = "Florian Bernd"
    url: str = "https://github.com/flobernd/paperless-chandra"
```

7. In `score()`, change the env var to `PAPERLESS_CHANDRA_SCORE` (keep `default=15` and the comment about beating Tesseract's 10, reworded to Chandra).

8. Replace the plugin-specific env block in `__init__` (everything from `self._engine: str = _env_choice(` down to the `self._lang_override` assignment) with:

```python
        # Plugin-specific env vars, read once per parser instance.
        self._server_url: str = (
            os.environ.get("PAPERLESS_CHANDRA_SERVER_URL", "") or ""
        ).strip()
        self._model_name: str = (
            os.environ.get("PAPERLESS_CHANDRA_MODEL_NAME", "chandra") or ""
        ).strip() or "chandra"
        self._api_key: str = (os.environ.get("PAPERLESS_CHANDRA_API_KEY", "") or "").strip()
        self._max_output_tokens: int = _env_int(
            "PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS",
            default=12384,
        )
        self._content_format: str = _env_choice(
            "PAPERLESS_CHANDRA_CONTENT_FORMAT",
            choices=("text", "markdown"),
            default="text",
        )
```

9. In `construct_ocrmypdf_parameters`, delete the `paddle_langs = resolve_paddle_languages(...)` call and replace the engine wiring section of `ocrmypdf_args`. The `language` entry (including the two comment lines above it about ocrmypdf joining multi-lang lists) becomes:

```python
            "language": self.settings.language or "eng",
```

and (replacing the five `paddle_*` entries and their section comment):

```python
            # ─ Chandra engine wiring ────────────────────────────────────
            "plugins": [_OCRMYPDF_PLUGIN_MODULE],
            # ocrmypdf 17.x removed the 'hocr' renderer and routes everything
            # through the fpdf2 renderer, which still calls our generate_hocr
            # and renders the invisible text layer itself. No renderer override
            # is required (and 'hocr' would be silently ignored).
            # Custom kwargs registered by our add_options hookimpl.
            "chandra_server_url": self._server_url,
            "chandra_model_name": self._model_name,
            "chandra_api_key": self._api_key,
            "chandra_max_output_tokens": self._max_output_tokens,
            "chandra_content_format": self._content_format,
```

10. In both `parse()` log-redaction lines, change `"paddle_vl_api_key"` to `"chandra_api_key"`.

11. In the two OCR-strategy debug logs in `parse()`, change `"OCR strategy: full OCR via PaddleOCR - OCR_MODE=%s, engine=%s", self.settings.mode, self._engine` to `"OCR strategy: full OCR via Chandra - OCR_MODE=%s", self.settings.mode` (drop the second argument; there is no engine variant).

12. Sanity sweep: `grep -in paddle paperless_chandra/parser.py` must return nothing. Fix any stragglers (comments referencing `tesseract.py` line numbers stay; they document the mirror relationship).

- [ ] **Step 3: Verify the file compiles and the suite still passes**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/python -m py_compile paperless_chandra/parser.py && echo COMPILES
.venv/bin/pytest tests/ -q
```

Expected: `COMPILES` and all tests pass. The parser cannot be imported here (django/paperless are not installed); full import verification is Task 9.

- [ ] **Step 4: Lint and commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add paperless_chandra/parser.py
git commit -m "Add paperless parser mirroring the Tesseract flow"
```

---

### Task 9: Smoke test inside the paperless-ngx beta image

**Files:**
- Create: `tests/smoke_paperless.py`

**Interfaces:**
- Consumes: entry point (Task 1), `PaperlessChandraParser` (Task 8), `ocrmypdf_plugin.get_ocr_engine` (Task 7).
- Produces: a script CI (Task 10) and local docker runs execute inside the paperless image; exit code 0 on success.

- [ ] **Step 1: Write `tests/smoke_paperless.py`**

```python
"""Smoke test executed inside the paperless-ngx Docker image.

This is *not* collected by pytest (filename is not ``test_*``); it runs
with the real paperless-ngx runtime on ``PYTHONPATH``. Its job is to catch
the failure modes the lightweight unit suite cannot:

* the ``paperless_ngx.parsers`` entry point is actually discoverable, and
  resolves to :class:`PaperlessChandraParser`;
* ``paperless_chandra.parser`` imports cleanly against the *real*
  paperless-ngx API surface (``documents.parsers``, ``paperless.config``,
  ``paperless.parsers.*``) - i.e. paperless has not renamed/moved anything
  the parser depends on;
* ``paperless_chandra.ocrmypdf_plugin`` imports and the ocrmypdf hookimpl
  wiring binds our engine.

Exit code is non-zero on the first failed check so callers fail loudly.
"""

from __future__ import annotations

import sys


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", flush=True)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


def check_entry_point() -> None:
    import importlib.metadata

    eps = importlib.metadata.entry_points(group="paperless_ngx.parsers")
    names = {ep.name: ep for ep in eps}
    if "chandra" not in names:
        _fail(f"entry point 'chandra' not found in paperless_ngx.parsers ({sorted(names)})")
    target = names["chandra"].value
    if "PaperlessChandraParser" not in target:
        _fail(f"entry point 'chandra' resolves to unexpected target: {target!r}")
    _ok(f"entry point discoverable: chandra -> {target}")


def check_plugin_wiring() -> None:
    from paperless_chandra.engine.engine import ChandraEngine
    from paperless_chandra.ocrmypdf_plugin import get_ocr_engine

    engine = get_ocr_engine()
    if not isinstance(engine, ChandraEngine):
        _fail(f"get_ocr_engine() returned {type(engine).__name__}, expected ChandraEngine")
    _ok("ocrmypdf plugin wiring OK")


def check_parser_imports() -> None:
    # Importing this module exercises every `from documents...` /
    # `from paperless...` import against the real paperless-ngx install.
    import django

    django.setup()

    from paperless_chandra.parser import PaperlessChandraParser

    mimes = PaperlessChandraParser.supported_mime_types()
    if "application/pdf" not in mimes:
        _fail(f"application/pdf missing from supported_mime_types(): {mimes}")
    score = PaperlessChandraParser.score("application/pdf", "doc.pdf")
    if score is None or score <= 10:
        _fail(f"score() should beat Tesseract's 10, got {score!r}")
    _ok(f"parser imports against real paperless-ngx; score={score}")


def main() -> None:
    print("== paperless-chandra smoke test ==", flush=True)
    check_entry_point()
    check_plugin_wiring()
    check_parser_imports()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test locally inside the beta image**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/python -m build --wheel --outdir dist
docker run --rm --entrypoint /bin/bash \
  -v "$PWD":/plugin:ro \
  -e PYTHONPATH=/usr/src/paperless/src \
  -e DJANGO_SETTINGS_MODULE=paperless.settings \
  -e PAPERLESS_REDIS=redis://localhost:6379 \
  -e PAPERLESS_DBENGINE=sqlite \
  -e PAPERLESS_SECRET_KEY=smoke-test-not-a-real-secret \
  ghcr.io/paperless-ngx/paperless-ngx:beta \
  -c 'set -e
      PY=/usr/src/paperless/.venv/bin/python
      [ -x "$PY" ] || PY=$(command -v python3)
      "$PY" -m pip install --quiet --break-system-packages /plugin/dist/*.whl
      "$PY" /plugin/tests/smoke_paperless.py'
```

Expected output ends with `== all smoke checks passed ==`. This pulls the beta image on first run (several hundred MB; be patient). If `pip install` hits dependency conflicts with paperless's pinned packages, record the exact conflict in the task summary - that is a real finding, not something to paper over.

- [ ] **Step 3: Commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check . && .venv/bin/ruff format --check .
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add tests/smoke_paperless.py
git commit -m "Add paperless-ngx smoke test and verify against beta image"
```

---

### Task 10: CI workflows

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/smoke.yml`

**Interfaces:**
- Consumes: test suite (Tasks 2-7), smoke script (Task 9), lint configs (Task 1).
- Produces: GitHub Actions gating pushes/PRs to `master`.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

Copy `/home/administrator/workspace/paperless-paddleocr/.github/workflows/ci.yml` and apply:

1. In the `mypy` job and the `static` job, replace the dependency install line
   `python -m pip install "ocrmypdf>=17.4" "lxml>=4.9" pillow numpy`
   with
   `python -m pip install "ocrmypdf>=17.4" "chandra-ocr>=0.2" "beautifulsoup4>=4.12" numpy`.
2. In both mypy invocations, replace the target `paperless_paddleocr` with `paperless_chandra`.
3. In the `tests` job, replace the install block with (chandra-ocr is a light pure-python dependency, so the full dependency set installs in CI; only HTTP is mocked):

```yaml
      - name: Install runtime + test deps
        run: |
          python -m pip install --upgrade pip
          python -m pip install . pytest pikepdf
```

   and delete the comment block above it about paddle wheels.
4. In the `build` job's wheel verification, replace `import paperless_paddleocr; print(paperless_paddleocr.__version__)` with `import paperless_chandra; print(paperless_chandra.__version__)` and change `--no-deps dist/*.whl` install to keep `--no-deps` (import of the bare package works without deps).

Everything else (job structure, actions versions, yamllint/markdownlint steps) stays identical.

- [ ] **Step 2: Write `.github/workflows/smoke.yml`**

Copy `/home/administrator/workspace/paperless-paddleocr/.github/workflows/smoke.yml` and apply:

1. Header comment: replace the paddle sentence with `chandra-ocr and its pure-python dependencies install alongside paperless's pinned packages; the smoke test exercises import wiring and entry-point discovery, not actual OCR.`
2. The install step drops `--no-deps` (the plugin imports chandra at plugin-load time, so dependencies must be present):

```yaml
      - name: Install package
        run: |
          "${{ steps.py.outputs.bin }}" -m pip install --break-system-packages .
```

Everything else stays identical (image, env block, run step).

- [ ] **Step 3: Validate YAML and run all local gates**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/pip install --quiet pyyaml
.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"
.venv/bin/pytest tests/ -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy --ignore-missing-imports paperless_chandra
```

Expected: YAML parses, tests pass, ruff clean. mypy: fix any errors it reports in plugin code (the copied modules are known-clean in the reference repo).

- [ ] **Step 4: Commit**

```bash
cd /home/administrator/workspace/paperless-chandra
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add .github
git commit -m "Add CI and smoke workflows"
```

---

### Task 11: Builder image, install recipes, bootstrap script

**Files:**
- Create: `docker/builder.Dockerfile`, `examples/Dockerfile`, `examples/docker-compose.vllm.yml`, `setup.sh`

**Interfaces:**
- Consumes: wheel build (Task 1), README stub.
- Produces: `docker/builder.Dockerfile` (local wheel build, used by Task 12's e2e image), `examples/Dockerfile` (GitHub-ref build for end users), compose example with a GPU vLLM sidecar.

- [ ] **Step 1: Write `docker/builder.Dockerfile`**

```dockerfile
# Builder image: produces the paperless-chandra wheel.
#
# The version-controlled build recipe for the plugin. Machine-specific
# deployment Dockerfiles consume the /dist output as a named build
# context, so the final paperless image installs a prebuilt wheel and
# needs no build tools of its own.
#
# Build context is the repo root.
FROM python:3.12-slim

WORKDIR /src

# Build inputs: the package sources plus the files pyproject.toml's metadata
# references (readme + license). Copied explicitly so unrelated repo content
# does not invalidate the layer cache.
COPY pyproject.toml README.md LICENSE ./
COPY paperless_chandra ./paperless_chandra

# Produce the wheel under /dist.
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist
```

- [ ] **Step 2: Write `examples/Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.7
#
# paperless-ngx + paperless-chandra.
# OCR recognition is delegated to a separate OpenAI-compatible inference
# server (vLLM, llama.cpp, Ollama) - see docker-compose.vllm.yml for the
# matching runtime config.
#
# Build:
#   docker build \
#     --build-arg PLUGIN_REF=v0.1.0 \
#     -f examples/Dockerfile \
#     -t paperless-chandra:latest .

ARG PLUGIN_REF=master
ARG PAPERLESS_TAG=beta

# --- Stage 1: build the plugin wheel from source --------------------------
FROM python:3.12-slim AS plugin
ARG PLUGIN_REF
ADD https://github.com/flobernd/paperless-chandra.git#${PLUGIN_REF} /src
WORKDIR /src
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

# --- Stage 2: layer the plugin onto paperless-ngx -------------------------
FROM ghcr.io/paperless-ngx/paperless-ngx:${PAPERLESS_TAG}

COPY --from=plugin /dist/*.whl /tmp/plugin/
RUN pip install --no-cache-dir /tmp/plugin/*.whl \
 && rm -rf /tmp/plugin
```

- [ ] **Step 3: Write `examples/docker-compose.vllm.yml`**

```yaml
# paperless-ngx (CPU-only) + Chandra on a local NVIDIA GPU via vLLM.
#
# Build + run from the examples/ directory:
#   docker compose -f docker-compose.vllm.yml build
#   docker compose -f docker-compose.vllm.yml up -d
#
# Required `.env` keys alongside this file:
#   CHANDRA_API_KEY=<some-random-string>
#   PAPERLESS_DB_USER=<db-user>
#   PAPERLESS_DB_PASSWORD=<strong-db-password>
# Optional: PLUGIN_REF=v0.1.0 (release tag for reproducible builds).
#
# The vLLM flags mirror the upstream chandra_vllm launcher, scaled for a
# 24 GB GPU (L4 / RTX 4090 class). For an 80 GB H100 raise
# --max-num-seqs to 64 and --max-num-batched-tokens to 8192.

services:
  paperless:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        PLUGIN_REF: ${PLUGIN_REF:-master}
        PAPERLESS_TAG: ${PAPERLESS_TAG:-beta}
    image: paperless-chandra:latest
    restart: unless-stopped
    depends_on:
      - paperless-db
      - paperless-redis
      - chandra-server
    ports:
      - "8000:8000"
    volumes:
      - paperless_data:/usr/src/paperless/data
      - paperless_media:/usr/src/paperless/media
      - ./consume:/usr/src/paperless/consume
    environment:
      PAPERLESS_REDIS: redis://paperless-redis:6379
      PAPERLESS_DBHOST: paperless-db
      PAPERLESS_DBUSER: ${PAPERLESS_DB_USER:?missing PAPERLESS_DB_USER in .env}
      PAPERLESS_DBPASS: ${PAPERLESS_DB_PASSWORD:?missing PAPERLESS_DB_PASSWORD in .env}
      PAPERLESS_OCR_LANGUAGE: eng
      PAPERLESS_CHANDRA_SERVER_URL: http://chandra-server:8000
      PAPERLESS_CHANDRA_MODEL_NAME: chandra
      PAPERLESS_CHANDRA_API_KEY: ${CHANDRA_API_KEY:?missing CHANDRA_API_KEY in .env}

  chandra-server:
    image: vllm/vllm-openai:v0.17.0
    restart: unless-stopped
    command:
      - --model=datalab-to/chandra-ocr-2
      - --served-model-name=chandra
      - --api-key=${CHANDRA_API_KEY:?missing CHANDRA_API_KEY in .env}
      - --dtype=bfloat16
      - --max-model-len=18000
      - --max-num-seqs=16
      - --max-num-batched-tokens=2048
      - --gpu-memory-utilization=0.85
      - --enable-prefix-caching
      - --no-enforce-eager
      - --mm-processor-kwargs={"min_pixels":3136,"max_pixels":6291456}
    expose:
      - "8000"
    ipc: host
    volumes:
      # Persist downloaded model weights (~10 GB) between restarts.
      - hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              count: 1

  paperless-db:
    image: postgres:16
    restart: unless-stopped
    volumes:
      - paperless_db:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: paperless
      POSTGRES_USER: ${PAPERLESS_DB_USER:?missing PAPERLESS_DB_USER in .env}
      POSTGRES_PASSWORD: ${PAPERLESS_DB_PASSWORD:?missing PAPERLESS_DB_PASSWORD in .env}

  paperless-redis:
    image: redis:8
    restart: unless-stopped
    volumes:
      - paperless_redis:/data

volumes:
  paperless_data:
  paperless_media:
  paperless_db:
  paperless_redis:
  hf_cache:
```

- [ ] **Step 4: Write `setup.sh`**

Adapt `/home/administrator/workspace/paperless-paddleocr/setup.sh` (read it first): keep the same `/custom-cont-init.d/` bootstrap structure and logging, but drop every paddle wheel/native-lib step - this plugin needs no native libraries and no special index. The install line becomes a plain `pip install` of the tarball/wheel found next to the script (`paperless-chandra.tar.gz` or `paperless_chandra-*.whl`). Keep the final "bootstrap complete" log line. Do not add a GPU variant; there is none.

- [ ] **Step 5: Verify builder image and compose file**

```bash
cd /home/administrator/workspace/paperless-chandra
docker build -q -f docker/builder.Dockerfile -t paperless-chandra-builder . \
  && docker run --rm paperless-chandra-builder ls /dist
CHANDRA_API_KEY=x PAPERLESS_DB_USER=x PAPERLESS_DB_PASSWORD=x \
  docker compose -f examples/docker-compose.vllm.yml config -q && echo COMPOSE-OK
bash -n setup.sh && echo SETUP-SYNTAX-OK
```

Expected: a `.whl` listed, `COMPOSE-OK`, `SETUP-SYNTAX-OK`. (`examples/Dockerfile` fetches from GitHub, which does not exist yet; it is validated implicitly by the identical-structure builder build and by compose `config`.)

- [ ] **Step 6: Commit**

```bash
cd /home/administrator/workspace/paperless-chandra
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add docker examples setup.sh
git commit -m "Add docker recipes, compose example, and bootstrap script"
```

---

### Task 12: End-to-end stack with stub server

**Files:**
- Create: `docker/e2e.Dockerfile`, `tests/e2e/stub_chandra.py`, `tests/e2e/docker-compose.stub.yml`, `tests/e2e/make_test_page.py`, `tests/e2e/assert_e2e.py`, `tests/e2e/run-stub.sh`

**Interfaces:**
- Consumes: the whole plugin (Tasks 1-9), builder pattern (Task 11).
- Produces: a scripted, deterministic full-pipeline test: consume dir -> paperless consumer -> parser -> ocrmypdf -> engine -> HTTP -> stub -> hOCR -> PDF/A archive -> API assertions. Task 13 swaps the stub service for llama.cpp and reuses `make_test_page.py` + `assert_e2e.py`.

- [ ] **Step 1: Write `docker/e2e.Dockerfile`**

```dockerfile
# paperless-ngx beta + the locally built plugin, for the e2e compose stacks.
# Build context is the repo root.
FROM python:3.12-slim AS plugin
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY paperless_chandra ./paperless_chandra
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

FROM ghcr.io/paperless-ngx/paperless-ngx:beta
COPY --from=plugin /dist/*.whl /tmp/plugin/
RUN pip install --no-cache-dir /tmp/plugin/*.whl \
 && rm -rf /tmp/plugin
```

- [ ] **Step 2: Write `tests/e2e/stub_chandra.py`**

```python
"""Minimal OpenAI-compatible stub returning canned Chandra layout HTML.

Runs on stdlib only so the compose service is a bare python:3.12-slim.
Every chat completion returns the same layout blocks; the e2e assertion
looks for STUB_MARKER in the ingested document's content.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

STUB_MARKER = "PAPERLESS CHANDRA STUB OK"

_PAGE_HTML = (
    f'<div data-label="Section-Header" data-bbox="60 40 940 120"><h1>{STUB_MARKER}</h1></div>'
    '<div data-label="Text" data-bbox="60 160 940 400">'
    "<p>Stub invoice 2026-0042<br>Second stub line</p></div>"
)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/").endswith("/models"):
            self._send_json({"object": "list", "data": [{"id": "chandra", "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        self._send_json(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": "chandra",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _PAGE_HTML},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def log_message(self, fmt, *args):  # quiet: compose logs stay readable
        pass


if __name__ == "__main__":
    print("stub chandra server listening on :8080", flush=True)
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
```

- [ ] **Step 3: Write `tests/e2e/make_test_page.py`**

```python
"""Generate a simple high-contrast test page for e2e ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main(out_path: str) -> None:
    img = Image.new("RGB", (1654, 2339), "white")  # A4 at 200 dpi
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.load_default(size=72)
    font_body = ImageFont.load_default(size=48)
    draw.text((120, 200), "RECHNUNG 2026-0042", fill="black", font=font_big)
    draw.text((120, 420), "PAPERLESS CHANDRA END TO END", fill="black", font=font_body)
    draw.text((120, 540), "Betrag: 123,45 EUR", fill="black", font=font_body)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, dpi=(200, 200))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Write `tests/e2e/assert_e2e.py`**

```python
"""Poll the paperless API until the ingested document appears, then assert.

stdlib + pypdf only (pypdf is a dev dependency of the repo venv). Auth uses
the admin credentials the compose stack creates via PAPERLESS_ADMIN_USER.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"


def _request(path: str, token: str | None = None, raw: bool = False):
    req = urllib.request.Request(f"{BASE}{path}")
    if token:
        req.add_header("Authorization", f"Token {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return data if raw else json.loads(data)


def _get_token(timeout_s: int) -> str:
    payload = json.dumps({"username": "admin", "password": "admin"}).encode()
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{BASE}/api/token/",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())["token"]
        except Exception as e:  # server still starting
            last_err = e
            time.sleep(5)
    raise SystemExit(f"paperless API never came up: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", action="append", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    token = _get_token(args.timeout)
    print("API up, token acquired", flush=True)

    deadline = time.time() + args.timeout
    doc = None
    while time.time() < deadline:
        docs = _request("/api/documents/?ordering=-added&page_size=1", token)
        if docs.get("count", 0) >= 1:
            doc = docs["results"][0]
            break
        time.sleep(10)
    if doc is None:
        raise SystemExit("document never appeared in paperless")

    # PDF text extraction and OCR post-processing can vary spacing, so
    # matching happens on whitespace-normalised text.
    content = " ".join((doc.get("content") or "").split())
    print(f"document {doc['id']} content: {content[:200]!r}", flush=True)
    for needle in args.expect:
        if needle not in content:
            raise SystemExit(f"expected {needle!r} in document content")
    print("content assertions passed", flush=True)

    archive = _request(f"/api/documents/{doc['id']}/download/", token, raw=True)
    if not archive.startswith(b"%PDF"):
        raise SystemExit("downloaded archive is not a PDF")
    from pypdf import PdfReader

    text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(archive)).pages)
    text = " ".join(text.split())
    print(f"archive text layer: {text[:200]!r}", flush=True)
    for needle in args.expect:
        if needle not in text:
            raise SystemExit(f"expected {needle!r} in archive PDF text layer")
    print("archive text layer assertions passed", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `tests/e2e/docker-compose.stub.yml`**

```yaml
# Full-pipeline e2e against a stub OpenAI-compatible server. Deterministic
# and CPU-cheap: verifies consumer -> parser -> ocrmypdf -> engine -> HTTP
# -> hOCR -> PDF/A -> API, not OCR accuracy.

services:
  paperless:
    build:
      context: ../..
      dockerfile: docker/e2e.Dockerfile
    restart: unless-stopped
    depends_on:
      - redis
      - stub
    ports:
      - "8000:8000"
    volumes:
      - ./consume:/usr/src/paperless/consume
    environment:
      PAPERLESS_REDIS: redis://redis:6379
      PAPERLESS_SECRET_KEY: e2e-not-a-real-secret
      PAPERLESS_ADMIN_USER: admin
      PAPERLESS_ADMIN_PASSWORD: admin
      PAPERLESS_OCR_LANGUAGE: eng
      PAPERLESS_CHANDRA_SERVER_URL: http://stub:8080
      PAPERLESS_CONSUMER_POLLING: "5"

  stub:
    image: python:3.12-slim
    restart: unless-stopped
    volumes:
      - ./stub_chandra.py:/srv/stub_chandra.py:ro
    command: ["python", "/srv/stub_chandra.py"]
    expose:
      - "8080"

  redis:
    image: redis:8
    restart: unless-stopped
```

- [ ] **Step 6: Write `tests/e2e/run-stub.sh`**

```bash
#!/usr/bin/env bash
# End-to-end pipeline test against the stub Chandra server.
# Usage: tests/e2e/run-stub.sh   (from anywhere; needs docker + repo venv)
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
PY="$REPO_ROOT/.venv/bin/python"

cleanup() { docker compose -f docker-compose.stub.yml down -v; }
trap cleanup EXIT

rm -rf consume && mkdir -p consume
docker compose -f docker-compose.stub.yml up -d --build

"$PY" make_test_page.py consume/e2e-test.png
"$PY" assert_e2e.py --expect "PAPERLESS CHANDRA STUB OK" --expect "Stub invoice 2026-0042" --timeout 600
echo "== stub e2e passed =="
```

```bash
chmod +x tests/e2e/run-stub.sh
```

- [ ] **Step 7: Run it**

```bash
cd /home/administrator/workspace/paperless-chandra
tests/e2e/run-stub.sh
```

Expected: `== stub e2e passed ==`. First run builds the paperless image and starts the stack; allow ~10 minutes. Debugging aids if it fails: `docker compose -f tests/e2e/docker-compose.stub.yml logs paperless | grep -i -E "chandra|parser|error"` - look for the `Loaded third-party parser` line and consumer errors. RAM is tight on this host (8 GB); stop unrelated containers first if the stack OOMs.

- [ ] **Step 8: Commit**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/ruff check tests/e2e && .venv/bin/ruff format --check tests/e2e
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add docker/e2e.Dockerfile tests/e2e
git commit -m "Add stub-server end-to-end pipeline test"
```

---

### Task 13: End-to-end with the real quantized model on CPU (llama.cpp)

**Files:**
- Create: `tests/e2e/docker-compose.llamacpp.yml`, `tests/e2e/download-model.sh`, `tests/e2e/run-llamacpp.sh`

**Interfaces:**
- Consumes: `docker/e2e.Dockerfile`, `make_test_page.py`, `assert_e2e.py` (Task 12).
- Produces: the real-model verification the user requested; also serves as documentation for CPU/quantized deployments.

- [ ] **Step 1: Locate the official Chandra GGUF**

The HF model card advertises llama.cpp-compatible quantizations. Find the repo and exact filenames:

```bash
curl -s "https://huggingface.co/api/models?search=chandra&author=datalab-to" | python3 -m json.tool | grep -i '"id"'
# then, for the GGUF repo found (expected name similar to datalab-to/chandra-ocr-2-GGUF):
curl -s "https://huggingface.co/api/models/<GGUF-REPO-ID>/tree/main" | python3 -m json.tool | grep -i gguf
```

Decision rule: prefer the `datalab-to` repo; if none exists, take the GGUF repo linked under "Quantizations" on `https://huggingface.co/datalab-to/chandra-ocr-2` (fetch that page and grep for `gguf`). Pick the `Q4_K_M` model file plus the `mmproj` file (usually `mmproj-*-f16.gguf`); a VLM GGUF is unusable without its mmproj. Record both exact URLs in `download-model.sh`.

- [ ] **Step 2: Write `tests/e2e/download-model.sh`**

```bash
#!/usr/bin/env bash
# Download the quantized Chandra GGUF + mmproj for the llama.cpp e2e stack.
# ~4 GB total; files land in tests/e2e/models/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

# URLs pinned by the implementation step that located the official GGUFs.
MODEL_URL="<exact-resolved-url-to-Q4_K_M.gguf>"
MMPROJ_URL="<exact-resolved-url-to-mmproj.gguf>"

[ -f models/chandra-q4_k_m.gguf ] || curl -L --fail -o models/chandra-q4_k_m.gguf "$MODEL_URL"
[ -f models/mmproj.gguf ] || curl -L --fail -o models/mmproj.gguf "$MMPROJ_URL"
ls -lh models/
```

Replace both placeholder URLs with the real `https://huggingface.co/<repo>/resolve/main/<file>` URLs found in Step 1 before committing; the script must not ship with placeholders (verify with `grep -c "exact-resolved" tests/e2e/download-model.sh` returning 0).

- [ ] **Step 3: Write `tests/e2e/docker-compose.llamacpp.yml`**

```yaml
# Full-pipeline e2e against the real quantized Chandra model on CPU via
# llama.cpp. Slow (minutes per page on a small host) but exercises true
# model output end to end. Run tests/e2e/download-model.sh first.

services:
  paperless:
    build:
      context: ../..
      dockerfile: docker/e2e.Dockerfile
    restart: unless-stopped
    depends_on:
      - redis
      - llamacpp
    ports:
      - "8000:8000"
    volumes:
      - ./consume:/usr/src/paperless/consume
    environment:
      PAPERLESS_REDIS: redis://redis:6379
      PAPERLESS_SECRET_KEY: e2e-not-a-real-secret
      PAPERLESS_ADMIN_USER: admin
      PAPERLESS_ADMIN_PASSWORD: admin
      PAPERLESS_OCR_LANGUAGE: eng
      PAPERLESS_CHANDRA_SERVER_URL: http://llamacpp:8080
      # Small budget keeps CPU generation inside the client's 600 s timeout.
      PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS: "4096"
      PAPERLESS_CONSUMER_POLLING: "5"

  llamacpp:
    image: ghcr.io/ggml-org/llama.cpp:server
    restart: unless-stopped
    volumes:
      - ./models:/models:ro
    command:
      - -m
      - /models/chandra-q4_k_m.gguf
      - --mmproj
      - /models/mmproj.gguf
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - --ctx-size
      - "8192"
      - --jinja
    expose:
      - "8080"

  redis:
    image: redis:8
    restart: unless-stopped
```

- [ ] **Step 4: Write `tests/e2e/run-llamacpp.sh`**

```bash
#!/usr/bin/env bash
# End-to-end pipeline test against real quantized Chandra on CPU.
# Slow by design; expect several minutes for the OCR step alone.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$(cd ../.. && pwd)
PY="$REPO_ROOT/.venv/bin/python"

./download-model.sh

avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
if [ "$avail_gb" -lt 5 ]; then
  echo "WARNING: only ${avail_gb} GB RAM available; the 5B Q4 model needs ~4 GB." >&2
  echo "Stop other containers/services first or expect OOM." >&2
fi

cleanup() { docker compose -f docker-compose.llamacpp.yml down; }
trap cleanup EXIT

rm -rf consume && mkdir -p consume
docker compose -f docker-compose.llamacpp.yml up -d --build

"$PY" make_test_page.py consume/e2e-test.png
"$PY" assert_e2e.py --expect "RECHNUNG" --expect "CHANDRA" --timeout 2400
echo "== llama.cpp e2e passed =="
```

```bash
chmod +x tests/e2e/download-model.sh tests/e2e/run-llamacpp.sh
```

- [ ] **Step 5: Run it**

```bash
cd /home/administrator/workspace/paperless-chandra
free -g
tests/e2e/run-llamacpp.sh
```

Expected: `== llama.cpp e2e passed ==` after a long wait (model download ~4 GB, then minutes of CPU inference). Known risks on this 4-core/8 GB host, and what to do:

- OOM or the llamacpp container gets killed: report it, keep the compose file and scripts committed as the documented manual path, and state clearly in the task summary that the real-model run did not complete locally and the stub e2e (Task 12) is the scripted verification.
- OpenAI-client timeout (600 s) during generation: retry once with `PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS: "2048"` in the compose file; if it still times out, same fallback as above.
- Model architecture unsupported by the llama.cpp server image: record the exact error; same fallback as above.

Do not mark this task complete on a failed run without documenting the failure mode; do not silently swap in the stub.

- [ ] **Step 6: Commit**

```bash
cd /home/administrator/workspace/paperless-chandra
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add tests/e2e/docker-compose.llamacpp.yml tests/e2e/download-model.sh tests/e2e/run-llamacpp.sh
git commit -m "Add real-model llama.cpp end-to-end test on CPU"
```

---

### Task 14: README and final polish

**Files:**
- Modify: `README.md` (replace the stub)

**Interfaces:**
- Consumes: everything; the spec at `docs/superpowers/specs/2026-07-12-paperless-chandra-design.md` is the content source of truth.

- [ ] **Step 1: Write the full README**

Mirror the section structure of `/home/administrator/workspace/paperless-paddleocr/README.md` (read it for tone and layout), with this outline and content sourced from the spec:

1. Title + one-paragraph intro (drop-in Chandra provider; paperless keeps driving ocrmypdf; `PAPERLESS_OCR_*` keeps working; PDF/A preserved). Keep the `> [!NOTE]` block: requires paperless-ngx 3.x (beta) with the `paperless_ngx.parsers` entry-point group.
2. Features list: drop-in replacement; honours `PAPERLESS_OCR_*`; 90+ languages and mixed scripts in a single pass with no configuration; complex layout/tables/handwriting via a ~5B VLM; content as plain text or markdown (`PAPERLESS_CHANDRA_CONTENT_FORMAT`); works with any OpenAI-compatible server (vLLM, llama.cpp, Ollama).
3. Architecture diagram (adapt the paddle one: parser -> ocrmypdf -> ChandraEngine -> remote server).
4. Requirements: a self-hosted inference server; GPU sizing paragraph (bf16 needs ~24 GB-class GPU with the documented flags; quantized GGUF via llama.cpp/Ollama runs on smaller GPUs or CPU at reduced speed).
5. Installation: A. custom docker image (`examples/Dockerfile`, build args `PLUGIN_REF` / `PAPERLESS_TAG`), B. bootstrap `setup.sh`, C. host install via `pip install git+...`.
6. Docker compose example: point at `examples/docker-compose.vllm.yml`; include the "Loaded third-party parser 'Paperless-ngx Chandra Parser'" log line to look for.
7. Environment reference: every `PAPERLESS_CHANDRA_*` var with default and description (SCORE 15, SERVER_URL required, MODEL_NAME chandra, API_KEY unset, MAX_OUTPUT_TOKENS 12384, CONTENT_FORMAT text|markdown) - copy semantics from the spec's Configuration section.
8. Honoured paperless-ngx settings table: copy the paddle table, then state the three differences verbatim from the spec (LANGUAGE pass-through single pass; DESKEW supported locally; ROTATE_PAGES not supported, with the zero-confidence no-op explanation).
9. Limitations section (verbatim content from the spec's Known constraints): estimated word/line boxes so viewer search-highlight is approximate; line-level boxes for CJK/Thai; model weights under modified OpenRAIL-M (free for research, personal use, startups under $2M funding/revenue - link https://www.datalab.to/pricing for commercial licensing) while the plugin code is MIT; 600 s client timeout note; concurrency note (threads may issue parallel page requests; remote servers batch them).
10. Performance notes: expect roughly 1-2 pages/s on a datacenter GPU with vLLM, tens of seconds per page on small GPUs, minutes per page on CPU llama.cpp; scale with `PAPERLESS_TASK_WORKERS`.
11. Troubleshooting: plugin not discovered (needs 3.x beta image); server unreachable / API key rejected (preflight error messages); slow CPU inference hitting the 600 s timeout (lower `PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS`, use a GPU); Ollama users must set `PAPERLESS_CHANDRA_MODEL_NAME` to their model tag.
12. License: MIT for the plugin, with the model-license pointer.

Style constraints: no em-dashes or en-dashes anywhere; GitHub-flavoured markdown; keep lists short and use prose for anything nuanced; must pass `.markdownlint.yaml`.

- [ ] **Step 2: Run every gate**

```bash
cd /home/administrator/workspace/paperless-chandra
.venv/bin/pytest tests/ -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy --ignore-missing-imports paperless_chandra
docker run --rm -v "$PWD":/work -w /work davidanson/markdownlint-cli2 "**/*.md" "!docs/superpowers/**" || true
```

Expected: tests pass, ruff and mypy clean. Fix any markdownlint findings in README.md (the `|| true` is only because the image pull may be blocked; if it runs, treat findings as failures).

- [ ] **Step 3: Commit**

```bash
cd /home/administrator/workspace/paperless-chandra
export SSH_AUTH_SOCK=/tmp/ssh-fH7oMDlprO0K/agent.22774
git add README.md
git commit -m "Write full README"
```

---

## Verification summary

After all tasks: `pytest` green, `ruff`/`mypy` clean, smoke test passed inside the real paperless-ngx beta image, stub e2e proves the full ingest pipeline, llama.cpp e2e proves real model output (or its failure mode is documented honestly), and the repo contains user-facing install recipes mirroring paperless-paddleocr.
