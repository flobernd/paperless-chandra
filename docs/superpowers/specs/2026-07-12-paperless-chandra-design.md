# paperless-chandra: Chandra OCR provider for paperless-ngx

Date: 2026-07-12
Status: approved

## Overview

`paperless-chandra` is a parser plugin for paperless-ngx 3.x (beta) that
replaces the built-in Tesseract OCR pipeline with [Chandra OCR](https://github.com/datalab-to/chandra),
Datalab's ~5B-parameter vision-language OCR model. The plugin mirrors the
architecture of `paperless-paddleocr` (same author, same repo structure): a
`paperless_ngx.parsers` entry point drives `ocrmypdf` with a custom
`OcrEngine`, so all `PAPERLESS_OCR_*` settings keep working and archive PDFs
remain searchable PDF/A.

Recognition is delegated to a self-hosted, OpenAI-compatible inference server
(vLLM is the official path; llama.cpp and Ollama serve the published
quantized checkpoints through the same API). The paperless container itself
stays CPU-only and lean.

## Decisions

These were settled during design review:

- **Backends: remote server only.** One engine, pointed at an
  OpenAI-compatible endpoint. Chandra's local HuggingFace backend (torch in
  the paperless image, GPU in the paperless container) is out of scope.
- **Content format: configurable, default plain text.**
  `PAPERLESS_CHANDRA_CONTENT_FORMAT=text|markdown` selects what paperless
  stores as document content. The PDF text layer is always plain text.
- **Client: depend on the `chandra-ocr` package.** Upstream prompts, retry
  and repeat-token detection, and HTML-to-blocks parsing stay in lock-step
  with model releases. The dependency is isolated behind one client module.
- **E2E test: real quantized model on CPU** via a llama.cpp server in the
  compose stack, with the stub-server variant as fallback if the host cannot
  fit the model.
- **Implementation is executed with subagents** (subagent-driven
  development) once the implementation plan exists.

## Architecture

```text
paperless consumer
        ▼
PaperlessChandraParser         ← paperless_ngx.parsers entry point, score 15
        ▼
ocrmypdf.ocr(plugins=["paperless_chandra.ocrmypdf_plugin"])
        ▼
ChandraEngine                  ← ocrmypdf OcrEngine
        ▼
chandra-ocr generate_vllm      ← upstream client: prompt, retries, repeat detection
        ▼
OpenAI-compatible /v1/chat/completions   (self-hosted vLLM / llama.cpp / Ollama)
```

`ocrmypdf` keeps handling rasterisation, PDF/A conversion, deskew rotation,
image cleaning, sidecar extraction, and encryption. Only the OCR engine is
replaced.

Compared to `paperless-paddleocr`, two whole layers disappear: there is no
language mapping (Chandra takes no language parameter; one pass covers 90+
languages and mixed scripts) and no multi-language merge engine. Block order
returned by the model is trusted as reading order, so the banded-column
re-sorting layer is not needed either.

## Repository layout

```text
paperless_chandra/
  __init__.py          version
  parser.py            mirror of the paddleocr parser: same MIME set, OCR_MODE
                       flow, safe-fallback retry, PDF/A no-OCR paths; no
                       language mapping
  ocrmypdf_plugin.py   add_options / check_options (+ preflight probe) /
                       get_ocr_engine hookimpls
  engine/
    engine.py          ChandraEngine: generate_hocr, generate_pdf, get_deskew,
                       get_orientation, version, creator_tag, languages
    client.py          the only module touching chandra-ocr: applies plugin
                       config to chandra's global settings once per process,
                       calls generate_vllm with the ocr_layout prompt, raises
                       on error flags, returns raw HTML
    blocks.py          chandra chunks → filtered text blocks → plain-text
                       lines (HTML→text), plus sidecar rendering (text or
                       markdown mode)
    hocr.py            typed Page/Block/Line/Word model + hOCR serialisation
                       (copied from paperless-paddleocr)
    geometry.py        BBox helpers + proportional word-box estimation (copied
                       subset)
    deskew.py          projection-profile skew estimator (copied; PIL + numpy)
    pdf.py             hOCR → invisible-text-only PDF for the generate_pdf
                       path (copied)
```

## Data flow per page

1. ocrmypdf rasterises the page and calls `ChandraEngine.generate_hocr` with
   the image path.
2. `client.py` loads the PIL image and calls `chandra.model.vllm.generate_vllm`
   with a `BatchInputItem` using the `ocr_layout` prompt. Upstream handles
   scale-to-fit, base64 encoding, the OpenAI request (temperature 0.0,
   top_p 0.1), retries with temperature bumps, and repeat-token detection.
3. The raw HTML response contains top-level `<div data-label="…"
   data-bbox="x0 y0 x1 y1">` blocks with bboxes normalised 0-1000. Upstream
   `parse_chunks` converts them to pixel-space blocks against the original
   image dimensions.
4. `blocks.py` filters to text-bearing labels and converts each block's inner
   HTML to plain-text lines:
   - kept: Text, Section-Header, Caption, Footnote, List-Group, Table,
     Form, Code-Block, Equation-Block, Table-Of-Contents, Bibliography,
     Complex-Block, Page-Header, Page-Footer (headers/footers are kept
     deliberately: invoice numbers and dates live there)
   - skipped: Image, Figure (alt-text descriptions must not pollute the
     text layer), Blank-Page, Diagram, Chemical-Block
   - line splitting: block-level HTML boundaries (`<p>`, `<br>`, `<li>`,
     `<h1>`-`<h5>`, `<tr>`); table rows become one line each with cells
     space-joined; math becomes its LaTeX source text
5. Line boxes are synthesized by dividing the block bbox evenly per line
   (per-line boundary arithmetic so rounding never accumulates); word boxes
   are estimated proportionally to token length within the line box. This is
   the same approach as paperless-paddleocr's VL "ocr mode" path.
6. The typed `Page` is serialised to hOCR plus the sidecar text; ocrmypdf's
   fpdf2 renderer draws the invisible text layer and grafts the original
   raster on top, producing the PDF/A archive.

## Configuration

Plugin-specific env vars, all prefixed `PAPERLESS_CHANDRA_`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVER_URL` | required | OpenAI-compatible endpoint; `/v1` appended when missing |
| `MODEL_NAME` | `chandra` | served model name (vLLM launcher default; Ollama users override) |
| `API_KEY` | unset | Bearer token; the vLLM convention `EMPTY` is sent when unset |
| `MAX_OUTPUT_TOKENS` | `12384` | upstream default; guards runaway generations |
| `CONTENT_FORMAT` | `text` | `text` or `markdown` document content |
| `SCORE` | `15` | parser-registry priority (Tesseract scores 10) |

`markdown` mode swaps only the sidecar/content rendering (upstream
`parse_markdown` with headers/footers included); the PDF text layer is
always plain text. When paperless extracts text from the archive PDF
instead of the sidecar (e.g. `redo` mode), content is plain text regardless;
this is documented in the README.

### Honoured PAPERLESS_OCR_* settings

Identical to paperless-paddleocr's table, with three differences:

- `PAPERLESS_OCR_LANGUAGE`: accepted but not mapped; Chandra is
  language-agnostic and runs a single pass regardless of how many codes are
  listed. The first code labels the hOCR `lang` attribute. The engine's
  `languages()` reports every requested code as supported so ocrmypdf
  preflight never aborts.
- `PAPERLESS_OCR_DESKEW`: supported via the copied projection-profile
  estimator (no model required).
- `PAPERLESS_OCR_ROTATE_PAGES`: not supported in v1. `get_orientation`
  returns a zero-confidence no-op, logs once per process, and the README
  documents the limitation. A fully rotated page will OCR poorly; mild skew
  is handled by deskew and by the model's own tolerance.

## Error handling

- `check_options` preflight fails fast with actionable messages when:
  chandra-ocr is not importable, `SERVER_URL` is unset, the server is
  unreachable, or it rejects the API key (probe `{base}/models` once per
  worker process, mirroring the paddleocr VL probe).
- Upstream `generate_vllm` swallows exceptions into `GenerationResult.error`
  after its retry budget; `client.py` raises on that flag so the parser's
  existing safe-fallback / ParseError flow engages.
- Malformed `data-bbox` values fall back upstream to a degenerate box;
  `blocks.py` widens degenerate boxes to the full page so text is never
  dropped from the layer.
- Empty model output → empty sidecar → the parser's `_NoTextFoundError`
  force-OCR retry, then empty content with a warning (identical to the
  paddleocr behaviour).

## Known constraints

- Word and line boxes are estimated within block boxes, so viewer
  search-highlighting is approximate. Extracted text and search are
  unaffected. For scripts without inter-word spaces (CJK, Thai) the boxes
  are effectively line-level.
- Chandra model weights are under a modified OpenRAIL-M license (free for
  research, personal use, and startups under $2M funding/revenue). The
  plugin code is MIT; the README carries the model-license note.
- The upstream client has a fixed 600 s request timeout (OpenAI client
  default); only very slow CPU inference approaches it.
- Concurrency is unthrottled: within one worker, ocrmypdf threads may issue
  parallel page requests, which remote servers batch efficiently. Noted in
  the README performance section.

## Testing

**Unit (pytest, CI):** blocks.py HTML→lines (paragraphs, `<br>`, lists,
tables, math, checkboxes), label filtering, degenerate-bbox widening,
line/word box synthesis, hOCR rendering, client wrapper against mocked
OpenAI responses (success, error flag, retry exhaustion), content-format
toggle, parser ocrmypdf-arg construction, check_options failure modes.
chandra-ocr installs in CI; only HTTP is mocked.

**Smoke (CI):** inside `ghcr.io/paperless-ngx/paperless-ngx:beta` with the
built wheel: entry point `chandra` discoverable and resolving to
`PaperlessChandraParser`, parser imports against the real paperless v3 API
surface, hookimpl wiring returns `ChandraEngine`.

**E2E (local script, not CI):** compose stack of paperless-ngx:beta with the
plugin baked in, plus a llama.cpp server running the official quantized
Chandra GGUF on CPU. The script ingests a one-page test document through the
consume directory and asserts: document created, content contains expected
strings, archive PDF/A produced, `pdftotext` output of the archive matches,
and overlay word boxes fall within the page. Host constraints (8 GB RAM,
4 cores) make this tight; if the model does not fit, the scripted test runs
against a stub OpenAI-compatible server returning canned Chandra HTML, and
the real-model compose file remains as the documented manual path.

## Packaging and examples

- `pyproject.toml`: dependencies `ocrmypdf>=17.4`, `chandra-ocr>=0.2`,
  `beautifulsoup4>=4.12` (used directly by blocks.py; also a chandra-ocr
  dependency), `numpy`; Python ≥3.12; entry point
  `chandra = paperless_chandra.parser:PaperlessChandraParser`; ruff/mypy
  config copied from paperless-paddleocr; license MIT.
- `examples/Dockerfile`: builds the wheel from a Git ref and layers it onto
  `ghcr.io/paperless-ngx/paperless-ngx:beta` (no extra native libs needed).
- `examples/docker-compose.vllm.yml`: full stack with a GPU sidecar running
  stock `vllm/vllm-openai` using upstream's serve flags (bfloat16,
  `--max-model-len 18000`, `--served-model-name chandra`, HF cache volume,
  API key). README documents the Ollama / llama.cpp alternative for smaller
  GPUs or CPU.
- `docker/builder.Dockerfile` for local wheel builds, `setup.sh` bootstrap
  for `/custom-cont-init.d/` deployments (no index gymnastics needed, unlike
  paddle wheels).
- CI: ruff + mypy + pytest workflow, plus the smoke workflow against the
  beta image.
- README structured like paperless-paddleocr's: features, architecture,
  installation methods, compose examples, env reference, honoured settings,
  limitations (bbox granularity, rotate-pages, model license),
  troubleshooting.

## Out of scope for v1

- Local HuggingFace inference inside the paperless container.
- `PAPERLESS_OCR_ROTATE_PAGES` orientation detection.
- Word-level boxes from nested `data-bbox` attributes (the model sometimes
  emits them but the open-source parser strips them; revisit if upstream
  makes them official).
- Datalab hosted API support.
