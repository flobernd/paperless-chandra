"""ocrmypdf plugin: registers the Chandra engine and its CLI options.

This module is the one loaded by ``ocrmypdf.ocr(plugins=[...])`` when the
paperless parser invokes ocrmypdf. The engine lives inside this package
rather than being published as a standalone ocrmypdf plugin, so ocrmypdf's
entry-point auto-discovery never registers it: calls from paperless's
built-in Tesseract parser are unaffected.

The ``validate`` hook writes the mask report of a run in a redo installation:
the pages on which the archive can keep text of the input. The parser reads the
report to decide whether the sidecar holds the whole text of the document.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ocrmypdf
import pikepdf
from ocrmypdf.exceptions import MissingDependencyError

from paperless_chandra import (
    CONTENT_FORMATS,
    DEFAULT_CONTENT_FORMAT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
)
from paperless_chandra.engine.client import ChandraClientError, probe_server, request_settings
from paperless_chandra.engine.engine import ChandraEngine, content_format

log = logging.getLogger("paperless.chandra.plugin")

#: ocrmypdf strips an old invisible text layer inside a Form XObject from this
#: release on. An older release leaves it in the archive, beside the new layer.
_STRIPS_NESTED_TEXT = (17, 11)

#: The operators of a content stream that show text.
_SHOWS_TEXT = "Tj TJ ' \""


# check_options runs once per document, and one successful probe per worker
# process is enough. functools.cache keeps nothing for a probe that raises, so
# the next document probes again.
@functools.cache
def _probe_server(url: str, api_key: str) -> None:
    probe_server(url, api_key)


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Register the ``--chandra-*`` arguments of the ocrmypdf command line.

    The Python API stores an unknown keyword such as ``chandra_server_url``
    without checking it against these arguments.
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
        "--chandra-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        dest="chandra_timeout",
        metavar="SECONDS",
        help=f"Seconds each wait for data from the server may last before the document fails, "
        f"and no limit for the whole request (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    chandra.add_argument(
        "--chandra-content-format",
        choices=list(CONTENT_FORMATS),
        default=DEFAULT_CONTENT_FORMAT,
        dest="chandra_content_format",
        help="Document content stored by paperless: plain text (default) or markdown.",
    )


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate the options and probe the inference server unless the run skips pages with text."""
    fmt = content_format(options)
    if fmt not in CONTENT_FORMATS:
        # argparse validates the CLI path via `choices=`; the Python API
        # (ocrmypdf.ocr(chandra_content_format=...)) skips that check.
        raise ValueError(
            f"Unknown chandra_content_format={fmt!r}. Valid choices: {list(CONTENT_FORMATS)}."
        )

    try:
        settings = request_settings(options)
        # A run with skip_text may send no page to the server, and a page that
        # does go there reports a server that is down.
        if not getattr(options, "skip_text", False):
            _probe_server(settings.url, settings.api_key)
    except ChandraClientError as e:
        raise MissingDependencyError(str(e)) from e


def _form_xobjects(holder: Any, seen: set[tuple[int, int]]) -> Iterator[Any]:
    """Each Form XObject that a page or a form can draw, each one once.

    These are the forms in its resources and the transparency group of each
    soft mask there.
    """
    # The page tree is not followed: the hook opens the document with
    # inherit_page_attributes, so each page holds the resources it inherits.
    resources = holder.get("/Resources", {})
    # Read from the dictionaries, because Page.form_xobjects of pikepdf finds
    # nothing in a stream without /Subtype /Form, and poppler draws a mask
    # group without it.
    forms = [
        xobject
        for xobject in resources.get("/XObject", {}).values()
        # A missing /Subtype counts as /Form. Ghostscript draws such a stream
        # as a form when the PDF/A conversion goes on after an error.
        if xobject.get("/Subtype", pikepdf.Name.Form) == pikepdf.Name.Form
    ]
    for state in resources.get("/ExtGState", {}).values():
        mask = state.get("/SMask")
        # The name /None in place of the dictionary stands for no mask.
        if isinstance(mask, pikepdf.Dictionary):
            forms.append(mask.G)
    for form in forms:
        if form.objgen not in seen:
            seen.add(form.objgen)
            yield form
            yield from _form_xobjects(form, seen)


def _keeps_text(pdf: Any, page: Any) -> bool:
    """Whether a redo run can leave text of the input on a page.

    ocrmypdf strips the invisible text of a page before it adds the new layer,
    and its releases differ in the text that they strip. So the stripper of the
    installed release runs here on the open document, which changes no file,
    and an operator that shows text afterwards is text that the archive keeps.
    The stripper reads the content of the page and the forms that the content
    draws. It does not reach the group of a soft mask, which pdftotext reads,
    so each operator there counts. The same holds for a stream without
    /Subtype, which Ghostscript can draw as a form. A page that has an
    annotation counts whatever the annotation holds.
    """
    # A reader can draw an annotation from its entries without a content
    # stream, and ocrmypdf strips no annotation, so a page that has one counts.
    # An empty array holds none.
    if page.obj.get("/Annots"):
        return True
    # A private function of ocrmypdf. A release that renames it fails this
    # import inside the guard of the hook, which costs the report. At the head
    # of the module the same failure would cost the plugin.
    from ocrmypdf._graft import strip_invisible_text

    strip_invisible_text(pdf, page)
    streams = [page, *_form_xobjects(page.obj, set())]
    return any(pikepdf.parse_content_stream(stream, _SHOWS_TEXT) for stream in streams)


def _masked_pages(pdfinfo: Any, options: Any) -> list[int] | None:
    """The pages, counted from 1, on which the archive can keep text of the input.

    ``None`` stands for a run that gets no report. A forced run turns each page
    into an image, which leaves no text of the input. A redo run keeps the
    visible text of a page, which ocrmypdf blanks on the image for the engine
    (``create_ocr_image``), and the invisible text that its stripper leaves in
    place. The sidecar holds neither. A page with an annotation or with text in
    a soft mask is listed as well, even where the page shows such text and the
    engine reads it: the page test does not look at an annotation and does not
    tell visible text in a mask from the invisible kind.
    """
    if getattr(options, "force_ocr", False):
        return []
    if not getattr(options, "redo_ocr", False):
        return None
    with pikepdf.open(pdfinfo.filename, inherit_page_attributes=True) as pdf:
        return [
            number
            for number, (info, page) in enumerate(
                zip(pdfinfo.pages, pdf.pages, strict=True), start=1
            )
            if info is None
            or any(True for _ in info.get_textareas(visible=True, corrupt=None))
            or _keeps_text(pdf, page)
        ]


# trylast, so that the mode is the one that the validate hook of each other
# plugin leaves behind.
@ocrmypdf.hookimpl(trylast=True)
def validate(pdfinfo: Any, options: Any) -> None:
    """Report to the parser the pages on which the archive can keep text of the input.

    In redo mode the parser stores the markdown sidecar only when the report
    lists no page, so a run without a report keeps the text of the archive.
    """
    path = getattr(options, "chandra_mask_report", None)
    if not path:
        return
    try:
        if content_format(options) != "markdown":
            return
        release = tuple(int(part) for part in ocrmypdf.__version__.split(".")[:2])
        if release < _STRIPS_NESTED_TEXT:
            return
        masked = _masked_pages(pdfinfo, options)
        if masked is not None:
            Path(path).write_text(json.dumps({"masked": masked}), encoding="utf-8")
    except Exception:
        log.exception(
            "Could not report the pages that keep text of the input. "
            "The stored content stays the text of the archive."
        )


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine() -> Any:
    """Return the Chandra engine.

    ``tryfirst=True`` makes this hookimpl run before the built-in
    Tesseract plugin's ``get_ocr_engine``; ocrmypdf's ``firstresult=True``
    policy then short-circuits, so Tesseract never claims the engine.
    """
    return ChandraEngine()
