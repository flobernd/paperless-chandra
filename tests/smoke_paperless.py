"""Smoke test executed inside the paperless-ngx Docker image.

This is *not* collected by pytest, because its filename is not ``test_*``.
It runs with the real paperless-ngx runtime on ``PYTHONPATH``. Its job is to
catch the failure modes the lightweight unit suite cannot:

* the ``paperless_ngx.parsers`` entry point is actually discoverable, and
  resolves to :class:`PaperlessChandraParser`.
* ``paperless_chandra.parser`` imports cleanly against the *real*
  paperless-ngx API surface (``paperless.parsers.tesseract``,
  ``paperless.version``) - i.e. paperless has not renamed/moved anything
  the parser depends on.
* ``paperless_chandra.ocrmypdf_plugin`` imports and the ocrmypdf hookimpl
  wiring binds our engine.
* the upstream builder, with the override on top, hands ocrmypdf the
  Chandra plugin once, no API key and the user's ``chandra_*`` values.
* the parser registry picks the parser for a PDF and passes it over when
  remote parsers are not allowed.
* one page goes through the parser, ocrmypdf and the plugin to a stub
  server and comes back as text in both content formats and as an archive
  PDF, and a server that is down at the probe or stops after it fails the
  parse with its URL.
* a redo installation with the markdown format stores the markdown sidecar
  for a scan, and keeps the text of the archive where ocrmypdf skipped a
  page or left old text in the archive.

Exit code is non-zero on the first failed check so callers fail loudly.
"""

from __future__ import annotations

import sys

#: The words of the old text layer that the redo checks give a scan.
OLD_LAYER = "OLDLAYER zebra quartz"


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
    # Importing this module exercises every `from paperless...` import
    # against the real paperless-ngx install.
    import django

    django.setup()

    from paperless_chandra.parser import PaperlessChandraParser

    mimes = PaperlessChandraParser.supported_mime_types()
    if "application/pdf" not in mimes:
        _fail(f"application/pdf missing from supported_mime_types(): {mimes}")
    score = PaperlessChandraParser.score("application/pdf", "doc.pdf")
    if score is None or score <= 10:
        _fail(f"score() should beat Tesseract's 10, got {score!r}")
    _ok(f"parser imports against real paperless-ngx with score {score}")


def check_parser_contract() -> None:
    import inspect
    import tempfile
    from pathlib import Path

    from django.test import override_settings
    from paperless.parsers.registry import get_parser_registry
    from paperless.parsers.tesseract import RasterisedDocumentParser
    from paperless.version import __version__ as host_version
    from PIL import Image

    from paperless_chandra.parser import PaperlessChandraParser

    if not issubclass(PaperlessChandraParser, RasterisedDocumentParser):
        _fail("PaperlessChandraParser does not subclass RasterisedDocumentParser")
    if "construct_ocrmypdf_parameters" not in PaperlessChandraParser.__dict__:
        _fail("PaperlessChandraParser does not override construct_ocrmypdf_parameters")
    _ok("parser subclasses RasterisedDocumentParser and overrides the builder")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        png = tmp / "page.png"
        Image.new("RGB", (300, 300), "white").save(png, dpi=(72, 72))

        with PaperlessChandraParser() as parser:
            args = parser.construct_ocrmypdf_parameters(
                png, "image/png", tmp / "out.pdf", tmp / "sidecar.txt"
            )
        if args.get("plugins") != ["paperless_chandra.ocrmypdf_plugin"]:
            _fail(f"plugins should be the Chandra module alone, got {args.get('plugins')!r}")
        if "chandra_api_key" in args:
            _fail("chandra_api_key is in the ocrmypdf arguments, which parse() logs")
        if args.get("chandra_timeout") != 600:
            _fail(f"chandra_timeout should default to 600, got {args.get('chandra_timeout')!r}")
        if args.get("chandra_content_format") != "text":
            _fail(
                "chandra_content_format should default to 'text', "
                f"got {args.get('chandra_content_format')!r}"
            )
        _ok("builder attaches the Chandra plugin with the default chandra_* options")

        with PaperlessChandraParser() as parser:
            args = parser.construct_ocrmypdf_parameters(
                png, "image/png", tmp / "out.pdf", tmp / "sidecar.txt", skip_text=True
            )
        if args.get("skip_text") is not True:
            _fail(f"the builder should hand skip_text to the parent, got {args.get('skip_text')!r}")
        _ok("builder hands its keywords to the parent")

        # paperless reads PAPERLESS_OCR_USER_ARGS into settings once at
        # import, so the user arguments reach OcrConfig through the setting.
        user_args = (
            '{"chandra_model_name": "custom", "chandra_api_key": "leak", '
            '"plugins": "paperless_chandra.ocrmypdf_plugin"}'
        )
        with override_settings(OCR_USER_ARGS=user_args), PaperlessChandraParser() as parser:
            args = parser.construct_ocrmypdf_parameters(
                png, "image/png", tmp / "out.pdf", tmp / "sidecar.txt"
            )
        if args.get("chandra_model_name") != "custom":
            _fail(
                f"the user's chandra_model_name should win, got {args.get('chandra_model_name')!r}"
            )
        if "chandra_api_key" in args:
            _fail("the user's chandra_api_key reached the ocrmypdf arguments, which parse() logs")
        if args.get("plugins") != ["paperless_chandra.ocrmypdf_plugin"]:
            _fail(
                "a bare plugins string should list the Chandra module once, "
                f"got {args.get('plugins')!r}"
            )
        _ok("user arguments keep their chandra_* values, lose the key and list the plugin once")

        # The builder imports no plugin, so other_plugin need not exist.
        user_args = '{"plugins": ["paperless_chandra.ocrmypdf_plugin", "other_plugin"]}'
        with override_settings(OCR_USER_ARGS=user_args), PaperlessChandraParser() as parser:
            args = parser.construct_ocrmypdf_parameters(
                png, "image/png", tmp / "out.pdf", tmp / "sidecar.txt"
            )
        if args.get("plugins") != ["paperless_chandra.ocrmypdf_plugin", "other_plugin"]:
            _fail(
                "the user's own plugin should follow the Chandra module, "
                f"got {args.get('plugins')!r}"
            )
        _ok("user arguments keep their own plugins after the Chandra module")

    registry = get_parser_registry()
    chosen = registry.get_parser_for_file("application/pdf", "doc.pdf")
    if chosen is not PaperlessChandraParser:
        _fail(f"the registry should pick the Chandra parser for a PDF, got {chosen!r}")
    # paperless-ngx 3.0 has no filter for remote parsers. A later release
    # without the keyword fails, or the latest leg would stop checking the flag.
    if "allow_remote" in inspect.signature(registry.get_parser_for_file).parameters:
        local = registry.get_parser_for_file("application/pdf", "doc.pdf", allow_remote=False)
        if local is PaperlessChandraParser:
            _fail("allow_remote=False should keep a PDF away from the Chandra parser")
        _ok(
            "the registry picks the Chandra parser for a PDF, "
            "and passes it over without remote parsers"
        )
    elif host_version[:2] == (3, 0):
        _ok("the registry picks the Chandra parser for a PDF")
    else:
        host = ".".join(map(str, host_version))
        _fail(
            f"the registry of paperless-ngx {host} has no allow_remote keyword, "
            "so the check cannot ask it to leave out remote parsers"
        )


def check_document_run() -> None:
    import os
    import tempfile
    from pathlib import Path
    from unittest import mock

    import pikepdf
    from documents.parsers import ParseError
    from e2e.make_test_page import write_page
    from e2e.stub_chandra import STUB_MARKER, STUB_PHRASE, serve_in_thread

    from paperless_chandra.engine.client import ChandraClientError
    from paperless_chandra.parser import PaperlessChandraParser

    with tempfile.TemporaryDirectory() as tmp_dir:
        png = write_page(Path(tmp_dir) / "page.png")

        server, port = serve_in_thread()
        try:
            stub_url = {"PAPERLESS_CHANDRA_SERVER_URL": f"http://127.0.0.1:{port}"}
            with mock.patch.dict(os.environ, stub_url), PaperlessChandraParser() as parser:
                parser.parse(png, "image/png")
                if STUB_MARKER not in parser.get_text():
                    _fail(f"the page text lacks {STUB_MARKER!r}: {parser.get_text()!r}")
                _ok("one page goes through ocrmypdf and the Chandra engine to the stub server")

                archive = parser.get_archive_path()
                if archive is None:
                    _fail("parse() produced no archive PDF")
                with pikepdf.open(archive) as pdf:
                    creator = str(pdf.docinfo.get("/Creator", ""))
                if "Chandra" not in creator:
                    _fail(f"the archive PDF's /Creator should name Chandra, got {creator!r}")
                _ok(f"the archive PDF names its creator: {creator}")

                # Only pdftotext shows whether the words of a paragraph keep their
                # spaces, because it finds a space by the shape of the glyphs.
                archive_text = parser.extract_text(None, archive) or ""
                if STUB_PHRASE not in archive_text:
                    _fail(f"the archive text lacks {STUB_PHRASE!r}: {archive_text!r}")
                _ok("the archive text keeps the spaces between the words of a paragraph")

            markdown = {**stub_url, "PAPERLESS_CHANDRA_CONTENT_FORMAT": "markdown"}
            with mock.patch.dict(os.environ, markdown), PaperlessChandraParser() as parser:
                parser.parse(png, "image/png")
                if "# PAPERLESS CHANDRA STUB OK" not in parser.get_text():
                    _fail(f"the markdown content lacks the stub's heading: {parser.get_text()!r}")
            _ok("the markdown content format reaches the engine")
        finally:
            server.shutdown()
            server.server_close()

        # The probe of the stub's URL succeeded above and stays cached, so this
        # run gets past check_options and fails on the page request.
        with mock.patch.dict(os.environ, stub_url), PaperlessChandraParser() as parser:
            try:
                parser.parse(png, "image/png")
            except ParseError as e:
                if not isinstance(e.__cause__, ChandraClientError):
                    _fail(f"the parse should fail on the page request, got {e!r}")
                if f"127.0.0.1:{port}" not in str(e):
                    _fail(f"the ParseError should name the server, got {e}")
            else:
                _fail("parse() should raise ParseError when the server stops after the probe")
        _ok("a page request to a stopped server fails the parse with a ParseError that names it")

        # Port 9 belongs to the discard service, which nothing runs here.
        stopped_url = {"PAPERLESS_CHANDRA_SERVER_URL": "http://127.0.0.1:9"}
        with mock.patch.dict(os.environ, stopped_url), PaperlessChandraParser() as parser:
            try:
                parser.parse(png, "image/png")
            except ParseError as e:
                if "127.0.0.1:9" not in str(e):
                    _fail(f"the ParseError should name the server, got {e}")
            else:
                _fail("parse() should raise ParseError when the server is down")
        _ok("a stopped server fails the parse with a ParseError that names it")


def _write_scan(path, pages: int = 1):
    """Write a PDF whose pages are images of the test page and hold no text."""
    from e2e.make_test_page import write_page
    from PIL import Image

    with Image.open(write_page(path.with_suffix(".png"))) as page:
        page.save(path, "PDF", resolution=300, save_all=True, append_images=[page] * (pages - 1))
    return path


def _add_old_layer(pdf_path, *, page: int = 0, where: str = "page", kept: bool = False) -> None:
    """Give a page the words of OLD_LAYER in a place where its image does not show them.

    ``where`` names the place. Each place but ``annotation`` holds the words
    as the invisible text layer of an earlier OCR run. Text rendering mode 3
    of the PDF specification draws nothing, which is how a text layer hides
    its words. ocrmypdf strips such a layer in a redo run. With ``kept`` the
    text object leaves that mode and shows a space before it ends, and
    ocrmypdf then leaves its words in the archive.

    * ``annotation``: a FreeText annotation without an appearance stream. A
      reader draws its words from its entries, here in white.
    * ``form``: the layer in a Form XObject that a second Form XObject draws.
    * ``mask``: the layer in a form that the transparency group of a soft mask
      draws. The group has no ``/Subtype``, which a reader of the archive does
      not need to draw it. ocrmypdf strips nothing there.
    * ``page``: the layer in the content of the page. The page also gets an
      empty ``/Annots`` array, which is no annotation. The check for the
      markdown of a scan needs the array, because it fails for a rule that
      takes the entry itself for an annotation.
    * ``stream``: the layer in a stream without ``/Subtype`` that the page
      draws like a form. ocrmypdf strips nothing there, and Ghostscript draws
      the stream as a form when the PDF/A conversion goes on after an error.
    """
    import pikepdf

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        target = pdf.pages[page]
        if where == "annotation":
            note = pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.FreeText,
                Rect=[72, 72, 372, 122],
                # The print flag. The PDF/A conversion drops an annotation without it.
                F=4,
                Contents=pikepdf.String(OLD_LAYER),
                # White, so that the page image does not show the words.
                DA=pikepdf.String("/Helv 14 Tf 1 g"),
            )
            target.obj.Annots = pikepdf.Array([pdf.make_indirect(note)])
            pdf.save(pdf_path)
            return

        visible_end = "0 Tr ( ) Tj " if kept else ""
        ops = f"BT /FSmoke 14 Tf 3 Tr 72 72 Td ({OLD_LAYER}) Tj {visible_end}ET\n"
        font = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica,
                Encoding=pikepdf.Name.WinAnsiEncoding,
            )
        )
        fonts = pikepdf.Dictionary(FSmoke=font)

        def form(content: bytes, **resources):
            stream = pdf.make_stream(content)
            stream.Type = pikepdf.Name.XObject
            stream.Subtype = pikepdf.Name.Form
            stream.BBox = target.mediabox
            stream.Resources = pikepdf.Dictionary(**resources)
            return stream

        if where == "form":
            inner = form(ops.encode("ascii"), Font=fonts)
            outer = form(b"q /Inner Do Q\n", XObject=pikepdf.Dictionary(Inner=inner))
            target.obj.Resources.XObject.OldLayer = outer
            target.contents_add(pdf.make_stream(b"q /OldLayer Do Q\n"))
        elif where == "mask":
            inner = form(ops.encode("ascii"), Font=fonts)
            group = form(b"q /Inner Do Q\n", XObject=pikepdf.Dictionary(Inner=inner))
            del group.Type, group.Subtype
            group.Group = pikepdf.Dictionary(
                S=pikepdf.Name.Transparency, CS=pikepdf.Name.DeviceGray
            )
            mask = pikepdf.Dictionary(S=pikepdf.Name.Luminosity, G=group)
            target.obj.Resources.ExtGState = pikepdf.Dictionary(
                OldLayer=pikepdf.Dictionary(SMask=mask)
            )
            # The PDF/A conversion drops a mask that nothing is painted through.
            target.contents_add(pdf.make_stream(b"q /OldLayer gs 1 g 0 0 1 1 re f Q\n"))
        elif where == "stream":
            layer = form(ops.encode("ascii"), Font=fonts)
            del layer.Type, layer.Subtype
            target.obj.Resources.XObject.OldLayer = layer
            target.contents_add(pdf.make_stream(b"q /OldLayer Do Q\n"))
        else:
            target.obj.Resources.Font = fonts
            target.contents_add(pdf.make_stream(b"q " + ops.encode("ascii") + b"Q\n"))
            target.obj.Annots = pikepdf.Array()
        pdf.save(pdf_path)


def _add_form_field(pdf_path) -> None:
    """Give the first page a fillable text field."""
    import pikepdf

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        page = pdf.pages[0]
        field = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.Widget,
                FT=pikepdf.Name.Tx,
                T=pikepdf.String("name"),
                Rect=[72, 100, 372, 122],
                P=page.obj,
            )
        )
        page.obj.Annots = pikepdf.Array([field])
        pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([field]))
        pdf.save(pdf_path)


def check_redo_content() -> None:
    import os
    import tempfile
    from pathlib import Path
    from unittest import mock

    from django.test import override_settings
    from documents.parsers import ParseError
    from e2e.stub_chandra import STUB_MARKER, serve_in_thread
    from paperless.version import __version__ as host_version

    from paperless_chandra.parser import PaperlessChandraParser

    server, port = serve_in_thread()
    markdown = {
        "PAPERLESS_CHANDRA_CONTENT_FORMAT": "markdown",
        "PAPERLESS_CHANDRA_SERVER_URL": f"http://127.0.0.1:{port}",
    }

    def redo(pdf, **ocr_settings) -> tuple[str, str]:
        """The stored content and the pdftotext text of the archive of one redo parse."""
        with (
            override_settings(OCR_MODE="redo", **ocr_settings),
            mock.patch.dict(os.environ, markdown),
            PaperlessChandraParser() as parser,
        ):
            try:
                parser.parse(pdf, "application/pdf")
            except ParseError as e:
                _fail(f"the redo parse of {pdf.name} failed: {e}")
            archive_text = parser.extract_text(None, parser.get_archive_path()) or ""
            return parser.get_text(), archive_text

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ocrmypdf leaves a page outside the page range as it is, old text layer
            # included, and the plugin learns that from the marker in the sidecar alone.
            skipped = _write_scan(tmp / "skipped.pdf", pages=2)
            _add_old_layer(skipped, page=1)
            content, _ = redo(skipped, OCR_USER_ARGS='{"pages": "1"}')
            if OLD_LAYER not in content:
                _fail(
                    f"the redo content lacks the text of a page that ocrmypdf skipped: {content!r}"
                )
            _ok("redo keeps the text of a page that ocrmypdf skipped")

            # With a page limit paperless-ngx asks ocrmypdf for no sidecar at all.
            content, _ = redo(skipped, OCR_PAGES=1)
            if OLD_LAYER not in content:
                _fail(f"the redo content lacks the text of a page past the page limit: {content!r}")
            _ok("redo keeps the text of a page past the page limit")

            # paperless-ngx ships ocrmypdf 17.11 or newer from 3.2 on. With an older
            # ocrmypdf, redo keeps the text of the archive for each document.
            if host_version[:2] >= (3, 2):
                # The sidecar never holds old text that ocrmypdf leaves in the archive.
                # ocrmypdf strips no annotation, no soft mask and no stream without
                # /Subtype, so the words stay there without the visible end of a kept layer.
                # The PDF/A conversion fails at a stream without /Subtype unless this user
                # argument lets it go on, which paperless-ngx advises in its log.
                go_on = {"OCR_USER_ARGS": '{"continue_on_soft_render_error": true}'}
                for where, place, kept, ocr_settings in (
                    ("page", "the page", True, {}),
                    ("form", "a Form XObject", True, {}),
                    ("annotation", "an annotation", False, {}),
                    ("mask", "a soft mask", False, {}),
                    ("stream", "a stream without /Subtype", False, go_on),
                ):
                    scan = _write_scan(tmp / f"kept-{where}.pdf")
                    _add_old_layer(scan, where=where, kept=kept)
                    content, archive_text = redo(scan, **ocr_settings)
                    if OLD_LAYER not in archive_text:
                        _fail(
                            f"the archive text lacks the old text in {place}, "
                            f"so this check needs another input: {archive_text!r}"
                        )
                    if OLD_LAYER not in content:
                        _fail(
                            f"the redo content lacks old text that ocrmypdf left in {place}: "
                            f"{content!r}"
                        )
                    _ok(f"redo keeps old text that ocrmypdf leaves in {place}")

                layered = _write_scan(tmp / "layered.pdf", pages=2)
                _add_old_layer(layered)
                content, _ = redo(layered)
                # The blank line ahead of the heading comes only from the end of the page before
                # it, so the check needs a scan of two pages and both line breaks.
                if f"\n\n# {STUB_MARKER}" not in content:
                    _fail(
                        f"the redo content of a scan is not the markdown of each page: {content!r}"
                    )
                _ok("redo stores the markdown of a scan with an old invisible text layer")

                # ocrmypdf refuses a redo run on a PDF with a form, and paperless-ngx
                # then forces OCR, which hands each page to the engine whole.
                form = _write_scan(tmp / "form.pdf")
                _add_form_field(form)
                content, _ = redo(form)
                if f"# {STUB_MARKER}" not in content:
                    _fail(f"the redo content of a PDF with a form is not markdown: {content!r}")
                _ok("redo stores the markdown of a PDF with a form field")
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    print("== paperless-chandra smoke test ==", flush=True)
    check_entry_point()
    check_plugin_wiring()
    check_parser_imports()
    check_parser_contract()
    check_document_run()
    check_redo_content()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
