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
    from e2e.stub_chandra import STUB_MARKER, serve_in_thread

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


def main() -> None:
    print("== paperless-chandra smoke test ==", flush=True)
    check_entry_point()
    check_plugin_wiring()
    check_parser_imports()
    check_parser_contract()
    check_document_run()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
