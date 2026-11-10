"""Smoke test executed inside the paperless-ngx Docker image.

This is *not* collected by pytest (filename is not ``test_*``); it runs
with the real paperless-ngx runtime on ``PYTHONPATH``. Its job is to catch
the failure modes the lightweight unit suite cannot:

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


def check_parser_contract() -> None:
    import tempfile
    from pathlib import Path

    from django.test import override_settings
    from paperless.parsers.tesseract import RasterisedDocumentParser
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
                f"the user's own plugin should follow the Chandra module, got {args.get('plugins')!r}"
            )
        _ok("user arguments keep their own plugins after the Chandra module")

    if PaperlessChandraParser.uses_remote_service is not True:
        _fail("PaperlessChandraParser.uses_remote_service should be True")
    _ok("parser declares itself a remote service")


def main() -> None:
    print("== paperless-chandra smoke test ==", flush=True)
    check_entry_point()
    check_plugin_wiring()
    check_parser_imports()
    check_parser_contract()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
