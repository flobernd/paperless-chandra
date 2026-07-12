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
