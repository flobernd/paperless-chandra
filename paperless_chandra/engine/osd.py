"""Page orientation detection via tesseract's orientation-and-script detection (OSD) mode.

Chandra has no orientation classifier of its own. The official paperless-ngx image already
ships the tesseract binary and osd.traineddata for the stock Tesseract OCR engine, so this
shells out to `tesseract --psm 0` to reuse that binary as a fast, local orientation probe;
Chandra still performs all text recognition. Output parsing and failure handling mirror
ocrmypdf's own tesseract engine (`ocrmypdf._exec.tesseract`) so the reported angle and
confidence use the scale the rest of the ocrmypdf pipeline already expects.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("paperless.chandra.osd")

#: OSD only classifies orientation, a fraction of the work of a full OCR pass; the bound only
#: exists so a hung tesseract process cannot stall the Celery worker indefinitely.
_TIMEOUT_SECONDS = 60.0

#: Warn about an unavailable/broken tesseract OSD once per process, not once per page.
_osd_warned = False


def _parse_tesseract_output(output: bytes) -> dict[str, str]:
    """Split ``key: value`` lines, same as ocrmypdf's ``_parse_tesseract_output``."""

    def gen():
        for line in output.decode(errors="replace").splitlines():
            line = line.strip()
            parts = line.split(":", maxsplit=2)
            if len(parts) == 2:
                yield parts[0].strip(), parts[1].strip()

    return dict(gen())


def detect_orientation(input_file: Path) -> tuple[int, float]:
    """Clockwise degrees (0/90/180/270) the text is rotated, with confidence."""
    global _osd_warned
    try:
        proc = subprocess.run(
            ["tesseract", "-l", "osd", "--psm", "0", str(input_file), "stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        if not _osd_warned:
            log.warning(
                "PAPERLESS_OCR_ROTATE_PAGES is set but the tesseract binary is not available "
                "for orientation detection; pages keep their stored orientation."
            )
            _osd_warned = True
        else:
            log.debug("tesseract binary still unavailable for orientation detection.")
        return 0, 0.0
    except subprocess.CalledProcessError as e:
        output = e.output or b""
        # Benign per-page conditions tesseract reports via a non-zero exit, not real failures.
        if b"Too few characters. Skipping this page" in output or b"Image too large" in output:
            log.debug(
                "tesseract OSD skipped this page: %s", output.decode(errors="replace").strip()
            )
            return 0, 0.0
        if not _osd_warned:
            excerpt = output.decode(errors="replace").strip()[:200]
            log.warning(
                "tesseract OSD failed (exit %s); pages keep their stored orientation. Output: %s",
                e.returncode,
                excerpt,
            )
            _osd_warned = True
        else:
            log.debug("tesseract OSD failed again (exit %s).", e.returncode)
        return 0, 0.0
    except subprocess.TimeoutExpired:
        log.debug("tesseract OSD timed out after %s seconds.", _TIMEOUT_SECONDS)
        return 0, 0.0

    fields = _parse_tesseract_output(proc.stdout)
    angle = int(fields.get("Orientation in degrees", 0))
    confidence = float(fields.get("Orientation confidence", 0.0))
    return angle, confidence
