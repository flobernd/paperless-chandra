"""Generate a simple high-contrast test page for e2e ingestion."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 300 dpi A4. Measured against the paperless-ngx image's tesseract OSD:
# at 200 dpi even a dozen body lines stayed under paperless's default
# rotate-pages confidence threshold (12.0); this resolution and line count
# clears it with margin (~16 on the rotated page) while keeping all the ink
# in the upper part of the page, which the e2e orientation assertion relies
# on to tell "rotated" from "not rotated".
_WIDTH, _HEIGHT = 2481, 3508
_BODY_BOTTOM = 1800  # keep text out of the lower half so the ink centroid stays top-heavy
_LINE_STEP = 75

_BODY_LINES = [
    "Vielen Dank fuer Ihren Einkauf bei Paperless Chandra.",
    "Bitte pruefen Sie die Rechnung auf Vollstaendigkeit.",
    "Zahlbar innerhalb von vierzehn Tagen ohne Abzug.",
    "Bei Rueckfragen wenden Sie sich an unseren Support.",
    "Diese Seite dient ausschliesslich Testzwecken.",
    "Alle Angaben ohne Gewaehr auf Richtigkeit.",
    "Der Versand erfolgte am zwoelften Juli zweitausendsechsundzwanzig.",
    "Lieferadresse entspricht der Rechnungsadresse des Kunden.",
    "Die Mehrwertsteuer ist in obigem Betrag bereits enthalten.",
    "Fuer Rueckgaben gilt eine Frist von dreissig Tagen.",
]


def main(out_path: str, rotate_deg: int = 0) -> None:
    img = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.load_default(size=108)
    font_body = ImageFont.load_default(size=72)
    draw.text((180, 300), "RECHNUNG 2026-0042", fill="black", font=font_big)
    draw.text((180, 630), "PAPERLESS CHANDRA END TO END", fill="black", font=font_body)
    draw.text((180, 810), "Betrag: 123,45 EUR", fill="black", font=font_body)
    y = 990
    for line in itertools.cycle(_BODY_LINES):
        if y >= _BODY_BOTTOM:
            break
        draw.text((180, y), line, fill="black", font=font_body)
        y += _LINE_STEP
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=False, fillcolor="white")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, dpi=(300, 300))
    print(f"wrote {out}")


if __name__ == "__main__":
    rotate = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(sys.argv[1], rotate)
