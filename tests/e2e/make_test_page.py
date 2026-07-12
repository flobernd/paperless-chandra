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
