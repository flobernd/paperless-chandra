"""Poll the paperless API until the ingested document(s) appear, then assert.

stdlib + pypdf + PIL only (both are dev dependencies of the repo venv). Auth
uses the admin credentials the compose stack creates via PAPERLESS_ADMIN_USER.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request

from PIL import Image
from pypdf import PdfReader

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


#: The test page carries all its text in the upper part of the page (see
#: make_test_page.py), so a correctly-oriented archive must be top-heavy.
#: An uncorrected upside-down page puts the centroid well above 0.55, which
#: gives this threshold a comfortable margin in both directions.
_UPRIGHT_CENTROID_MAX = 0.45


def _ink_centroid_y_fraction(image: Image.Image) -> float:
    """Vertical position (0..1) of the image's ink-weighted centroid.

    Collapses each row to its mean brightness via a box-filter resize to
    width 1, then weights by darkness (255 - value). Row width is constant
    across the image, so it cancels out of the weighted average: this gives
    the same result as summing every pixel, without the pure-Python
    per-pixel loop.
    """
    gray = image.convert("L")
    height = gray.size[1]
    column = gray.resize((1, height), Image.Resampling.BOX)
    # tobytes() on a mode-L image yields one byte per pixel top to bottom;
    # getdata() would do the same but is deprecated for removal in Pillow 14.
    weights = [255 - v for v in column.tobytes()]
    total = sum(weights)
    if total == 0:
        raise SystemExit("page has no ink to compute an orientation centroid from")
    weighted = sum(w * y for y, w in enumerate(weights))
    return weighted / total / height


def _assert_upright(label: str, reader: PdfReader) -> None:
    page = reader.pages[0]
    images = list(page.images)
    if not images:
        raise SystemExit(f"{label}: archive PDF page 0 has no embedded images")
    image = max(images, key=lambda im: im.image.size[0] * im.image.size[1]).image

    rotation = page.rotation
    if rotation not in (0, 90, 180, 270):
        raise SystemExit(f"{label}: unexpected page rotation {rotation!r}")
    if rotation:
        # PDF /Rotate is clockwise, PIL rotate() counter-clockwise; only 0 and
        # 180 occur here, where the two conventions agree, so the sign is moot.
        image = image.rotate(-rotation, expand=True)

    centroid = _ink_centroid_y_fraction(image)
    print(f"{label}: page rotation {rotation}, ink centroid y = {centroid:.4f}", flush=True)
    if not centroid < _UPRIGHT_CENTROID_MAX:
        raise SystemExit(
            f"{label}: expected upright page (ink centroid y < {_UPRIGHT_CENTROID_MAX}), "
            f"got {centroid:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", action="append", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--expect-docs", type=int, default=1)
    parser.add_argument("--assert-upright", action="append", default=[])
    args = parser.parse_args()

    token = _get_token(args.timeout)
    print("API up, token acquired", flush=True)

    deadline = time.time() + args.timeout
    docs: list[dict] = []
    while time.time() < deadline:
        resp = _request(f"/api/documents/?ordering=-added&page_size={args.expect_docs}", token)
        if resp.get("count", 0) >= args.expect_docs:
            docs = resp["results"]
            break
        time.sleep(10)
    if not docs:
        raise SystemExit(f"expected {args.expect_docs} document(s), none appeared in time")

    seen_titles = set()
    for doc in docs:
        label = f"document {doc['id']} ({doc.get('title')!r})"
        seen_titles.add(doc.get("title"))

        # PDF text extraction and OCR post-processing can vary spacing, so
        # matching happens on whitespace-normalised text.
        content = " ".join((doc.get("content") or "").split())
        print(f"{label} content: {content[:200]!r}", flush=True)
        for needle in args.expect:
            if needle not in content:
                raise SystemExit(f"expected {needle!r} in {label} content")

        archive = _request(f"/api/documents/{doc['id']}/download/", token, raw=True)
        if not archive.startswith(b"%PDF"):
            raise SystemExit(f"{label}: downloaded archive is not a PDF")
        reader = PdfReader(io.BytesIO(archive))

        text = "".join(p.extract_text() or "" for p in reader.pages)
        text = " ".join(text.split())
        print(f"{label} archive text layer: {text[:200]!r}", flush=True)
        for needle in args.expect:
            if needle not in text:
                raise SystemExit(f"expected {needle!r} in {label} archive PDF text layer")

        if doc.get("title") in args.assert_upright:
            _assert_upright(label, reader)
    print("content and archive text layer assertions passed for all documents", flush=True)

    missing = [t for t in args.assert_upright if t not in seen_titles]
    if missing:
        raise SystemExit(f"--assert-upright title(s) not found among fetched documents: {missing}")

    sys.exit(0)


if __name__ == "__main__":
    main()
