"""Poll the paperless API until the ingested document appears, then assert.

stdlib + pypdf only (pypdf is a dev dependency of the repo venv). Auth uses
the admin credentials the compose stack creates via PAPERLESS_ADMIN_USER.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", action="append", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    token = _get_token(args.timeout)
    print("API up, token acquired", flush=True)

    deadline = time.time() + args.timeout
    doc = None
    while time.time() < deadline:
        docs = _request("/api/documents/?ordering=-added&page_size=1", token)
        if docs.get("count", 0) >= 1:
            doc = docs["results"][0]
            break
        time.sleep(10)
    if doc is None:
        raise SystemExit("document never appeared in paperless")

    # PDF text extraction and OCR post-processing can vary spacing, so
    # matching happens on whitespace-normalised text.
    content = " ".join((doc.get("content") or "").split())
    print(f"document {doc['id']} content: {content[:200]!r}", flush=True)
    for needle in args.expect:
        if needle not in content:
            raise SystemExit(f"expected {needle!r} in document content")
    print("content assertions passed", flush=True)

    archive = _request(f"/api/documents/{doc['id']}/download/", token, raw=True)
    if not archive.startswith(b"%PDF"):
        raise SystemExit("downloaded archive is not a PDF")
    from pypdf import PdfReader

    text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(archive)).pages)
    text = " ".join(text.split())
    print(f"archive text layer: {text[:200]!r}", flush=True)
    for needle in args.expect:
        if needle not in text:
            raise SystemExit(f"expected {needle!r} in archive PDF text layer")
    print("archive text layer assertions passed", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
