"""Minimal OpenAI-compatible stub returning canned Chandra layout HTML.

Runs on stdlib only so the compose service is a bare python:3.12-slim.
Every chat completion returns the same layout blocks; the e2e assertion
looks for STUB_MARKER in the ingested document's content.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

STUB_MARKER = "PAPERLESS CHANDRA STUB OK"

_PAGE_HTML = (
    f'<div data-label="Section-Header" data-bbox="60 40 940 120"><h1>{STUB_MARKER}</h1></div>'
    '<div data-label="Text" data-bbox="60 160 940 400">'
    "<p>Stub invoice 2026-0042<br>Second stub line</p></div>"
)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/").endswith("/models"):
            self._send_json({"object": "list", "data": [{"id": "chandra", "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        self._send_json(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": "chandra",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": _PAGE_HTML},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def log_message(self, fmt, *args):  # quiet: compose logs stay readable
        pass


if __name__ == "__main__":
    print("stub chandra server listening on :8080", flush=True)
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
