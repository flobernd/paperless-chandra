"""Minimal OpenAI-compatible stub returning canned Chandra layout HTML.

Runs on stdlib only so the compose service is a bare python:3.12-slim.
Every chat completion returns the same layout blocks; the e2e assertion
looks for STUB_MARKER in the ingested document's content.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

STUB_MARKER = "PAPERLESS CHANDRA STUB OK"

#: Words of the paragraph below, which the model returns as one reading line.
STUB_PHRASE = "several lines of running text"

_PAGE_HTML = (
    f'<div data-label="Section-Header" data-bbox="60 40 940 120"><h1>{STUB_MARKER}</h1></div>'
    '<div data-label="Text" data-bbox="60 160 940 400">'
    "<p>Stub invoice 2026-0042<br>Second stub line</p></div>"
    '<div data-label="Text" data-bbox="60 440 940 560">'
    f"<p>This paragraph stands for {STUB_PHRASE}. The model returns such a block as one "
    "reading line with the box of the whole block, and the text layer has to split it into "
    "the lines that the page shows. Each word must come back from the archive with the "
    "spaces around it, in the order in which the page holds it, so that a search for a "
    "phrase of several words finds the document again.</p></div>"
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


def serve_in_thread() -> tuple[HTTPServer, int]:
    """Serve on an ephemeral loopback port and return the server with its port.

    The thread is a daemon, so a caller that exits on a failed check does
    not wait for ``serve_forever`` to return.
    """
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


if __name__ == "__main__":
    print("stub chandra server listening on :8080", flush=True)
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
