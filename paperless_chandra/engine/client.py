"""Thin wrapper around the chandra-ocr remote client.

The only module that touches the upstream package's configuration.
``chandra.settings.settings`` is a module-global pydantic object; the
model name and API key can only be passed through it (``generate_vllm``
accepts the base URL as a kwarg but reads the rest from settings at call
time). The values derive from env vars and are constant per worker
process, so re-applying them before every call is idempotent; the lock
only guards the theoretical mixed-write window between threads.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from chandra.model.schema import BatchInputItem
from chandra.model.vllm import generate_vllm
from chandra.settings import settings as chandra_settings
from PIL import Image

log = logging.getLogger("paperless.chandra.client")

DEFAULT_MODEL_NAME = "chandra"
DEFAULT_MAX_OUTPUT_TOKENS = 12384

_SETTINGS_LOCK = threading.Lock()


class ChandraClientError(RuntimeError):
    """Chandra inference failed or is misconfigured."""


def normalize_server_url(raw: str) -> str:
    """Normalised OpenAI-compatible base URL.

    chandra hands the URL to the OpenAI SDK, which appends
    ``/chat/completions``; servers route under ``/v1``. Accept the URL
    with or without the version suffix.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ChandraClientError(
            "Chandra requires PAPERLESS_CHANDRA_SERVER_URL (or "
            "--chandra-server-url) to point at an OpenAI-compatible "
            "inference server."
        )
    url = cleaned.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url = f"{url}/v1"
    return url


def ocr_image(image: Image.Image, options: Any) -> str:
    """OCR one page image via the remote server; return Chandra's raw HTML."""
    url = normalize_server_url(getattr(options, "chandra_server_url", "") or "")
    model = (getattr(options, "chandra_model_name", "") or "").strip() or DEFAULT_MODEL_NAME
    api_key = (getattr(options, "chandra_api_key", "") or "").strip() or "EMPTY"
    max_tokens = (
        int(getattr(options, "chandra_max_output_tokens", 0) or 0) or DEFAULT_MAX_OUTPUT_TOKENS
    )

    with _SETTINGS_LOCK:
        chandra_settings.VLLM_MODEL_NAME = model
        chandra_settings.VLLM_API_KEY = api_key

    results = generate_vllm(
        [BatchInputItem(image=image, prompt_type="ocr_layout")],
        max_output_tokens=max_tokens,
        vllm_api_base=url,
    )
    if not results:
        raise ChandraClientError("Chandra returned no result for the page.")
    result = results[0]
    if result.error:
        raise ChandraClientError(
            f"Chandra inference failed after retries against {url}. "
            "Check the inference server logs and the PAPERLESS_CHANDRA_* settings."
        )
    log.debug("Chandra returned %d output tokens for one page.", result.token_count)
    return result.raw or ""
