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

#: Warn about a thinking-wrapped response once per process, not once per page.
_think_warned = False


class ChandraClientError(RuntimeError):
    """Chandra inference failed or is misconfigured."""


def _warn_thinking_response(reason: str) -> None:
    """Warn (once) that the server wrapped the OCR output in a reasoning block.

    Chandra 2 is a no-think fine-tune of a hybrid reasoning model, but GGUF
    conversions embed the base model's chat template, which turns thinking
    back on. The result reaches this client either as an unclosed ``<think>``
    wrapper in the content or as empty content (the server diverted the whole
    output to ``reasoning_content``); both parse to an empty page.
    """
    global _think_warned
    if not _think_warned:
        log.warning(
            "%s. The inference server is likely applying a thinking-enabled chat "
            "template (chandra-ocr-2 GGUF conversions embed one). Restart "
            "llama-server with --chat-template-kwargs '{\"enable_thinking\":false}' "
            "so the OCR output is returned as regular message content.",
            reason,
        )
        _think_warned = True
    else:
        log.debug("%s (thinking-template warning already emitted).", reason)


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
    raw = result.raw or ""
    if "<think>" in raw and "</think>" not in raw:
        _warn_thinking_response("Chandra returned OCR output inside an unclosed <think> block")
    elif not raw.strip() and result.token_count:
        _warn_thinking_response(
            f"Chandra returned empty content although the server generated "
            f"{result.token_count} tokens (output likely diverted to reasoning_content)"
        )
    log.debug("Chandra returned %d output tokens for one page.", result.token_count)
    return raw
