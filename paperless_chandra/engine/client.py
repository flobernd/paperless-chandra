"""Page request to the Chandra inference server.

chandra-ocr knows the model: the prompt, the image scaling, the base64
encoding and the repeat detection. The OpenAI SDK knows the transport.
The library's batch call leaves the timeout at the SDK default and
retries every failure the same way, so the request lives here, where the
plugin sets the timeout and the retry policy.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from chandra.model.util import detect_repeat_token, scale_to_fit
from chandra.model.vllm import image_to_base64
from chandra.prompts import PROMPT_MAPPING
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from PIL import Image

from paperless_chandra import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TIMEOUT_SECONDS,
)

log = logging.getLogger("paperless.chandra.client")

_CONNECTION_RETRIES = 2
_REPEAT_RETRIES = 2

_THINKING_HINT = (
    "The inference server is likely applying a thinking-enabled chat "
    "template (chandra-ocr-2 GGUF conversions embed one). Restart "
    "llama-server with --chat-template-kwargs '{\"enable_thinking\":false}' "
    "so the OCR output is returned as regular message content."
)

_FILTER_HINT = "Adjust the content filter of the inference server or of the gateway in front of it."

#: Warn about a thinking-wrapped response once per process, not once per page.
_think_warned = False


class ChandraClientError(RuntimeError):
    """Chandra inference failed or is misconfigured."""


@dataclass(frozen=True)
class RequestSettings:
    """Server, model and limits for one page request."""

    url: str
    model: str
    api_key: str
    max_tokens: int
    timeout: int


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
        log.warning("%s. %s", reason, _THINKING_HINT)
        _think_warned = True
    else:
        log.debug("%s (thinking-template warning already emitted).", reason)


def normalize_server_url(raw: str) -> str:
    """Normalised OpenAI-compatible base URL.

    The OpenAI SDK appends ``/chat/completions`` to the base URL, and
    servers route under ``/v1``. Accept the URL with or without the version
    suffix.
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


def resolve_api_key(options: Any) -> str:
    """Bearer token for the server.

    The option wins when set. The environment covers ocrmypdf options that
    do not carry the key, and ``EMPTY`` is the vLLM convention for a server
    without auth.
    """
    option = (getattr(options, "chandra_api_key", "") or "").strip()
    return option or os.environ.get("PAPERLESS_CHANDRA_API_KEY", "").strip() or "EMPTY"


def request_settings(options: Any) -> RequestSettings:
    """Settings from the ``chandra_*`` options, with the package defaults."""
    return RequestSettings(
        url=normalize_server_url(getattr(options, "chandra_server_url", "") or ""),
        model=(getattr(options, "chandra_model_name", "") or "").strip() or DEFAULT_MODEL_NAME,
        api_key=resolve_api_key(options),
        max_tokens=(
            int(getattr(options, "chandra_max_output_tokens", 0) or 0) or DEFAULT_MAX_OUTPUT_TOKENS
        ),
        timeout=int(getattr(options, "chandra_timeout", 0) or 0) or DEFAULT_TIMEOUT_SECONDS,
    )


def _has_unclosed_think(raw: str) -> bool:
    return "<think>" in raw and "</think>" not in raw


def _is_repeat_loop(raw: str) -> bool:
    # The library's own pair of checks. The second skips the last 50
    # characters, so it also finds a loop the model closed with end tags.
    return bool(
        detect_repeat_token(raw) or (len(raw) > 50 and detect_repeat_token(raw, cut_from_end=50))
    )


def _status_error(error: APIStatusError, settings: RequestSettings) -> ChandraClientError:
    status = error.status_code
    if status in (401, 403):
        return ChandraClientError(
            f"The Chandra server at {settings.url} rejected the API key (HTTP {status}). "
            "Check PAPERLESS_CHANDRA_API_KEY against the server configuration."
        )
    if status == 404:
        return ChandraClientError(
            f"The Chandra server at {settings.url} does not serve the model "
            f"{settings.model!r} (HTTP 404: {error.message}). Set "
            "PAPERLESS_CHANDRA_MODEL_NAME to the name the server advertises."
        )
    if status == 400:
        return ChandraClientError(
            f"The Chandra server at {settings.url} rejected the request (HTTP 400): "
            f"{error.message}. A PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS budget above the "
            "model's context window is a common cause."
        )
    return ChandraClientError(
        f"The Chandra server at {settings.url} answered HTTP {status}: {error.message}"
    )


def ocr_image(image: Image.Image, options: Any) -> str:
    """OCR one page image via the remote server. The result is Chandra's raw HTML."""
    settings = request_settings(options)
    # The SDK's own retries would run beneath the policy below and stretch
    # one page to several timeouts. Without the block, the connection of a
    # page stays open until the garbage collector reaches the client.
    with OpenAI(
        api_key=settings.api_key,
        base_url=settings.url,
        timeout=settings.timeout,
        max_retries=0,
    ) as client:
        image_url = f"data:image/png;base64,{image_to_base64(scale_to_fit(image))}"
        temperature, top_p = 0.0, 0.1
        connection_retries = repeat_retries = 0
        while True:
            try:
                completion = client.chat.completions.create(
                    model=settings.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": PROMPT_MAPPING["ocr_layout"]},
                            ],
                        }
                    ],
                    max_tokens=settings.max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            # APITimeoutError subclasses APIConnectionError, so it is caught first.
            except APITimeoutError as e:
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} did not answer within "
                    f"{settings.timeout} seconds. Raise PAPERLESS_CHANDRA_TIMEOUT for slow "
                    "inference. Lower PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS only for a generation "
                    "that repeats itself until the timeout, so the plugin gets the output in time "
                    "to retry it."
                ) from e
            except APIConnectionError as e:
                cause = e.__cause__ or e
                if connection_retries == _CONNECTION_RETRIES:
                    raise ChandraClientError(
                        f"The Chandra server at {settings.url} is not reachable ({cause}). "
                        "Check PAPERLESS_CHANDRA_SERVER_URL and that the inference server "
                        "is running."
                    ) from e
                connection_retries += 1
                delay = 2 * connection_retries
                log.warning(
                    "Chandra server at %s not reachable (%s), retrying in %d s.",
                    settings.url,
                    cause,
                    delay,
                )
                time.sleep(delay)
                continue
            except APIStatusError as e:
                raise _status_error(e, settings) from e

            # The SDK does not validate a 200 body: an HTML page arrives as a str,
            # and a JSON body without choices as a completion whose choices are None.
            choices = getattr(completion, "choices", None)
            if not choices:
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} answered without a chat completion. "
                    "Check that PAPERLESS_CHANDRA_SERVER_URL points at an OpenAI-compatible "
                    "inference server."
                )
            choice = choices[0]
            # A document must not be stored without the text of a filtered or
            # refused page. These checks precede the loop check, so such a page is
            # not retried.
            if choice.finish_reason == "content_filter":
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} filtered the output for the page "
                    f"(finish reason content_filter). {_FILTER_HINT}"
                )
            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} refused the page ({refusal}). "
                    f"{_FILTER_HINT}"
                )
            raw = choice.message.content or ""
            token_count = completion.usage.completion_tokens if completion.usage else 0
            # A loop at temperature 0 usually runs into the budget, so the loop
            # check comes before the length check.
            if repeat_retries < _REPEAT_RETRIES and _is_repeat_loop(raw):
                repeat_retries += 1
                temperature, top_p = min(0.2 * repeat_retries, 0.8), 0.95
                log.info("Chandra output repeats, retrying at temperature %.1f.", temperature)
                continue
            if choice.finish_reason == "length":
                message = (
                    f"Chandra's output for the page hit the budget of {settings.max_tokens} "
                    "tokens and was cut off. The plugin retries a detected repetition loop "
                    "twice before it raises this error. Raise PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS "
                    "when the page holds more text than the budget allows."
                )
                # A thinking template spends the budget on reasoning, and a larger
                # budget only delays the same failure.
                if not raw.strip() or _has_unclosed_think(raw):
                    message += (
                        f" The output is empty or an unclosed <think> block. {_THINKING_HINT}"
                    )
                raise ChandraClientError(message)
            break

    if _has_unclosed_think(raw):
        _warn_thinking_response("Chandra returned OCR output inside an unclosed <think> block")
    elif not raw.strip() and token_count:
        _warn_thinking_response(
            f"Chandra returned empty content although the server generated "
            f"{token_count} tokens (output likely diverted to reasoning_content)"
        )
    log.debug("Chandra returned %d output tokens for one page.", token_count)
    return raw
