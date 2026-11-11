"""Page request and preflight probe for the Chandra inference server.

chandra-ocr knows the model: the prompt, the image scaling, the base64
encoding and the repeat detection. The OpenAI SDK knows the transport.
The library's batch call leaves the timeout at the SDK default and
retries every failure the same way, so the request lives here, where the
plugin sets the timeout and the retry policy.
"""

from __future__ import annotations

import contextlib
import email.utils
import logging
import math
import os
import re
import time
from collections.abc import Mapping
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

# A server that is loading its model or is overloaded is not ready. It can
# answer with one of these statuses, itself or through a gateway in front of
# it, or a connection to it can fail.
_NOT_READY_STATUSES = frozenset({429, 502, 503, 504})
_NOT_READY_WAITS = (1.0, 2.0, 4.0, 8.0)
_RETRY_AFTER_CEILING = 30.0
_NOT_READY_WAIT_LIMIT = 60.0

_REJECTED_KEY_STATUSES = frozenset({401, 403})

_REPEAT_RETRIES = 2

#: A vLLM server under load answers /models slowly. The plugin keeps a
#: successful probe for the worker process, so a slow server costs the wait once.
_PROBE_TIMEOUT_SECONDS = 30

_THINKING_HINT = (
    "The inference server is likely applying a thinking-enabled chat "
    "template (chandra-ocr-2 GGUF conversions embed one). Restart "
    "llama-server with --chat-template-kwargs '{\"enable_thinking\":false}' "
    "so the OCR output is returned as regular message content."
)

_FILTER_HINT = "Adjust the content filter of the inference server or of the gateway in front of it."

_SERVER_URL_ADVICE = (
    "Check that PAPERLESS_CHANDRA_SERVER_URL points at an OpenAI-compatible inference server."
)

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
    output to ``reasoning_content``). Both parse to an empty page.
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


def text_option(options: Any, name: str) -> str:
    """The option as it was set, or ``""`` when it is absent or empty."""
    value = getattr(options, name, None)
    if not value:
        return ""
    if not isinstance(value, str):
        if name == "chandra_api_key":
            # The value is a secret, and the error reaches the log.
            raise ValueError(f"{name} must be a string.")
        raise ValueError(f"{name} must be a string, got {value!r}.")
    return value


def _int_option(options: Any, name: str, default: int) -> int:
    value = getattr(options, name, None)
    if value is None:
        return default
    # NaN fails both comparisons, so the range rejects it too.
    if not isinstance(value, (int, float)) or not 1 <= value < math.inf:
        raise ValueError(f"{name} must be a number of at least 1, got {value!r}.")
    return int(value)


def resolve_api_key(options: Any) -> str:
    """Bearer token for the server.

    The option wins when set. The environment covers ocrmypdf options that
    do not carry the key, and ``EMPTY`` is the vLLM convention for a server
    without auth.
    """
    option = text_option(options, "chandra_api_key").strip()
    return option or os.environ.get("PAPERLESS_CHANDRA_API_KEY", "").strip() or "EMPTY"


def request_settings(options: Any) -> RequestSettings:
    """Settings from the ``chandra_*`` options, with the package defaults.

    The values are checked here because the Python API of ocrmypdf hands
    over the user's values unchecked and argparse's ``type=int`` sets no
    lower bound.
    """
    return RequestSettings(
        url=normalize_server_url(text_option(options, "chandra_server_url")),
        model=text_option(options, "chandra_model_name").strip() or DEFAULT_MODEL_NAME,
        api_key=resolve_api_key(options),
        max_tokens=_int_option(options, "chandra_max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        timeout=_int_option(options, "chandra_timeout", DEFAULT_TIMEOUT_SECONDS),
    )


def _has_unclosed_think(raw: str) -> bool:
    return "<think>" in raw and "</think>" not in raw


def _is_repeat_loop(raw: str) -> bool:
    # The library's own pair of checks. The second skips the last 50
    # characters, so it also finds a loop the model closed with end tags.
    return bool(
        detect_repeat_token(raw) or (len(raw) > 50 and detect_repeat_token(raw, cut_from_end=50))
    )


def _rejected_key_message(url: str, status: int) -> str:
    """Error text for a refused key, shared by the preflight probe and the page request."""
    return (
        f"The Chandra server at {url} rejected the API key (HTTP {status}). "
        "Check PAPERLESS_CHANDRA_API_KEY against the server configuration."
    )


def _not_reachable_message(url: str, cause: object) -> str:
    """Error text for a failed connection, shared by the preflight probe and the page request."""
    return (
        f"The Chandra server at {url} is not reachable ({cause}). "
        "Check PAPERLESS_CHANDRA_SERVER_URL and that the inference server is running."
    )


def _body_message(error: APIStatusError) -> str | None:
    """The ``message`` of the server's error body, or ``None`` when the body holds none."""
    body = error.body
    if isinstance(body, Mapping):
        message = body.get("message")
        if isinstance(message, str) and message:
            return message
    return None


def _server_message(error: APIStatusError) -> str:
    """The ``message`` of the server's error body, else the SDK's text.

    The SDK's text for a JSON body is ``Error code: N - `` and a dump of the
    whole body, so it is only the fallback for a body that holds no message,
    such as FastAPI's ``{"detail": ...}`` or a proxy's HTML page.
    """
    return _body_message(error) or error.message


def _status_error(error: APIStatusError, settings: RequestSettings) -> ChandraClientError:
    status = error.status_code
    if status in _REJECTED_KEY_STATUSES:
        return ChandraClientError(_rejected_key_message(settings.url, status))
    if status == 404:
        return ChandraClientError(
            f"The Chandra server at {settings.url} does not serve the model "
            f"{settings.model!r} (HTTP 404: {_server_message(error)}). Set "
            "PAPERLESS_CHANDRA_MODEL_NAME to the name the server advertises."
        )
    if status == 400:
        return ChandraClientError(
            f"The Chandra server at {settings.url} rejected the request (HTTP 400): "
            f"{_server_message(error)}. A PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS budget above the "
            "model's context window is a common cause."
        )
    return ChandraClientError(
        f"The Chandra server at {settings.url} answered HTTP {status}: {_server_message(error)}"
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The wait in seconds that an answer asks for, capped at the ceiling, or ``None``.

    ``retry-after-ms`` comes first, then ``Retry-After`` as seconds and then as
    an HTTP date. A form that does not read or that asks for no wait counts as
    absent.
    """
    retry_after = headers.get("retry-after", "")
    waits: list[float] = []
    with contextlib.suppress(ValueError):
        waits.append(float(headers.get("retry-after-ms", "")) / 1000)
    with contextlib.suppress(ValueError):
        waits.append(float(retry_after))
    date = email.utils.parsedate_tz(retry_after)
    if date is not None:
        # A date that parses can still lie outside the calendar.
        with contextlib.suppress(OverflowError, ValueError):
            waits.append(email.utils.mktime_tz(date) - time.time())
    return next((min(wait, _RETRY_AFTER_CEILING) for wait in waits if wait > 0), None)


def _open_client(url: str, api_key: str, timeout: float) -> OpenAI:
    # The callers own the retry policy. The SDK's own retries would run
    # beneath it and stretch one request to several timeouts.
    return OpenAI(api_key=api_key, base_url=url, timeout=timeout, max_retries=0)


def probe_server(url: str, api_key: str) -> None:
    """Fail fast unless the Chandra server answers ``/models`` with a model list.

    Without this, a wrong URL or key surfaces only at first inference,
    deep inside a celery task.
    """
    no_model_list = (
        f"The Chandra server at {url} answered without a model list. {_SERVER_URL_ADVICE}"
    )
    try:
        with _open_client(url, api_key, _PROBE_TIMEOUT_SECONDS) as client:
            page = client.models.list()
    except APIStatusError as e:
        if e.status_code in _REJECTED_KEY_STATUSES:
            raise ChandraClientError(_rejected_key_message(url, e.status_code)) from e
        # The probe quotes the body's own message and leaves out the SDK's
        # text, which for a body that is no JSON is the whole body. A wrong
        # URL often reaches a web server that answers with an HTML page.
        message = _body_message(e)
        quoted = f": {message}" if message else ""
        raise ChandraClientError(
            f"The Chandra server at {url} answered HTTP {e.status_code} at /models{quoted}. "
            f"{_SERVER_URL_ADVICE}"
        ) from e
    except APIConnectionError as e:
        raise ChandraClientError(_not_reachable_message(url, e.__cause__ or e)) from e
    # The SDK does not validate a 200 body. A page of text reaches its list
    # parser as a str and fails there with an error of its own.
    except Exception as e:
        raise ChandraClientError(no_model_list) from e
    if not isinstance(getattr(page, "data", None), list):
        raise ChandraClientError(no_model_list)


def ocr_image(image: Image.Image, options: Any) -> str:
    """OCR one page image via the remote server. The result is Chandra's raw HTML."""
    settings = request_settings(options)
    # Without the block, the connection of a page stays open until the
    # garbage collector reaches the client.
    with _open_client(settings.url, settings.api_key, settings.timeout) as client:
        image_url = f"data:image/png;base64,{image_to_base64(scale_to_fit(image))}"
        temperature, top_p = 0.0, 0.1
        not_ready_retries = repeat_retries = 0
        not_ready_waited = 0.0
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
                    f"The Chandra server at {settings.url} sent no data for "
                    f"{settings.timeout} seconds. Raise PAPERLESS_CHANDRA_TIMEOUT for slow "
                    "inference. Lower PAPERLESS_CHANDRA_MAX_OUTPUT_TOKENS only for a generation "
                    "that repeats itself until the timeout, so the plugin gets the output in time "
                    "to retry it."
                ) from e
            except (APIConnectionError, APIStatusError) as e:
                wait: float | None = None
                if isinstance(e, APIStatusError):
                    error = _status_error(e, settings)
                    if e.status_code not in _NOT_READY_STATUSES:
                        raise error from e
                    cause = f"HTTP {e.status_code}"
                    wait = _retry_after(e.response.headers)
                else:
                    cause = str(e.__cause__ or e)
                    error = ChandraClientError(_not_reachable_message(settings.url, cause))
                if not_ready_retries == len(_NOT_READY_WAITS):
                    raise error from e
                if wait is None:
                    wait = _NOT_READY_WAITS[not_ready_retries]
                if not_ready_waited + wait > _NOT_READY_WAIT_LIMIT:
                    raise error from e
                not_ready_retries += 1
                not_ready_waited += wait
                log.warning(
                    "Chandra server at %s is not ready (%s), retrying in %g s.",
                    settings.url,
                    cause,
                    wait,
                )
                time.sleep(wait)
                continue

            # The SDK does not validate a 200 body: an HTML page arrives as a str,
            # and a JSON body without choices as a completion whose choices are None.
            choices = getattr(completion, "choices", None)
            if not choices:
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} answered without a chat completion. "
                    f"{_SERVER_URL_ADVICE}"
                )
            choice = choices[0]
            refusal = getattr(choice.message, "refusal", None)
            # A document must not be stored without the text of a filtered or
            # refused page. These checks precede the loop check, so such a page is
            # not retried.
            if choice.finish_reason == "content_filter":
                refused = f" and refused it ({refusal})" if refusal else ""
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} filtered the output for the page "
                    f"(finish reason content_filter){refused}. {_FILTER_HINT}"
                )
            if refusal:
                raise ChandraClientError(
                    f"The Chandra server at {settings.url} refused the page "
                    f"({refusal}). {_FILTER_HINT}"
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
