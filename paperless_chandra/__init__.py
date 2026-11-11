"""Chandra OCR provider for paperless-ngx."""

__version__ = "0.2.0"

#: Option defaults shared by the client, the parser and the plugin. They
#: live here because paperless imports the parser in every process and
#: that import must not pull chandra-ocr in.
CONTENT_FORMATS = ("text", "markdown")
DEFAULT_CONTENT_FORMAT = "text"
DEFAULT_MAX_OUTPUT_TOKENS = 12384
DEFAULT_MODEL_NAME = "chandra"
DEFAULT_TIMEOUT_SECONDS = 600
