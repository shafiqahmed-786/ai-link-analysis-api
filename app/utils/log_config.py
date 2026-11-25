"""Structured (JSON) logging with secret redaction.

Every log line is a single JSON object.  Before a line is emitted, configured
secret values and common API-key shapes are replaced by ``***`` so that a
credential cannot leak through an exception message or a URL.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Iterable

_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"tvly-[0-9A-Za-z_\-]{8,}"),  # Tavily keys
    re.compile(r"sk-[0-9A-Za-z_\-]{16,}"),  # OpenAI-style keys
    re.compile(r"(?i)(bearer\s+)[0-9A-Za-z._\-]{8,}"),  # Authorization headers
    re.compile(r"(?i)(x-goog-api-key['\"]?\s*[:=]\s*['\"]?)[0-9A-Za-z_\-]{8,}"),
)
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Replace configured secrets and key-shaped tokens in ``text``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "***", text)
    return text


class JsonFormatter(logging.Formatter):
    """Render records as redacted single-line JSON."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():  # fields passed via ``extra=``
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        line = json.dumps(payload, default=str, ensure_ascii=False)
        return redact(line, self._secrets)


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(secrets))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Library request logs include full URLs; keep them out of INFO output.
    for noisy in ("httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers = []
        uv.propagate = True  # route uvicorn's own logs through the JSON handler
    logging.getLogger("uvicorn.access").disabled = True  # app.main logs every request itself
