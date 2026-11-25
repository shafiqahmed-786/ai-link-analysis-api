"""LLM integration layer.

``LLMService`` is provider-agnostic: it builds a JSON Schema from a Pydantic
model, asks a provider for JSON, validates the answer, retries transient
failures with exponential backoff and makes at most one *repair* attempt when
the model returns malformed or schema-violating JSON.

Adding a provider
-----------------
Subclass :class:`LLMProvider`, implement :meth:`LLMProvider.generate` and register
the class in :data:`PROVIDER_REGISTRY`.  Select it with ``LLM_PROVIDER`` /
``LLM_MODEL``.  Only the Gemini adapter ships with this project (see
``docs/LLM_RESEARCH.md`` for why).

The Gemini adapter talks to the public REST endpoint with HTTPX (no vendor SDK
dependency).  The API key is sent in the ``x-goog-api-key`` header, never in the
URL, so it cannot appear in request logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.exceptions import (
    LLMAuthError,
    LLMError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 20.0
MAX_HONOURED_RETRY_AFTER = 30.0  # longer provider waits are surfaced to the caller instead
_SCHEMA_STRIP_KEYS = frozenset({"title", "default", "examples", "minLength", "maxLength", "pattern", "uniqueItems"})


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None  # includes thinking tokens (they are billed as output)
    thinking_tokens: int | None = None

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        def add(a: int | None, b: int | None) -> int | None:
            return None if a is None and b is None else (a or 0) + (b or 0)

        return LLMUsage(
            add(self.input_tokens, other.input_tokens),
            add(self.output_tokens, other.output_tokens),
            add(self.thinking_tokens, other.thinking_tokens),
        )


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: str | None = None


@dataclass
class StructuredResult(Generic[T]):
    data: T
    provider: str
    model: str
    usage: LLMUsage
    attempts: int  # provider calls made, including retries and the repair call
    repaired: bool
    latency_ms: int


class SchemaRejectedError(LLMProviderError):
    """The provider refused the JSON Schema itself (HTTP 400 mentioning the schema)."""


# --------------------------------------------------------------------------- #
# Provider abstraction
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Minimal interface every provider adapter implements."""

    name: str
    model: str

    @abstractmethod
    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None
    ) -> ProviderResponse:
        """Return the model's JSON text for the prompt (raise :class:`LLMError` on failure)."""


class GeminiProvider(LLMProvider):
    """Google Gemini via the ``generateContent`` REST endpoint."""

    name = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if settings.gemini_api_key is None:
            raise LLMNotConfiguredError()
        self._api_key = settings.gemini_api_key.get_secret_value()
        self.model = settings.effective_llm_model
        self._timeout = settings.llm_timeout
        self._max_output_tokens = settings.llm_max_output_tokens
        self._thinking_level = settings.llm_thinking_level
        self._temperature = settings.llm_temperature
        self._transport = transport

    def build_payload(self, system: str, user: str, json_schema: dict[str, Any] | None) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "maxOutputTokens": self._max_output_tokens,
        }
        if json_schema is not None:
            generation_config["responseJsonSchema"] = json_schema
        if self._thinking_level:
            generation_config["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        if self._temperature is not None:
            generation_config["temperature"] = self._temperature
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

    async def generate(
        self, *, system: str, user: str, json_schema: dict[str, Any] | None
    ) -> ProviderResponse:
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self._api_key, "content-type": "application/json"}
        payload = self.build_payload(system, user, json_schema)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout), transport=self._transport
            ) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                "Could not reach the language model provider.", retryable=True
            ) from exc

        if response.status_code != 200:
            raise map_gemini_error(response.status_code, response.text, dict(response.headers))
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError("The provider returned a non-JSON response.", retryable=True) from exc
        return parse_gemini_response(data)


PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {"gemini": GeminiProvider}


def build_provider(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> LLMProvider:
    """Instantiate the configured provider or raise :class:`LLMNotConfiguredError`."""
    provider_cls = PROVIDER_REGISTRY.get(settings.llm_provider)
    if provider_cls is None:
        supported = ", ".join(sorted(PROVIDER_REGISTRY))
        raise LLMNotConfiguredError(
            f"LLM_PROVIDER '{settings.llm_provider}' is not supported. Supported providers: {supported}."
        )
    if not settings.effective_llm_model:
        raise LLMNotConfiguredError("LLM_MODEL is not set and the provider has no default model.")
    return provider_cls(settings, transport=transport)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Gemini response / error parsing (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def parse_gemini_response(data: dict[str, Any]) -> ProviderResponse:
    """Extract text, finish reason and token usage from a ``generateContent`` reply."""
    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        raise LLMInvalidResponseError(
            "The language model refused to process this content.", details={"block_reason": block_reason}
        )
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMInvalidResponseError("The language model returned no candidates.")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(
        part.get("text", "") for part in parts if isinstance(part, dict) and not part.get("thought")
    )
    meta = data.get("usageMetadata") or {}
    candidates_tokens, thought_tokens = meta.get("candidatesTokenCount"), meta.get("thoughtsTokenCount")
    output_tokens = (
        None
        if candidates_tokens is None and thought_tokens is None
        else (candidates_tokens or 0) + (thought_tokens or 0)
    )
    usage = LLMUsage(meta.get("promptTokenCount"), output_tokens, thought_tokens)
    return ProviderResponse(text=text, usage=usage, finish_reason=candidate.get("finishReason"))


def _error_message(body: str) -> str:
    try:
        message = json.loads(body).get("error", {}).get("message")
    except (ValueError, AttributeError):
        message = None
    return str(message or body)[:300]


def _retry_after(headers: dict[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
    return None


def map_gemini_error(status: int, body: str, headers: dict[str, str]) -> LLMError:
    """Translate a Gemini HTTP error into an application exception.

    The upstream message is logged (redacted) but never returned to API users.
    """
    message = _error_message(body)
    lowered = message.lower()
    logger.warning("gemini error", extra={"status": status, "upstream_message": message})
    if status in (401, 403) or (status == 400 and "api key" in lowered):
        return LLMAuthError()
    if status == 429:
        return LLMRateLimitError(retry_after=_retry_after(headers))
    if status == 404:
        return LLMProviderError(
            "The configured language model was not found. Check LLM_MODEL.", retryable=False
        )
    if status == 400 and "schema" in lowered:
        return SchemaRejectedError("The provider rejected the response schema.")
    if status in (408, 500, 502, 503, 504):
        return LLMProviderError(retryable=True, retry_after=_retry_after(headers))
    return LLMProviderError(f"The provider rejected the request (HTTP {status}).")


# --------------------------------------------------------------------------- #
# Schema and JSON helpers
# --------------------------------------------------------------------------- #
def build_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for ``model`` without keywords providers commonly reject.

    Length/pattern constraints are removed from the schema sent to the model but
    remain enforced by Pydantic when the answer is validated.
    """
    schema = model.model_json_schema()
    return _clean_schema_tree(schema)


def _clean_schema_tree(node: Any, *, in_properties: bool = False) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if not in_properties and key in _SCHEMA_STRIP_KEYS:
                continue
            out[key] = _clean_schema_tree(value, in_properties=(key == "properties" and not in_properties))
        return out
    if isinstance(node, list):
        return [_clean_schema_tree(item) for item in node]
    return node


def extract_json(text: str) -> Any:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty response")
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def summarize_validation_error(exc: ValidationError | ValueError, limit: int = 6) -> str:
    """Short, prompt-safe description of why an answer was rejected."""
    if isinstance(exc, ValidationError):
        items = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:limit]]
        extra = exc.error_count() - len(items)
        return "; ".join(items) + (f"; and {extra} more" if extra > 0 else "")
    return str(exc)[:300]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class LLMService:
    """Structured generation with retries, validation and a repair attempt."""

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._sleep = sleep

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = build_provider(self._settings)
        return self._provider

    def ensure_configured(self) -> None:
        """Raise :class:`LLMNotConfiguredError` if the provider cannot be built."""
        self._get_provider()

    async def generate_structured(
        self, *, system: str, user: str, response_model: type[T]
    ) -> StructuredResult[T]:
        """Return a validated ``response_model`` instance produced by the LLM."""
        provider = self._get_provider()
        schema: dict[str, Any] | None = build_json_schema(response_model)
        prompt = user
        usage = LLMUsage()
        calls = 0
        started = time.monotonic()
        problem = ""

        for round_number in (1, 2):  # 1 = first answer, 2 = single repair attempt
            if round_number == 2:
                prompt = (
                    f"{user}\n\nYour previous answer was rejected: {problem}\n"
                    "Return ONLY one corrected JSON object that satisfies the schema."
                )
            response, used_calls, schema = await self._call(provider, system, prompt, schema, response_model)
            usage, calls = usage + response.usage, calls + used_calls

            if response.finish_reason == "MAX_TOKENS":
                raise LLMInvalidResponseError(
                    "The model output was cut off. Increase LLM_MAX_OUTPUT_TOKENS or reduce the page size.",
                    details={"finish_reason": response.finish_reason},
                )
            try:
                parsed = response_model.model_validate(extract_json(response.text))
            except (ValueError, ValidationError) as exc:
                problem = summarize_validation_error(exc)
                logger.warning(
                    "llm output rejected",
                    extra={"round": round_number, "problem": problem[:500], "provider": provider.name},
                )
                continue

            latency_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "llm analysis complete",
                extra={
                    "provider": provider.name,
                    "model": provider.model,
                    "calls": calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "latency_ms": latency_ms,
                },
            )
            return StructuredResult(
                data=parsed,
                provider=provider.name,
                model=provider.model,
                usage=usage,
                attempts=calls,
                repaired=round_number == 2,
                latency_ms=latency_ms,
            )

        raise LLMInvalidResponseError(
            "The language model returned malformed output twice.", details={"problem": problem[:500]}
        )

    async def _call(
        self,
        provider: LLMProvider,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        response_model: type[T],
    ) -> tuple[ProviderResponse, int, dict[str, Any] | None]:
        """One logical provider call with transient-error retries.

        If the provider rejects the JSON Schema itself, retry once without it and
        describe the shape in the prompt instead (Pydantic still validates).
        """
        calls = 0
        max_attempts = self._settings.llm_max_retries + 1
        attempt = 0
        while True:
            attempt += 1
            calls += 1
            try:
                response = await provider.generate(system=system, user=user, json_schema=schema)
                return response, calls, schema
            except SchemaRejectedError:
                if schema is None:
                    raise
                logger.warning("provider rejected schema; retrying with schema in prompt")
                user = f"{user}\n\nJSON Schema of the required output:\n{json.dumps(schema, separators=(',', ':'))}"
                schema = None
                attempt -= 1  # not a transient failure
            except LLMError as exc:
                if not exc.retryable or attempt >= max_attempts:
                    raise
                delay = self._backoff_delay(attempt, exc.retry_after)
                if delay is None:
                    raise
                logger.warning(
                    "llm call failed; retrying",
                    extra={"code": exc.code, "attempt": attempt, "wait_s": round(delay, 2)},
                )
                await self._sleep(delay)

    @staticmethod
    def _backoff_delay(attempt: int, retry_after: float | None) -> float | None:
        """Exponential backoff with jitter; honour ``Retry-After`` when reasonable."""
        if retry_after is not None:
            return None if retry_after > MAX_HONOURED_RETRY_AFTER else retry_after
        delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        return min(delay + random.uniform(0, delay * 0.25), BACKOFF_MAX_SECONDS)
