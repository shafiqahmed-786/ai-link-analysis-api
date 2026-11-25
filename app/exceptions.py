"""Application exceptions.

Every failure mode the API reports to clients is a subclass of :class:`AppError`.
Each carries an HTTP status code and a stable machine-readable ``code`` so that
``app.main`` can translate it into a uniform JSON error envelope without leaking
stack traces or internal details.

This module deliberately has no third-party imports so that it can be used by
the dependency-free URL validator.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all errors that are safe to show to API clients."""

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details or {}
        self.headers = headers or {}
        super().__init__(self.message)


# --------------------------------------------------------------------------- #
# URL validation (including SSRF protection)
# --------------------------------------------------------------------------- #
class InvalidURLError(AppError):
    """The supplied string is not a well-formed website URL."""

    status_code = 400
    code = "invalid_url"
    default_message = "The supplied URL is not valid."


class UnsafeURLError(AppError):
    """The URL is well-formed but must not be fetched (SSRF protection)."""

    status_code = 400
    code = "url_not_allowed"
    default_message = "This URL cannot be analyzed because it is not a public website address."


# --------------------------------------------------------------------------- #
# Website fetching / extraction
# --------------------------------------------------------------------------- #
class WebsiteFetchError(AppError):
    """Base class for problems retrieving the target website."""

    status_code = 502
    code = "website_error"
    default_message = "The website could not be retrieved."


class WebsiteInaccessibleError(WebsiteFetchError):
    """DNS failure, connection failure, HTTP error status or redirect loop."""

    code = "website_inaccessible"
    default_message = "The website could not be reached."


class WebsiteTimeoutError(WebsiteFetchError):
    """The website did not respond within the configured time budget."""

    status_code = 504
    code = "website_timeout"
    default_message = "The website did not respond in time."


class UnsupportedContentTypeError(WebsiteFetchError):
    """The URL does not serve an HTML document."""

    status_code = 422
    code = "unsupported_content_type"
    default_message = "The URL does not return an HTML page."


class ResponseTooLargeError(WebsiteFetchError):
    """The website response exceeded the configured size limit."""

    status_code = 422
    code = "response_too_large"
    default_message = "The website response is too large to analyze."


class EmptyContentError(WebsiteFetchError):
    """The page was fetched but contained no usable content."""

    status_code = 422
    code = "empty_content"
    default_message = (
        "The page contains no readable content. It may be rendered with JavaScript "
        "or blocked for automated access."
    )


# --------------------------------------------------------------------------- #
# LLM integration
# --------------------------------------------------------------------------- #
class LLMError(AppError):
    """Base class for LLM problems.

    ``retryable`` tells :class:`app.services.llm_service.LLMService` whether the
    request may be repeated with exponential backoff.
    """

    status_code = 502
    code = "llm_error"
    default_message = "The language model request failed."
    retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, details=details, headers=headers)
        if retryable is not None:
            self.retryable = retryable
        self.retry_after = retry_after


class LLMNotConfiguredError(LLMError):
    """No usable API key / provider configuration."""

    status_code = 503
    code = "llm_not_configured"
    default_message = (
        "The language model is not configured. Set the provider API key "
        "(for example GEMINI_API_KEY) in the server environment."
    )


class LLMAuthError(LLMError):
    """The provider rejected the configured credentials."""

    code = "llm_auth_failed"
    default_message = (
        "The language model provider rejected the configured API key. "
        "Check the key and its permissions."
    )


class LLMRateLimitError(LLMError):
    """The provider returned HTTP 429 (or an equivalent quota error)."""

    status_code = 429
    code = "llm_rate_limited"
    default_message = "The language model rate limit was reached. Please retry shortly."
    retryable = True


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""

    status_code = 504
    code = "llm_timeout"
    default_message = "The language model did not respond in time."
    retryable = True


class LLMProviderError(LLMError):
    """Any other provider-side failure (5xx, unknown model, rejected request)."""

    code = "llm_provider_error"
    default_message = "The language model provider returned an error."


class LLMInvalidResponseError(LLMError):
    """The model output was empty, truncated, not JSON or failed schema validation."""

    code = "llm_invalid_response"
    default_message = "The language model returned an invalid or incomplete analysis."


# --------------------------------------------------------------------------- #
# Abuse protection
# --------------------------------------------------------------------------- #
class RateLimitExceededError(AppError):
    """The caller exceeded the per-client request limit of this API."""

    status_code = 429
    code = "rate_limit_exceeded"
    default_message = "Too many requests. Please slow down and retry shortly."
