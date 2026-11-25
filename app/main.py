"""FastAPI application: ``POST /link-analysis``, ``GET /health``, ``GET /``, ``/docs``."""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import Settings, get_settings
from app.exceptions import AppError, LLMError, RateLimitExceededError
from app.schemas import (
    AnalysisResponse,
    ComponentStatus,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    LinkAnalysisRequest,
)
from app.services.analysis_service import AnalysisService
from app.utils.log_config import configure_logging
from app.utils.rate_limit import SlidingWindowRateLimiter

logger = logging.getLogger("app.main")
_REQUEST_ID = re.compile(r"[A-Za-z0-9\-]{1,64}")

DESCRIPTION = """\
Analyzes a website with an LLM and returns five structured sections: **competitor vs brand details**, \
**top 5 target search queries**, **SEO improvements**, **top 5 services** and **target audience**.

* Website content is fetched and extracted server-side (SSRF-protected).
* Competitor entries carry `evidence_type`, `source_urls` and `confidence`; unverified AI suggestions are \
labelled as such.
* Search volume / difficulty are reported as `unavailable` - they are never estimated.
"""

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid URL or URL not allowed (SSRF protection)."},
    413: {"model": ErrorResponse, "description": "Request body too large."},
    422: {"model": ErrorResponse, "description": "Invalid request, unsupported/empty page or response too large."},
    429: {"model": ErrorResponse, "description": "Rate limit reached (this API or the LLM provider)."},
    502: {"model": ErrorResponse, "description": "Website or LLM provider failure / invalid LLM output."},
    503: {"model": ErrorResponse, "description": "LLM not configured."},
    504: {"model": ErrorResponse, "description": "Website or LLM timeout."},
}


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
@lru_cache
def get_analysis_service() -> AnalysisService:
    return AnalysisService(get_settings())


@lru_cache
def get_rate_limiter() -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(get_settings().rate_limit_per_minute)


def client_key(request: Request, trust_proxy: bool) -> str:
    """Identify the caller (X-Forwarded-For is only trusted behind a known proxy)."""
    if trust_proxy:
        first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(
    request: Request,
    limiter: SlidingWindowRateLimiter = Depends(get_rate_limiter),
    settings: Settings = Depends(get_settings),
) -> None:
    wait = limiter.check(client_key(request, settings.trust_proxy_headers))
    if wait is not None:
        raise RateLimitExceededError(headers={"Retry-After": str(max(1, math.ceil(wait)))})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details or None, request_id=request_id)
    )
    merged = dict(headers or {})
    if request_id:
        merged["X-Request-ID"] = request_id
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"), headers=merged)


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.secret_values())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "service starting",
            extra={
                "environment": settings.app_env,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.effective_llm_model,
                "llm_configured": settings.has_llm_key,
                "research_configured": settings.has_research_key,
            },
        )
        if not settings.has_llm_key:
            logger.warning(
                "No LLM API key configured: POST /link-analysis will return 503 until GEMINI_API_KEY is set."
            )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # ---- middleware -----------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:16]
        started = time.perf_counter()

        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > settings.max_request_bytes:
            response = error_response(request, 413, "request_too_large", "The request body is too large.")
        else:
            response = await call_next(request)

        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "request_id": request.state.request_id,
            },
        )
        return response

    # ---- exception handlers ----------------------------------------------
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        headers = dict(exc.headers)
        retry_after = getattr(exc, "retry_after", None) if isinstance(exc, LLMError) else None
        if retry_after:
            headers.setdefault("Retry-After", str(max(1, math.ceil(retry_after))))
        logger.warning(
            "request failed",
            extra={"code": exc.code, "status": exc.status_code, "request_id": getattr(request.state, "request_id", None)},
        )
        return error_response(
            request, exc.status_code, exc.code, exc.message, details=exc.details, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body") or "body", "message": err["msg"]}
            for err in exc.errors()
        ]  # the offending input is deliberately not echoed back
        return error_response(
            request, 422, "invalid_request", "The request body is invalid.", details={"errors": problems}
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return error_response(
            request, exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail)
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled error",
            exc_info=exc,
            extra={"request_id": getattr(request.state, "request_id", None), "path": request.url.path},
        )
        return error_response(
            request, 500, "internal_error", "An unexpected error occurred. Please try again later."
        )

    # ---- routes ------------------------------------------------------------
    @app.get("/", tags=["meta"], summary="API information")
    async def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "endpoints": {
                "GET /health": "Service health and configuration status",
                "POST /link-analysis": "Analyze a website URL (body: {\"url\": \"https://example.com\"})",
                "GET /docs": "Interactive API documentation",
            },
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Health check")
    async def health(current: Settings = Depends(get_settings)) -> HealthResponse:
        return HealthResponse(
            service=current.app_name,
            version=__version__,
            environment=current.app_env,
            llm=ComponentStatus(
                provider=current.llm_provider,
                model=current.effective_llm_model or None,
                configured=current.has_llm_key,
            ),
            research=ComponentStatus(
                provider="tavily" if current.has_research_key else None, configured=current.has_research_key
            ),
        )

    @app.post(
        "/link-analysis",
        response_model=AnalysisResponse,
        responses=ERROR_RESPONSES,
        tags=["analysis"],
        summary="Analyze a website with an LLM",
        dependencies=[Depends(enforce_rate_limit)],
    )
    async def link_analysis(
        body: LinkAnalysisRequest, service: AnalysisService = Depends(get_analysis_service)
    ) -> AnalysisResponse:
        return await service.analyze(body.url)

    return app


app = create_app()

if __name__ == "__main__":  # local run: python -m app.main  (port from PORT, default 4000)
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=get_settings().port)  # noqa: S104
