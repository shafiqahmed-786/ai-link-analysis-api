"""HTTP-level behaviour: routes, validation, status codes and error envelopes.

The real ``AnalysisService`` is used with fake extractor / LLM / research
components injected through FastAPI dependency overrides, so URL validation and
error mapping are the production code paths.  No network or paid API is used.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.exceptions import (
    EmptyContentError,
    LLMAuthError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    ResponseTooLargeError,
    UnsupportedContentTypeError,
    WebsiteInaccessibleError,
    WebsiteTimeoutError,
)
from app.main import app, get_analysis_service, get_rate_limiter
from app.services.analysis_service import AnalysisService
from app.utils.cache import TTLCache
from app.utils.rate_limit import SlidingWindowRateLimiter
from tests.conftest import FakeExtractor, FakeLLM, FakeResearch, make_settings

VALID = {"url": "https://example.com"}


def build_service(*, llm=None, extractor=None) -> AnalysisService:
    return AnalysisService(
        make_settings(),
        extractor=extractor or FakeExtractor(),
        llm=llm or FakeLLM(),
        research=FakeResearch(),
        cache=TTLCache(0),
    )


@pytest.fixture
def make_client():
    def factory(service=None, limiter=None, raise_server_exceptions=True) -> TestClient:
        chosen_service = service or build_service()
        chosen_limiter = limiter or SlidingWindowRateLimiter(0)
        app.dependency_overrides[get_analysis_service] = lambda: chosen_service
        app.dependency_overrides[get_rate_limiter] = lambda: chosen_limiter
        return TestClient(app, raise_server_exceptions=raise_server_exceptions)

    yield factory
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Meta endpoints
# --------------------------------------------------------------------------- #
def test_health_reports_status_without_exposing_secrets(make_client):
    response = make_client().get("/health")
    body = response.json()
    assert response.status_code == 200 and body["status"] == "ok"
    assert isinstance(body["llm"]["configured"], bool) and isinstance(body["research"]["configured"], bool)
    assert body["llm"]["provider"] == "groq"
    assert "api_key" not in response.text.lower() and "aiza" not in response.text.lower()


def test_root_describes_the_api(make_client):
    body = make_client().get("/").json()
    assert body["docs"] == "/docs" and "POST /link-analysis" in body["endpoints"]


def test_interactive_docs_and_openapi_are_available(make_client):
    client = make_client()
    docs = client.get("/docs")
    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    schema = client.get("/openapi.json").json()
    assert "/link-analysis" in schema["paths"] and "post" in schema["paths"]["/link-analysis"]


# --------------------------------------------------------------------------- #
# POST /link-analysis - success
# --------------------------------------------------------------------------- #
def test_successful_analysis_returns_the_complete_document(make_client):
    response = make_client().post("/link-analysis", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True and body["website"] == "https://example.com"
    assert {"name", "category", "summary"} <= set(body["brand"])
    assert set(body["analysis"]) == {
        "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience",
    }
    assert len(body["analysis"]["target_search_queries"]) == 5 and len(body["analysis"]["top_services"]) == 5
    competitor = body["analysis"]["competitor_analysis"]["competitors"][0]
    assert {"evidence_type", "source_urls", "confidence"} <= set(competitor)
    assert body["metadata"]["llm"]["model"] == "fake-model"


def test_request_id_is_generated_or_echoed_when_valid(make_client):
    client = make_client()
    assert client.get("/health").headers["x-request-id"]
    assert client.get("/health", headers={"X-Request-ID": "trace-123"}).headers["x-request-id"] == "trace-123"
    assert client.get("/health", headers={"X-Request-ID": "bad id!"}).headers["x-request-id"] != "bad id!"


# --------------------------------------------------------------------------- #
# POST /link-analysis - bad input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body", [{}, {"url": ""}, {"url": 123}, {"url": None}, {"link": "https://example.com"}, []])
def test_invalid_request_bodies_return_422(make_client, body):
    response = make_client().post("/link-analysis", json=body)
    payload = response.json()
    assert response.status_code == 422 and payload["success"] is False
    assert payload["error"]["code"] == "invalid_request" and payload["error"]["details"]["errors"]


def test_non_json_body_returns_422(make_client):
    response = make_client().post("/link-analysis", content="not json", headers={"content-type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"


def test_validation_errors_do_not_echo_the_submitted_input(make_client):
    response = make_client().post("/link-analysis", json={"url": 123456789})
    assert "123456789" not in response.text


def test_malformed_url_returns_400(make_client):
    response = make_client().post("/link-analysis", json={"url": "not a url"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_url"


@pytest.mark.parametrize(
    "url",
    ["http://localhost", "http://127.0.0.1:80/admin", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5",
     "http://[::1]/", "ftp://example.com", "file:///etc/passwd", "https://user:pw@example.com"],
)
def test_ssrf_attempts_return_400_url_not_allowed(make_client, url):
    extractor = FakeExtractor()
    response = make_client(build_service(extractor=extractor)).post("/link-analysis", json={"url": url})
    assert response.status_code == 400 and response.json()["error"]["code"] == "url_not_allowed"
    assert extractor.calls == 0  # nothing was fetched


# --------------------------------------------------------------------------- #
# POST /link-analysis - upstream failures
# --------------------------------------------------------------------------- #
WEBSITE_FAILURES = [
    (WebsiteInaccessibleError(), 502, "website_inaccessible"),
    (WebsiteTimeoutError(), 504, "website_timeout"),
    (UnsupportedContentTypeError(), 422, "unsupported_content_type"),
    (EmptyContentError(), 422, "empty_content"),
    (ResponseTooLargeError(), 422, "response_too_large"),
]


@pytest.mark.parametrize("error, status, code", WEBSITE_FAILURES, ids=[c for _, _, c in WEBSITE_FAILURES])
def test_website_failures_map_to_clear_status_codes(make_client, error, status, code):
    response = make_client(build_service(extractor=FakeExtractor(error=error))).post("/link-analysis", json=VALID)
    body = response.json()
    assert response.status_code == status and body["success"] is False and body["error"]["code"] == code
    assert body["error"]["message"] and body["error"]["request_id"]


LLM_FAILURES = [
    (LLMNotConfiguredError(), 503, "llm_not_configured"),
    (LLMAuthError(), 502, "llm_auth_failed"),
    (LLMRateLimitError(), 429, "llm_rate_limited"),
    (LLMTimeoutError(), 504, "llm_timeout"),
    (LLMInvalidResponseError(), 502, "llm_invalid_response"),
    (LLMProviderError(), 502, "llm_provider_error"),
]


@pytest.mark.parametrize("error, status, code", LLM_FAILURES, ids=[c for _, _, c in LLM_FAILURES])
def test_llm_failures_map_to_clear_status_codes(make_client, error, status, code):
    response = make_client(build_service(llm=FakeLLM(error=error))).post("/link-analysis", json=VALID)
    assert response.status_code == status and response.json()["error"]["code"] == code


def test_missing_api_key_message_is_actionable(make_client):
    response = make_client(build_service(llm=FakeLLM(configured=False))).post("/link-analysis", json=VALID)
    assert response.status_code == 503 and "GROQ_API_KEY" in response.json()["error"]["message"]


def test_llm_rate_limit_exposes_retry_after(make_client):
    service = build_service(llm=FakeLLM(error=LLMRateLimitError(retry_after=3.0)))
    response = make_client(service).post("/link-analysis", json=VALID)
    assert response.status_code == 429 and response.headers["retry-after"] == "3"


def test_unexpected_errors_return_a_generic_500_without_internals(make_client):
    extractor = FakeExtractor(error=RuntimeError("boom-secret-internal-detail /etc/passwd"))
    client = make_client(build_service(extractor=extractor), raise_server_exceptions=False)
    response = client.post("/link-analysis", json=VALID)
    assert response.status_code == 500 and response.json()["error"]["code"] == "internal_error"
    for leaked in ("boom-secret", "Traceback", "/etc/passwd", "RuntimeError"):
        assert leaked not in response.text


# --------------------------------------------------------------------------- #
# Abuse protection and framework errors
# --------------------------------------------------------------------------- #
def test_per_client_rate_limit_returns_429_with_retry_after(make_client):
    client = make_client(limiter=SlidingWindowRateLimiter(2, 60))
    assert client.post("/link-analysis", json=VALID).status_code == 200
    assert client.post("/link-analysis", json=VALID).status_code == 200
    limited = client.post("/link-analysis", json=VALID)
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(limited.headers["retry-after"]) >= 1
    assert client.get("/health").status_code == 200  # only the analysis route is limited


def test_oversized_request_bodies_are_rejected_early(make_client):
    response = make_client().post("/link-analysis", content="x" * 9000, headers={"content-type": "application/json"})
    assert response.status_code == 413 and response.json()["error"]["code"] == "request_too_large"


def test_unknown_routes_and_methods_use_the_error_envelope(make_client):
    client = make_client()
    missing = client.get("/nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    wrong_method = client.get("/link-analysis")
    assert wrong_method.status_code == 405 and wrong_method.json()["success"] is False
