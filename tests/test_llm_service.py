"""LLM layer: retries, validation/repair, schema handling and the Groq adapter.

No real API is called: a scripted fake provider or ``httpx.MockTransport`` stands in.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from app.exceptions import (
    LLMAuthError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.schemas import LLMAnalysis
from app.services import llm_service as llm
from tests.conftest import make_llm_payload, make_settings

pytestmark = pytest.mark.anyio


class Toy(BaseModel):
    n: int = Field(ge=5, le=5)  # the only valid answer is n == 5
    label: str = Field(default="x", max_length=5, title="Label")


class ScriptedProvider(llm.LLMProvider):
    name, model = "fake", "fake-1"

    def __init__(self, *steps):
        self.steps, self.calls = list(steps), []

    async def generate(self, *, system, user, json_schema):
        self.calls.append({"user": user, "schema": json_schema})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def reply(text='{"n": 5}', usage=llm.LLMUsage(100, 50, 10), finish="STOP"):
    return llm.ProviderResponse(text, usage, finish)


def make_service(provider, **settings):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    return llm.LLMService(make_settings(**settings), provider, sleep=fake_sleep), sleeps


async def generate(service):
    return await service.generate_structured(system="sys", user="usr", response_model=Toy)


# --------------------------------------------------------------------------- #
# Validation and repair
# --------------------------------------------------------------------------- #
async def test_valid_answer_is_returned_with_usage():
    service, _ = make_service(ScriptedProvider(reply()))
    result = await generate(service)
    assert result.data == Toy(n=5) and result.attempts == 1 and not result.repaired
    assert result.usage == llm.LLMUsage(100, 50, 10) and result.provider == "fake"


async def test_malformed_json_triggers_one_repair_attempt():
    provider = ScriptedProvider(reply("not json at all"), reply())
    service, _ = make_service(provider)
    result = await generate(service)
    assert result.repaired and result.attempts == 2
    assert "rejected" in provider.calls[1]["user"]
    assert result.usage.input_tokens == 200  # usage is summed across calls


async def test_schema_violation_is_reported_to_the_model_on_repair():
    provider = ScriptedProvider(reply('{"n": 4}'), reply())
    service, _ = make_service(provider)
    await generate(service)
    assert "n:" in provider.calls[1]["user"]


async def test_two_invalid_answers_raise_invalid_response():
    provider = ScriptedProvider(reply("{}"), reply("[]"))
    service, _ = make_service(provider)
    with pytest.raises(LLMInvalidResponseError):
        await generate(service)
    assert len(provider.calls) == 2  # no endless retry loop


async def test_truncated_output_fails_fast_with_an_actionable_message():
    provider = ScriptedProvider(reply('{"n": 5', finish="MAX_TOKENS"))
    service, _ = make_service(provider)
    with pytest.raises(LLMInvalidResponseError, match="LLM_MAX_OUTPUT_TOKENS"):
        await generate(service)
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------- #
# Retry and backoff
# --------------------------------------------------------------------------- #
async def test_transient_errors_are_retried_with_exponential_backoff():
    provider = ScriptedProvider(LLMProviderError(retryable=True), LLMTimeoutError(), reply())
    service, sleeps = make_service(provider)
    result = await generate(service)
    assert result.attempts == 3 and len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 1.25 and 2.0 <= sleeps[1] <= 2.5  # 1s then 2s, plus <=25% jitter


async def test_gives_up_after_configured_retries():
    provider = ScriptedProvider(LLMTimeoutError(), LLMTimeoutError(), LLMTimeoutError())
    service, sleeps = make_service(provider, llm_max_retries=2)
    with pytest.raises(LLMTimeoutError):
        await generate(service)
    assert len(provider.calls) == 3 and len(sleeps) == 2


async def test_retry_after_header_is_honoured():
    provider = ScriptedProvider(LLMRateLimitError(retry_after=2.0), reply())
    service, sleeps = make_service(provider)
    await generate(service)
    assert sleeps == [2.0]


async def test_very_long_retry_after_is_surfaced_instead_of_waited_for():
    provider = ScriptedProvider(LLMRateLimitError(retry_after=120.0), reply())
    service, sleeps = make_service(provider)
    with pytest.raises(LLMRateLimitError):
        await generate(service)
    assert sleeps == [] and len(provider.calls) == 1


async def test_authentication_errors_are_not_retried():
    provider = ScriptedProvider(LLMAuthError(), reply())
    service, _ = make_service(provider)
    with pytest.raises(LLMAuthError):
        await generate(service)
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------- #
# JSON Schema handling
# --------------------------------------------------------------------------- #
async def test_schema_is_sent_to_the_provider_without_unsupported_keywords():
    provider = ScriptedProvider(reply())
    service, _ = make_service(provider)
    await generate(service)
    sent = json.dumps(provider.calls[0]["schema"])
    for keyword in ('"maxLength"', '"title"', '"default"'):
        assert keyword not in sent
    assert provider.calls[0]["schema"]["properties"]["label"]  # field named like a keyword survives


async def test_provider_schema_rejection_falls_back_to_a_prompt_embedded_schema():
    provider = ScriptedProvider(llm.SchemaRejectedError("rejected"), reply())
    service, _ = make_service(provider)
    result = await generate(service)
    assert provider.calls[0]["schema"] is not None and provider.calls[1]["schema"] is None
    assert "JSON Schema of the required output" in provider.calls[1]["user"]
    assert result.data.n == 5


def test_llm_analysis_schema_is_sendable_and_keeps_structure():
    schema = llm.build_json_schema(LLMAnalysis)
    flat = json.dumps(schema)
    assert "$defs" in schema and '"maxLength"' not in flat and '"pattern"' not in flat
    queries = schema["properties"]["target_search_queries"]
    assert queries["minItems"] == 5 and queries["maxItems"] == 5  # "exactly five" is enforced in the schema too


@pytest.mark.parametrize(
    "text, expected",
    [('{"a": 1}', {"a": 1}), ('```json\n{"a": 1}\n```', {"a": 1}), ('Sure! {"a": 1} hope that helps', {"a": 1})],
)
def test_extract_json_tolerates_fences_and_prose(text, expected):
    assert llm.extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "no braces here"])
def test_extract_json_rejects_non_json(text):
    with pytest.raises(ValueError):
        llm.extract_json(text)


# --------------------------------------------------------------------------- #
# Groq response / error parsing
# --------------------------------------------------------------------------- #
def test_groq_response_parsing_extracts_text_and_counts_reasoning_as_output():
    data = {
        "choices": [{"message": {"role": "assistant", "content": '{"n": 5}'}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 800,
            "total_tokens": 2000,
            "completion_tokens_details": {"reasoning_tokens": 300},
        },
    }
    parsed = llm.parse_groq_response(data)
    assert parsed.text == '{"n": 5}' and parsed.finish_reason == "stop"
    assert parsed.usage == llm.LLMUsage(1200, 800, 300)  # reasoning tokens are billed as output


@pytest.mark.parametrize(
    "data",
    [
        {"choices": []},
        {},
        {"choices": [{"message": {"content": None, "refusal": "I cannot help with that."}}]},
        {"choices": [{"message": {"content": ""}}]},
    ],
)
def test_groq_empty_or_refused_responses_are_invalid(data):
    with pytest.raises(LLMInvalidResponseError):
        llm.parse_groq_response(data)


def test_groq_error_mapping():
    err = llm.map_groq_error
    assert isinstance(err(401, "", {}), LLMAuthError) and isinstance(err(403, "", {}), LLMAuthError)
    limited = err(429, "", {"Retry-After": "7"})
    assert isinstance(limited, LLMRateLimitError) and limited.retry_after == 7.0
    assert isinstance(err(400, json.dumps({"error": {"message": "Invalid JSON schema"}}), {}), llm.SchemaRejectedError)
    assert isinstance(err(400, json.dumps({"error": {"message": "bad request"}}), {}), LLMProviderError)
    assert not err(404, "", {}).retryable and err(503, "", {}).retryable and err(500, "", {}).retryable
    assert not err(418, "", {}).retryable
    assert "SECRET-UPSTREAM-BODY" not in err(500, "SECRET-UPSTREAM-BODY", {}).message  # never shown to API users


# --------------------------------------------------------------------------- #
# Groq adapter over a mocked transport
# --------------------------------------------------------------------------- #
def groq_reply(payload, *, status=200):
    body = {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9000, "completion_tokens": 3500, "total_tokens": 12500},
    }
    return httpx.Response(status, json=body)


async def test_groq_request_shape_and_api_key_handling():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return groq_reply(make_llm_payload())

    settings = make_settings(groq_api_key="test-key-not-real", llm_model="")
    service = llm.LLMService(settings, llm.GroqProvider(settings, transport=httpx.MockTransport(handler)))
    result = await service.generate_structured(system="SYS", user="USR", response_model=LLMAnalysis)

    request = seen[0]
    assert request.url.path == "/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key-not-real"
    assert "test-key-not-real" not in str(request.url)  # key never appears in a URL (and so never in logs)
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-oss-120b"  # default model
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USR"}
    assert "max_completion_tokens" in body and "temperature" not in body
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]  # schema derived from the Pydantic model

    assert result.data.brand.name == "Acme Cloud"
    assert result.usage == llm.LLMUsage(9000, 3500, None) and result.model == "openai/gpt-oss-120b"


async def test_temperature_is_configurable_or_omitted():
    settings = make_settings(llm_temperature=0.4, llm_model="custom-model")
    payload = llm.GroqProvider(settings).build_payload("s", "u", None)
    assert payload["temperature"] == 0.4
    assert payload["response_format"] == {"type": "json_object"}  # no schema -> JSON mode
    default = llm.GroqProvider(make_settings(llm_model="custom-model")).build_payload("s", "u", None)
    assert "temperature" not in default


def test_make_strict_schema_enforces_required_and_additional_properties():
    schema = llm.build_json_schema(LLMAnalysis)
    strict = llm.make_strict_schema(schema)
    flat = json.dumps(strict)
    assert '"minItems"' not in flat and '"maxItems"' not in flat  # unsupported keywords removed for strict mode
    assert '"additionalProperties": false' in flat
    assert set(strict["required"]) == set(strict["properties"])  # every field is required


@pytest.mark.parametrize(
    "make_response, expected",
    [(lambda: httpx.Response(401, json={"error": {"message": "bad key"}}), LLMAuthError),
     (lambda: httpx.Response(404, json={}), LLMProviderError)],
)
async def test_groq_non_retryable_http_errors(make_response, expected):
    calls = []

    def handler(request):
        calls.append(1)
        return make_response()

    settings = make_settings()
    service, _ = make_service(llm.GroqProvider(settings, transport=httpx.MockTransport(handler)))
    with pytest.raises(expected):
        await service.generate_structured(system="s", user="u", response_model=Toy)
    assert len(calls) == 1


async def test_groq_rate_limit_is_retried_then_surfaced():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "quota"}})

    settings = make_settings(llm_max_retries=2)
    service, sleeps = make_service(llm.GroqProvider(settings, transport=httpx.MockTransport(handler)))
    with pytest.raises(LLMRateLimitError):
        await service.generate_structured(system="s", user="u", response_model=Toy)
    assert len(calls) == 3 and sleeps == [1.0, 1.0]


async def test_groq_timeout_maps_to_llm_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    settings = make_settings(llm_max_retries=0)
    service, _ = make_service(llm.GroqProvider(settings, transport=httpx.MockTransport(handler)), llm_max_retries=0)
    with pytest.raises(LLMTimeoutError):
        await service.generate_structured(system="s", user="u", response_model=Toy)


async def test_groq_invalid_json_body_is_a_provider_error():
    settings = make_settings(llm_max_retries=0)
    provider = llm.GroqProvider(settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>oops")))
    with pytest.raises(LLMProviderError):
        await provider.generate(system="s", user="u", json_schema=None)


# --------------------------------------------------------------------------- #
# Configuration errors
# --------------------------------------------------------------------------- #
def test_missing_api_key_gives_an_actionable_error():
    service = llm.LLMService(make_settings(groq_api_key=None))
    with pytest.raises(LLMNotConfiguredError, match="GROQ_API_KEY"):
        service.ensure_configured()


def test_unsupported_provider_is_rejected_with_the_supported_list():
    service = llm.LLMService(make_settings(llm_provider="openai"))
    with pytest.raises(LLMNotConfiguredError, match="not supported.*groq"):
        service.ensure_configured()


def test_model_can_be_changed_through_configuration():
    provider = llm.build_provider(make_settings(llm_model="openai/gpt-oss-20b"))
    assert provider.model == "openai/gpt-oss-20b"
