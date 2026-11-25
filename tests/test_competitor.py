"""Competitor research (Tavily, mocked) and server-side provenance enforcement."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.schemas import Competitor, Confidence, EvidenceType, LLMAnalysis, LLMCompetitor, ResearchStatus
from app.services.competitor_service import (
    MAX_COMPETITORS,
    CompetitorService,
    ResearchResult,
    apply_provenance,
    normalize_source_url,
)
from tests.conftest import make_llm_payload, make_research, make_settings

SITE = "https://example.com"


def candidates() -> list[LLMCompetitor]:
    return LLMAnalysis.model_validate(make_llm_payload()).competitor_analysis.competitors


def run(cands=None, research=None, brand="Acme Cloud"):
    return apply_provenance(
        cands if cands is not None else candidates(),
        research if research is not None else make_research(),
        brand_name=brand,
        site_url=SITE,
    )


def make_candidate(name, *, urls=(), website=None, **extra) -> LLMCompetitor:
    base = {
        "name": name,
        "website_url": website,
        "overview": "Overview.",
        "comparison_with_brand": "Comparison.",
        "evidence_type": "retrieved_from_source",
        "source_urls": list(urls),
        "confidence": "high",
        "strengths": ["S"],
        "weaknesses_or_gaps": ["W"],
        "source_backed_facts": ["F"],
    }
    base.update(extra)
    return LLMCompetitor.model_validate(base)


# --------------------------------------------------------------------------- #
# Provenance enforcement
# --------------------------------------------------------------------------- #
def test_only_competitors_backed_by_retrieved_sources_stay_labelled_as_retrieved():
    competitors, status, _ = run()
    by_name = {c.name: c for c in competitors}

    rival = by_name["Rival Cloud Partners"]
    assert rival.evidence_type is EvidenceType.RETRIEVED_FROM_SOURCE
    assert rival.source_urls == ["https://reviews.example/top-aws-partners"]
    assert rival.website_url == "https://rivalcloud.example"  # host was surfaced by the search itself
    assert rival.strengths == ["Large team"] and rival.source_backed_facts

    assert by_name["Invented Corp"].evidence_type is EvidenceType.AI_UNVERIFIED  # cited URL was never retrieved
    assert by_name["Third Option Ltd"].evidence_type is EvidenceType.AI_UNVERIFIED
    assert status is ResearchStatus.PARTIALLY_SOURCED


def test_unverified_competitors_lose_website_sources_strengths_weaknesses_and_facts():
    invented = {c.name: c for c in run()[0]}["Invented Corp"]
    assert invented.website_url is None and invented.source_urls == []
    assert invented.strengths == [] and invented.weaknesses_or_gaps == [] and invented.source_backed_facts == []
    assert invented.target_market is None
    assert invented.confidence is Confidence.LOW
    assert "not verified" in invented.verification_note


def test_the_models_own_confidence_label_is_never_trusted():
    rival = {c.name: c for c in run()[0]}["Rival Cloud Partners"]
    assert rival.confidence is Confidence.LOW  # the LLM claimed "high"; one source host only


def test_confidence_is_medium_with_two_independent_source_hosts():
    cand = make_candidate(
        "Rival Cloud Partners",
        urls=["https://reviews.example/top-aws-partners", "https://rivalcloud.example/about"],
        website="https://rivalcloud.example",
    )
    competitors, status, _ = run([cand])
    assert competitors[0].confidence is Confidence.MEDIUM and status is ResearchStatus.EXTERNALLY_SOURCED


def test_competitor_must_be_named_in_the_cited_source():
    cand = make_candidate("Totally Different Name", urls=["https://reviews.example/top-aws-partners"])
    competitors, _, _ = run([cand])
    assert competitors[0].evidence_type is EvidenceType.AI_UNVERIFIED


def test_url_matching_ignores_fragments_www_and_trailing_slashes():
    cand = make_candidate("Rival Cloud Partners", urls=["https://www.reviews.example/top-aws-partners/#section"])
    assert run([cand])[0][0].evidence_type is EvidenceType.RETRIEVED_FROM_SOURCE
    assert normalize_source_url("HTTPS://WWW.Example.com/a/#x") == "https://example.com/a"


def test_websites_not_surfaced_by_the_research_are_dropped():
    cand = make_candidate("Rival Cloud Partners", urls=["https://reviews.example/top-aws-partners"], website="https://other-site.example")
    assert run([cand])[0][0].website_url is None


def test_without_research_every_competitor_is_an_unverified_suggestion():
    research = ResearchResult(status="unavailable", note="External research is not configured (set TAVILY_API_KEY to enable it).")
    competitors, status, note = run(research=research)
    assert len(competitors) == 3
    assert all(c.evidence_type is EvidenceType.AI_UNVERIFIED and c.website_url is None and c.source_urls == [] for c in competitors)
    assert status is ResearchStatus.AI_UNVERIFIED and "TAVILY_API_KEY" in note and "unverified" in note


def test_failed_research_is_explained():
    research = ResearchResult(status="failed", note="External research produced no sources: the research API key was rejected.")
    _, status, note = run(research=research)
    assert status is ResearchStatus.AI_UNVERIFIED and "rejected" in note


def test_brand_itself_duplicates_and_same_site_entries_are_removed():
    cands = [
        make_candidate("Acme Cloud"),  # the brand listed as its own competitor
        make_candidate("Rival", website="https://www.example.com/pricing"),  # the brand's own website
        make_candidate("Dup"),
        make_candidate("dup"),
        make_candidate("Real"),
    ]
    names = [c.name for c in run(cands, ResearchResult(status="unavailable"))[0]]
    assert names == ["Dup", "Real"]


def test_competitor_list_is_capped():
    cands = [make_candidate(f"Company {i}") for i in range(6)]
    assert len(run(cands, ResearchResult(status="unavailable"))[0]) <= MAX_COMPETITORS


def test_no_competitors_means_unavailable():
    competitors, status, note = run([], ResearchResult(status="unavailable"))
    assert competitors == [] and status is ResearchStatus.UNAVAILABLE and "No competitors" in note


def test_final_competitor_model_rejects_inconsistent_provenance():
    base = {"name": "X", "overview": "o", "comparison_with_brand": "c", "confidence": "low", "verification_note": "n"}
    with pytest.raises(ValidationError):  # "retrieved" without any source
        Competitor(**base, evidence_type="retrieved_from_source", source_urls=[])
    with pytest.raises(ValidationError):  # "unverified" must not carry a website
        Competitor(**base, evidence_type="ai_unverified_suggestion", website_url="https://x.example")


# --------------------------------------------------------------------------- #
# Tavily retrieval (mocked)
# --------------------------------------------------------------------------- #
def tavily(handler, key="tvly-test-key"):
    return CompetitorService(make_settings(tavily_api_key=key), transport=httpx.MockTransport(handler))


async def research(service):
    return await service.research(brand_name="Acme Cloud", site_url=SITE, category_hint="cloud devops")


RESULTS = {
    "results": [
        {"title": "Top AWS partners", "url": "https://reviews.example/top", "content": "Rival Cloud Partners is a consultancy."},
        {"title": "Acme Cloud | Home", "url": "https://www.example.com/home", "content": "own site"},  # must be excluded
        {"title": "No scheme", "url": "javascript:alert(1)", "content": "bad"},
    ]
}


@pytest.mark.anyio
async def test_research_is_unavailable_without_an_api_key():
    service = CompetitorService(make_settings(tavily_api_key=None))
    result = await research(service)
    assert not service.enabled and result.status == "unavailable" and "TAVILY_API_KEY" in result.note


@pytest.mark.anyio
async def test_research_retrieves_deduplicates_and_excludes_the_brands_own_site():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=RESULTS)

    result = await research(tavily(handler))
    assert result.status == "retrieved" and result.provider == "tavily"
    assert [s.url for s in result.sources] == ["https://reviews.example/top"]  # same hit from 2 queries -> once
    assert len(seen) == 2 and len(result.queries) == 2
    request = seen[0]
    assert request.headers["authorization"] == "Bearer tvly-test-key"
    body = json.loads(request.content)
    assert body["search_depth"] == "basic" and "Acme Cloud" in body["query"] and body["include_answer"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "make_response, expected_word",
    [
        (lambda: httpx.Response(401, json={}), "rejected"),
        (lambda: httpx.Response(429, json={}), "rate limit"),
        (lambda: httpx.Response(500, json={}), "HTTP 500"),
        (lambda: httpx.Response(200, content=b"not json"), "malformed"),
    ],
)
async def test_research_failures_never_raise_and_never_leak_the_key(make_response, expected_word):
    result = await research(tavily(lambda request: make_response()))
    assert result.status == "failed" and expected_word in result.note
    assert "tvly-test-key" not in result.note and result.sources == []


@pytest.mark.anyio
async def test_research_timeout_is_reported_as_failed():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    result = await research(tavily(handler))
    assert result.status == "failed" and "timed out" in result.note


@pytest.mark.anyio
async def test_one_failing_query_does_not_discard_the_other():
    def handler(request):
        if "vs" in json.loads(request.content)["query"] or "similar" in json.loads(request.content)["query"]:
            return httpx.Response(500, json={})
        return httpx.Response(200, json=RESULTS)

    assert (await research(tavily(handler))).status == "retrieved"
