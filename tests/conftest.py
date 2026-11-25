"""Shared fixtures and builders.  Nothing here touches the network."""

from __future__ import annotations

import copy
import socket
from typing import Any

import pytest

from app.config import Settings
from app.services.competitor_service import ResearchResult, ResearchSource
from app.services.html_parser import parse_html
from app.services.llm_service import LLMUsage, StructuredResult
from app.services.website_extractor import ExtractionResult

PUBLIC_IP = "93.184.216.34"

GOOD_HTML = """<!doctype html>
<html lang="en"><head>
<title>Acme Cloud | DevOps and Cloud Migration Experts</title>
<meta name="description" content="Acme helps mid-size companies migrate to AWS and automate delivery pipelines.">
<meta name="keywords" content="cloud migration, devops, aws consulting">
<meta property="og:site_name" content="Acme Cloud">
<link rel="canonical" href="https://example.com/">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Cloud Ltd","telephone":"+1 415 555 0100"}</script>
</head><body>
<header><nav><a href="/services/cloud-migration">Cloud Migration</a><a href="/services/devops">DevOps Automation</a></nav></header>
<main>
<h1>Cloud migration without the downtime</h1>
<p>We help mid-size companies move their workloads to AWS and automate how software is delivered, so
engineering teams can ship every day instead of every quarter.</p>
<h2>Our services</h2>
<ul><li>Cloud Migration</li><li>DevOps Automation</li><li>Managed Kubernetes</li></ul>
<p>Our consultants are certified AWS partners with more than ten years of experience in regulated industries
such as finance and healthcare, and we operate with fixed-scope engagements and transparent pricing.</p>
<h3>Contact</h3><p>Email <a href="mailto:hello@acme.test">hello@acme.test</a> to book a free assessment call.</p>
</main>
<footer>(c) 2026 Acme Cloud. All rights reserved.</footer>
</body></html>"""


def make_settings(**overrides: Any) -> Settings:
    """Settings that ignore any local .env file and the real environment."""
    values: dict[str, Any] = {
        "groq_api_key": "test-key-not-real",
        "tavily_api_key": None,
        "llm_max_retries": 2,
        "cache_ttl_seconds": 0,
        "rate_limit_per_minute": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_resolver(mapping: dict[str, list[str]]):
    """Async DNS resolver stand-in; unknown hosts fail like real DNS."""

    async def resolver(host: str, port: int) -> list[str]:
        if host not in mapping:
            raise socket.gaierror(-2, "Name or service not known")
        return mapping[host]

    return resolver


def make_extraction(url: str = "https://example.com", html: str = GOOD_HTML) -> ExtractionResult:
    page = parse_html(html, url)
    return ExtractionResult(
        requested_url=url,
        final_url=url,
        page=page,
        status_code=200,
        content_type="text/html",
        redirects=[],
        bytes_fetched=len(html),
        fetch_ms=25,
        content_quality="full" if len(page.text) >= 300 else "limited",
        warnings=list(page.warnings),
    )


def _query(text: str, platforms: list[str], intent: str) -> dict[str, Any]:
    return {
        "query": text,
        "platforms": platforms,
        "search_intent": intent,
        "business_relevance": "Matches a core service of the business.",
        "target_audience": "CTOs and engineering managers",
    }


def _service(name: str, origin: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name} for mid-size companies.",
        "target_industry": "Software and fintech",
        "target_customer": "Engineering leaders",
        "main_benefit": "Faster, safer delivery.",
        "evidence": "Listed under 'Our services' on the homepage." if origin == "extracted" else "Implied by the DevOps focus.",
        "origin": origin,
    }


def make_llm_payload() -> dict[str, Any]:
    """A complete, schema-valid LLM answer (fresh copy on every call)."""
    segment = {"name": "Mid-size SaaS companies", "description": "Teams of 50-500 engineers.", "why_relevant": "They run on AWS."}
    return copy.deepcopy(
        {
            "brand": {
                "name": "Acme Cloud",
                "category": "Cloud and DevOps consulting",
                "summary": "Acme helps mid-size companies migrate to AWS and automate software delivery.",
                "main_products": [],
                "main_services": ["Cloud Migration", "DevOps Automation", "Managed Kubernetes"],
                "target_market": "Mid-size companies in regulated industries",
            },
            "competitor_analysis": {
                "brand_overview": "A cloud consultancy focused on AWS migration and delivery automation.",
                "brand_competitive_advantages": ["Certified AWS partner", "Fixed-scope engagements"],
                "competitors": [
                    {
                        "name": "Rival Cloud Partners",
                        "website_url": "https://rivalcloud.example",
                        "overview": "AWS consultancy.",
                        "main_offerings": ["AWS migration"],
                        "target_market": "Enterprises",
                        "source_backed_facts": ["Listed as an AWS consulting partner."],
                        "strengths": ["Large team"],
                        "weaknesses_or_gaps": ["Higher prices"],
                        "comparison_with_brand": "Larger but less focused on mid-size firms.",
                        "positioning_opportunities": ["Emphasise fixed pricing"],
                        "evidence_type": "retrieved_from_source",
                        "source_urls": ["https://reviews.example/top-aws-partners"],
                        "confidence": "high",
                    },
                    {
                        "name": "Invented Corp",
                        "website_url": "https://invented.example",
                        "overview": "Another consultancy.",
                        "main_offerings": [],
                        "target_market": "SMBs",
                        "source_backed_facts": ["Claims to have 500 clients."],
                        "strengths": ["Brand awareness"],
                        "weaknesses_or_gaps": ["Slow delivery"],
                        "comparison_with_brand": "Competes on price.",
                        "positioning_opportunities": [],
                        "evidence_type": "retrieved_from_source",
                        "source_urls": ["https://made-up.example/page"],
                        "confidence": "high",
                    },
                    {
                        "name": "Third Option Ltd",
                        "website_url": None,
                        "overview": "A DevOps agency.",
                        "main_offerings": [],
                        "target_market": None,
                        "source_backed_facts": [],
                        "strengths": [],
                        "weaknesses_or_gaps": [],
                        "comparison_with_brand": "Overlaps on DevOps services.",
                        "positioning_opportunities": [],
                        "evidence_type": "ai_unverified_suggestion",
                        "source_urls": [],
                        "confidence": "low",
                    },
                ],
                "market_positioning_opportunities": ["Own the 'migration without downtime' message"],
            },
            "target_search_queries": [
                _query("aws migration consultant for mid-size company", ["google", "linkedin"], "commercial"),
                _query("how to migrate to aws without downtime", ["google"], "informational"),
                _query("devops automation services", ["google", "linkedin"], "commercial"),
                _query("managed kubernetes provider", ["google", "x"], "transactional"),
                _query("aws certified partner near me", ["google", "facebook"], "local"),
            ],
            "seo_improvements": {
                "keyword_recommendations": {
                    "primary_keyword": "aws migration services",
                    "secondary_keywords": ["devops consulting", "cloud migration"],
                    "long_tail_keywords": ["aws migration without downtime", "devops for regulated industries"],
                    "business_keywords": ["managed kubernetes"],
                },
                "seo_title": {
                    "existing_title": "MODEL-INVENTED TITLE",
                    "recommended_title": "AWS Migration Services | Acme Cloud",
                    "primary_keyword": "aws migration services",
                    "reason": "Leads with the primary keyword and keeps the brand.",
                },
                "meta_description": {
                    "existing_description": "MODEL-INVENTED DESCRIPTION",
                    "recommended_description": "Move to AWS with zero downtime. Acme Cloud delivers fixed-scope migrations.",
                    "keywords": ["aws migration"],
                    "relevance_explanation": "Highlights the main differentiator.",
                },
                "supporting_content": {
                    "service_landing_pages": ["AWS migration"],
                    "industry_pages": ["Cloud for fintech"],
                    "faq_topics": ["How long does an AWS migration take?"],
                    "blog_topics": ["Migration checklist"],
                    "location_pages": [],
                    "supporting_articles": ["Cost of cloud migration"],
                },
            },
            "top_services": [
                _service("Cloud Migration", "extracted"),
                _service("DevOps Automation", "extracted"),
                _service("Managed Kubernetes", "extracted"),
                _service("Quantum Teleportation Consulting", "extracted"),
                _service("Cloud Cost Optimization", "inferred"),
            ],
            "target_audience": {
                "primary_audience": segment,
                "secondary_audience": {**segment, "name": "Regulated-industry IT teams"},
                "industry_segments": [segment],
                "business_types": ["SaaS", "Fintech"],
                "customer_profiles": [
                    {"profile": "CTO", "description": "Owns delivery speed.", "needs": ["Faster releases"], "challenges": ["Legacy infrastructure"]}
                ],
                "geographic_focus": {"regions": [], "basis": "No geographic focus is evident on the page."},
                "customer_requirements": ["Zero-downtime migration"],
                "customer_challenges": ["Skills shortage"],
                "potential_interests": ["Kubernetes", "FinOps"],
                "recommended_marketing_segments": [
                    {"name": "AWS-hosted SaaS CTOs", "description": "Target on LinkedIn.", "suggested_channels": ["linkedin"], "why_relevant": "Decision makers."}
                ],
            },
        }
    )


def make_research() -> ResearchResult:
    """Retrieved sources covering one of the two 'retrieved' competitors above."""
    return ResearchResult(
        status="retrieved",
        queries=["Acme Cloud competitors and alternatives"],
        sources=[
            ResearchSource(
                title="Top AWS partners",
                url="https://reviews.example/top-aws-partners",
                snippet="Rival Cloud Partners is a leading AWS consultancy.",
            ),
            ResearchSource(
                title="Rival Cloud Partners | About",
                url="https://rivalcloud.example/about",
                snippet="Rival Cloud Partners helps enterprises move to AWS.",
            ),
        ],
        note="2 third-party source(s) retrieved.",
        provider="tavily",
    )


class FakeExtractor:
    def __init__(self, result: ExtractionResult | None = None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, 0

    async def extract(self, url: str) -> ExtractionResult:
        self.calls += 1
        if self.error:
            raise self.error
        return self.result or make_extraction(url)


class FakeLLM:
    """Stands in for LLMService; validates the payload with the real schema."""

    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None, configured: bool = True) -> None:
        self.payload = payload if payload is not None else make_llm_payload()
        self.error, self.configured = error, configured
        self.calls: list[dict[str, str]] = []

    def ensure_configured(self) -> None:
        if not self.configured:
            from app.exceptions import LLMNotConfiguredError

            raise LLMNotConfiguredError()

    async def generate_structured(self, *, system: str, user: str, response_model: Any) -> StructuredResult:
        self.calls.append({"system": system, "user": user})
        if self.error:
            raise self.error
        return StructuredResult(
            data=response_model.model_validate(self.payload),
            provider="fake",
            model="fake-model",
            usage=LLMUsage(10_000, 5_000, 1_000),
            attempts=1,
            repaired=False,
            latency_ms=12,
        )


class FakeResearch:
    def __init__(self, result: ResearchResult | None = None) -> None:
        self.result = result or ResearchResult(status="unavailable", note="External research is not configured (set TAVILY_API_KEY to enable it).")
        self.calls = 0

    async def research(self, *, brand_name: str | None, site_url: str, category_hint: str | None) -> ResearchResult:
        self.calls += 1
        return self.result


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    return make_settings()
