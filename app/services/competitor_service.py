"""Competitor research and provenance enforcement.

Why this module exists
----------------------
A language model asked for "competitors" will happily produce plausible-looking
names, URLs and "facts".  This module makes the output honest:

* :class:`CompetitorService` optionally retrieves *real* web sources (Tavily) for
  the brand.  Retrieval failures never fail the analysis; they only downgrade it.
* :func:`apply_provenance` re-checks every competitor the LLM proposed against
  what was actually retrieved.  The LLM's own ``evidence_type`` label is ignored.

  - A competitor is ``retrieved_from_source`` only if it cites at least one URL
    that was really retrieved **and** its name appears in that source's text.
  - Anything else is forced to ``ai_unverified_suggestion``: website, sources,
    strengths, weaknesses and "source-backed facts" are removed.
  - ``confidence`` is derived from the number of independent source hosts, never
    taken from the model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urldefrag, urlsplit

import httpx

from app.config import Settings
from app.schemas import (
    Competitor,
    Confidence,
    EvidenceType,
    LLMCompetitor,
    ResearchStatus,
)

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
MAX_COMPETITORS = 5
SNIPPET_CHARS = 600

UNVERIFIED_NOTE = (
    "AI-generated candidate; not verified against any external source. Website, strengths and "
    "weaknesses are intentionally omitted."
)


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    snippet: str


@dataclass
class ResearchResult:
    status: Literal["retrieved", "unavailable", "failed"]
    queries: list[str] = field(default_factory=list)
    sources: list[ResearchSource] = field(default_factory=list)
    note: str = ""
    provider: str | None = None


class ResearchFailure(Exception):
    """Internal: a single search call failed (message is safe to log and show)."""


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #
def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def normalize_source_url(url: str) -> str:
    """Comparable form of a URL (no fragment, no www, no trailing slash)."""
    parts = urlsplit(urldefrag(url.strip())[0])
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{host_of(url)}{path}{query}"


def _same_site(host: str, own_host: str) -> bool:
    return bool(host) and (host == own_host or host.endswith("." + own_host))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class CompetitorService:
    """Optional external research for competitor identification."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
        self._timeout = settings.research_timeout
        self._max_results = settings.research_max_results
        self._transport = transport

    @property
    def enabled(self) -> bool:
        return self._key is not None

    async def research(self, *, brand_name: str | None, site_url: str, category_hint: str | None) -> ResearchResult:
        """Retrieve third-party sources about the brand's competitors."""
        if not self.enabled:
            return ResearchResult(
                status="unavailable",
                note="External research is not configured (set TAVILY_API_KEY to enable it).",
            )
        own_host = host_of(site_url)
        name = (brand_name or own_host).strip()
        queries = [f"{name} competitors and alternatives"]
        queries.append(f"companies similar to {name} {category_hint}" if category_hint else f"{name} vs")

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout), transport=self._transport) as client:
            outcomes = await asyncio.gather(
                *(self._search(client, q) for q in queries), return_exceptions=True
            )

        sources: dict[str, ResearchSource] = {}
        errors: list[str] = []
        for outcome in outcomes:
            if isinstance(outcome, ResearchFailure):
                errors.append(str(outcome))
            elif isinstance(outcome, BaseException):
                errors.append("unexpected research error")
                logger.warning("research error", extra={"error_type": type(outcome).__name__})
            else:
                for source in outcome:
                    if _same_site(host_of(source.url), own_host):
                        continue  # the brand's own pages are not competitor evidence
                    sources.setdefault(normalize_source_url(source.url), source)

        if sources:
            return ResearchResult(
                status="retrieved", queries=queries, sources=list(sources.values()),
                note=f"{len(sources)} third-party source(s) retrieved.", provider="tavily",
            )
        reason = errors[0] if errors else "the search returned no usable third-party results"
        return ResearchResult(
            status="failed", queries=queries, note=f"External research produced no sources: {reason}.",
            provider="tavily",
        )

    async def _search(self, client: httpx.AsyncClient, query: str) -> list[ResearchSource]:
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": self._max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            response = await client.post(
                TAVILY_URL, headers={"Authorization": f"Bearer {self._key}"}, json=payload
            )
        except httpx.TimeoutException as exc:
            raise ResearchFailure("the research provider timed out") from exc
        except httpx.HTTPError as exc:
            raise ResearchFailure("the research provider could not be reached") from exc

        if response.status_code in (401, 403):
            raise ResearchFailure("the research API key was rejected")
        if response.status_code == 429:
            raise ResearchFailure("the research provider rate limit was reached")
        if response.status_code != 200:
            raise ResearchFailure(f"the research provider returned HTTP {response.status_code}")
        try:
            results = response.json().get("results", [])
        except (ValueError, AttributeError) as exc:
            raise ResearchFailure("the research provider returned malformed data") from exc

        sources: list[ResearchSource] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            url, title, content = item.get("url"), item.get("title"), item.get("content")
            if isinstance(url, str) and urlsplit(url).scheme in ("http", "https"):
                sources.append(
                    ResearchSource(
                        title=" ".join(str(title or url).split())[:200],
                        url=url,
                        snippet=" ".join(str(content or "").split())[:SNIPPET_CHARS],
                    )
                )
        return sources


# --------------------------------------------------------------------------- #
# Provenance enforcement
# --------------------------------------------------------------------------- #
def _mentions(name: str, source: ResearchSource) -> bool:
    return name.casefold() in f"{source.title} {source.snippet}".casefold()


def _unverified(candidate: LLMCompetitor) -> Competitor:
    """Strip everything an unverified competitor must not claim."""
    return Competitor(
        name=candidate.name,
        website_url=None,
        overview=candidate.overview,
        main_offerings=candidate.main_offerings,
        target_market=None,
        source_backed_facts=[],
        strengths=[],
        weaknesses_or_gaps=[],
        comparison_with_brand=candidate.comparison_with_brand,
        positioning_opportunities=candidate.positioning_opportunities,
        evidence_type=EvidenceType.AI_UNVERIFIED,
        source_urls=[],
        confidence=Confidence.LOW,
        verification_note=UNVERIFIED_NOTE,
    )


def apply_provenance(
    candidates: list[LLMCompetitor],
    research: ResearchResult,
    *,
    brand_name: str,
    site_url: str,
) -> tuple[list[Competitor], ResearchStatus, str]:
    """Return verified competitors, the overall research status and a note."""
    own_host = host_of(site_url)
    sources_by_url = {normalize_source_url(s.url): s for s in research.sources}
    retrieved_hosts = {host_of(s.url) for s in research.sources}
    research_usable = research.status == "retrieved" and bool(sources_by_url)

    competitors: list[Competitor] = []
    seen: set[str] = {brand_name.casefold()}
    for candidate in candidates:
        key = candidate.name.casefold()
        if key in seen or _same_site(host_of(candidate.website_url or ""), own_host):
            continue  # duplicates and the brand listing itself
        seen.add(key)

        cited: list[ResearchSource] = []
        if research_usable:
            for url in candidate.source_urls:
                source = sources_by_url.get(normalize_source_url(url))
                if source is not None and source not in cited:
                    cited.append(source)

        if cited and any(_mentions(candidate.name, s) for s in cited):
            website = candidate.website_url
            if not website or urlsplit(website).scheme not in ("http", "https") or host_of(website) not in retrieved_hosts:
                website = None  # only keep a URL that the search itself surfaced
            distinct_hosts = {host_of(s.url) for s in cited}
            competitors.append(
                Competitor(
                    **candidate.model_dump(exclude={"website_url", "source_urls", "evidence_type", "confidence"}),
                    website_url=website,
                    source_urls=[s.url for s in cited][:4],
                    evidence_type=EvidenceType.RETRIEVED_FROM_SOURCE,
                    confidence=Confidence.MEDIUM if len(distinct_hosts) >= 2 else Confidence.LOW,
                    verification_note=(
                        f"Named in {len(cited)} retrieved source(s) from {len(distinct_hosts)} site(s). "
                        "'source_backed_facts' come from those sources; strengths, weaknesses and the "
                        "comparison are AI interpretation."
                    ),
                )
            )
        else:
            competitors.append(_unverified(candidate))
        if len(competitors) == MAX_COMPETITORS:
            break

    return competitors, *_overall_status(competitors, research)


def _overall_status(competitors: list[Competitor], research: ResearchResult) -> tuple[ResearchStatus, str]:
    if not competitors:
        return ResearchStatus.UNAVAILABLE, "No competitors could be identified reliably from the available evidence."
    sourced = sum(c.evidence_type is EvidenceType.RETRIEVED_FROM_SOURCE for c in competitors)
    if sourced == len(competitors):
        return ResearchStatus.EXTERNALLY_SOURCED, f"All competitors are backed by retrieved sources. {research.note}"
    if sourced:
        return (
            ResearchStatus.PARTIALLY_SOURCED,
            f"{sourced} of {len(competitors)} competitors are backed by retrieved sources; the rest are "
            "AI-generated, unverified candidates.",
        )
    reason = (
        "none of the suggested competitors could be matched to a retrieved source"
        if research.status == "retrieved"
        else research.note.rstrip(".")
    )
    return (
        ResearchStatus.AI_UNVERIFIED,
        f"Competitors are AI-generated candidates and are unverified: {reason}.",
    )
