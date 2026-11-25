"""Pydantic models.

Layout
------
* ``LinkAnalysisRequest`` – request body.
* ``LLM*`` models and :class:`LLMAnalysis` – the *only* shape the language model
  is asked to produce.  They are sent to the provider as a JSON Schema and used
  to validate its reply.  They contain no server-controlled facts.
* The un-prefixed models (``Competitor``, ``SearchQuery``, ``TopService`` ...) are
  the public API shapes.  Where the server must add or override information
  (provenance, search-metric placeholders, existing title/description, contact
  details) the public model *extends* the LLM model.
* :class:`AnalysisResponse` – the complete response document.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Platform(str, Enum):
    GOOGLE = "google"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    X = "x"


class SearchIntent(str, Enum):
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"
    LOCAL = "local"


class EvidenceType(str, Enum):
    """How a competitor profile is supported."""

    RETRIEVED_FROM_SOURCE = "retrieved_from_source"  # named in retrieved external sources
    AI_UNVERIFIED = "ai_unverified_suggestion"  # LLM suggestion, not verified


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ServiceOrigin(str, Enum):
    EXTRACTED = "extracted"  # stated on the website
    INFERRED = "inferred"  # reasoned from the website's content


class ResearchStatus(str, Enum):
    EXTERNALLY_SOURCED = "externally_sourced"  # every competitor backed by retrieved sources
    PARTIALLY_SOURCED = "partially_sourced"
    AI_UNVERIFIED = "ai_unverified"  # no external research: AI-suggested candidates only
    UNAVAILABLE = "unavailable"  # no competitors could be identified


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
class LinkAnalysisRequest(BaseModel):
    """Body of ``POST /link-analysis``."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{"url": "https://example.com"}]},
    )

    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Public http(s) URL of the website to analyze.",
    )


# --------------------------------------------------------------------------- #
# Brand
# --------------------------------------------------------------------------- #
class BrandDetails(BaseModel):
    """Brand facts the LLM derives from the website."""

    name: str = Field(max_length=120, description="Brand or company name.")
    category: str = Field(max_length=160, description="Business category / industry.")
    summary: str = Field(
        max_length=700, description="Evidence-based summary of what the business does (2-3 sentences)."
    )
    main_products: list[str] = Field(
        default_factory=list, max_length=8, description="Products named on the site; empty if none."
    )
    main_services: list[str] = Field(
        default_factory=list, max_length=8, description="Services named on the site; empty if none."
    )
    target_market: str | None = Field(
        default=None, max_length=300, description="Who the business sells to, if evident."
    )


class ContactInfo(BaseModel):
    """Contact details extracted directly from the page (never LLM-generated)."""

    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    addresses: list[str] = Field(default_factory=list)


class BrandInfo(BrandDetails):
    """Public brand block: LLM facts plus extracted contact/social data."""

    contact: ContactInfo = Field(default_factory=ContactInfo)
    social_links: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Section 1 - competitor analysis
# --------------------------------------------------------------------------- #
class LLMCompetitor(BaseModel):
    """Competitor as proposed by the LLM (provenance is re-checked server-side)."""

    name: str = Field(max_length=120)
    website_url: str | None = Field(
        default=None,
        max_length=300,
        description="Official URL, only if it appears in the provided research; otherwise null.",
    )
    overview: str = Field(max_length=500)
    main_offerings: list[str] = Field(default_factory=list, max_length=6)
    target_market: str | None = Field(default=None, max_length=250)
    source_backed_facts: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Statements directly supported by the provided research sources; else empty.",
    )
    strengths: list[str] = Field(default_factory=list, max_length=5)
    weaknesses_or_gaps: list[str] = Field(default_factory=list, max_length=5)
    comparison_with_brand: str = Field(max_length=500)
    positioning_opportunities: list[str] = Field(default_factory=list, max_length=4)
    evidence_type: EvidenceType
    source_urls: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="URLs copied verbatim from the provided research; empty if none.",
    )
    confidence: Confidence


class Competitor(LLMCompetitor):
    """Competitor after server-side provenance verification."""

    verification_note: str = Field(description="Plain-language statement of how well this entry is supported.")

    @model_validator(mode="after")
    def _enforce_provenance(self) -> "Competitor":
        if self.evidence_type is EvidenceType.RETRIEVED_FROM_SOURCE and not self.source_urls:
            raise ValueError("retrieved_from_source competitors must list at least one source URL")
        if self.evidence_type is EvidenceType.AI_UNVERIFIED and (self.website_url or self.source_urls):
            raise ValueError("ai_unverified_suggestion competitors must not carry a website or sources")
        return self


class LLMCompetitorAnalysis(BaseModel):
    brand_overview: str = Field(max_length=800)
    brand_competitive_advantages: list[str] = Field(default_factory=list, max_length=6)
    competitors: list[LLMCompetitor] = Field(default_factory=list, max_length=6)
    market_positioning_opportunities: list[str] = Field(default_factory=list, max_length=6)


class YourBrandProfile(BaseModel):
    """The submitted brand, as shown beside its competitors."""

    name: str
    overview: str
    category: str
    main_products: list[str]
    main_services: list[str]
    target_market: str | None
    competitive_advantages: list[str]


class CompetitorAnalysis(BaseModel):
    your_brand: YourBrandProfile
    research_status: ResearchStatus
    research_note: str
    competitors: list[Competitor]
    market_positioning_opportunities: list[str]


# --------------------------------------------------------------------------- #
# Section 2 - target search queries
# --------------------------------------------------------------------------- #
class LLMSearchQuery(BaseModel):
    query: str = Field(min_length=3, max_length=160)
    platforms: list[Platform] = Field(min_length=1, max_length=5)
    search_intent: SearchIntent
    business_relevance: str = Field(max_length=300)
    target_audience: str = Field(max_length=200)

    @field_validator("platforms")
    @classmethod
    def _unique_platforms(cls, value: list[Platform]) -> list[Platform]:
        return list(dict.fromkeys(value))


class SearchMetrics(BaseModel):
    """Explicit placeholders: no search-performance data source is integrated."""

    search_volume: Literal["unavailable"] = "unavailable"
    keyword_difficulty: Literal["unavailable"] = "unavailable"
    ranking: Literal["unavailable"] = "unavailable"
    note: str = "No keyword-data provider is integrated; performance figures are intentionally not estimated."


class SearchQuery(LLMSearchQuery):
    metrics: SearchMetrics = Field(default_factory=SearchMetrics)


# --------------------------------------------------------------------------- #
# Section 3 - SEO improvements
# --------------------------------------------------------------------------- #
class KeywordRecommendations(BaseModel):
    primary_keyword: str = Field(max_length=80)
    secondary_keywords: list[str] = Field(min_length=2, max_length=8)
    long_tail_keywords: list[str] = Field(min_length=2, max_length=8)
    business_keywords: list[str] = Field(default_factory=list, max_length=8)


class SEOTitle(BaseModel):
    existing_title: str | None = Field(default=None, max_length=400, description="Set by the server.")
    recommended_title: str = Field(max_length=80, description="Aim for 60 characters or fewer.")
    primary_keyword: str = Field(max_length=80)
    reason: str = Field(max_length=400)


class MetaDescription(BaseModel):
    existing_description: str | None = Field(default=None, max_length=800, description="Set by the server.")
    recommended_description: str = Field(max_length=200, description="Aim for 155 characters or fewer.")
    keywords: list[str] = Field(default_factory=list, max_length=8)
    relevance_explanation: str = Field(max_length=400)


class SupportingContent(BaseModel):
    service_landing_pages: list[str] = Field(default_factory=list, max_length=6)
    industry_pages: list[str] = Field(default_factory=list, max_length=6)
    faq_topics: list[str] = Field(default_factory=list, max_length=6)
    blog_topics: list[str] = Field(default_factory=list, max_length=6)
    location_pages: list[str] = Field(
        default_factory=list, max_length=6, description="Only when the business is location-dependent."
    )
    supporting_articles: list[str] = Field(default_factory=list, max_length=6)


class SEOImprovements(BaseModel):
    keyword_recommendations: KeywordRecommendations
    seo_title: SEOTitle
    meta_description: MetaDescription
    supporting_content: SupportingContent


# --------------------------------------------------------------------------- #
# Section 4 - top services
# --------------------------------------------------------------------------- #
class LLMTopService(BaseModel):
    name: str = Field(max_length=120)
    description: str = Field(max_length=400)
    target_industry: str = Field(max_length=160)
    target_customer: str = Field(max_length=200)
    main_benefit: str = Field(max_length=300)
    evidence: str = Field(
        max_length=400,
        description="Where on the site the service appears (extracted) or why it is inferred.",
    )
    origin: ServiceOrigin


class TopService(LLMTopService):
    verification_note: str | None = Field(
        default=None, description="Set when the server changed the origin label."
    )


# --------------------------------------------------------------------------- #
# Section 5 - target audience
# --------------------------------------------------------------------------- #
class AudienceSegment(BaseModel):
    name: str = Field(max_length=120)
    description: str = Field(max_length=350)
    why_relevant: str = Field(max_length=350)


class CustomerProfile(BaseModel):
    profile: str = Field(max_length=120)
    description: str = Field(max_length=300)
    needs: list[str] = Field(default_factory=list, max_length=4)
    challenges: list[str] = Field(default_factory=list, max_length=4)


class GeographicFocus(BaseModel):
    regions: list[str] = Field(default_factory=list, max_length=6)
    basis: str = Field(
        max_length=300, description="Evidence for the focus, or a statement that none is evident."
    )


class MarketingSegment(BaseModel):
    name: str = Field(max_length=120)
    description: str = Field(max_length=300)
    suggested_channels: list[str] = Field(default_factory=list, max_length=5)
    why_relevant: str = Field(max_length=300)


class TargetAudience(BaseModel):
    primary_audience: AudienceSegment
    secondary_audience: AudienceSegment
    industry_segments: list[AudienceSegment] = Field(min_length=1, max_length=5)
    business_types: list[str] = Field(min_length=1, max_length=6)
    customer_profiles: list[CustomerProfile] = Field(min_length=1, max_length=3)
    geographic_focus: GeographicFocus
    customer_requirements: list[str] = Field(min_length=1, max_length=6)
    customer_challenges: list[str] = Field(min_length=1, max_length=6)
    potential_interests: list[str] = Field(min_length=1, max_length=6)
    recommended_marketing_segments: list[MarketingSegment] = Field(min_length=1, max_length=4)


# --------------------------------------------------------------------------- #
# LLM output (the schema sent to the model)
# --------------------------------------------------------------------------- #
class LLMAnalysis(BaseModel):
    """Complete structured output requested from the language model."""

    brand: BrandDetails
    competitor_analysis: LLMCompetitorAnalysis
    target_search_queries: list[LLMSearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: list[LLMTopService] = Field(min_length=5, max_length=5)
    target_audience: TargetAudience


# --------------------------------------------------------------------------- #
# Response metadata
# --------------------------------------------------------------------------- #
class ExtractionMetadata(BaseModel):
    final_url: str
    status_code: int
    content_type: str | None
    redirects: list[str]
    bytes_fetched: int
    fetch_ms: int
    text_chars: int
    text_truncated: bool
    content_quality: Literal["full", "limited"]
    warnings: list[str]


class LLMMetadata(BaseModel):
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    attempts: int
    repaired: bool
    latency_ms: int
    estimated_cost_usd: float | None
    cost_note: str


class ResearchSourceRef(BaseModel):
    title: str
    url: str


class ResearchMetadata(BaseModel):
    provider: str | None
    status: Literal["retrieved", "unavailable", "failed"]
    queries: list[str]
    sources: list[ResearchSourceRef]
    note: str


class AnalysisMetadata(BaseModel):
    analyzed_at: datetime
    cached: bool = False
    extraction: ExtractionMetadata
    llm: LLMMetadata
    research: ResearchMetadata
    limitations: list[str]


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #
class AnalysisSections(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: list[TopService] = Field(min_length=5, max_length=5)
    target_audience: TargetAudience


class AnalysisResponse(BaseModel):
    """Response of ``POST /link-analysis``."""

    success: Literal[True] = True
    website: str
    brand: BrandInfo
    analysis: AnalysisSections
    metadata: AnalysisMetadata


# --------------------------------------------------------------------------- #
# Errors & health
# --------------------------------------------------------------------------- #
class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody


class ComponentStatus(BaseModel):
    provider: str | None
    model: str | None = None
    configured: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
    llm: ComponentStatus
    research: ComponentStatus
