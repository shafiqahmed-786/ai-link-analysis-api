"""Pure HTML -> structured data conversion.

Kept separate from :mod:`app.services.website_extractor` (which owns network
access and SSRF protection) so that parsing can be tested with plain strings.

The parser never raises on malformed markup; lxml repairs broken HTML and every
extraction step is defensive about missing elements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag, UnicodeDammit
from bs4.element import NavigableString, PreformattedString

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
HEADING_LIMITS = {"h1": 10, "h2": 30, "h3": 30}
MAX_HEADING_CHARS = 200
MAX_LD_ITEMS = 8
MAX_LD_STRING = 300
MAX_RELEVANT_LINKS = 30
MAX_SOCIAL_LINKS = 10
MAX_SERVICE_HINTS = 20
MAX_NAME_CANDIDATES = 6

#: Elements that never contain visible page content.
_NOISE_TAGS = (
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed",
    "form", "button", "select", "option", "nav", "footer", "aside", "dialog",
)
_NOISE_ROLES = {"navigation", "contentinfo", "dialog", "alertdialog"}
_NOISE_ATTR = re.compile(r"cookie|consent|gdpr|onetrust|cookiebot|cc-window|cc-banner", re.IGNORECASE)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
_BLOCK_TAGS = frozenset(
    "address article blockquote body dd details div dl dt fieldset figcaption figure h1 h2 h3 h4 h5 h6 "
    "header hr li main ol p pre section summary table tbody td tfoot th thead tr ul".split()
)
_BOILERPLATE_LINE = re.compile(r"^(?:©|\(c\)|copyright\b)|all rights reserved", re.IGNORECASE)

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,24}")
_FILE_SUFFIXES = {"png", "jpg", "jpeg", "gif", "svg", "webp", "avif", "css", "js"}

_SOCIAL_DOMAINS = (
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "github.com", "tiktok.com",
)
_RELEVANT_PATH_KEYWORDS = (
    "service", "solution", "product", "about", "contact", "pricing", "industr", "blog", "case-stud",
    "portfolio", "platform", "feature", "expertise", "capabilit", "offering", "faq", "resources",
    "customers", "what-we-do", "work",
)
_SERVICE_PATH_KEYWORDS = (
    "service", "solution", "product", "what-we-do", "offering", "capabilit", "expertise", "platform",
)
_LD_TYPE_HINTS = (
    "organization", "business", "corporation", "store", "service", "product", "website", "offer",
    "brand", "faqpage", "softwareapplication", "restaurant", "hotel", "school",
)
_TITLE_SEPARATORS = re.compile(r"\s[|\-–—·:]\s")

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts?|rules)", re.I),
    re.compile(r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|your)\s+(?:instructions|prompts?|rules)", re.I),
    re.compile(r"(?:reveal|print|show|output)\s+(?:your\s+)?(?:system\s+prompt|instructions|api[\s_-]?keys?)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?(?:developer|dan|jailbreak)", re.I),
    re.compile(r"<\|(?:im_start|im_end|system)\|>", re.I),
)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class PageLink:
    url: str
    text: str


@dataclass
class BusinessInfo:
    name_candidates: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    service_hints: list[str] = field(default_factory=list)


@dataclass
class ParsedPage:
    """Everything extracted from one HTML document."""

    title: str | None = None
    meta_description: str | None = None
    meta_keywords: list[str] = field(default_factory=list)
    open_graph: dict[str, str] = field(default_factory=dict)
    canonical_url: str | None = None
    language: str | None = None
    headings: dict[str, list[str]] = field(default_factory=lambda: {"h1": [], "h2": [], "h3": []})
    structured_data: list[dict[str, Any]] = field(default_factory=list)
    relevant_links: list[PageLink] = field(default_factory=list)
    social_links: list[str] = field(default_factory=list)
    business: BusinessInfo = field(default_factory=BusinessInfo)
    text: str = ""
    text_truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def evidence_chars(self) -> int:
        """Rough amount of usable content (text + description + headings)."""
        heading_chars = sum(len(h) for items in self.headings.values() for h in items)
        return len(self.text) + len(self.meta_description or "") + heading_chars


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def decode_html(body: bytes, declared_encoding: str | None = None) -> str:
    """Decode raw bytes using the declared charset, then sniffing, then UTF-8."""
    if declared_encoding:
        try:
            return body.decode(declared_encoding, errors="replace")
        except LookupError:
            pass
    dammit = UnicodeDammit(body, is_html=True)
    return dammit.unicode_markup or body.decode("utf-8", errors="replace")


def parse_html(html: str, base_url: str, *, max_text_chars: int = 50_000) -> ParsedPage:
    """Parse ``html`` (already decoded) into a :class:`ParsedPage`."""
    soup = BeautifulSoup(html or "", "lxml")
    page = ParsedPage()

    root = soup.find("html")
    lang = root.get("lang") if isinstance(root, Tag) else None
    page.language = _clean(lang) if isinstance(lang, str) else None

    script_count = sum(
        1 for s in soup.find_all("script") if "ld+json" not in str(s.get("type") or "").lower()
    )
    page.title = _clean(soup.title.get_text()) if soup.title else None
    _read_meta(soup, page)
    page.canonical_url = _read_canonical(soup, base_url)
    page.structured_data = _read_json_ld(soup, page)
    all_links = _read_links(soup, base_url, page)  # before nav/footer are removed

    _strip_noise(soup)
    page.headings = _read_headings(soup)
    lines = _visible_lines(soup)
    page.text, page.text_truncated = _join_limited(lines, max_text_chars)

    page.business = _read_business(soup, page, all_links, lines)
    _add_warnings(page, script_count)
    return page


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _clean(value: str | None, limit: int | None = None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text[:limit] if limit else text


def _read_meta(soup: BeautifulSoup, page: ParsedPage) -> None:
    for meta in soup.find_all("meta"):
        content = _clean(meta.get("content"), 1000) if isinstance(meta.get("content"), str) else None
        if not content:
            continue
        name = str(meta.get("name") or "").lower()
        prop = str(meta.get("property") or "").lower()
        if name == "description" and page.meta_description is None:
            page.meta_description = content
        elif name == "keywords" and not page.meta_keywords:
            page.meta_keywords = [k for k in (_clean(p, 80) for p in content.split(",")) if k][:20]
        elif prop.startswith("og:") and prop not in page.open_graph:
            page.open_graph[prop] = content


def _read_canonical(soup: BeautifulSoup, base_url: str) -> str | None:
    link = soup.find("link", rel="canonical")
    href = link.get("href") if isinstance(link, Tag) else None
    if not isinstance(href, str) or not href.strip():
        return None
    absolute = urldefrag(urljoin(base_url, href.strip()))[0]
    return absolute if urlsplit(absolute).scheme in ("http", "https") else None


# --------------------------------------------------------------------------- #
# JSON-LD
# --------------------------------------------------------------------------- #
def _read_json_ld(soup: BeautifulSoup, page: ParsedPage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for script in soup.find_all("script", type=re.compile(r"ld\+json", re.IGNORECASE)):
        raw = script.string if script.string is not None else script.get_text()
        try:
            data = json.loads(raw.strip())
        except (ValueError, AttributeError):
            page.warnings.append("A JSON-LD block was invalid and was ignored.")
            continue
        for node in _flatten_ld(data):
            compact = _compact_ld(node)
            if compact:
                items.append(compact)
            if len(items) >= MAX_LD_ITEMS:
                return items
    return items


def _flatten_ld(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [n for item in data for n in _flatten_ld(item)]
    if isinstance(data, dict):
        if isinstance(data.get("@graph"), list):
            return [n for item in data["@graph"] for n in _flatten_ld(item)]
        return [data]
    return []


def _ld_types(node: dict[str, Any]) -> list[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return [str(v) for v in values if isinstance(v, str)]


def _ld_names(value: Any, limit: int = 8) -> list[str]:
    """Names from a string, a dict with ``name`` or a list of either."""
    values = value if isinstance(value, list) else [value]
    names: list[str] = []
    for item in values:
        if isinstance(item, str):
            text = _clean(item, MAX_LD_STRING)
        elif isinstance(item, dict):
            text = _clean(item.get("name") if isinstance(item.get("name"), str) else None, MAX_LD_STRING)
            if text is None and isinstance(item.get("itemListElement"), list):
                names.extend(_ld_names(item["itemListElement"], limit))
                continue
            if text is None and isinstance(item.get("itemOffered"), dict):
                text = _clean(item["itemOffered"].get("name"), MAX_LD_STRING)
        else:
            text = None
        if text:
            names.append(text)
    return list(dict.fromkeys(names))[:limit]


def _ld_address(value: Any) -> str | None:
    if isinstance(value, str):
        return _clean(value, 200)
    if isinstance(value, dict):
        parts = [value.get(k) for k in ("streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry")]
        flat = [p if isinstance(p, str) else (p.get("name") if isinstance(p, dict) else None) for p in parts]
        return _clean(", ".join(p for p in flat if p), 200)
    return None


def _compact_ld(node: dict[str, Any]) -> dict[str, Any] | None:
    types = _ld_types(node)
    if not any(hint in t.lower() for t in types for hint in _LD_TYPE_HINTS):
        return None
    out: dict[str, Any] = {"@type": types[0] if len(types) == 1 else types}
    for key in ("name", "legalName", "alternateName", "description", "url", "telephone", "email",
                "slogan", "serviceType", "category", "foundingDate"):
        text = _clean(node.get(key), MAX_LD_STRING) if isinstance(node.get(key), str) else None
        if text:
            out[key] = text
    address = _ld_address(node.get("address"))
    if address:
        out["address"] = address
    for key in ("areaServed", "brand", "knowsAbout", "makesOffer", "hasOfferCatalog", "offers"):
        names = _ld_names(node.get(key))
        if names:
            out[key] = names
    same_as = node.get("sameAs")
    if isinstance(same_as, str):
        same_as = [same_as]
    if isinstance(same_as, list):
        links = [s for s in same_as if isinstance(s, str)][:10]
        if links:
            out["sameAs"] = links
    return out if len(out) > 1 else None


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #
def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _read_links(soup: BeautifulSoup, base_url: str, page: ParsedPage) -> list[PageLink]:
    base_host = _host(base_url)
    seen: set[str] = set()
    social: list[str] = []
    internal: list[PageLink] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:", "sms:")):
            continue
        absolute = urldefrag(urljoin(base_url, href))[0]
        if urlsplit(absolute).scheme not in ("http", "https") or absolute in seen:
            continue
        seen.add(absolute)
        host = _host(absolute)
        if any(host == d or host.endswith("." + d) for d in _SOCIAL_DOMAINS):
            if host != base_host and len(social) < MAX_SOCIAL_LINKS:
                social.append(absolute)
            continue
        if host == base_host:
            text = _clean(anchor.get_text(" "), 80) or ""
            internal.append(PageLink(url=absolute, text=text))

    page.social_links = social
    relevant = [
        link for link in internal
        if any(k in urlsplit(link.url).path.lower() or k in link.text.lower() for k in _RELEVANT_PATH_KEYWORDS)
    ]
    page.relevant_links = relevant[:MAX_RELEVANT_LINKS]
    return internal


# --------------------------------------------------------------------------- #
# Cleaning and text
# --------------------------------------------------------------------------- #
def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove non-content elements in place (``extract`` is safe for nested nodes)."""
    for tag in soup.find_all(_NOISE_TAGS):
        tag.extract()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.extract()
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.extract()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.extract()
    for tag in soup.find_all(attrs={"role": True}):
        if str(tag.get("role", "")).lower() in _NOISE_ROLES:
            tag.extract()
    for tag in soup.find_all(True):
        if tag.attrs is None:  # already detached together with an ancestor
            continue
        identity = " ".join([str(tag.get("id") or ""), " ".join(tag.get("class") or [])])
        if identity.strip() and _NOISE_ATTR.search(identity):
            tag.extract()
    for comment in soup.find_all(string=lambda s: isinstance(s, PreformattedString)):
        comment.extract()


def _read_headings(soup: BeautifulSoup) -> dict[str, list[str]]:
    headings: dict[str, list[str]] = {}
    for level, limit in HEADING_LIMITS.items():
        texts = (_clean(h.get_text(" "), MAX_HEADING_CHARS) for h in soup.find_all(level))
        headings[level] = list(dict.fromkeys(t for t in texts if t))[:limit]
    return headings


def _visible_lines(soup: BeautifulSoup) -> list[str]:
    """Visible text, one line per block element, de-duplicated and normalized."""
    root = soup.body or soup
    raw: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            raw.append(" ".join("".join(buffer).split()))
            buffer.clear()

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if not isinstance(child, PreformattedString):
                    buffer.append(str(child))
            elif isinstance(child, Tag):
                if child.name in _BLOCK_TAGS or child.name == "br":
                    flush()
                    if child.name != "br":
                        walk(child)
                        flush()
                else:
                    walk(child)

    try:
        walk(root)
    except RecursionError:  # pathologically deep markup
        raw = [" ".join(line.split()) for line in root.get_text("\n").splitlines()]
    flush()

    lines: list[str] = []
    seen: set[str] = set()
    for line in raw:
        if len(line) < 3 or not re.search(r"\w", line) or _BOILERPLATE_LINE.search(line):
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return lines


def _join_limited(lines: list[str], limit: int) -> tuple[str, bool]:
    out: list[str] = []
    total = 0
    for line in lines:
        cost = len(line) + 1
        if total + cost > limit:
            remaining = limit - total
            if remaining > 40:  # keep a meaningful partial line
                out.append(line[:remaining].rstrip())
            return "\n".join(out), True
        out.append(line)
        total += cost
    return "\n".join(out), False


# --------------------------------------------------------------------------- #
# Business information
# --------------------------------------------------------------------------- #
def _read_business(
    soup: BeautifulSoup, page: ParsedPage, internal_links: list[PageLink], lines: list[str]
) -> BusinessInfo:
    info = BusinessInfo()

    names: list[str] = []
    for node in page.structured_data:
        types = " ".join(node["@type"]) if isinstance(node["@type"], list) else str(node["@type"])
        if any(k in types.lower() for k in ("organization", "business", "corporation", "store", "brand")):
            names += [node[k] for k in ("name", "legalName") if k in node]
    if page.open_graph.get("og:site_name"):
        names.append(page.open_graph["og:site_name"])
    if page.title:
        segments = [s.strip() for s in _TITLE_SEPARATORS.split(page.title) if s.strip()]
        if segments:
            names += [segments[0]] + ([segments[-1]] if len(segments) > 1 else [])
    info.name_candidates = list(dict.fromkeys(names))[:MAX_NAME_CANDIDATES]

    emails: list[str] = []
    for anchor in soup.find_all("a", href=re.compile(r"^mailto:", re.IGNORECASE)):
        address = str(anchor.get("href", ""))[7:].split("?")[0].strip()
        if _EMAIL.fullmatch(address):
            emails.append(address.lower())
    for match in _EMAIL.findall("\n".join(lines)):
        if match.rsplit(".", 1)[-1].lower() not in _FILE_SUFFIXES:
            emails.append(match.lower())
    info.emails = list(dict.fromkeys(emails))[:5]

    phones = [str(a.get("href", ""))[4:].strip() for a in soup.find_all("a", href=re.compile(r"^tel:", re.IGNORECASE))]
    phones += [n["telephone"] for n in page.structured_data if "telephone" in n]
    cleaned = [p for p in (_clean(p, 40) for p in phones) if p and 7 <= sum(c.isdigit() for c in p) <= 15]
    by_digits: dict[str, str] = {}
    for phone in cleaned:  # same number in different formats counts once
        by_digits.setdefault("".join(c for c in phone if c.isdigit()), phone)
    info.phones = list(by_digits.values())[:5]

    addresses = [n["address"] for n in page.structured_data if "address" in n]
    addresses += [t for t in (_clean(a.get_text(" "), 200) for a in soup.find_all("address")) if t]
    info.addresses = list(dict.fromkeys(addresses))[:3]

    hints: list[str] = []
    for node in page.structured_data:
        for key in ("makesOffer", "hasOfferCatalog", "offers", "serviceType", "knowsAbout"):
            value = node.get(key)
            hints += value if isinstance(value, list) else ([value] if value else [])
        if any(t in str(node["@type"]).lower() for t in ("service", "product")) and "name" in node:
            hints.append(node["name"])
    for link in internal_links:
        if link.text and any(k in urlsplit(link.url).path.lower() for k in _SERVICE_PATH_KEYWORDS):
            hints.append(link.text)
    info.service_hints = list(dict.fromkeys(h for h in hints if h))[:MAX_SERVICE_HINTS]
    return info


def _add_warnings(page: ParsedPage, script_count: int) -> None:
    if len(page.text) < 300 and script_count >= 3:
        page.warnings.append(
            "Very little server-rendered text was found; the site may build its content with "
            "JavaScript, so the analysis can be incomplete."
        )
    if not page.title:
        page.warnings.append("The page has no <title> element.")
    if not page.meta_description:
        page.warnings.append("The page has no meta description.")
    haystack = "\n".join([page.text, page.meta_description or ""])
    if any(p.search(haystack) for p in _INJECTION_PATTERNS):
        page.warnings.append(
            "The page contains text that resembles instructions to an AI system; it was treated as untrusted data."
        )
    if page.text_truncated:
        page.warnings.append("Page text was truncated to the configured maximum length.")
