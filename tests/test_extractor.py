"""Website fetching and extraction, including SSRF defences.

All HTTP traffic goes through ``httpx.MockTransport`` and DNS through a fake
resolver, so these tests never touch the network.
"""

from __future__ import annotations

import httpx
import pytest

from app.exceptions import (
    EmptyContentError,
    InvalidURLError,
    ResponseTooLargeError,
    UnsafeURLError,
    UnsupportedContentTypeError,
    WebsiteInaccessibleError,
    WebsiteTimeoutError,
)
from app.services.website_extractor import WebsiteExtractor
from tests.conftest import GOOD_HTML, PUBLIC_IP, make_resolver, make_settings

pytestmark = pytest.mark.anyio

HTML_HEADERS = {"content-type": "text/html; charset=utf-8"}


def build(handler, *, mapping=None, **settings) -> WebsiteExtractor:
    resolver = make_resolver(mapping if mapping is not None else {"example.com": [PUBLIC_IP]})
    return WebsiteExtractor(make_settings(**settings), transport=httpx.MockTransport(handler), resolver=resolver)


def page_handler(html: str = GOOD_HTML, *, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers or HTML_HEADERS, content=html.encode("utf-8"))

    return handler


class Counter:
    """Wraps a handler and counts the requests that actually reached the transport."""

    def __init__(self, handler):
        self.handler, self.requests = handler, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


# --------------------------------------------------------------------------- #
# Successful extraction
# --------------------------------------------------------------------------- #
async def test_extracts_content_and_connects_to_the_pinned_ip():
    transport = Counter(page_handler())
    result = await build(transport).extract("https://example.com")

    page = result.page
    assert page.title == "Acme Cloud | DevOps and Cloud Migration Experts"
    assert page.meta_description.startswith("Acme helps mid-size companies")
    assert page.headings["h1"] == ["Cloud migration without the downtime"]
    assert "Managed Kubernetes" in page.text
    assert page.canonical_url == "https://example.com/"
    assert page.structured_data[0]["name"] == "Acme Cloud Ltd"
    assert result.final_url == "https://example.com" and result.status_code == 200
    assert result.content_type == "text/html" and result.content_quality == "full"
    assert result.redirects == [] and result.bytes_fetched > 0

    request = transport.requests[0]
    assert request.url.host == PUBLIC_IP  # DNS pinning: we connect to the validated IP ...
    assert request.headers["host"] == "example.com"  # ... but keep the original Host header
    assert request.extensions["sni_hostname"] == "example.com"  # ... and TLS SNI / cert verification
    assert "AILinkAnalysisBot" in request.headers["user-agent"]


async def test_ipv6_addresses_are_pinned_too():
    transport = Counter(page_handler())
    ipv6 = "2606:2800:220:1:248:1893:25c8:1946"
    await build(transport, mapping={"example.com": [ipv6]}).extract("https://example.com")
    assert transport.requests[0].url.host == ipv6


async def test_falls_back_to_the_second_address_when_connecting_fails():
    def handler(request):
        if request.url.host == PUBLIC_IP:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, headers=HTML_HEADERS, content=GOOD_HTML.encode())

    result = await build(handler, mapping={"example.com": [PUBLIC_IP, "93.184.216.35"]}).extract("https://example.com")
    assert result.page.title


async def test_missing_title_is_reported_not_fatal():
    html = GOOD_HTML.replace("<title>Acme Cloud | DevOps and Cloud Migration Experts</title>", "")
    result = await build(page_handler(html)).extract("https://example.com")
    assert result.page.title is None
    assert any("no <title>" in w for w in result.warnings)


async def test_missing_metadata_is_tolerated():
    html = "<html><body><p>" + "We repair vintage bicycles and ship parts worldwide. " * 4 + "</p></body></html>"
    result = await build(page_handler(html)).extract("https://example.com")
    page = result.page
    assert page.title is None and page.meta_description is None and page.canonical_url is None
    assert page.structured_data == [] and "vintage bicycles" in page.text


async def test_invalid_html_is_still_extracted():
    html = "<div><p>Unclosed paragraph describing our bakery and the sourdough we bake each morning <b>bold <i>text<div>more words"
    result = await build(page_handler(html)).extract("https://example.com")
    assert "sourdough" in result.page.text


async def test_little_text_is_flagged_as_limited_quality():
    html = "<html><head><title>Bakery</title><meta name='description' content='Fresh sourdough bread baked daily in town for local families, nearby offices, weekend visitors, and community events.'></head><body><p>Open daily.</p></body></html>"
    result = await build(page_handler(html)).extract("https://example.com")
    assert result.content_quality == "limited"
    assert any("small amount of page text" in w for w in result.warnings)


async def test_text_is_truncated_to_max_content_length():
    html = "<body>" + "".join(f"<p>Paragraph number {i} with filler words for length.</p>" for i in range(400)) + "</body>"
    result = await build(page_handler(html), max_content_length=1000).extract("https://example.com")
    assert result.page.text_truncated and len(result.page.text) <= 1000


async def test_declared_charset_is_respected():
    body = "<p>Café au lait is served daily at our bakery location downtown with fresh pastries.</p>".encode("latin-1")
    headers = {"content-type": "text/html; charset=iso-8859-1"}
    result = await build(lambda r: httpx.Response(200, headers=headers, content=body)).extract("https://example.com")
    assert "Café" in result.page.text


async def test_missing_content_type_header_is_accepted():
    result = await build(lambda r: httpx.Response(200, content=GOOD_HTML.encode())).extract("https://example.com")
    assert result.content_type is None and result.page.title


# --------------------------------------------------------------------------- #
# Empty / unusable pages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("html", ["", "   \n", "<html><body></body></html>", "<html><head><title></title></head></html>"])
async def test_empty_page_raises_empty_content_error(html):
    with pytest.raises(EmptyContentError):
        await build(page_handler(html)).extract("https://example.com")


async def test_javascript_only_shell_raises_empty_content_error_with_hint():
    html = ('<html><head><script src="a.js"></script><script src="b.js"></script><script src="c.js"></script>'
            '</head><body><div id="root"></div></body></html>')
    with pytest.raises(EmptyContentError) as info:
        await build(page_handler(html)).extract("https://example.com")
    assert any("JavaScript" in hint for hint in info.value.details["hints"])


# --------------------------------------------------------------------------- #
# HTTP-level failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [403, 404, 500, 503])
async def test_http_error_status_raises_website_inaccessible(status):
    with pytest.raises(WebsiteInaccessibleError) as info:
        await build(lambda r: httpx.Response(status)).extract("https://example.com")
    assert info.value.details["status_code"] == status


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.WriteTimeout("slow")])
async def test_timeouts_raise_website_timeout(error):
    def handler(request):
        raise error

    with pytest.raises(WebsiteTimeoutError):
        await build(handler).extract("https://example.com")


async def test_connection_failure_raises_website_inaccessible():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(WebsiteInaccessibleError):
        await build(handler).extract("https://example.com")


async def test_dns_failure_raises_website_inaccessible():
    with pytest.raises(WebsiteInaccessibleError):
        await build(page_handler(), mapping={}).extract("https://example.com")


async def test_oversized_response_declared_by_content_length():
    def handler(request):
        return httpx.Response(200, headers={**HTML_HEADERS, "content-length": "999999"}, content=b"x")

    with pytest.raises(ResponseTooLargeError):
        await build(handler, max_response_bytes=10_000).extract("https://example.com")


async def test_oversized_response_detected_while_streaming():
    class Chunks(httpx.AsyncByteStream):  # no Content-Length header: size is only known while reading
        async def __aiter__(self):
            for _ in range(5):
                yield b"a" * 4000

    def handler(request):
        return httpx.Response(200, headers=HTML_HEADERS, stream=Chunks())

    with pytest.raises(ResponseTooLargeError):
        await build(handler, max_response_bytes=10_000).extract("https://example.com")


@pytest.mark.parametrize("content_type", ["application/pdf", "application/json", "image/png", "text/plain"])
async def test_unsupported_content_type(content_type):
    with pytest.raises(UnsupportedContentTypeError):
        await build(lambda r: httpx.Response(200, headers={"content-type": content_type}, content=b"data")).extract("https://example.com")


# --------------------------------------------------------------------------- #
# SSRF protection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    ["http://localhost", "http://127.0.0.1", "http://10.0.0.5", "http://169.254.169.254/latest/meta-data/",
     "http://[::1]/", "http://metadata.google.internal/", "ftp://example.com", "https://user:pw@example.com", "http://example.com:22"],
)
async def test_unsafe_urls_are_rejected_before_any_request(url):
    transport = Counter(page_handler())
    with pytest.raises(UnsafeURLError):
        await build(transport).extract(url)
    assert transport.requests == []


async def test_invalid_url_is_rejected():
    with pytest.raises(InvalidURLError):
        await build(page_handler()).extract("not a url")


async def test_hostname_resolving_to_a_private_address_is_blocked():
    transport = Counter(page_handler())
    with pytest.raises(UnsafeURLError):
        await build(transport, mapping={"example.com": ["10.0.0.5"]}).extract("https://example.com")
    assert transport.requests == []


async def test_mixed_public_and_private_dns_answers_are_blocked():
    transport = Counter(page_handler())  # DNS-rebinding style answer
    with pytest.raises(UnsafeURLError):
        await build(transport, mapping={"example.com": [PUBLIC_IP, "127.0.0.1"]}).extract("https://example.com")
    assert transport.requests == []


async def test_safe_redirects_are_followed_and_recorded():
    def handler(request):
        if request.headers["host"] == "example.com":
            return httpx.Response(302, headers={"location": "https://www.example.org/landing"})
        return httpx.Response(200, headers=HTML_HEADERS, content=GOOD_HTML.encode())

    mapping = {"example.com": [PUBLIC_IP], "www.example.org": ["93.184.216.35"]}
    result = await build(handler, mapping=mapping).extract("https://example.com")
    assert result.final_url == "https://www.example.org/landing"
    assert result.redirects == ["https://www.example.org/landing"]


async def test_relative_redirects_are_resolved():
    def handler(request):
        if request.url.path == "/":
            return httpx.Response(301, headers={"location": "/home"})
        return httpx.Response(200, headers=HTML_HEADERS, content=GOOD_HTML.encode())

    result = await build(handler).extract("https://example.com")
    assert result.final_url == "https://example.com/home"


@pytest.mark.parametrize(
    "location",
    ["http://internal.test/admin", "http://169.254.169.254/latest/meta-data/", "http://localhost/", "http://127.0.0.1/",
     "http://[::1]/", "file:///etc/passwd", "gopher://example.com/"],
)
async def test_unsafe_redirect_destinations_are_blocked(location):
    transport = Counter(lambda r: httpx.Response(302, headers={"location": location}))
    mapping = {"example.com": [PUBLIC_IP], "internal.test": ["10.0.0.5"]}
    with pytest.raises(UnsafeURLError):
        await build(transport, mapping=mapping).extract("https://example.com")
    assert len(transport.requests) == 1  # the unsafe hop was never requested


async def test_redirect_loops_are_capped():
    transport = Counter(lambda r: httpx.Response(302, headers={"location": "https://example.com/next"}))
    with pytest.raises(WebsiteInaccessibleError):
        await build(transport, max_redirects=3).extract("https://example.com")
    assert len(transport.requests) == 4  # first request + 3 redirects


async def test_redirect_without_location_is_an_error():
    with pytest.raises(WebsiteInaccessibleError):
        await build(lambda r: httpx.Response(302)).extract("https://example.com")
