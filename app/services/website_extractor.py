"""Fetch a website safely and extract analysis-ready content.

Security model (defence in depth)
---------------------------------
1. :func:`app.utils.validators.parse_url` rejects bad schemes, credentials,
   non-web ports, internal hostnames and non-public literal IPs.
2. Hostnames are resolved by *us* and every returned address must be public.
3. **DNS pinning**: the connection is opened to the validated IP address while the
   ``Host`` header and TLS SNI / certificate verification keep using the original
   hostname.  A second DNS lookup by the HTTP client (the DNS-rebinding window)
   therefore never happens.
4. Redirects are *not* followed automatically.  Each ``Location`` is re-validated
   (steps 1-3) and the number of hops is capped.
5. ``trust_env=False`` so proxy environment variables cannot redirect traffic.
6. Hard limits on total time, response bytes (measured after decompression) and
   content type.

Residual risk: validation happens in this process.  For production, also restrict
the container's *egress* network so that it cannot reach private ranges at all.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal
from urllib.parse import urljoin, urlsplit

import httpx

from app.config import Settings
from app.exceptions import (
    EmptyContentError,
    InvalidURLError,
    ResponseTooLargeError,
    UnsafeURLError,
    UnsupportedContentTypeError,
    WebsiteInaccessibleError,
    WebsiteTimeoutError,
)
from app.services.html_parser import ParsedPage, decode_html, parse_html
from app.utils.validators import ParsedURL, ensure_public_ips, parse_url

logger = logging.getLogger(__name__)

Resolver = Callable[[str, int], Awaitable[list[str]]]

ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MIN_EVIDENCE_CHARS = 80  # below this the page is considered empty
LIMITED_TEXT_CHARS = 300  # below this the analysis is flagged as based on limited content
MAX_IP_ATTEMPTS = 2
USER_AGENT = "Mozilla/5.0 (compatible; AILinkAnalysisBot/1.0; +https://example.com/bot)"


async def system_resolver(host: str, port: int) -> list[str]:
    """Resolve ``host`` with the operating system resolver (non-blocking)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


@dataclass
class FetchResult:
    html: str
    final_url: str
    status_code: int
    content_type: str | None
    redirects: list[str] = field(default_factory=list)
    bytes_fetched: int = 0
    elapsed_ms: int = 0


@dataclass
class ExtractionResult:
    """Parsed page plus fetch metadata (the raw HTML is not retained)."""

    requested_url: str
    final_url: str
    page: ParsedPage
    status_code: int
    content_type: str | None
    redirects: list[str]
    bytes_fetched: int
    fetch_ms: int
    content_quality: Literal["full", "limited"]
    warnings: list[str]


@dataclass
class _Hop:
    redirect_to: str | None = None
    body: bytes = b""
    status_code: int = 0
    content_type: str | None = None
    charset: str | None = None


class WebsiteExtractor:
    """Downloads a page and converts it to :class:`ExtractionResult`."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._timeout = settings.request_timeout
        self._max_redirects = settings.max_redirects
        self._max_bytes = settings.max_response_bytes
        self._max_chars = settings.max_content_length
        self._transport = transport
        self._resolver = resolver or system_resolver

    # ------------------------------------------------------------------ #
    async def extract(self, url: str) -> ExtractionResult:
        """Fetch ``url`` and return its extracted content."""
        requested = parse_url(url).url
        fetched = await self.fetch(requested)
        page = parse_html(fetched.html, fetched.final_url, max_text_chars=self._max_chars)
        fetched.html = ""  # release memory early

        if page.evidence_chars < MIN_EVIDENCE_CHARS:
            raise EmptyContentError(details={"hints": page.warnings})

        warnings = list(page.warnings)
        quality: Literal["full", "limited"] = "full"
        if len(page.text) < LIMITED_TEXT_CHARS:
            quality = "limited"
            warnings.append("Only a small amount of page text was available; conclusions rely mostly on metadata.")

        logger.info(
            "website extracted",
            extra={
                "host": urlsplit(fetched.final_url).hostname,
                "status": fetched.status_code,
                "bytes": fetched.bytes_fetched,
                "text_chars": len(page.text),
                "quality": quality,
            },
        )
        return ExtractionResult(
            requested_url=requested,
            final_url=fetched.final_url,
            page=page,
            status_code=fetched.status_code,
            content_type=fetched.content_type,
            redirects=fetched.redirects,
            bytes_fetched=fetched.bytes_fetched,
            fetch_ms=fetched.elapsed_ms,
            content_quality=quality,
            warnings=warnings,
        )

    async def fetch(self, url: str) -> FetchResult:
        """Download ``url`` following (validated) redirects manually."""
        target = parse_url(url)
        redirects: list[str] = []
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout):
                async with self._client() as client:
                    for _ in range(self._max_redirects + 1):
                        hop = await self._fetch_hop(client, target)
                        if hop.redirect_to is None:
                            html = decode_html(hop.body, hop.charset)
                            return FetchResult(
                                html=html,
                                final_url=target.url,
                                status_code=hop.status_code,
                                content_type=hop.content_type,
                                redirects=redirects,
                                bytes_fetched=len(hop.body),
                                elapsed_ms=int((time.monotonic() - started) * 1000),
                            )
                        target = self._next_target(target, hop.redirect_to)
                        redirects.append(target.url)
                    raise WebsiteInaccessibleError(
                        f"The website redirected more than {self._max_redirects} times.",
                        details={"max_redirects": self._max_redirects},
                    )
        except TimeoutError as exc:
            raise WebsiteTimeoutError(
                f"The website did not respond within {self._timeout:g} seconds."
            ) from exc

    # ------------------------------------------------------------------ #
    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            follow_redirects=False,  # redirects are validated manually
            trust_env=False,  # ignore proxy environment variables
            transport=self._transport,
            timeout=httpx.Timeout(self._timeout),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                "Accept-Language": "en;q=0.9,*;q=0.5",
            },
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
        )

    @staticmethod
    def _next_target(current: ParsedURL, location: str) -> ParsedURL:
        """Validate a redirect target exactly like a user-supplied URL."""
        try:
            return parse_url(urljoin(current.url, location.strip()))
        except UnsafeURLError as exc:
            raise UnsafeURLError(
                "The website redirected to an address that cannot be analyzed.",
                details={"stage": "redirect", **exc.details},
            ) from exc
        except InvalidURLError as exc:
            raise WebsiteInaccessibleError("The website redirected to an invalid URL.") from exc

    async def _resolve(self, target: ParsedURL) -> list[str]:
        if target.host_is_ip:  # literal IPs were already validated by parse_url
            return [target.host]
        try:
            addresses = await asyncio.wait_for(
                self._resolver(target.host, target.port), timeout=self._timeout
            )
        except TimeoutError:
            raise
        except (OSError, UnicodeError) as exc:
            raise WebsiteInaccessibleError("The website's hostname could not be resolved.") from exc
        if not addresses:
            raise WebsiteInaccessibleError("The website's hostname could not be resolved.")
        return ensure_public_ips(target.host, addresses)

    async def _fetch_hop(self, client: httpx.AsyncClient, target: ParsedURL) -> _Hop:
        addresses = await self._resolve(target)
        extensions = {} if target.host_is_ip else {"sni_hostname": target.host}
        last_error: Exception | None = None
        for ip in addresses[:MAX_IP_ATTEMPTS]:
            try:
                async with client.stream(
                    "GET",
                    target.pinned_url(ip),
                    headers={"Host": target.host_header},
                    extensions=extensions,
                ) as response:
                    return await self._read_response(response)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc  # try the next resolved address
            except httpx.TimeoutException as exc:
                raise WebsiteTimeoutError(
                    f"The website did not respond within {self._timeout:g} seconds."
                ) from exc
            except httpx.HTTPError as exc:
                raise WebsiteInaccessibleError("The website request failed.") from exc
        if isinstance(last_error, httpx.ConnectTimeout):
            raise WebsiteTimeoutError("Connecting to the website timed out.") from last_error
        raise WebsiteInaccessibleError("Could not connect to the website.") from last_error

    async def _read_response(self, response: httpx.Response) -> _Hop:
        status = response.status_code
        if status in REDIRECT_STATUSES:
            location = response.headers.get("location")
            if not location:
                raise WebsiteInaccessibleError("The website sent a redirect without a destination.")
            return _Hop(redirect_to=location, status_code=status)
        if not 200 <= status < 300:
            raise WebsiteInaccessibleError(
                f"The website returned HTTP {status}.", details={"status_code": status}
            )

        raw_type = response.headers.get("content-type", "")
        content_type = raw_type.split(";", 1)[0].strip().lower() or None
        if content_type is not None and content_type not in ALLOWED_CONTENT_TYPES:
            raise UnsupportedContentTypeError(
                f"The URL returned '{content_type}', but only HTML pages can be analyzed.",
                details={"content_type": content_type},
            )

        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            raise ResponseTooLargeError(details={"limit_bytes": self._max_bytes})

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():  # decoded bytes: also bounds decompression bombs
            total += len(chunk)
            if total > self._max_bytes:
                raise ResponseTooLargeError(details={"limit_bytes": self._max_bytes})
            chunks.append(chunk)
        return _Hop(
            body=b"".join(chunks),
            status_code=status,
            content_type=content_type,
            charset=response.charset_encoding,
        )
