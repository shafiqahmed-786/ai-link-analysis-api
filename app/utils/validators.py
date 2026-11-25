"""URL validation and SSRF protection helpers.

Everything here is pure and uses only the standard library so it can be unit
tested without network access.  The module is the single source of truth for
"may we fetch this URL?":

* :func:`parse_url` validates *syntax* and every property of the URL that can be
  judged without DNS (scheme, credentials, port, hostname shape, literal IPs).
* :func:`ensure_public_ips` validates the *resolved* addresses of a hostname.

Callers must run both, for the initial URL and for every redirect target.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit

from app.exceptions import InvalidURLError, UnsafeURLError

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}
#: Only standard web ports are fetched; this blocks internal port scanning.
ALLOWED_PORTS = frozenset({80, 443})

BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
        "instance-data.ec2.internal",
        "kubernetes",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)
BLOCKED_SUFFIXES = (
    ".localhost",
    ".local",
    ".localdomain",
    ".internal",
    ".intranet",
    ".lan",
    ".home",
    ".corp",
    ".private",
    ".home.arpa",
    ".in-addr.arpa",
    ".ip6.arpa",
)

#: Cloud metadata endpoints. Most are link-local/private (already blocked by the
#: generic checks) but 168.63.129.16 (Azure) is a *public* address, so the list
#: is explicit as defence in depth.
CLOUD_METADATA_IPS = frozenset(
    ipaddress.ip_address(value)
    for value in (
        "169.254.169.254",  # AWS / GCP / Azure / OpenStack IMDS
        "169.254.170.2",  # AWS ECS task metadata
        "100.100.100.200",  # Alibaba Cloud
        "168.63.129.16",  # Azure WireServer
        "192.0.0.192",  # Oracle Cloud (legacy)
        "fd00:ec2::254",  # AWS IMDS over IPv6
    )
)

_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_FORBIDDEN_CHARS = re.compile(r"[\s\x00-\x1f\x7f\\]")
_HOST_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
# Decimal / hex / octal IPv4 obfuscation such as 2130706433, 0x7f.1, 0177.0.0.1
_NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+))*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedURL:
    """A syntactically valid, policy-compliant URL."""

    url: str  # normalized (no fragment, no credentials)
    scheme: str
    host: str  # lower-case ASCII (IDNA) host without brackets
    port: int  # effective port
    path_and_query: str  # always starts with "/"
    host_is_ip: bool

    @property
    def host_header(self) -> str:
        """Value for the HTTP ``Host`` header (port only if non-default)."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        if self.port != DEFAULT_PORTS[self.scheme]:
            return f"{host}:{self.port}"
        return host

    def pinned_url(self, ip: str) -> str:
        """URL that connects to ``ip`` directly (DNS pinning)."""
        address = ipaddress.ip_address(ip.split("%", 1)[0])
        literal = f"[{address}]" if address.version == 6 else str(address)
        return f"{self.scheme}://{literal}:{self.port}{self.path_and_query}"


def unsafe_reason(ip: IPAddress) -> str | None:
    """Return why ``ip`` must not be contacted, or ``None`` if it is public."""
    if ip in CLOUD_METADATA_IPS:
        return "cloud metadata address"

    for embedded in _embedded_ipv4(ip):
        reason = unsafe_reason(embedded)
        if reason:
            return f"embedded IPv4: {reason}"

    if ip.is_loopback:
        return "loopback address"
    if ip.is_unspecified:
        return "unspecified address"
    if ip.is_link_local:
        return "link-local address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_private:
        return "private network address"
    if not ip.is_global:
        return "non-public address"
    return None


def _embedded_ipv4(ip: IPAddress) -> list[ipaddress.IPv4Address]:
    """IPv4 addresses hidden inside IPv6 forms (mapped, 6to4, Teredo, NAT64)."""
    if ip.version != 6:
        return []
    found: list[ipaddress.IPv4Address] = []
    if ip.ipv4_mapped:
        found.append(ip.ipv4_mapped)
    if ip.sixtofour:
        found.append(ip.sixtofour)
    if ip.teredo:
        found.append(ip.teredo[1])
    if ip in _NAT64_PREFIX:
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def parse_ip(value: str) -> IPAddress:
    """Parse an IP string (dropping any IPv6 zone id)."""
    try:
        return ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError as exc:
        raise InvalidURLError("Resolved address is not a valid IP address.") from exc


def ensure_public_ips(host: str, addresses: Iterable[str]) -> list[str]:
    """Validate resolved addresses; reject if *any* of them is non-public.

    Rejecting mixed answers (public + private) prevents an attacker from
    relying on connection fallback to reach an internal address.
    """
    safe: list[str] = []
    for value in addresses:
        reason = unsafe_reason(parse_ip(value))
        if reason:
            raise UnsafeURLError(details={"host": host, "reason": reason})
        safe.append(value)
    if not safe:
        raise InvalidURLError("The hostname did not resolve to any address.")
    return safe


def parse_url(raw: str) -> ParsedURL:
    """Validate ``raw`` and return its normalized form.

    A missing scheme defaults to ``https://`` for convenience.

    Raises:
        InvalidURLError: the string is not a usable URL.
        UnsafeURLError: the URL is valid but forbidden (scheme, credentials,
            port, internal hostname or non-public literal IP).
    """
    if not isinstance(raw, str):
        raise InvalidURLError("URL must be a string.")
    candidate = raw.strip()
    if not candidate:
        raise InvalidURLError("URL must not be empty.")
    if len(candidate) > MAX_URL_LENGTH:
        raise InvalidURLError(f"URL is longer than {MAX_URL_LENGTH} characters.")
    if _FORBIDDEN_CHARS.search(candidate):
        raise InvalidURLError("URL contains whitespace, control characters or backslashes.")

    if "://" not in candidate:
        candidate = f"https://{candidate}"

    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise InvalidURLError("URL could not be parsed.") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError("Only http and https URLs can be analyzed.", details={"scheme": scheme})

    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UnsafeURLError("URLs containing credentials are not allowed.")

    try:
        port = parts.port
    except ValueError as exc:
        raise InvalidURLError("URL contains an invalid port.") from exc
    effective_port = port if port is not None else DEFAULT_PORTS[scheme]
    if effective_port not in ALLOWED_PORTS:
        raise UnsafeURLError(
            "Only the standard web ports 80 and 443 are allowed.", details={"port": effective_port}
        )

    hostname = parts.hostname
    if not hostname:
        raise InvalidURLError("URL does not contain a hostname.")
    hostname = hostname.rstrip(".")  # "localhost." must not bypass the checks
    if not hostname:
        raise InvalidURLError("URL does not contain a hostname.")

    host, host_is_ip = _validate_host(hostname)

    path = parts.path or ""
    query = f"?{parts.query}" if parts.query else ""
    port_part = f":{port}" if port is not None and port != DEFAULT_PORTS[scheme] else ""
    host_part = f"[{host}]" if host_is_ip and ":" in host else host
    normalized = f"{scheme}://{host_part}{port_part}{path}{query}"
    return ParsedURL(
        url=normalized,
        scheme=scheme,
        host=host,
        port=effective_port,
        path_and_query=(path or "/") + query,
        host_is_ip=host_is_ip,
    )


def _validate_host(hostname: str) -> tuple[str, bool]:
    """Validate a hostname or literal IP; return (normalized_host, is_ip)."""
    try:
        ip = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        ip = None
    if ip is not None:
        reason = unsafe_reason(ip)
        if reason:
            raise UnsafeURLError(details={"reason": reason})
        return str(ip), True

    try:
        ascii_host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise InvalidURLError("Hostname is not valid.") from exc

    if len(ascii_host) > 253:
        raise InvalidURLError("Hostname is too long.")
    if _NUMERIC_HOST.match(ascii_host):
        raise UnsafeURLError(details={"reason": "ambiguous numeric hostname"})
    labels = ascii_host.split(".")
    if not all(_HOST_LABEL.match(label) for label in labels):
        raise InvalidURLError("Hostname contains invalid characters.")
    if len(labels) < 2:
        raise UnsafeURLError(details={"reason": "single-label (internal) hostname"})
    if ascii_host in BLOCKED_HOSTNAMES or ascii_host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeURLError(details={"reason": "internal hostname"})
    return ascii_host, False
