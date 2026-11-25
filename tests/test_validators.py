"""URL validation and SSRF protection (pure, no network)."""

import pytest

from app.exceptions import InvalidURLError, UnsafeURLError
from app.utils.validators import ensure_public_ips, parse_url, unsafe_reason
import ipaddress


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://example.com", "https://example.com"),
        ("http://example.com/path?q=1#fragment", "http://example.com/path?q=1"),
        ("example.com", "https://example.com"),  # missing scheme defaults to https
        ("  HTTPS://EXAMPLE.COM.  ", "https://example.com"),  # trimmed, lower-cased, trailing dot removed
        ("https://sub.example.co.uk:443/a", "https://sub.example.co.uk/a"),
        ("https://bücher.de", "https://xn--bcher-kva.de"),  # IDN -> punycode
        ("https://93.184.216.34/", "https://93.184.216.34/"),  # public literal IPs are allowed
    ],
)
def test_valid_urls_are_normalized(raw, expected):
    assert parse_url(raw).url == expected


def test_valid_http_and_https_defaults():
    https, http = parse_url("https://example.com"), parse_url("http://example.com")
    assert (https.scheme, https.port) == ("https", 443)
    assert (http.scheme, http.port) == ("http", 80)
    assert https.path_and_query == "/"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not a url",
        "https://",
        "https://exa mple.com",
        "http://evil.com\\@127.0.0.1/",
        "https://a..b.com",
        "https://-bad.com",
        "javascript:alert(1)",
        "https://example.com:abc",
        "https://" + "a" * 2100,
    ],
)
def test_malformed_urls_are_invalid(raw):
    with pytest.raises(InvalidURLError):
        parse_url(raw)


@pytest.mark.parametrize("raw", ["ftp://example.com", "file:///etc/passwd", "gopher://example.com", "data://x"])
def test_unsupported_schemes_are_blocked(raw):
    with pytest.raises(UnsafeURLError):
        parse_url(raw)


@pytest.mark.parametrize(
    "raw",
    ["https://user:pass@example.com", "https://user@example.com", "https://example.com@evil.com"],
)
def test_embedded_credentials_are_blocked(raw):
    with pytest.raises(UnsafeURLError):
        parse_url(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "http://localhost",
        "http://localhost:4000/x",
        "http://LOCALHOST./",  # trailing dot must not bypass
        "http://foo.localhost",
        "http://printer.local",
        "http://svc.internal",
        "http://intranet",  # single-label internal name
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_localhost_and_internal_hostnames_are_blocked(raw):
    with pytest.raises(UnsafeURLError):
        parse_url(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "http://127.0.0.1",
        "http://10.0.0.5",
        "http://172.16.0.1",
        "http://192.168.1.1",
        "http://100.64.0.1",  # carrier-grade NAT
        "http://0.0.0.0",
        "http://169.254.169.254/latest/meta-data/",  # AWS/GCP/Azure metadata
        "http://168.63.129.16",  # Azure WireServer (a *public* address)
        "http://224.0.0.1",
        "http://255.255.255.255",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[::]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "http://[::ffff:7f00:1]/",
        "http://[64:ff9b::7f00:1]/",  # NAT64 wrapping 127.0.0.1
        "http://[2002:7f00:1::]/",  # 6to4 wrapping 127.0.0.1
        "http://[fd00:ec2::254]/",  # AWS IMDS over IPv6
    ],
)
def test_private_loopback_link_local_and_metadata_ips_are_blocked(raw):
    with pytest.raises(UnsafeURLError):
        parse_url(raw)


@pytest.mark.parametrize("raw", ["http://2130706433/", "http://0x7f.0.0.1/", "http://0177.0.0.1/", "http://127.1/"])
def test_obfuscated_numeric_hosts_are_blocked(raw):
    with pytest.raises(UnsafeURLError):
        parse_url(raw)


@pytest.mark.parametrize("raw", ["http://example.com:22", "https://example.com:8443", "http://example.com:6379"])
def test_non_web_ports_are_blocked(raw):
    with pytest.raises(UnsafeURLError):
        parse_url(raw)


def test_resolved_addresses_must_all_be_public():
    assert ensure_public_ips("example.com", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
    for bad in (["93.184.216.34", "127.0.0.1"], ["10.0.0.1"], ["::1"], ["::ffff:10.0.0.1"]):
        with pytest.raises(UnsafeURLError):
            ensure_public_ips("example.com", bad)  # one bad address poisons the whole answer
    with pytest.raises(InvalidURLError):
        ensure_public_ips("example.com", [])


def test_unsafe_reason_for_public_address_is_none():
    assert unsafe_reason(ipaddress.ip_address("93.184.216.34")) is None


def test_dns_pinning_helpers():
    parsed = parse_url("https://example.com:443/a/b?x=1")
    assert parsed.pinned_url("93.184.216.34") == "https://93.184.216.34:443/a/b?x=1"
    assert parsed.host_header == "example.com"  # default port is omitted
    assert parse_url("http://example.com").pinned_url("2606:2800:220:1::1") == "http://[2606:2800:220:1::1]:80/"
