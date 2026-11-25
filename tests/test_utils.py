"""Rate limiter, TTL cache and log redaction (pure, no network)."""

import json
import logging

from app.utils.cache import TTLCache
from app.utils.log_config import JsonFormatter, redact
from app.utils.rate_limit import SlidingWindowRateLimiter


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_rate_limiter_blocks_after_limit_and_reports_wait():
    clock = Clock()
    limiter = SlidingWindowRateLimiter(3, 60, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    assert limiter.check("a") == 60.0
    assert limiter.check("b") is None  # keys are independent
    clock.now = 30
    assert limiter.check("a") == 30.0
    clock.now = 61
    assert limiter.check("a") is None  # window slides


def test_rate_limiter_disabled_when_limit_is_zero():
    limiter = SlidingWindowRateLimiter(0)
    assert not limiter.enabled and limiter.check("x") is None


def test_rate_limiter_purges_stale_keys_at_capacity():
    clock = Clock()
    limiter = SlidingWindowRateLimiter(1, 10, clock=clock, max_keys=3)
    clock.now = 100
    for key in "abc":
        limiter.check(key)
    clock.now = 200
    limiter.check("d")
    assert list(limiter._hits) == ["d"]


def test_cache_expires_and_evicts_oldest():
    clock = Clock()
    cache = TTLCache(10, 2, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("a") is None and cache.get("b") == 2 and cache.get("c") == 3
    clock.now = 11
    assert cache.get("b") is None


def test_cache_with_zero_ttl_is_disabled():
    cache = TTLCache(0)
    cache.set("x", 1)
    assert cache.get("x") is None


def test_redact_replaces_configured_secrets_and_key_shapes():
    # Key-shaped fakes are assembled at runtime so no key-like literal sits in the source
    # (secret scanners, including GitHub push protection, pattern-match such literals).
    google_like = "AI" + "za" + "Sy" + "A1234567890abcdefghijklmnopqrstu"
    tavily_like = "tv" + "ly-" + "abcdefgh12345"
    text = f"key=SUPERSECRET123 Bearer abcdefghijkl {google_like} {tavily_like}"
    cleaned = redact(text, ["SUPERSECRET123"])
    for leaked in ("SUPERSECRET123", "abcdefghijkl", google_like, tavily_like):
        assert leaked not in cleaned


def test_json_formatter_emits_redacted_single_line_json_with_extras():
    record = logging.LogRecord("t", logging.INFO, "f.py", 1, "calling %s", ("SUPERSECRET123",), None)
    record.request_id = "abc"
    line = JsonFormatter(["SUPERSECRET123"]).format(record)
    payload = json.loads(line)
    assert "\n" not in line and "SUPERSECRET123" not in line
    assert payload["message"] == "calling ***" and payload["request_id"] == "abc" and payload["level"] == "INFO"
