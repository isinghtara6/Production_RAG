"""
Rate limiting.

Token-bucket per (api_key_id, scope) so a caller ingesting a large batch of
documents doesn't starve their own query traffic against a shared limit.
This in-memory implementation is correct for a single process; for a
horizontally-scaled deployment, swap the `_buckets` dict for Redis
(INCR + EXPIRE or a Lua-scripted token bucket) behind the same interface —
call sites don't need to change.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.core.exceptions import RateLimitExceededError


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._refill_rate = max_requests / window_seconds  # tokens per second
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.max_requests - 1, last_refill=now)
                self._buckets[key] = bucket
                return

            elapsed = now - bucket.last_refill
            bucket.tokens = min(self.max_requests, bucket.tokens + elapsed * self._refill_rate)
            bucket.last_refill = now

            if bucket.tokens < 1:
                retry_after = max(0.0, (1 - bucket.tokens) / self._refill_rate)
                raise RateLimitExceededError(
                    "Rate limit exceeded.",
                    details={
                        "limit": self.max_requests,
                        "window_seconds": self.window_seconds,
                        "retry_after_seconds": round(retry_after, 2),
                    },
                )
            bucket.tokens -= 1
