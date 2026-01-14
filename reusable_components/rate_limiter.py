"""Simple token bucket rate limiter for outbound model calls."""
from __future__ import annotations
import threading, time

class RateLimiter:
    def __init__(self, rate_per_minute: int = 120, burst: int | None = None):
        self._lock = threading.Lock()
        self.configure(rate_per_minute, burst)

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        add = elapsed * self.rate_per_sec
        if add > 0:
            self.tokens = min(self.capacity, self.tokens + add)
            self.last_refill = now

    def configure(self, rate_per_minute: int, burst: int | None = None):
        with self._lock:
            self.capacity = burst or rate_per_minute
            if self.capacity <= 0:
                self.capacity = 1
            self.tokens = getattr(self, 'tokens', self.capacity)
            self.rate_per_sec = max(0.01, rate_per_minute / 60.0)
            self.last_refill = time.time()

    def acquire(self, estimated_tokens: int = 1):
        weight = max(1, int(estimated_tokens / 200))  # crude weight scaling
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
            time.sleep(0.1)

# Global singleton configured via env later if needed
global_rate_limiter = RateLimiter()

__all__ = ["RateLimiter", "global_rate_limiter"]