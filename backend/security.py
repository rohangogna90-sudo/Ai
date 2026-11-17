"""Small security helpers (standard library only)."""
import hmac
import time
from collections import defaultdict, deque
from typing import Callable, Optional


def verify_password(supplied: str, expected: str) -> bool:
    """Constant-time password check.

    Comparing bytes (not str) avoids a TypeError crash when the password contains
    non-ASCII characters such as Hindi or emoji.
    """
    if not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


class RateLimiter:
    """Sliding-window limiter kept in memory (fine for one small server instance)."""

    def __init__(self, limit: int, window_seconds: int, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def _prune(self, key: str) -> deque:
        hits = self._hits[key]
        cutoff = self._clock() - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if not hits:
            self._hits.pop(key, None)
            return self._hits[key]
        return hits

    def retry_after(self, key: str) -> Optional[int]:
        """Seconds to wait if `key` is over the limit, else None. Does not record a hit."""
        hits = self._prune(key)
        if len(hits) < self.limit:
            return None
        return max(1, int(hits[0] + self.window - self._clock()) + 1)

    def hit(self, key: str) -> None:
        self._prune(key).append(self._clock())

    def allow(self, key: str) -> Optional[int]:
        """Record a hit unless over the limit. Returns None if allowed, else seconds to wait."""
        wait = self.retry_after(key)
        if wait is None:
            self.hit(key)
        return wait

    def reset(self, key: str) -> None:
        self._hits.pop(key, None)
