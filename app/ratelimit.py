"""Per-client rate limiting.

Every question costs at least two Hugging Face Inference calls (gate, then
generation). Without a limit, one script can exhaust the token's quota — or its
budget — in minutes. This is a fixed-window counter kept in process, which is
the right scope here because the app deliberately runs a single uvicorn worker
to keep the idle-unload singleton coherent.
"""

import threading
import time
from typing import Callable, Dict, Tuple


class RateLimiter:
    """Fixed-window request counter keyed by client identity."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        prune_threshold: int = 10_000,
    ):
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if prune_threshold <= 0:
            raise ValueError("prune_threshold must be positive")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.prune_threshold = prune_threshold
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: Dict[str, Tuple[float, int]] = {}

    def check(self, key: str) -> Tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        # Self-prune: without this the map grows for every distinct key ever
        # seen, which is an unbounded-memory path in a long-lived process.
        if len(self._windows) >= self.prune_threshold:
            self.prune()

        now = self._clock()
        with self._lock:
            started, count = self._windows.get(key, (now, 0))

            if now - started >= self.window_seconds:
                self._windows[key] = (now, 1)
                return True, 0.0

            if count >= self.max_requests:
                return False, max(0.0, self.window_seconds - (now - started))

            self._windows[key] = (started, count + 1)
            return True, 0.0

    def prune(self) -> int:
        """Drop expired windows so the map cannot grow without bound. Returns
        how many entries were removed."""
        now = self._clock()
        with self._lock:
            stale = [
                k for k, (started, _) in self._windows.items()
                if now - started >= self.window_seconds
            ]
            for k in stale:
                del self._windows[k]
        return len(stale)
