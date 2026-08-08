import pytest

from app.ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_allows_up_to_the_limit():
    limiter = RateLimiter(3, 60, clock=Clock())
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, True]


def test_blocks_beyond_the_limit():
    limiter = RateLimiter(2, 60, clock=Clock())
    limiter.check("a")
    limiter.check("a")
    allowed, retry_after = limiter.check("a")
    assert allowed is False
    assert retry_after > 0


def test_window_resets_after_expiry():
    clock = Clock()
    limiter = RateLimiter(1, 60, clock=clock)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False
    clock.advance(60)
    assert limiter.check("a")[0] is True


def test_clients_are_tracked_independently():
    limiter = RateLimiter(1, 60, clock=Clock())
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False


def test_retry_after_shrinks_as_the_window_elapses():
    clock = Clock()
    limiter = RateLimiter(1, 60, clock=clock)
    limiter.check("a")
    _, first = limiter.check("a")
    clock.advance(30)
    _, second = limiter.check("a")
    assert second < first


def test_prune_removes_expired_windows_only():
    clock = Clock()
    limiter = RateLimiter(5, 60, clock=clock)
    limiter.check("old")
    clock.advance(61)
    limiter.check("fresh")
    assert limiter.prune() == 1
    assert limiter.check("fresh")[0] is True


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        RateLimiter(0, 60)
    with pytest.raises(ValueError):
        RateLimiter(5, 0)


def test_windows_map_is_pruned_automatically():
    """Without self-pruning the map grows for every distinct key ever seen."""
    clock = Clock()
    limiter = RateLimiter(5, 60, clock=clock, prune_threshold=10)
    for i in range(10):
        limiter.check(f"client-{i}")
    assert len(limiter._windows) == 10
    clock.advance(61)
    limiter.check("trigger")          # crosses the threshold -> prunes expired
    assert len(limiter._windows) == 1


def test_rejects_invalid_prune_threshold():
    with pytest.raises(ValueError):
        RateLimiter(5, 60, prune_threshold=0)
