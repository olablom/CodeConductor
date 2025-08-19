from codeconductor.utils.lru_cache import LRUCacheTTL


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_lru_cache_basic_hit_miss_and_ttl():
    clock = FakeClock(100.0)
    cache = LRUCacheTTL(
        max_entries=2, ttl_seconds=10, time_provider=clock.now, namespace="ns"
    )
    key = cache.make_key(
        prompt="p",
        persona="coder",
        policy="latency",
        model="m",
        params={"temperature": 0.1, "top_p": 0.9},
    )
    assert cache.get(key) is None
    cache.set(key, {"answer": 42})
    assert cache.get(key) == {"answer": 42}
    # advance beyond TTL
    clock.advance(11)
    assert cache.get(key) is None


def test_lru_cache_evicts_oldest():
    clock = FakeClock(0.0)
    cache = LRUCacheTTL(
        max_entries=2, ttl_seconds=1000, time_provider=clock.now, namespace="ns"
    )
    k1 = cache.make_key(
        prompt="p1", persona="coder", policy="latency", model="m1", params={}
    )
    k2 = cache.make_key(
        prompt="p2", persona="coder", policy="latency", model="m2", params={}
    )
    k3 = cache.make_key(
        prompt="p3", persona="coder", policy="latency", model="m3", params={}
    )
    cache.set(k1, 1)
    cache.set(k2, 2)
    # This should evict k1 when adding k3
    cache.set(k3, 3)
    assert cache.get(k1) is None
    assert cache.get(k2) == 2
    assert cache.get(k3) == 3
