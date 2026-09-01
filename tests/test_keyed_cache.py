"""
KeyedCache backs both the idempotency cache and the pending/ready
explanation store in api/main.py. Every test class runs against both
backends (in-process dict, Redis via fakeredis) via the cache_factory
fixture, proving they share the same get/put/clear contract — the same
approach as test_entity_memory.py.
"""
import fakeredis
import pytest

from redis_utils import KeyedCache


@pytest.fixture(params=["in_process", "redis"])
def cache_factory(request):
    """Returns a callable(prefix, ttl_seconds, max_size=1000) -> KeyedCache
    for the backend named by this fixture's current param."""
    if request.param == "in_process":
        return lambda prefix, ttl_seconds, max_size=1000: KeyedCache(None, prefix, ttl_seconds, max_size)

    client = fakeredis.FakeRedis(decode_responses=True)
    return lambda prefix, ttl_seconds, max_size=1000: KeyedCache(client, prefix, ttl_seconds, max_size)


class TestGetPut:
    def test_missing_key_returns_none(self, cache_factory):
        cache = cache_factory("test", ttl_seconds=60)
        assert cache.get("nope") is None

    def test_put_then_get_round_trips_the_value(self, cache_factory):
        cache = cache_factory("test", ttl_seconds=60)
        cache.put("k1", {"status": "ready", "verdict": {"action": "ALLOW"}})
        assert cache.get("k1") == {"status": "ready", "verdict": {"action": "ALLOW"}}

    def test_overwriting_a_key_replaces_the_value(self, cache_factory):
        cache = cache_factory("test", ttl_seconds=60)
        cache.put("k1", {"status": "pending"})
        cache.put("k1", {"status": "ready"})
        assert cache.get("k1") == {"status": "ready"}

    def test_different_keys_do_not_collide(self, cache_factory):
        cache = cache_factory("test", ttl_seconds=60)
        cache.put("k1", {"v": 1})
        cache.put("k2", {"v": 2})
        assert cache.get("k1") == {"v": 1}
        assert cache.get("k2") == {"v": 2}


class TestClear:
    def test_clear_removes_everything(self, cache_factory):
        cache = cache_factory("test", ttl_seconds=60)
        cache.put("k1", {"v": 1})
        cache.put("k2", {"v": 2})
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_clear_only_touches_this_caches_own_prefix(self, cache_factory):
        # Two caches with different prefixes (mirroring idempotency vs
        # explanations in api/main.py) must not step on each other.
        cache_a = cache_factory("prefix-a", ttl_seconds=60)
        cache_b = cache_factory("prefix-b", ttl_seconds=60)
        cache_a.put("k1", {"v": "a"})
        cache_b.put("k1", {"v": "b"})

        cache_a.clear()

        assert cache_a.get("k1") is None
        assert cache_b.get("k1") == {"v": "b"}


class TestInProcessLRUEviction:
    """Eviction-on-max-size only applies to the in-process backend — Redis
    relies on TTL expiry instead, which fakeradis's clock doesn't simulate
    for a real time-based test, so this isn't parametrized."""

    def test_oldest_entry_is_evicted_beyond_max_size(self):
        cache = KeyedCache(None, prefix="test", ttl_seconds=60, max_size=2)
        cache.put("k1", {"v": 1})
        cache.put("k2", {"v": 2})
        cache.put("k3", {"v": 3})  # evicts k1

        assert cache.get("k1") is None
        assert cache.get("k2") == {"v": 2}
        assert cache.get("k3") == {"v": 3}

    def test_getting_an_entry_marks_it_as_recently_used(self):
        cache = KeyedCache(None, prefix="test", ttl_seconds=60, max_size=2)
        cache.put("k1", {"v": 1})
        cache.put("k2", {"v": 2})
        cache.get("k1")            # k1 is now more recently used than k2
        cache.put("k3", {"v": 3})  # should evict k2, not k1

        assert cache.get("k1") == {"v": 1}
        assert cache.get("k2") is None
        assert cache.get("k3") == {"v": 3}


class TestRedisTTL:
    """TTL is a Redis-only concept — the in-process backend has no
    expiry (only size-based eviction, covered above)."""

    def test_put_sets_a_ttl(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        cache = KeyedCache(client, prefix="test", ttl_seconds=3600)
        cache.put("k1", {"v": 1})
        ttl = client.ttl("test:k1")
        assert 0 < ttl <= 3600
