"""
Single place that decides whether the app talks to Redis or falls back
to in-process state. Every Redis-backed store in this project (entity
memory, idempotency cache, explanation cache) goes through
get_redis_client() so "is Redis configured" is answered once, the same
way, everywhere.
"""
import json
import os
from collections import OrderedDict

import redis as redis_lib


def get_redis_client(redis_url: str | None = None) -> "redis_lib.Redis | None":
    """Returns a Redis client if a URL is configured (REDIS_URL env var,
    or passed explicitly — mainly for tests), else None. Constructing a
    redis.Redis does not eagerly connect, so this is safe to call even
    if the server isn't actually reachable yet; the first real command
    is what would raise.
    """
    url = redis_url if redis_url is not None else os.environ.get("REDIS_URL")
    if not url:
        return None
    return redis_lib.Redis.from_url(url, decode_responses=True)


class KeyedCache:
    """A simple key -> JSON-serializable-dict cache, used for both the
    idempotency cache and the pending/ready explanation store in
    api/main.py. Redis-backed (SET with an expiry, so entries expire on
    their own — survives restarts, shared across workers) when a
    redis_client is given; otherwise an in-process OrderedDict with LRU
    eviction — today's exact behavior, unchanged, and still the default.
    """

    def __init__(self, redis_client, prefix: str, ttl_seconds: int, max_size: int = 1000):
        self._redis = redis_client
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._store: "OrderedDict[str, dict]" = OrderedDict()

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> dict | None:
        if self._redis is not None:
            raw = self._redis.get(self._key(key))
            return json.loads(raw) if raw is not None else None

        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: dict) -> None:
        if self._redis is not None:
            self._redis.set(self._key(key), json.dumps(value), ex=self._ttl_seconds)
            return

        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Test-only. Redis-backed clear is scoped to this cache's own
        prefix via SCAN — never a FLUSHALL that could wipe an unrelated
        key someone else put in the same Redis instance."""
        if self._redis is not None:
            for k in self._redis.scan_iter(match=f"{self._prefix}:*"):
                self._redis.delete(k)
            return
        self._store.clear()
