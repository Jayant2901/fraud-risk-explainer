"""
Human review queue — the piece that turns a model score into an actual
workflow. Every flagged verdict (REVIEW/BLOCK) lands here; a reviewer
disposes it as CONFIRMED_FRAUD or FALSE_POSITIVE, and that disposition
closes the feedback loop between the model's decision and a confirmed
outcome. GET /api/review-queue/metrics recomputes Phase 0's escalation-
ablation comparison (escalated-flag precision vs. non-escalated-flag
precision) from these LIVE dispositions instead of only the offline test
set — see src/escalation_ablation.py for the offline version.

Same REDIS_URL-optional design as entity_memory.py/redis_utils.py:
in-process by default (a dict), Redis-backed (a hash of items + a set of
pending ids) when a redis_client is given — see create_review_queue().
KeyedCache's plain key->value shape doesn't fit "list all pending items
sorted by risk_score" well, so this is a small parallel structure
following the same in-process/Redis-optional principle rather than
forcing reuse.
"""
import json
from datetime import datetime, timezone

REDIS_KEY_PREFIX = "riskmgr:review_queue"

CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
FALSE_POSITIVE = "FALSE_POSITIVE"
VALID_DISPOSITIONS = {CONFIRMED_FRAUD, FALSE_POSITIVE}


class UnknownVerdictError(Exception):
    """No review-queue item exists for this verdict_id."""


class AlreadyDisposedError(Exception):
    """This item already has a disposition — don't silently overwrite a
    reviewer's decision (mirrors the idempotency cache's care about
    double-processing)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_metrics(items: list[dict]) -> dict:
    """Overall precision, plus — the important one — precision split by
    whether entity escalation is what triggered the flag. This is Phase
    0's ablation comparison, recomputed from real reviewer input."""
    disposed = [i for i in items if i["disposition"] is not None]
    escalated = [i for i in disposed if i["escalated_due_to_history"]]
    non_escalated = [i for i in disposed if not i["escalated_due_to_history"]]

    def precision(subset: list[dict]) -> float | None:
        if not subset:
            return None
        confirmed = sum(1 for i in subset if i["disposition"] == CONFIRMED_FRAUD)
        return round(confirmed / len(subset), 4)

    return {
        "total_disposed": len(disposed),
        "overall_precision": precision(disposed),
        "escalated_count": len(escalated),
        "escalated_precision": precision(escalated),
        "non_escalated_count": len(non_escalated),
        "non_escalated_precision": precision(non_escalated),
    }


class ReviewQueue:
    """In-process default — session-scoped, resets when the process
    restarts. No setup required, same tradeoff as EntityRiskMemory."""

    def __init__(self):
        self._items: dict[str, dict] = {}

    def add(self, item: dict) -> None:
        self._items[item["verdict_id"]] = dict(item)

    def get(self, verdict_id: str) -> dict | None:
        return self._items.get(verdict_id)

    def list_pending(self) -> list[dict]:
        pending = [i for i in self._items.values() if i["disposition"] is None]
        return sorted(pending, key=lambda i: i["risk_score"], reverse=True)

    def dispose(self, verdict_id: str, disposition: str) -> dict:
        item = self._items.get(verdict_id)
        if item is None:
            raise UnknownVerdictError(verdict_id)
        if item["disposition"] is not None:
            raise AlreadyDisposedError(verdict_id)
        item["disposition"] = disposition
        item["disposed_at"] = _now_iso()
        return item

    def metrics(self) -> dict:
        return _compute_metrics(list(self._items.values()))

    def reset(self) -> None:
        """Test-only."""
        self._items.clear()


class RedisReviewQueue:
    """Same contract as ReviewQueue, backed by Redis: each item is a JSON
    string keyed by verdict_id, plus a Redis set tracking which verdict_ids
    are still pending (so list_pending doesn't have to scan every item
    ever seen). `redis_client` must be a redis.Redis (or fakeredis.FakeRedis
    for tests) constructed with decode_responses=True."""

    def __init__(self, redis_client):
        self._redis = redis_client

    def _item_key(self, verdict_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:item:{verdict_id}"

    def _pending_key(self) -> str:
        return f"{REDIS_KEY_PREFIX}:pending"

    def _all_key(self) -> str:
        return f"{REDIS_KEY_PREFIX}:all"

    def add(self, item: dict) -> None:
        verdict_id = item["verdict_id"]
        pipe = self._redis.pipeline()
        pipe.set(self._item_key(verdict_id), json.dumps(item))
        pipe.sadd(self._pending_key(), verdict_id)
        pipe.sadd(self._all_key(), verdict_id)
        pipe.execute()

    def get(self, verdict_id: str) -> dict | None:
        raw = self._redis.get(self._item_key(verdict_id))
        return json.loads(raw) if raw is not None else None

    def list_pending(self) -> list[dict]:
        ids = self._redis.smembers(self._pending_key())
        items = [self.get(vid) for vid in ids]
        items = [i for i in items if i is not None]
        return sorted(items, key=lambda i: i["risk_score"], reverse=True)

    def dispose(self, verdict_id: str, disposition: str) -> dict:
        item = self.get(verdict_id)
        if item is None:
            raise UnknownVerdictError(verdict_id)
        if item["disposition"] is not None:
            raise AlreadyDisposedError(verdict_id)
        item["disposition"] = disposition
        item["disposed_at"] = _now_iso()

        pipe = self._redis.pipeline()
        pipe.set(self._item_key(verdict_id), json.dumps(item))
        pipe.srem(self._pending_key(), verdict_id)
        pipe.execute()
        return item

    def metrics(self) -> dict:
        ids = self._redis.smembers(self._all_key())
        items = [self.get(vid) for vid in ids]
        items = [i for i in items if i is not None]
        return _compute_metrics(items)

    def reset(self) -> None:
        """Test-only, scoped to this module's own key prefix — never a
        FLUSHALL that could wipe an unrelated key in the same Redis
        instance (same care as KeyedCache.clear())."""
        for key in self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}:*"):
            self._redis.delete(key)


def create_review_queue(redis_client=None):
    """Factory used by api/main.py: Redis-backed if a client is given
    (i.e. REDIS_URL was configured), the in-process default otherwise —
    same pattern as create_entity_memory()."""
    if redis_client is not None:
        return RedisReviewQueue(redis_client)
    return ReviewQueue()
