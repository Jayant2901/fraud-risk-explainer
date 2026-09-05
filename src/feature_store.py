"""
Online feature store — the live counterpart to the causal features that
data_utils.add_causal_entity_history and
graph_features.add_causal_device_graph_features compute offline.

The gap this closes: those two functions build history-dependent features
over a static dataframe at training/evaluation time. The live scoring
path then scores a transaction against features derived from that frozen
snapshot, so an entity's 40th transaction today was being scored with
graph features that did not include its 39th from ten minutes ago. The
offline construction is correct and leakage-free; it simply had no online
twin, which quietly made the live system less capable than the evaluation
implied.

This maintains the same aggregates incrementally:

  per entity      entity_prior_txn_count, entity_prior_fraud_count,
                  entity_prior_fraud_rate
  per fingerprint shared_device_prior_entity_count (distinct entities),
                  shared_device_prior_fraud_rate

**Ordering is the whole correctness argument.** Features for a
transaction must be READ before that transaction is RECORDED. Record
first and the transaction counts itself — which is precisely the leakage
the offline functions were written with a shift(1) to avoid.
ScoringService enforces the order in one place, and
tests/test_feature_store.py asserts it directly.

**Labels arrive later than transactions.** Offline, every prior row's
isFraud is known. Online it is not: a transaction's true label comes from
a reviewer disposition or a chargeback, days later. record() therefore
takes is_fraud=None for the live path (the transaction counts toward
volume immediately, and toward fraud counts only once labelled) and an
explicit label for historical replay. This is a real difference between
the offline and online feature distributions, not a bug, and it is why
apply_label() exists as a separate call.

Redis-backed when available, in-process otherwise — same create_*
factory convention as entity_memory.py.
"""
import logging
import os

import pandas as pd

from graph_features import build_device_fingerprint

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "riskmgr:features"

# Entities and devices go stale; a store that never forgets grows without
# bound. 90 days is long enough to cover any window these features look
# at and short enough to bound the keyspace.
FEATURE_TTL_SECONDS = int(os.environ.get("FEATURE_TTL_SECONDS", 90 * 24 * 60 * 60))

ZERO_FEATURES = {
    "entity_prior_txn_count": 0.0,
    "entity_prior_fraud_count": 0.0,
    "entity_prior_fraud_rate": 0.0,
    "shared_device_prior_entity_count": 0.0,
    "shared_device_prior_fraud_rate": 0.0,
}


def fingerprint_for(txn: dict) -> str | None:
    """The same device/address fingerprint graph_features builds offline,
    for a single transaction dict. Delegates to build_device_fingerprint
    on a one-row frame rather than restating its precedence rules, so the
    two can't drift."""
    import pandas as pd

    value = build_device_fingerprint(pd.DataFrame([txn])).iloc[0]
    return None if value is None or pd.isna(value) else str(value)


def _rate(numerator: float, denominator: float) -> float:
    """Matches the offline `.replace(0, np.nan)` / `.fillna(0.0)` idiom:
    no priors means a rate of 0.0, not a division by zero."""
    return float(numerator) / float(denominator) if denominator else 0.0


class FeatureStore:
    """In-process default — zero setup, resets with the process, same
    tradeoff as EntityRiskMemory."""

    def __init__(self):
        self._entity_txns: dict[str, int] = {}
        self._entity_frauds: dict[str, int] = {}
        self._fp_txns: dict[str, int] = {}
        self._fp_frauds: dict[str, int] = {}
        self._fp_entities: dict[str, set[str]] = {}

    def features_for(self, entity_id: str | None, fingerprint: str | None) -> dict:
        """The feature values as of *before* this transaction. Read this
        first; call record() only afterwards."""
        features = dict(ZERO_FEATURES)
        if entity_id:
            txns = self._entity_txns.get(entity_id, 0)
            frauds = self._entity_frauds.get(entity_id, 0)
            features["entity_prior_txn_count"] = float(txns)
            features["entity_prior_fraud_count"] = float(frauds)
            features["entity_prior_fraud_rate"] = _rate(frauds, txns)
        if fingerprint:
            features["shared_device_prior_entity_count"] = float(
                len(self._fp_entities.get(fingerprint, ()))
            )
            features["shared_device_prior_fraud_rate"] = _rate(
                self._fp_frauds.get(fingerprint, 0), self._fp_txns.get(fingerprint, 0)
            )
        return features

    def record(self, entity_id: str | None, fingerprint: str | None, is_fraud: int | None = None) -> None:
        """Fold this transaction into the aggregates. Must run AFTER
        features_for() for the same transaction."""
        if entity_id:
            self._entity_txns[entity_id] = self._entity_txns.get(entity_id, 0) + 1
            if is_fraud:
                self._entity_frauds[entity_id] = self._entity_frauds.get(entity_id, 0) + 1
        if fingerprint:
            self._fp_txns[fingerprint] = self._fp_txns.get(fingerprint, 0) + 1
            if is_fraud:
                self._fp_frauds[fingerprint] = self._fp_frauds.get(fingerprint, 0) + 1
            if entity_id:
                self._fp_entities.setdefault(fingerprint, set()).add(entity_id)

    def apply_label(self, entity_id: str | None, fingerprint: str | None, is_fraud: bool) -> None:
        """Record a label that arrived after the fact — a reviewer
        disposition or a chargeback. Volume was already counted by
        record(); this adds only the fraud side."""
        if not is_fraud:
            return
        if entity_id:
            self._entity_frauds[entity_id] = self._entity_frauds.get(entity_id, 0) + 1
        if fingerprint:
            self._fp_frauds[fingerprint] = self._fp_frauds.get(fingerprint, 0) + 1

    def stats(self) -> dict:
        return {
            "entities_tracked": len(self._entity_txns),
            "fingerprints_tracked": len(self._fp_txns),
            "transactions_recorded": sum(self._entity_txns.values()),
        }

    def reset(self) -> None:
        """Test-only."""
        self._entity_txns.clear()
        self._entity_frauds.clear()
        self._fp_txns.clear()
        self._fp_frauds.clear()
        self._fp_entities.clear()


class RedisFeatureStore:
    """Same contract, backed by Redis so the aggregates survive restarts
    and are shared across the API workers and the stream consumer — which
    matters more here than anywhere else in this project, since a
    per-worker view of an entity's history would mean the same entity
    scored differently depending on which worker took the request."""

    def __init__(self, redis_client, ttl_seconds: int = FEATURE_TTL_SECONDS):
        self._redis = redis_client
        self._ttl = ttl_seconds

    def _entity_key(self, entity_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:entity:{entity_id}"

    def _fp_key(self, fingerprint: str) -> str:
        return f"{REDIS_KEY_PREFIX}:fp:{fingerprint}"

    def _fp_entities_key(self, fingerprint: str) -> str:
        return f"{REDIS_KEY_PREFIX}:fp-entities:{fingerprint}"

    def features_for(self, entity_id: str | None, fingerprint: str | None) -> dict:
        features = dict(ZERO_FEATURES)
        if entity_id:
            counts = self._redis.hgetall(self._entity_key(entity_id)) or {}
            txns = int(counts.get("txns", 0))
            frauds = int(counts.get("frauds", 0))
            features["entity_prior_txn_count"] = float(txns)
            features["entity_prior_fraud_count"] = float(frauds)
            features["entity_prior_fraud_rate"] = _rate(frauds, txns)
        if fingerprint:
            counts = self._redis.hgetall(self._fp_key(fingerprint)) or {}
            features["shared_device_prior_entity_count"] = float(
                self._redis.scard(self._fp_entities_key(fingerprint)) or 0
            )
            features["shared_device_prior_fraud_rate"] = _rate(
                int(counts.get("frauds", 0)), int(counts.get("txns", 0))
            )
        return features

    def record(self, entity_id: str | None, fingerprint: str | None, is_fraud: int | None = None) -> None:
        pipe = self._redis.pipeline()
        if entity_id:
            key = self._entity_key(entity_id)
            pipe.hincrby(key, "txns", 1)
            if is_fraud:
                pipe.hincrby(key, "frauds", 1)
            pipe.expire(key, self._ttl)
        if fingerprint:
            key = self._fp_key(fingerprint)
            pipe.hincrby(key, "txns", 1)
            if is_fraud:
                pipe.hincrby(key, "frauds", 1)
            pipe.expire(key, self._ttl)
            if entity_id:
                entities_key = self._fp_entities_key(fingerprint)
                pipe.sadd(entities_key, entity_id)
                pipe.expire(entities_key, self._ttl)
        pipe.execute()

    def apply_label(self, entity_id: str | None, fingerprint: str | None, is_fraud: bool) -> None:
        if not is_fraud:
            return
        pipe = self._redis.pipeline()
        if entity_id:
            pipe.hincrby(self._entity_key(entity_id), "frauds", 1)
        if fingerprint:
            pipe.hincrby(self._fp_key(fingerprint), "frauds", 1)
        pipe.execute()

    def stats(self) -> dict:
        entities = sum(1 for _ in self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}:entity:*"))
        fingerprints = sum(1 for _ in self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}:fp:*"))
        return {
            "entities_tracked": entities,
            "fingerprints_tracked": fingerprints,
            "transactions_recorded": None,  # not cheaply countable in Redis
        }

    def reset(self) -> None:
        """Test-only, scoped to this module's own prefix — never a
        FLUSHALL (same care as KeyedCache.clear())."""
        for key in self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}:*"):
            self._redis.delete(key)


def create_feature_store(redis_client=None):
    """Factory used by api/main.py and the stream consumer — Redis-backed
    if a client is given, in-process otherwise. Same pattern as
    create_entity_memory()/create_review_queue()."""
    if redis_client is not None:
        return RedisFeatureStore(redis_client)
    return FeatureStore()


def seed_from_history(store, df, limit: int | None = None) -> dict:
    """Warm the store from historical transactions so entities aren't
    cold on first run.

    Feeds rows in chronological order, exactly as a live stream would
    arrive, using each row's known label. Returns counts for
    /api/health's feature-store block, which distinguishes seeded rows
    from live ones — a store that looks populated but was never seeded
    would score early transactions against empty history.
    """
    ordered = df.sort_values("TransactionDT")
    if limit is not None:
        ordered = ordered.head(limit)

    # build_device_fingerprint is a purely elementwise function (no
    # cross-row aggregation), so computing it once over the whole batch
    # gives byte-identical results to fingerprint_for()'s one-row-at-a-time
    # calls — but avoids constructing a fresh DataFrame per row, which was
    # the dominant cost of seeding a large sample (minutes, blocking the
    # API from accepting any connection while the lifespan handler ran).
    fingerprints = build_device_fingerprint(ordered).tolist()

    seeded = 0
    for row, raw_fingerprint in zip(ordered.to_dict("records"), fingerprints):
        entity_id = row.get("entity_id")
        fingerprint = None if pd.isna(raw_fingerprint) else str(raw_fingerprint)
        store.record(entity_id, fingerprint, int(row.get("isFraud", 0) or 0))
        seeded += 1
    logger.info("Feature store seeded from history", extra={"rows": seeded})
    return {"seeded_rows": seeded}
