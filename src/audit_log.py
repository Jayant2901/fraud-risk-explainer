"""
Immutable, tamper-evident audit trail for every scoring verdict.

review_queue.py already tracks who confirmed or reversed a decision.
What was missing is the decision itself: an append-only record of every
verdict this system ever produced, with each entry's hash folding in the
PREVIOUS entry's hash — so altering, deleting, or reordering a past
entry breaks the chain from that point on, verifiably, rather than
silently. `python -m src.audit verify` walks the whole log and reports
the first place it doesn't check out.

Same REDIS_URL-optional dual-backend design as the rest of this
codebase, but the two backends solve different problems here:
- Local file (default, data/audit_log.jsonl): a single writer (this
  process), so Python's own sequencing is the only ordering guarantee
  needed. Appended as JSON Lines so the trail survives a restart without
  any extra infrastructure.
- Redis: the API and src/stream_consumer.py both score transactions, so
  two workers can try to append at the same moment — computing "hash
  chained onto the CURRENT last hash" needs real compare-and-swap, not
  just an atomic RPUSH. Done here with WATCH/MULTI on the last-hash key,
  retried on a conflicting concurrent writer.

Always on, unlike the feedback loop / escalation alerts / shadow scoring
above: an audit trail that only sometimes exists isn't one. There's
nothing to opt into — create_audit_log() always returns a working
instance, local-file-backed with zero configuration required.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import redis as redis_lib

REDIS_KEY_PREFIX = "riskmgr:audit"
DEFAULT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "data/audit_log.jsonl")

GENESIS_HASH = "0" * 64
MAX_CAS_RETRIES = 10


def _canonical(record: dict) -> str:
    """Deterministic serialization — the same fields always hash the same
    way regardless of dict insertion order, which matters because the
    hash is recomputed independently by verify()."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _entry_hash(prev_hash: str, record: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(record)).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    def __init__(self, redis_client=None, log_path: str = DEFAULT_LOG_PATH):
        self._redis = redis_client
        self._log_path = log_path

    def append(self, event: dict) -> dict:
        """event: verdict_id, entity_id, risk_score, action, escalation
        state, model_version, ... — whatever ScoringService knows about
        the verdict. Returns the full stored entry, including its own
        hash and the previous entry's hash."""
        record = {**event, "at": _now_iso()}
        if self._redis is None:
            return self._append_local(record)
        return self._append_redis(record)

    def _append_local(self, record: dict) -> dict:
        prev_hash = self._local_last_hash()
        entry = {**record, "prev_hash": prev_hash, "hash": _entry_hash(prev_hash, record)}
        log_dir = os.path.dirname(self._log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def _local_last_hash(self) -> str:
        last_hash = GENESIS_HASH
        for entry in self._iter_local():
            last_hash = entry["hash"]
        return last_hash

    def _append_redis(self, record: dict) -> dict:
        last_hash_key = f"{REDIS_KEY_PREFIX}:last_hash"
        log_key = f"{REDIS_KEY_PREFIX}:entries"
        index_key = f"{REDIS_KEY_PREFIX}:by_verdict"
        with self._redis.pipeline() as pipe:
            for _ in range(MAX_CAS_RETRIES):
                pipe.watch(last_hash_key)
                prev_hash = pipe.get(last_hash_key) or GENESIS_HASH
                entry = {**record, "prev_hash": prev_hash, "hash": _entry_hash(prev_hash, record)}
                pipe.multi()
                pipe.rpush(log_key, json.dumps(entry))
                pipe.set(last_hash_key, entry["hash"])
                if "verdict_id" in entry:
                    pipe.hset(index_key, entry["verdict_id"], json.dumps(entry))
                try:
                    pipe.execute()
                    return entry
                except redis_lib.WatchError:
                    continue  # another worker appended between our watch and our multi — retry
        raise RuntimeError("audit log append failed after retries — persistent concurrent writers")

    def get(self, verdict_id: str) -> dict | None:
        if self._redis is not None:
            raw = self._redis.hget(f"{REDIS_KEY_PREFIX}:by_verdict", verdict_id)
            return json.loads(raw) if raw else None
        for entry in self._iter_local():
            if entry.get("verdict_id") == verdict_id:
                return entry
        return None

    def entries(self) -> list[dict]:
        if self._redis is not None:
            return [json.loads(raw) for raw in self._redis.lrange(f"{REDIS_KEY_PREFIX}:entries", 0, -1)]
        return list(self._iter_local())

    def _iter_local(self):
        if not os.path.exists(self._log_path):
            return
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def reset(self) -> None:
        """Test-only. A real audit trail is never truncated in
        production — that's the entire point of this module."""
        if self._redis is not None:
            self._redis.delete(
                f"{REDIS_KEY_PREFIX}:entries", f"{REDIS_KEY_PREFIX}:last_hash", f"{REDIS_KEY_PREFIX}:by_verdict"
            )
            return
        if os.path.exists(self._log_path):
            os.remove(self._log_path)


def verify_chain(entries: list[dict]) -> dict:
    """Walks the chain in order, recomputing each entry's hash from its
    own fields plus the previous entry's hash. Tampering, corruption, and
    a reordered or deleted entry all show up the same way: the first
    entry whose recomputed hash disagrees with what's stored."""
    prev_hash = GENESIS_HASH
    for i, entry in enumerate(entries):
        record = {k: v for k, v in entry.items() if k not in ("hash", "prev_hash")}
        if entry.get("prev_hash") != prev_hash:
            return {
                "ok": False, "broken_at": i, "verdict_id": entry.get("verdict_id"),
                "reason": "prev_hash does not match the previous entry's actual hash",
            }
        expected_hash = _entry_hash(prev_hash, record)
        if entry.get("hash") != expected_hash:
            return {
                "ok": False, "broken_at": i, "verdict_id": entry.get("verdict_id"),
                "reason": "entry hash does not match its recomputed hash - content was altered",
            }
        prev_hash = entry["hash"]
    return {"ok": True, "entries_verified": len(entries)}


def create_audit_log(redis_client=None, log_path: str = DEFAULT_LOG_PATH) -> AuditLog:
    return AuditLog(redis_client, log_path)
