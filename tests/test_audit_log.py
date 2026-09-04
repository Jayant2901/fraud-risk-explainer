"""
AuditLog tests, run against both backends (local file, via tmp_path, and
Redis via fakeredis) — the dual-implementation pattern
test_circuit_breaker.py established.

The hash chain is the entire point of this module, so it gets its own
class independent of storage backend: tampering with an in-memory list
of entries is enough to prove verify_chain() catches it, without needing
to hand-edit a file or a Redis list in every test.
"""
import fakeredis
import pytest

from audit_log import AuditLog, GENESIS_HASH, verify_chain


@pytest.fixture(params=["local_file", "redis"])
def log_factory(request, tmp_path):
    if request.param == "local_file":
        return lambda: AuditLog(redis_client=None, log_path=str(tmp_path / "audit_log.jsonl"))
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    return lambda: AuditLog(redis_client=redis_client)


class TestAppend:
    def test_first_entry_chains_onto_the_genesis_hash(self, log_factory):
        log = log_factory()
        entry = log.append({"verdict_id": "v1", "risk_score": 50.0})

        assert entry["prev_hash"] == GENESIS_HASH
        assert entry["verdict_id"] == "v1"
        assert "hash" in entry

    def test_second_entry_chains_onto_the_first_entrys_hash(self, log_factory):
        log = log_factory()
        first = log.append({"verdict_id": "v1"})
        second = log.append({"verdict_id": "v2"})

        assert second["prev_hash"] == first["hash"]

    def test_two_appends_of_identical_content_still_differ(self, log_factory):
        # Same fields twice would collide were "at" (the timestamp) not
        # folded into the hashed record — this is the guard against that.
        log = log_factory()
        first = log.append({"verdict_id": "v1", "risk_score": 50.0})
        second = log.append({"verdict_id": "v1", "risk_score": 50.0})

        assert first["hash"] != second["hash"]


class TestGet:
    def test_returns_the_stored_entry_by_verdict_id(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1", "risk_score": 50.0})
        log.append({"verdict_id": "v2", "risk_score": 90.0})

        entry = log.get("v2")
        assert entry["risk_score"] == 90.0

    def test_unknown_verdict_id_returns_none(self, log_factory):
        assert log_factory().get("does-not-exist") is None


class TestEntriesAndReset:
    def test_entries_returns_everything_in_append_order(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1"})
        log.append({"verdict_id": "v2"})

        assert [e["verdict_id"] for e in log.entries()] == ["v1", "v2"]

    def test_an_empty_log_has_no_entries(self, log_factory):
        assert log_factory().entries() == []

    def test_reset_clears_everything(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1"})

        log.reset()

        assert log.entries() == []
        assert log.get("v1") is None


class TestSharedState:
    def test_two_processes_on_one_redis_share_the_chain(self):
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        worker_a = AuditLog(redis_client)
        worker_b = AuditLog(redis_client)

        first = worker_a.append({"verdict_id": "v1"})
        second = worker_b.append({"verdict_id": "v2"})

        # worker_b's entry chains onto worker_a's — not its own separate
        # chain — proving the last-hash compare-and-swap actually shares
        # state across the two AuditLog instances.
        assert second["prev_hash"] == first["hash"]
        assert [e["verdict_id"] for e in worker_a.entries()] == ["v1", "v2"]

    def test_concurrent_redis_writers_never_corrupt_the_chain(self):
        # The real case the WATCH/MULTI compare-and-swap in _append_redis
        # exists for: the API and src/stream_consumer.py can both be
        # appending at once. Real threads racing against one fakeredis
        # instance, not a sequential proxy for concurrency.
        import threading

        redis_client = fakeredis.FakeRedis(decode_responses=True)
        workers = [AuditLog(redis_client) for _ in range(4)]
        entries_per_worker = 15

        def run(worker, worker_id):
            for i in range(entries_per_worker):
                worker.append({"verdict_id": f"w{worker_id}-{i}"})

        threads = [
            threading.Thread(target=run, args=(worker, i)) for i, worker in enumerate(workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = workers[0].entries()
        assert len(entries) == 4 * entries_per_worker
        assert verify_chain(entries) == {"ok": True, "entries_verified": len(entries)}

    def test_local_file_logs_do_not_share_state(self, tmp_path):
        a = AuditLog(redis_client=None, log_path=str(tmp_path / "a.jsonl"))
        b = AuditLog(redis_client=None, log_path=str(tmp_path / "b.jsonl"))

        a.append({"verdict_id": "v1"})

        assert a.entries() != []
        assert b.entries() == []


class TestVerifyChain:
    def test_an_empty_chain_is_valid(self):
        assert verify_chain([]) == {"ok": True, "entries_verified": 0}

    def test_an_untampered_chain_verifies(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1", "risk_score": 50.0})
        log.append({"verdict_id": "v2", "risk_score": 90.0})
        log.append({"verdict_id": "v3", "risk_score": 10.0})

        result = verify_chain(log.entries())

        assert result == {"ok": True, "entries_verified": 3}

    def test_altering_a_field_in_an_entry_is_caught(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1", "risk_score": 50.0})
        log.append({"verdict_id": "v2", "risk_score": 90.0})
        entries = log.entries()
        entries[0]["risk_score"] = 5.0  # tamper, without recomputing the hash

        result = verify_chain(entries)

        assert result["ok"] is False
        assert result["broken_at"] == 0
        assert result["verdict_id"] == "v1"

    def test_deleting_an_entry_breaks_the_next_ones_prev_hash_link(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1"})
        log.append({"verdict_id": "v2"})
        log.append({"verdict_id": "v3"})
        entries = log.entries()
        del entries[1]  # remove v2 without re-chaining v3 onto v1

        result = verify_chain(entries)

        assert result["ok"] is False
        assert result["broken_at"] == 1
        assert result["verdict_id"] == "v3"

    def test_reordering_entries_breaks_the_chain(self, log_factory):
        log = log_factory()
        log.append({"verdict_id": "v1"})
        log.append({"verdict_id": "v2"})
        entries = log.entries()
        entries[0], entries[1] = entries[1], entries[0]

        result = verify_chain(entries)

        assert result["ok"] is False
        assert result["broken_at"] == 0
