"""
Every test class below runs TWICE — once against ReviewQueue (in-process
dict) and once against RedisReviewQueue (backed by fakeredis) — via the
queue_factory fixture's parametrization, the same approach as
test_entity_memory.py and test_keyed_cache.py.
"""
import fakeredis
import pytest

from review_queue import (
    ReviewQueue,
    RedisReviewQueue,
    UnknownVerdictError,
    AlreadyDisposedError,
    CONFIRMED_FRAUD,
    FALSE_POSITIVE,
)


@pytest.fixture(params=["in_process", "redis"])
def queue_factory(request):
    """Returns a callable() -> queue instance for the backend named by
    this fixture's current param."""
    if request.param == "in_process":
        return lambda: ReviewQueue()

    client = fakeredis.FakeRedis(decode_responses=True)
    return lambda: RedisReviewQueue(client)


def make_item(
    verdict_id: str,
    risk_score: float,
    escalated: bool = False,
    entity_id: str = "e1",
    created_at: str = "2024-01-01T00:00:00+00:00",
) -> dict:
    return {
        "verdict_id": verdict_id,
        "entity_id": entity_id,
        "txn_index": 0,
        "risk_score": risk_score,
        "decision": {"action": "REVIEW", "escalated_due_to_history": escalated},
        "baseline_decision": {"action": "REVIEW", "escalated_due_to_history": False},
        "escalated_due_to_history": escalated,
        "disposition": None,
        "disposed_at": None,
        "created_at": created_at,
        "notes": [],
    }


class TestAddAndGet:
    def test_unknown_verdict_returns_none(self, queue_factory):
        queue = queue_factory()
        assert queue.get("nope") is None

    def test_add_then_get_round_trips_the_item(self, queue_factory):
        queue = queue_factory()
        item = make_item("v1", 55.0)
        queue.add(item)
        assert queue.get("v1")["risk_score"] == 55.0
        assert queue.get("v1")["disposition"] is None


class TestListPending:
    def test_sorted_by_risk_score_descending(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("low", 30.0))
        queue.add(make_item("high", 90.0))
        queue.add(make_item("mid", 55.0))

        pending = queue.list_pending()
        assert [i["verdict_id"] for i in pending] == ["high", "mid", "low"]

    def test_disposed_items_are_excluded(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))
        queue.add(make_item("v2", 90.0))
        queue.dispose("v1", CONFIRMED_FRAUD)

        pending = queue.list_pending()
        assert [i["verdict_id"] for i in pending] == ["v2"]

    def test_empty_queue_returns_empty_list(self, queue_factory):
        queue = queue_factory()
        assert queue.list_pending() == []


class TestDispose:
    def test_unknown_verdict_raises(self, queue_factory):
        queue = queue_factory()
        with pytest.raises(UnknownVerdictError):
            queue.dispose("nope", CONFIRMED_FRAUD)

    def test_dispose_sets_disposition_and_timestamp(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))
        result = queue.dispose("v1", CONFIRMED_FRAUD)

        assert result["disposition"] == CONFIRMED_FRAUD
        assert result["disposed_at"] is not None
        assert queue.get("v1")["disposition"] == CONFIRMED_FRAUD

    def test_disposing_twice_raises_and_does_not_overwrite(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))
        queue.dispose("v1", CONFIRMED_FRAUD)

        with pytest.raises(AlreadyDisposedError):
            queue.dispose("v1", FALSE_POSITIVE)

        # the original disposition must survive the rejected second call
        assert queue.get("v1")["disposition"] == CONFIRMED_FRAUD


class TestMetrics:
    def test_no_disposed_items_yields_none_precisions(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))

        metrics = queue.metrics()
        assert metrics["total_disposed"] == 0
        assert metrics["overall_precision"] is None
        assert metrics["escalated_precision"] is None
        assert metrics["non_escalated_precision"] is None

    def test_overall_precision_across_disposed_items(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))
        queue.add(make_item("v2", 60.0))
        queue.add(make_item("v3", 70.0))
        queue.dispose("v1", CONFIRMED_FRAUD)
        queue.dispose("v2", CONFIRMED_FRAUD)
        queue.dispose("v3", FALSE_POSITIVE)

        metrics = queue.metrics()
        assert metrics["total_disposed"] == 3
        assert metrics["overall_precision"] == pytest.approx(2 / 3, abs=1e-4)

    def test_escalated_vs_non_escalated_precision_computed_separately(self, queue_factory):
        queue = queue_factory()
        # 2 escalated flags: 1 confirmed fraud, 1 false positive -> 0.5 precision
        queue.add(make_item("e1", 50.0, escalated=True))
        queue.add(make_item("e2", 55.0, escalated=True))
        # 2 non-escalated flags: both confirmed fraud -> 1.0 precision
        queue.add(make_item("n1", 90.0, escalated=False))
        queue.add(make_item("n2", 95.0, escalated=False))

        queue.dispose("e1", CONFIRMED_FRAUD)
        queue.dispose("e2", FALSE_POSITIVE)
        queue.dispose("n1", CONFIRMED_FRAUD)
        queue.dispose("n2", CONFIRMED_FRAUD)

        metrics = queue.metrics()
        assert metrics["escalated_count"] == 2
        assert metrics["escalated_precision"] == pytest.approx(0.5)
        assert metrics["non_escalated_count"] == 2
        assert metrics["non_escalated_precision"] == pytest.approx(1.0)

    def test_pending_items_are_excluded_from_metrics(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))
        queue.add(make_item("v2", 60.0))
        queue.dispose("v1", CONFIRMED_FRAUD)
        # v2 stays pending, undisposed

        metrics = queue.metrics()
        assert metrics["total_disposed"] == 1
        assert metrics["overall_precision"] == 1.0


class TestReset:
    def test_reset_clears_everything(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))
        queue.dispose("v1", CONFIRMED_FRAUD)
        queue.add(make_item("v2", 60.0))

        queue.reset()

        assert queue.get("v1") is None
        assert queue.list_pending() == []
        assert queue.metrics()["total_disposed"] == 0


class TestNotes:
    def test_unknown_verdict_raises(self, queue_factory):
        queue = queue_factory()
        with pytest.raises(UnknownVerdictError):
            queue.add_note("nope", "Reviewer", "note text")

    def test_add_note_appends_and_is_visible_on_get(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))

        note = queue.add_note("v1", "Alice", "Looks like a stolen card.")
        assert note["author"] == "Alice"
        assert note["text"] == "Looks like a stolen card."
        assert note["at"] is not None

        assert queue.get("v1")["notes"] == [note]

    def test_multiple_notes_accumulate_in_order(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0))

        queue.add_note("v1", "Alice", "first note")
        queue.add_note("v1", "Bob", "second note")

        notes = queue.get("v1")["notes"]
        assert [n["text"] for n in notes] == ["first note", "second note"]


class TestRelated:
    def test_unknown_verdict_raises(self, queue_factory):
        queue = queue_factory()
        with pytest.raises(UnknownVerdictError):
            queue.related("nope")

    def test_no_related_items_returns_empty_list(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0, entity_id="e1"))
        assert queue.related("v1") == []

    def test_only_same_entity_items_are_related_and_self_is_excluded(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0, entity_id="e1", created_at="2024-01-01T00:00:00+00:00"))
        queue.add(make_item("v2", 60.0, entity_id="e1", created_at="2024-01-02T00:00:00+00:00"))
        queue.add(make_item("v3", 70.0, entity_id="e2", created_at="2024-01-03T00:00:00+00:00"))

        related = queue.related("v1")
        assert [i["verdict_id"] for i in related] == ["v2"]

    def test_related_includes_disposed_items_sorted_most_recent_first(self, queue_factory):
        queue = queue_factory()
        queue.add(make_item("v1", 50.0, entity_id="e1", created_at="2024-01-01T00:00:00+00:00"))
        queue.add(make_item("v2", 60.0, entity_id="e1", created_at="2024-01-03T00:00:00+00:00"))
        queue.add(make_item("v3", 70.0, entity_id="e1", created_at="2024-01-02T00:00:00+00:00"))
        queue.dispose("v2", CONFIRMED_FRAUD)

        related = queue.related("v1")
        assert [i["verdict_id"] for i in related] == ["v2", "v3"]


class TestRedisSpecific:
    """Behavior that only makes sense to assert for the Redis backend
    directly — mirrors test_entity_memory.py's TestRedisSpecific."""

    def test_uses_namespaced_keys(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        queue = RedisReviewQueue(client)
        queue.add(make_item("v1", 50.0))
        assert client.exists("riskmgr:review_queue:item:v1")
        assert client.sismember("riskmgr:review_queue:pending", "v1")

    def test_dispose_removes_from_pending_set_but_keeps_the_item(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        queue = RedisReviewQueue(client)
        queue.add(make_item("v1", 50.0))
        queue.dispose("v1", CONFIRMED_FRAUD)

        assert not client.sismember("riskmgr:review_queue:pending", "v1")
        assert client.exists("riskmgr:review_queue:item:v1")

    def test_reset_only_touches_this_projects_keys(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        client.set("some_other_apps_key", "do-not-touch")

        queue = RedisReviewQueue(client)
        queue.add(make_item("v1", 50.0))
        queue.reset()

        assert client.get("some_other_apps_key") == "do-not-touch"
        assert queue.get("v1") is None
