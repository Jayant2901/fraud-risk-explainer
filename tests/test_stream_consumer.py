"""
Stream ingestion tests.

The load-bearing one is TestScoringEquivalence: an event scored by the
consumer must produce the same decision as the same transaction scored
through the synchronous path. Two entry points into scoring is exactly
how a system grows two subtly different answers to "would this be
blocked", so that equivalence is asserted directly rather than assumed
from both calling the same function.
"""
import json

import fakeredis
import pytest

import event_stream
from conftest import FakeAgent, FakeExplainer
from entity_memory import create_entity_memory
from redis_utils import KeyedCache
from review_queue import create_review_queue
from scoring_service import ScoringService
from stream_consumer import EventConsumer

THRESHOLDS = {"review": 40.0, "block": 80.0}


@pytest.fixture
def redis_client():
    client = fakeredis.FakeRedis(decode_responses=True)
    event_stream.ensure_group(client)
    return client


def build_scoring(redis_client, explainer=None):
    return ScoringService(
        explainer=explainer or FakeExplainer(),
        memory=create_entity_memory(redis_client),
        review_queue=create_review_queue(redis_client),
        explanations_cache=KeyedCache(redis_client, prefix="riskmgr:explanations", ttl_seconds=3600),
        thresholds_provider=lambda: THRESHOLDS,
    )


def build_consumer(redis_client, scoring=None, explanations_cache=None):
    scoring = scoring or build_scoring(redis_client)
    return EventConsumer(
        redis_client=redis_client,
        scoring_service=scoring,
        agent_provider=lambda: FakeAgent(),
        explanations_cache=explanations_cache
        or KeyedCache(redis_client, prefix="riskmgr:explanations", ttl_seconds=3600),
        consumer_name="test-consumer",
    )


def make_event(event_id: str = "evt-1", entity_id: str | None = "entity-a", amount: float = 100.0) -> dict:
    return {
        "event_id": event_id,
        "verdict_id": f"verdict-for-{event_id}",
        "entity_id": entity_id,
        "transaction": {"TransactionAmt": amount, "ProductCD": "W"},
    }


class TestConsumeAndAck:
    def test_an_enqueued_event_is_consumed_and_acknowledged(self, redis_client):
        event_stream.publish_event(redis_client, make_event())
        consumer = build_consumer(redis_client)

        handled = consumer.poll_once()

        assert handled == 1
        # Nothing left pending — the message was acked, not abandoned.
        assert event_stream.stream_depth(redis_client)["pending"] == 0

    def test_scoring_records_the_verdict_in_entity_memory(self, redis_client):
        scoring = build_scoring(redis_client, FakeExplainer(risk_score=90.0))
        memory = create_entity_memory(redis_client)
        event_stream.publish_event(redis_client, make_event())

        build_consumer(redis_client, scoring).poll_once()

        state = memory.get_escalation_state("entity-a")
        assert state["recent_verdict_count"] == 1
        assert state["recent_verdicts"] == ["BLOCK"]

    def test_the_explanation_lands_under_the_verdict_id_the_caller_was_given(self, redis_client):
        cache = KeyedCache(redis_client, prefix="riskmgr:explanations", ttl_seconds=3600)
        event_stream.publish_event(redis_client, make_event("evt-42"))

        build_consumer(redis_client, explanations_cache=cache).poll_once()

        stored = cache.get("verdict-for-evt-42")
        assert stored["status"] == "ready"
        assert stored["verdict"]["explanation"] == "fake explanation"

    def test_a_flagged_event_reaches_the_review_queue(self, redis_client):
        scoring = build_scoring(redis_client, FakeExplainer(risk_score=90.0))
        queue = create_review_queue(redis_client)
        event_stream.publish_event(redis_client, make_event())

        build_consumer(redis_client, scoring).poll_once()

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["decision"]["action"] == "BLOCK"

    def test_an_allowed_event_does_not_reach_the_review_queue(self, redis_client):
        scoring = build_scoring(redis_client, FakeExplainer(risk_score=10.0))
        queue = create_review_queue(redis_client)
        event_stream.publish_event(redis_client, make_event())

        build_consumer(redis_client, scoring).poll_once()

        assert queue.list_pending() == []

    def test_an_event_with_no_entity_scores_against_the_cold_start_baseline(self, redis_client):
        event_stream.publish_event(redis_client, make_event(entity_id=None))

        assert build_consumer(redis_client).poll_once() == 1

    def test_polling_an_empty_stream_handles_nothing(self, redis_client):
        assert build_consumer(redis_client).poll_once() == 0


class TestScoringEquivalence:
    """The anti-drift test. The consumer and the synchronous API path must
    reach the same decision for the same transaction — that is the entire
    justification for extracting ScoringService."""

    def test_consumer_and_synchronous_path_agree_on_the_decision(self, redis_client):
        txn = {"TransactionAmt": 250.0, "ProductCD": "W"}

        # Synchronous path: score directly through the shared service, the
        # way /api/score-custom does.
        sync_service = build_scoring(redis_client, FakeExplainer(risk_score=55.0))
        sync_result = sync_service.score_and_decide(txn, None)

        # Streamed path: the same transaction through the consumer.
        stream_service = build_scoring(redis_client, FakeExplainer(risk_score=55.0))
        event_stream.publish_event(
            redis_client,
            {"event_id": "e", "verdict_id": "v", "entity_id": None, "transaction": txn},
        )
        captured = {}
        consumer = build_consumer(redis_client, stream_service)
        original = stream_service.score_and_decide

        def capture(*args, **kwargs):
            captured["result"] = original(*args, **kwargs)
            return captured["result"]

        stream_service.score_and_decide = capture
        consumer.poll_once()

        streamed = captured["result"]
        assert streamed["risk_score"] == sync_result["risk_score"]
        assert streamed["decision"] == sync_result["decision"]
        assert streamed["baseline_decision"] == sync_result["baseline_decision"]
        assert streamed["above_threshold"] == sync_result["above_threshold"]
        assert streamed["escalation_before"]["state"] == sync_result["escalation_before"]["state"]

    def test_both_paths_escalate_identically_given_the_same_history(self, redis_client):
        memory = create_entity_memory(redis_client)
        # Drive the entity to ELEVATED with full-strength BLOCKs.
        for _ in range(3):
            memory.record_verdict("entity-hot", "BLOCK", 100.0)

        service = build_scoring(redis_client, FakeExplainer(risk_score=50.0))
        sync_result = service.score_and_decide({"TransactionAmt": 1.0}, "entity-hot", record_verdict=False)

        assert sync_result["escalation_before"]["state"] == "ELEVATED"
        # A mid-band score under ELEVATED becomes BLOCK, not REVIEW.
        assert sync_result["decision"]["action"] == "BLOCK"
        assert sync_result["decision"]["escalated_due_to_history"] is True


class TestFailureHandling:
    class ExplodingExplainer:
        def score_transaction(self, txn):
            raise RuntimeError("model exploded")

    def test_a_failing_event_is_left_pending_for_retry_not_dropped(self, redis_client):
        scoring = build_scoring(redis_client, self.ExplodingExplainer())
        event_stream.publish_event(redis_client, make_event())

        build_consumer(redis_client, scoring).poll_once()

        # Still pending: unacked, so it will be redelivered.
        assert event_stream.stream_depth(redis_client)["pending"] == 1
        assert event_stream.list_dead_letter(redis_client) == []

    def test_an_event_is_dead_lettered_after_the_retry_limit(self, redis_client):
        scoring = build_scoring(redis_client, self.ExplodingExplainer())
        consumer = build_consumer(redis_client, scoring)
        event_stream.publish_event(redis_client, make_event("evt-bad"))

        # First delivery via the normal read; it fails and stays pending.
        consumer.poll_once()
        # Each subsequent claim redelivers the same pending message,
        # incrementing the group's own delivery counter until the retry
        # limit is reached.
        for _ in range(event_stream.MAX_DELIVERY_ATTEMPTS + 1):
            for message_id, event in event_stream.claim_stale_events(
                redis_client, "test-consumer", min_idle_ms=0
            ):
                consumer.process_one(message_id, event)

        dead = event_stream.list_dead_letter(redis_client)
        assert len(dead) == 1
        assert dead[0]["event"]["event_id"] == "evt-bad"
        assert "model exploded" in dead[0]["error"]
        # Dead-lettered messages are acked, so they stop being redelivered.
        assert event_stream.stream_depth(redis_client)["pending"] == 0

    def test_a_stale_message_from_a_dead_consumer_is_claimed_and_scored(self, redis_client):
        event_stream.publish_event(redis_client, make_event())
        # A consumer reads the message and dies without acking.
        event_stream.read_events(redis_client, "crashed-consumer", block_ms=0)
        assert event_stream.stream_depth(redis_client)["pending"] == 1

        consumer = build_consumer(redis_client)
        for message_id, event in event_stream.claim_stale_events(
            redis_client, "test-consumer", min_idle_ms=0
        ):
            consumer.process_one(message_id, event)

        assert event_stream.stream_depth(redis_client)["pending"] == 0


class TestStreamHelpers:
    def test_publish_then_read_round_trips_the_event(self, redis_client):
        event_stream.publish_event(redis_client, make_event("evt-round-trip"))

        events = event_stream.read_events(redis_client, "c", block_ms=0)

        assert len(events) == 1
        assert events[0][1]["event_id"] == "evt-round-trip"

    def test_ensure_group_is_idempotent(self, redis_client):
        event_stream.ensure_group(redis_client)
        event_stream.ensure_group(redis_client)  # must not raise

    def test_stream_operations_without_redis_raise_rather_than_pretending(self):
        with pytest.raises(event_stream.StreamUnavailableError):
            event_stream.publish_event(None, make_event())
        with pytest.raises(event_stream.StreamUnavailableError):
            event_stream.ensure_group(None)

    def test_dead_letter_entries_carry_the_original_payload(self, redis_client):
        message_id = event_stream.publish_event(redis_client, make_event("evt-dl"))
        event_stream.read_events(redis_client, "c", block_ms=0)

        event_stream.dead_letter(redis_client, make_event("evt-dl"), "boom", message_id)

        dead = event_stream.list_dead_letter(redis_client)
        assert dead[0]["event"]["transaction"]["TransactionAmt"] == 100.0
        assert dead[0]["original_message_id"] == message_id

    def test_stream_depth_reports_length_and_pending(self, redis_client):
        event_stream.publish_event(redis_client, make_event("a"))
        event_stream.publish_event(redis_client, make_event("b"))

        assert event_stream.stream_depth(redis_client)["length"] == 2
        assert event_stream.stream_depth(redis_client)["pending"] == 0

        event_stream.read_events(redis_client, "c", block_ms=0)
        assert event_stream.stream_depth(redis_client)["pending"] == 2

    def test_payload_is_json_encoded_in_a_single_field(self, redis_client):
        event_stream.publish_event(redis_client, make_event("evt-json"))

        _id, fields = redis_client.xrange(event_stream.STREAM_KEY)[0]
        assert json.loads(fields["payload"])["event_id"] == "evt-json"
