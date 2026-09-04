"""
Stream consumer — scores transaction events off the Redis Stream that
POST /api/events/transaction publishes to.

Run:
    python -m src.stream_consumer

This is the process that makes ingestion real: the API accepts an event
and returns 202 in milliseconds, and this worker does the actual scoring,
out of band, whether or not anyone has a browser open.

It scores through src/scoring_service.py — the same ScoringService the
synchronous endpoints use — so a transaction scored here and the same
transaction scored through /api/score-custom produce the same decision.
tests/test_stream_consumer.py asserts that equivalence directly, because
two divergent copies of the scoring logic would make this whole phase
worthless.

Unlike the API, this worker DOES generate the LLM explanation inline:
it is a background process with nothing waiting on it, so there's no
request latency to protect. The explanation lands in the same cache the
frontend reads, via the same batch path.
"""
import logging
import os
import signal
import sys
import time

sys.path.append(os.path.dirname(__file__))

from decision_rules import load_decision_thresholds
from entity_memory import create_entity_memory
from feature_store import create_feature_store
from notifications import create_notifier
from event_stream import (
    MAX_DELIVERY_ATTEMPTS,
    ack,
    claim_stale_events,
    dead_letter,
    delivery_count,
    ensure_group,
    read_events,
)
from llm_agent import RiskExplainerAgent
from logging_utils import configure_logging
from redis_utils import KeyedCache, get_redis_client
from review_queue import create_review_queue
from risk_explainer import RiskExplainer
from scoring_service import ScoringService, generate_explanation

logger = logging.getLogger(__name__)

# Matches api/main.py's cache configuration exactly — the consumer writes
# explanations into the same keyspace the API serves them from, so these
# must not drift.
EXPLANATION_TTL_SECONDS = 60 * 60

# How long a message may sit read-but-unacked before another consumer may
# claim it. Long enough that a slow-but-alive worker isn't robbed of its
# message mid-scoring; short enough that a crashed worker's message is
# picked up promptly.
STALE_CLAIM_IDLE_MS = 60_000

BATCH_SIZE = 10
BLOCK_MS = 5_000

# The consumer scores transactions too — real ingestion volume mostly
# flows through here, not the synchronous API — so its
# riskmgr_decisions_total/riskmgr_escalation_transitions_total counters
# (src/domain_metrics.py) need their own scrape target: this process
# never otherwise serves HTTP. 0 disables it (e.g. under pytest, which
# never calls main()).
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9101"))


class EventConsumer:
    """One consumer in the group. Constructed with its collaborators so
    tests can drive it with fakeredis and fake models."""

    def __init__(self, redis_client, scoring_service, agent_provider, explanations_cache,
                 consumer_name: str = "consumer-1"):
        self._redis = redis_client
        self._scoring = scoring_service
        self._agent_provider = agent_provider
        self._explanations_cache = explanations_cache
        self._consumer_name = consumer_name
        self._running = False

    def process_one(self, message_id: str, event: dict) -> bool:
        """Score a single event. Returns True if it was handled (acked),
        False if it was dead-lettered.

        A failure must neither be silently dropped nor redelivered
        forever, so retry accounting rides on the consumer group's own
        pending-entry delivery count rather than state of our own.
        """
        try:
            scored = self._scoring.score_and_decide(
                event.get("transaction", {}),
                event.get("entity_id"),
                # An event carries no index into any cached entity
                # sequence — it's a live transaction, not a replay.
                txn_index=0,
            )
        except Exception as exc:
            attempts = delivery_count(self._redis, message_id)
            if attempts >= MAX_DELIVERY_ATTEMPTS:
                dead_letter(self._redis, event, f"{type(exc).__name__}: {exc}", message_id)
                return False
            # Leave it unacked: it stays in the pending list and is
            # redelivered to whichever consumer claims it next.
            logger.warning(
                "Scoring failed, leaving event for retry",
                extra={"event_id": event.get("event_id"), "attempt": attempts, "error": str(exc)},
            )
            return False

        # The event's verdict_id was handed to the caller in the 202
        # response, so the caller can poll for exactly this verdict.
        requested_verdict_id = event.get("verdict_id")
        if requested_verdict_id:
            self._explanations_cache.put(requested_verdict_id, {"status": "pending"})

        generate_explanation(
            self._agent_provider,
            self._explanations_cache,
            requested_verdict_id or scored["verdict_id"],
            scored["risk_score"],
            scored["top_factors"],
            scored["escalation_before"],
        )

        ack(self._redis, message_id)
        logger.info(
            "Event scored",
            extra={
                "event_id": event.get("event_id"),
                "verdict_id": requested_verdict_id or scored["verdict_id"],
                "action": scored["decision"]["action"],
            },
        )
        return True

    def poll_once(self) -> int:
        """One read/process cycle. Returns the number of messages handled.
        Stale messages abandoned by a dead consumer are claimed first, so
        a crashed worker's in-flight transaction still gets scored."""
        handled = 0
        for message_id, event in claim_stale_events(
            self._redis, self._consumer_name, min_idle_ms=STALE_CLAIM_IDLE_MS, count=BATCH_SIZE
        ):
            self.process_one(message_id, event)
            handled += 1

        for message_id, event in read_events(
            self._redis, self._consumer_name, count=BATCH_SIZE, block_ms=BLOCK_MS
        ):
            self.process_one(message_id, event)
            handled += 1
        return handled

    def run(self) -> None:
        self._running = True
        logger.info("Stream consumer started", extra={"consumer": self._consumer_name})
        while self._running:
            try:
                self.poll_once()
            except Exception:
                # A failure in the loop itself (Redis blip) must not kill
                # the worker — back off briefly and carry on.
                logger.exception("Consumer loop error; retrying")
                time.sleep(1)
        logger.info("Stream consumer stopped", extra={"consumer": self._consumer_name})

    def stop(self) -> None:
        self._running = False


def build_consumer(redis_client, consumer_name: str) -> EventConsumer:
    thresholds = load_decision_thresholds()
    agent = RiskExplainerAgent(
        review_threshold=thresholds["review"], block_threshold=thresholds["block"]
    )
    explanations_cache = KeyedCache(
        redis_client, prefix="riskmgr:explanations", ttl_seconds=EXPLANATION_TTL_SECONDS
    )
    scoring = ScoringService(
        explainer=RiskExplainer(),
        memory=create_entity_memory(redis_client),
        review_queue=create_review_queue(redis_client),
        explanations_cache=explanations_cache,
        thresholds_provider=lambda: thresholds,
        # Shared with the API via Redis, so an entity's history is the
        # same whichever process scores its next transaction.
        feature_store=create_feature_store(redis_client),
        notifier=create_notifier(redis_client),
    )
    return EventConsumer(
        redis_client=redis_client,
        scoring_service=scoring,
        agent_provider=lambda: agent,
        explanations_cache=explanations_cache,
        consumer_name=consumer_name,
    )


def main() -> int:
    configure_logging()
    if METRICS_PORT:
        from prometheus_client import start_http_server
        start_http_server(METRICS_PORT)
    redis_client = get_redis_client()
    if redis_client is None:
        logger.error("REDIS_URL is not set — the stream consumer requires Redis")
        return 1

    ensure_group(redis_client)
    consumer = build_consumer(redis_client, os.environ.get("CONSUMER_NAME", f"consumer-{os.getpid()}"))

    # Ack the message in flight and exit cleanly on SIGTERM, so a compose
    # stop or a rolling restart doesn't strand a pending entry.
    def _handle_signal(signum, _frame):
        logger.info("Received signal, shutting down", extra={"signal": signum})
        consumer.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    consumer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
