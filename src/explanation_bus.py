"""
Delivery of explanation deltas from whoever is calling the LLM to
whoever is holding the SSE connection.

With more than one worker those are different processes: uvicorn routes
the POST /api/score that starts the LLM call to worker A, and the
subsequent GET /api/verdicts/{id}/stream to worker B. An in-process
queue would work perfectly in development and silently deliver nothing
in production, which is the worst possible failure shape.

So deltas go over Redis pub/sub, keyed by verdict_id, when Redis is
configured. Without Redis there is exactly one worker by definition of
the deployment this project documents, and the in-process fallback is
correct — unlike RT-1's durability question, where falling back would
have been a lie. The difference is documented in the README.
"""
import json
import logging
import queue
import threading

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "riskmgr:explanation-stream"

# How long a subscriber waits for the next delta before emitting a
# keepalive instead. Proxies commonly close an idle connection at 30-60s.
POLL_TIMEOUT_SECONDS = 15.0


def channel_for(verdict_id: str) -> str:
    return f"{CHANNEL_PREFIX}:{verdict_id}"


class ExplanationBus:
    """publish() from the producer, subscribe() from the SSE handler.

    Messages are the same dicts llm_agent.explain_stream yields
    ({"type": "delta"|"complete", ...}), so the transport adds no shape of
    its own.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        # Only used in the no-Redis case: verdict_id -> list of queues.
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._lock = threading.Lock()

    def publish(self, verdict_id: str, message: dict) -> None:
        if self._redis is not None:
            self._redis.publish(channel_for(verdict_id), json.dumps(message))
            return
        with self._lock:
            for subscriber in self._subscribers.get(verdict_id, []):
                subscriber.put(message)

    def subscription(self, verdict_id: str) -> "Subscription":
        """Attach to this verdict's channel immediately, before any
        messages are read.

        The producer (the LLM call) starts as soon as the score response
        goes out, and the client's EventSource connects a few milliseconds
        later — so the first deltas are routinely published before anyone
        is listening, and pub/sub has no retention. Attaching up front
        lets the caller re-check the explanation cache *after* it is
        subscribed: anything published from that moment on is captured,
        and anything published before it is already in the cache. Without
        this split there is a window where a message is in neither place.
        """
        return Subscription(self, verdict_id)

    def subscribe(self, verdict_id: str):
        """Yields messages for this verdict until a terminal 'complete'
        arrives. Yields None on timeout so the caller can emit a keepalive
        and notice a client that has gone away."""
        yield from self.subscription(verdict_id).messages()

    def _attach_local(self, verdict_id: str) -> queue.Queue:
        inbox: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(verdict_id, []).append(inbox)
        return inbox

    def _detach_local(self, verdict_id: str, inbox: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(verdict_id, [])
            if inbox in subscribers:
                subscribers.remove(inbox)
            if not subscribers:
                self._subscribers.pop(verdict_id, None)


class Subscription:
    """An attached listener for one verdict's messages.

    Constructing it subscribes; messages() then yields what arrives.
    Splitting those two steps is what lets a caller safely re-check
    durable state (the explanation cache) after attaching but before
    blocking — see ExplanationBus.subscription.
    """

    def __init__(self, bus: ExplanationBus, verdict_id: str):
        self._bus = bus
        self._verdict_id = verdict_id
        self._pubsub = None
        self._inbox = None

        if bus._redis is not None:
            self._pubsub = bus._redis.pubsub(ignore_subscribe_messages=True)
            self._pubsub.subscribe(channel_for(verdict_id))
        else:
            self._inbox = bus._attach_local(verdict_id)

    def messages(self):
        """Yields each message until a terminal 'complete', and None on
        each idle timeout so the caller can emit a keepalive."""
        try:
            while True:
                message = self._next()
                yield message
                if message is not None and message.get("type") == "complete":
                    return
        finally:
            self.close()

    def _next(self) -> dict | None:
        if self._pubsub is not None:
            raw = self._pubsub.get_message(timeout=POLL_TIMEOUT_SECONDS)
            if raw is None:
                return None
            try:
                return json.loads(raw["data"])
            except (TypeError, ValueError):
                return None
        try:
            return self._inbox.get(timeout=POLL_TIMEOUT_SECONDS)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._pubsub is not None:
            try:
                self._pubsub.close()
            except Exception:
                logger.debug("pubsub close failed", exc_info=True)
            self._pubsub = None
        elif self._inbox is not None:
            self._bus._detach_local(self._verdict_id, self._inbox)
            self._inbox = None
