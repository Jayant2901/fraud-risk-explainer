"""
Escalation-transition alerting.

entity_memory computes NORMAL/WATCH/ELEVATED on every scored transaction,
but nothing ever surfaces the *transition* -- the moment an entity crosses
into a worse state, which is exactly the moment a fraud team would want
to know about. This module is that surface.

Only escalating transitions page (NORMAL->WATCH, WATCH->ELEVATED,
NORMAL->ELEVATED): a de-escalation is just the rolling window aging old
verdicts out, not a new event, and alerting on it would be noise.

Delivery must never slow down or fail a scoring request:
- No ESCALATION_WEBHOOK_URL configured -> every transition is still
  logged, but nothing is sent anywhere (opt-in, same as the feedback
  loop -- absence of config must never surprise anyone with a new
  dependency).
- Configured -> the POST happens off the request thread (a daemon
  thread, not the one handling the score), with a short timeout and
  every exception swallowed and logged. A broken alerting endpoint is
  the alerting system's problem, never the scoring path's.

Per-entity cooldown (default 5 minutes) so an entity oscillating across a
threshold doesn't page on every transaction. Cooldown state lives in
Redis when available (shared across workers, same dual-backend pattern as
circuit_breaker.py) and in-process otherwise.
"""
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "riskmgr:escalation_alert"

WEBHOOK_URL = os.environ.get("ESCALATION_WEBHOOK_URL")
DEFAULT_COOLDOWN_SECONDS = float(os.environ.get("ESCALATION_ALERT_COOLDOWN_SECONDS", "300"))
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("ESCALATION_ALERT_TIMEOUT_SECONDS", "5"))

# Ordering used to tell an escalation (alert-worthy) from a de-escalation
# (not) -- higher number is worse.
SEVERITY = {"NORMAL": 0, "WATCH": 1, "ELEVATED": 2}


class EscalationNotifier:
    """Watches entity escalation-state transitions and fires alerts for
    the escalating ones, subject to a per-entity cooldown.

    send: callable(payload: dict) -> None, invoked once a transition
    clears the transition/cooldown checks. Swappable so a webhook, a
    no-op, or a test spy all share the same decision logic below.
    """

    def __init__(self, send, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
                 redis_client=None, clock=time.time):
        self._send = send
        self._cooldown_seconds = cooldown_seconds
        self._redis = redis_client
        self._clock = clock
        self._last_sent: dict[str, float] = {}

    def _key(self, entity_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:last_sent:{entity_id}"

    def _get_last_sent(self, entity_id: str) -> float | None:
        if self._redis is None:
            return self._last_sent.get(entity_id)
        raw = self._redis.get(self._key(entity_id))
        return float(raw) if raw else None

    def _set_last_sent(self, entity_id: str, value: float) -> None:
        if self._redis is None:
            self._last_sent[entity_id] = value
            return
        # TTL a little past the cooldown itself: the key only needs to
        # outlive the window it's suppressing, and letting Redis expire
        # it avoids an unbounded key per entity ever scored.
        self._redis.set(self._key(entity_id), value, ex=max(1, int(self._cooldown_seconds) + 5))

    def notify_transition(self, entity_id: str, from_state: str | None, to_state: str | None,
                           risk_score: float, verdict_id: str) -> None:
        if not entity_id or from_state == to_state:
            return
        if SEVERITY.get(to_state, 0) <= SEVERITY.get(from_state, 0):
            return  # de-escalation, or an unrecognized state -- not alert-worthy

        now = self._clock()
        last_sent = self._get_last_sent(entity_id)
        if last_sent is not None and now - last_sent < self._cooldown_seconds:
            logger.debug("Escalation alert suppressed by cooldown", extra={"entity_id": entity_id})
            return
        self._set_last_sent(entity_id, now)

        payload = {
            "entity_id": entity_id,
            "from_state": from_state,
            "to_state": to_state,
            "risk_score": risk_score,
            "verdict_id": verdict_id,
            "at": now,
        }
        logger.warning("Escalation transition", extra=payload)
        try:
            self._send(payload)
        except Exception:
            # send() for the webhook backend only ever launches a thread,
            # so this really is "something is very wrong" -- but a
            # scoring request must survive it regardless.
            logger.exception("Escalation alert delivery failed", extra={"entity_id": entity_id})


def _post_webhook(url: str, payload: dict, timeout: float) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=timeout)
    except (urllib.error.URLError, OSError):
        logger.warning("Escalation webhook unreachable", extra={"url": url}, exc_info=True)


def _webhook_send(url: str, timeout: float):
    def send(payload: dict) -> None:
        # Off the request thread: a slow or hanging webhook endpoint must
        # never add latency to the scoring request that triggered it.
        threading.Thread(target=_post_webhook, args=(url, payload, timeout), daemon=True).start()
    return send


def _noop_send(payload: dict) -> None:
    pass


def create_notifier(redis_client=None) -> EscalationNotifier:
    """ESCALATION_WEBHOOK_URL unset (the default): transitions are still
    logged by notify_transition, just never sent anywhere. Set: posted as
    JSON, fire-and-forget, to that URL."""
    send = _webhook_send(WEBHOOK_URL, DEFAULT_TIMEOUT_SECONDS) if WEBHOOK_URL else _noop_send
    return EscalationNotifier(send, redis_client=redis_client)
