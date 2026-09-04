"""
A circuit breaker for the LLM dependency.

llm_agent.py handles LLM *errors* well and had no defense against LLM
*slowness*. A call that hangs occupies a worker thread for its full
duration, and enough of those starve the thread pool — at which point
scoring, which does not need the LLM at all, starts queueing behind
explanations. A risk system must never let its explanation layer degrade
its decision layer.

So: after N consecutive failures the breaker opens, every call
short-circuits to the fallback for a cooldown window, and one call is
allowed through afterwards to test recovery.

Roughly forty lines of logic and no dependency, because that is all this
needs. The counter lives in Redis when it is available so the breaker is
shared across workers — one worker learning the API is down should not
leave the other five each discovering it independently — and in-process
otherwise, the same dual-backend pattern as entity_memory.py.
"""
import logging
import os
import time

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "riskmgr:circuit"

# Consecutive failures before the breaker opens. Low enough to stop
# wasting wall-clock on a dependency that is clearly down, high enough
# that a single blip doesn't disable explanations.
DEFAULT_FAILURE_THRESHOLD = int(os.environ.get("LLM_BREAKER_THRESHOLD", "5"))

# How long to stay open before letting one call through to test recovery.
DEFAULT_COOLDOWN_SECONDS = float(os.environ.get("LLM_BREAKER_COOLDOWN_SECONDS", "60"))

CLOSED = "closed"
OPEN = "open"


class CircuitBreaker:
    """Tracks consecutive failures for one named dependency.

    allow() before calling; record_success()/record_failure() after.
    Nothing here calls the dependency itself — the breaker gates, it does
    not wrap, so a caller can keep its own error handling intact.
    """

    def __init__(
        self,
        name: str = "llm",
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        redis_client=None,
        clock=time.time,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._redis = redis_client
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    # --- state access, backed by Redis when configured ----------------
    def _key(self, suffix: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{self.name}:{suffix}"

    def _get_failures(self) -> int:
        if self._redis is None:
            return self._failures
        raw = self._redis.get(self._key("failures"))
        return int(raw) if raw else 0

    def _set_failures(self, value: int) -> None:
        if self._redis is None:
            self._failures = value
            return
        self._redis.set(self._key("failures"), value)

    def _get_opened_at(self) -> float | None:
        if self._redis is None:
            return self._opened_at
        raw = self._redis.get(self._key("opened_at"))
        return float(raw) if raw else None

    def _set_opened_at(self, value: float | None) -> None:
        if self._redis is None:
            self._opened_at = value
            return
        if value is None:
            self._redis.delete(self._key("opened_at"))
        else:
            self._redis.set(self._key("opened_at"), value)

    # --- the breaker itself -------------------------------------------
    def allow(self) -> bool:
        """False while the breaker is open and still cooling down."""
        opened_at = self._get_opened_at()
        if opened_at is None:
            return True
        if self._clock() - opened_at >= self.cooldown_seconds:
            # Cooldown elapsed: close and let the next call through to
            # test whether the dependency recovered.
            self._close()
            return True
        return False

    def record_success(self) -> None:
        if self._get_failures() or self._get_opened_at() is not None:
            self._close()

    def record_failure(self) -> None:
        failures = self._get_failures() + 1
        self._set_failures(failures)
        if failures >= self.failure_threshold and self._get_opened_at() is None:
            self._set_opened_at(self._clock())
            logger.warning(
                "Circuit breaker opened",
                extra={
                    "breaker": self.name,
                    "consecutive_failures": failures,
                    "cooldown_seconds": self.cooldown_seconds,
                },
            )

    def _close(self) -> None:
        was_open = self._get_opened_at() is not None
        self._set_failures(0)
        self._set_opened_at(None)
        if was_open:
            logger.info("Circuit breaker closed", extra={"breaker": self.name})

    def state(self) -> dict:
        """What /api/health reports — this is what turns 'the explanations
        stopped working' into a diagnosable condition."""
        opened_at = self._get_opened_at()
        failures = self._get_failures()
        if opened_at is None:
            return {
                "state": CLOSED,
                "consecutive_failures": failures,
                "seconds_until_retry": 0.0,
            }
        remaining = max(0.0, self.cooldown_seconds - (self._clock() - opened_at))
        return {
            "state": OPEN,
            "consecutive_failures": failures,
            "seconds_until_retry": round(remaining, 1),
        }

    def reset(self) -> None:
        """Test-only."""
        self._close()
