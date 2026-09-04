"""
Circuit breaker tests, run against both backends (in-process and Redis
via fakeredis) — the dual-implementation pattern test_entity_memory.py
established. The Redis path matters because one worker learning the LLM
is down should spare the others from each rediscovering it.

Time is injected rather than slept, so cooldown behavior is asserted
exactly instead of approximately.
"""
import fakeredis
import pytest

from circuit_breaker import CLOSED, OPEN, CircuitBreaker


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(params=["in-process", "redis"])
def breaker_factory(request):
    redis_client = (
        fakeredis.FakeRedis(decode_responses=True) if request.param == "redis" else None
    )

    def build(threshold: int = 3, cooldown: float = 60.0, clock=None):
        return CircuitBreaker(
            name="llm-test",
            failure_threshold=threshold,
            cooldown_seconds=cooldown,
            redis_client=redis_client,
            clock=clock or FakeClock(),
        )

    return build


class TestOpening:
    def test_starts_closed_and_allows_calls(self, breaker_factory):
        breaker = breaker_factory()
        assert breaker.allow() is True
        assert breaker.state()["state"] == CLOSED

    def test_stays_closed_below_the_threshold(self, breaker_factory):
        breaker = breaker_factory(threshold=3)
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.allow() is True
        assert breaker.state()["consecutive_failures"] == 2

    def test_opens_on_the_nth_consecutive_failure(self, breaker_factory):
        breaker = breaker_factory(threshold=3)
        for _ in range(3):
            breaker.record_failure()

        assert breaker.allow() is False
        assert breaker.state()["state"] == OPEN

    def test_a_success_resets_the_failure_run(self, breaker_factory):
        breaker = breaker_factory(threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()

        # Consecutive, not cumulative — two before the success don't count.
        assert breaker.allow() is True
        assert breaker.state()["consecutive_failures"] == 1


class TestCooldown:
    def test_short_circuits_during_cooldown(self, breaker_factory):
        clock = FakeClock()
        breaker = breaker_factory(threshold=2, cooldown=60.0, clock=clock)
        breaker.record_failure()
        breaker.record_failure()

        clock.advance(59)

        assert breaker.allow() is False
        assert breaker.state()["seconds_until_retry"] == 1.0

    def test_closes_once_the_window_elapses(self, breaker_factory):
        clock = FakeClock()
        breaker = breaker_factory(threshold=2, cooldown=60.0, clock=clock)
        breaker.record_failure()
        breaker.record_failure()

        clock.advance(60)

        assert breaker.allow() is True
        assert breaker.state()["state"] == CLOSED
        # The failure count resets too, so recovery starts from scratch.
        assert breaker.state()["consecutive_failures"] == 0

    def test_reopens_immediately_if_the_probe_call_fails_again(self, breaker_factory):
        clock = FakeClock()
        breaker = breaker_factory(threshold=1, cooldown=30.0, clock=clock)
        breaker.record_failure()
        assert breaker.allow() is False

        clock.advance(30)
        assert breaker.allow() is True  # one probe allowed through
        breaker.record_failure()

        assert breaker.allow() is False

    def test_reports_time_until_retry_while_open(self, breaker_factory):
        clock = FakeClock()
        breaker = breaker_factory(threshold=1, cooldown=60.0, clock=clock)
        breaker.record_failure()

        clock.advance(15)

        assert breaker.state()["seconds_until_retry"] == 45.0

    def test_a_closed_breaker_reports_no_wait(self, breaker_factory):
        assert breaker_factory().state()["seconds_until_retry"] == 0.0


class TestSharedState:
    """The Redis-backed breaker is shared: a failure recorded by one
    worker must be visible to another."""

    def test_two_breakers_on_one_redis_share_the_failure_count(self):
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        worker_a = CircuitBreaker("llm", failure_threshold=2, redis_client=redis_client)
        worker_b = CircuitBreaker("llm", failure_threshold=2, redis_client=redis_client)

        worker_a.record_failure()
        worker_b.record_failure()

        # Worker B's failure was the second overall, so both see it open.
        assert worker_a.allow() is False
        assert worker_b.allow() is False

    def test_in_process_breakers_do_not_share_state(self):
        # The documented tradeoff of running without Redis.
        a = CircuitBreaker("llm", failure_threshold=1)
        b = CircuitBreaker("llm", failure_threshold=1)

        a.record_failure()

        assert a.allow() is False
        assert b.allow() is True

    def test_breakers_with_different_names_are_independent(self):
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        llm = CircuitBreaker("llm", failure_threshold=1, redis_client=redis_client)
        other = CircuitBreaker("other", failure_threshold=1, redis_client=redis_client)

        llm.record_failure()

        assert llm.allow() is False
        assert other.allow() is True
