"""
Escalation-alert tests, run against both backends (in-process and Redis
via fakeredis) — the dual-implementation pattern test_circuit_breaker.py
established.

send() is a spy rather than a real webhook: what's under test is the
transition/cooldown decision logic, not urllib. test_notifications_webhook
below covers the actual HTTP delivery path separately, with the request
never leaving the process.
"""
from unittest.mock import patch

import fakeredis
import pytest

from notifications import EscalationNotifier, _webhook_send, create_notifier


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(params=["in-process", "redis"])
def notifier_factory(request):
    redis_client = (
        fakeredis.FakeRedis(decode_responses=True) if request.param == "redis" else None
    )

    def build(cooldown: float = 300.0, clock=None):
        sent = []
        notifier = EscalationNotifier(
            sent.append, cooldown_seconds=cooldown, redis_client=redis_client, clock=clock or FakeClock()
        )
        return notifier, sent

    return build


class TestTransitionDirection:
    def test_escalating_transition_sends(self, notifier_factory):
        notifier, sent = notifier_factory()
        notifier.notify_transition("e1", "NORMAL", "WATCH", 55.0, "v1")

        assert len(sent) == 1
        assert sent[0] == {
            "entity_id": "e1", "from_state": "NORMAL", "to_state": "WATCH",
            "risk_score": 55.0, "verdict_id": "v1", "at": 1000.0,
        }

    def test_watch_to_elevated_sends(self, notifier_factory):
        notifier, sent = notifier_factory()
        notifier.notify_transition("e1", "WATCH", "ELEVATED", 90.0, "v1")

        assert len(sent) == 1

    def test_de_escalation_does_not_send(self, notifier_factory):
        notifier, sent = notifier_factory()
        notifier.notify_transition("e1", "ELEVATED", "WATCH", 10.0, "v1")

        assert sent == []

    def test_no_transition_does_not_send(self, notifier_factory):
        notifier, sent = notifier_factory()
        notifier.notify_transition("e1", "WATCH", "WATCH", 40.0, "v1")

        assert sent == []

    def test_missing_entity_id_does_not_send(self, notifier_factory):
        notifier, sent = notifier_factory()
        notifier.notify_transition(None, "NORMAL", "WATCH", 40.0, "v1")

        assert sent == []

    def test_missing_from_state_is_treated_as_normal(self, notifier_factory):
        # A first-ever verdict for an entity has no prior escalation
        # state to compare against; None must not crash the comparison.
        notifier, sent = notifier_factory()
        notifier.notify_transition("e1", None, "WATCH", 40.0, "v1")

        assert len(sent) == 1


class TestCooldown:
    def test_second_escalation_within_cooldown_is_suppressed(self, notifier_factory):
        clock = FakeClock()
        notifier, sent = notifier_factory(cooldown=300.0, clock=clock)
        notifier.notify_transition("e1", "NORMAL", "WATCH", 40.0, "v1")
        clock.advance(100)
        notifier.notify_transition("e1", "WATCH", "ELEVATED", 90.0, "v2")

        assert len(sent) == 1

    def test_escalation_after_cooldown_elapses_sends_again(self, notifier_factory):
        clock = FakeClock()
        notifier, sent = notifier_factory(cooldown=300.0, clock=clock)
        notifier.notify_transition("e1", "NORMAL", "WATCH", 40.0, "v1")
        clock.advance(300)
        notifier.notify_transition("e1", "WATCH", "ELEVATED", 90.0, "v2")

        assert len(sent) == 2

    def test_cooldown_is_per_entity(self, notifier_factory):
        notifier, sent = notifier_factory(cooldown=300.0)
        notifier.notify_transition("e1", "NORMAL", "WATCH", 40.0, "v1")
        notifier.notify_transition("e2", "NORMAL", "WATCH", 40.0, "v2")

        assert len(sent) == 2


class TestDeliveryFailureIsolation:
    def test_a_raising_send_does_not_propagate(self, notifier_factory):
        notifier, _ = notifier_factory()

        def boom(payload):
            raise RuntimeError("webhook endpoint is down")

        notifier._send = boom  # swap in a failing sender after construction

        notifier.notify_transition("e1", "NORMAL", "WATCH", 40.0, "v1")  # must not raise


class TestFactory:
    def test_no_webhook_url_yields_a_notifier_that_never_sends_anywhere(self, monkeypatch):
        monkeypatch.delenv("ESCALATION_WEBHOOK_URL", raising=False)
        notifier = create_notifier()

        with patch("notifications._post_webhook") as post:
            notifier.notify_transition("e1", "NORMAL", "WATCH", 40.0, "v1")
            post.assert_not_called()


class TestWebhookDelivery:
    """_webhook_send fires the HTTP call on a background thread; assert
    the thread is launched with the right target rather than opening a
    real socket."""

    def test_send_launches_a_daemon_thread_against_post_webhook(self):
        send = _webhook_send("http://example.invalid/alert", timeout=5.0)

        with patch("notifications.threading.Thread") as thread_cls:
            send({"entity_id": "e1"})

        thread_cls.assert_called_once()
        _, kwargs = thread_cls.call_args
        assert kwargs["daemon"] is True
        assert kwargs["args"] == ("http://example.invalid/alert", {"entity_id": "e1"}, 5.0)
