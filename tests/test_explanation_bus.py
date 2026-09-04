"""
ExplanationBus tests, run against both backends (in-process and Redis via
fakeredis) — the same dual-implementation pattern test_entity_memory.py
uses, because the whole point of the Redis path is that a delta published
by one worker reaches a subscriber in another.
"""
import threading

import fakeredis
import pytest

from explanation_bus import ExplanationBus, channel_for


@pytest.fixture(params=["in-process", "redis"])
def bus_factory(request):
    if request.param == "redis":
        client = fakeredis.FakeRedis(decode_responses=True)
        return lambda: ExplanationBus(client), request.param
    return lambda: ExplanationBus(None), request.param


def collect(bus, verdict_id, out, ready):
    subscription = bus.subscribe(verdict_id)
    ready.set()
    for message in subscription:
        if message is None:
            continue  # keepalive tick
        out.append(message)
        if message["type"] == "complete":
            return


class TestPublishSubscribe:
    def test_a_subscriber_receives_deltas_then_the_terminal_message(self, bus_factory):
        make_bus, backend = bus_factory
        bus = make_bus()
        # The Redis-backed bus is a real pub/sub: a publisher and a
        # subscriber in the same process still need separate threads,
        # because subscribe() blocks.
        received, ready = [], threading.Event()
        subscriber = threading.Thread(
            target=collect, args=(bus, "v1", received, ready), daemon=True
        )
        subscriber.start()
        ready.wait(timeout=5)

        bus.publish("v1", {"type": "delta", "text": "Hello "})
        bus.publish("v1", {"type": "delta", "text": "world"})
        bus.publish("v1", {"type": "complete", "verdict": {"action": "REVIEW"}})

        subscriber.join(timeout=10)
        assert [m["type"] for m in received] == ["delta", "delta", "complete"]
        assert "".join(m["text"] for m in received if m["type"] == "delta") == "Hello world"

    def test_a_subscriber_only_receives_its_own_verdicts_messages(self, bus_factory):
        make_bus, _ = bus_factory
        bus = make_bus()
        received, ready = [], threading.Event()
        subscriber = threading.Thread(
            target=collect, args=(bus, "wanted", received, ready), daemon=True
        )
        subscriber.start()
        ready.wait(timeout=5)

        bus.publish("other", {"type": "delta", "text": "not mine"})
        bus.publish("wanted", {"type": "complete", "verdict": {"action": "ALLOW"}})

        subscriber.join(timeout=10)
        assert len(received) == 1
        assert received[0]["type"] == "complete"

    def test_publishing_with_no_subscriber_does_not_raise(self, bus_factory):
        make_bus, _ = bus_factory
        make_bus().publish("nobody-listening", {"type": "delta", "text": "x"})

    def test_two_subscribers_both_receive_the_same_stream(self, bus_factory):
        make_bus, _ = bus_factory
        bus = make_bus()
        first, second = [], []
        ready_a, ready_b = threading.Event(), threading.Event()
        threads = [
            threading.Thread(target=collect, args=(bus, "v", first, ready_a), daemon=True),
            threading.Thread(target=collect, args=(bus, "v", second, ready_b), daemon=True),
        ]
        for t in threads:
            t.start()
        ready_a.wait(timeout=5)
        ready_b.wait(timeout=5)

        bus.publish("v", {"type": "complete", "verdict": {"action": "BLOCK"}})

        for t in threads:
            t.join(timeout=10)
        assert len(first) == 1 and len(second) == 1


class TestChannelNaming:
    def test_channels_are_namespaced_per_verdict(self):
        assert channel_for("abc") != channel_for("def")
        assert channel_for("abc").endswith("abc")
        # Namespaced like every other key this project puts in Redis.
        assert channel_for("abc").startswith("riskmgr:")
