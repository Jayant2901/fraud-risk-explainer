"""
DomainStateCollector tests.

describe() and collect() are tested separately on purpose: describe()
exists ONLY to keep CollectorRegistry.register()'s one-time name-clash
check from calling review_queue/breaker_provider at all (see the comment
on describe() itself) — a test that let collect()'s behavior stand in
for describe()'s would miss a regression back to that exact bug.
"""
from unittest.mock import Mock

from review_queue import ReviewQueue, CONFIRMED_FRAUD, FALSE_POSITIVE
from domain_metrics import DomainStateCollector


def make_item(verdict_id, disposition=None, escalated=False, risk_score=55.0):
    return {
        "verdict_id": verdict_id,
        "entity_id": "e1",
        "risk_score": risk_score,
        "escalated_due_to_history": escalated,
        "disposition": disposition,
    }


class TestDescribe:
    def test_declares_names_without_touching_its_collaborators(self):
        review_queue = Mock()
        breaker_provider = Mock()
        collector = DomainStateCollector(review_queue, breaker_provider)

        names = {family.name for family in collector.describe()}

        assert names == {
            "riskmgr_review_queue_pending",
            "riskmgr_review_precision",
            "riskmgr_llm_breaker_open",
        }
        review_queue.list_pending.assert_not_called()
        review_queue.metrics.assert_not_called()
        breaker_provider.assert_not_called()


class TestCollectPending:
    def test_reports_the_live_pending_count(self):
        queue = ReviewQueue()
        queue.add(make_item("v1"))
        queue.add(make_item("v2"))
        collector = DomainStateCollector(queue, lambda: {"state": "closed"})

        families = {f.name: f for f in collector.collect()}

        assert families["riskmgr_review_queue_pending"].samples[0].value == 2

    def test_disposed_items_do_not_count_as_pending(self):
        queue = ReviewQueue()
        queue.add(make_item("v1"))
        queue.dispose("v1", CONFIRMED_FRAUD)
        collector = DomainStateCollector(queue, lambda: {"state": "closed"})

        families = {f.name: f for f in collector.collect()}

        assert families["riskmgr_review_queue_pending"].samples[0].value == 0


class TestCollectPrecision:
    def test_reports_precision_by_segment(self):
        queue = ReviewQueue()
        queue.add(make_item("v1", escalated=True))
        queue.add(make_item("v2", escalated=False))
        queue.dispose("v1", CONFIRMED_FRAUD)
        queue.dispose("v2", FALSE_POSITIVE)
        collector = DomainStateCollector(queue, lambda: {"state": "closed"})

        families = {f.name: f for f in collector.collect()}
        by_segment = {s.labels["segment"]: s.value for s in families["riskmgr_review_precision"].samples}

        assert by_segment == {"overall": 0.5, "escalated": 1.0, "non_escalated": 0.0}

    def test_a_segment_with_nothing_disposed_is_omitted_rather_than_faked_as_zero(self):
        queue = ReviewQueue()
        collector = DomainStateCollector(queue, lambda: {"state": "closed"})

        families = {f.name: f for f in collector.collect()}

        assert families["riskmgr_review_precision"].samples == []


class TestCollectBreaker:
    def test_reports_one_when_open(self):
        queue = ReviewQueue()
        collector = DomainStateCollector(queue, lambda: {"state": "open"})

        families = {f.name: f for f in collector.collect()}

        assert families["riskmgr_llm_breaker_open"].samples[0].value == 1.0

    def test_reports_zero_when_closed(self):
        queue = ReviewQueue()
        collector = DomainStateCollector(queue, lambda: {"state": "closed"})

        families = {f.name: f for f in collector.collect()}

        assert families["riskmgr_llm_breaker_open"].samples[0].value == 0.0

    def test_a_raising_provider_yields_no_sample_rather_than_crashing_the_scrape(self):
        queue = ReviewQueue()

        def boom():
            raise RuntimeError("no credentials")

        collector = DomainStateCollector(queue, boom)

        families = {f.name: f for f in collector.collect()}  # must not raise

        assert families["riskmgr_llm_breaker_open"].samples == []
