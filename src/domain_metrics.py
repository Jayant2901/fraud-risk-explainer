"""
Domain-specific Prometheus metrics.

prometheus-fastapi-instrumentator (wired in api/main.py) already exposes
GET /metrics with request count/latency per route — HTTP-shaped
observability. Nobody operating a fraud system cares only about HTTP
though; they care about fraud-shaped questions: how many transactions
were blocked in the last hour, is the review queue backing up, is the
LLM breaker open, did an entity just escalate. This module answers those
on the same /metrics endpoint, in the same text format, so one Grafana
dashboard (ops/grafana-dashboard.json) covers both.

Two different metric shapes for two different questions:
- Counters (decisions_total, escalation_transitions_total) for RATES —
  incremented once per real event at the event site in
  scoring_service.py, independent of any alerting cooldown, so they
  never under-count relative to what actually happened.
- A custom Collector for CURRENT STATE (queue depth, live reviewer
  precision, LLM breaker state) — computed fresh on every scrape from
  the same sources the API itself reads (review_queue.metrics(), the
  circuit breaker), rather than incrementally tracked and liable to
  drift from the truth those endpoints report.

Single-process counters: prometheus-fastapi-instrumentator isn't
configured for multiprocess mode here, so under multiple worker
processes each worker's counters cover only the requests it handled —
the same tradeoff the HTTP-level metrics it ships with already carry.
"""
import logging

from prometheus_client import REGISTRY, Counter
from prometheus_client.core import GaugeMetricFamily

logger = logging.getLogger(__name__)

decisions_total = Counter(
    "riskmgr_decisions_total", "Scoring decisions, by action.", ["action"]
)

escalation_transitions_total = Counter(
    "riskmgr_escalation_transitions_total",
    "Entity escalation-state transitions (both directions).",
    ["from_state", "to_state"],
)


class DomainStateCollector:
    """Registered once at process startup; queried on every GET /metrics
    scrape rather than maintaining its own running totals — so
    riskmgr_review_queue_pending can never disagree with what GET
    /api/review-queue?status=pending itself would return right now."""

    def __init__(self, review_queue, breaker_provider):
        self._review_queue = review_queue
        self._breaker_provider = breaker_provider

    def describe(self):
        """CollectorRegistry.register() calls collect() once immediately
        to check for name clashes — which would run breaker_provider()
        (get_agent(), constructing the real LLM agent) as a side effect
        of module import, before api/main.py has even finished defining
        get_agent. A describe() that declares names without touching
        review_queue/breaker_provider is the documented prometheus_client
        way to avoid that: register() prefers it over collect()."""
        yield GaugeMetricFamily(
            "riskmgr_review_queue_pending", "Items currently awaiting reviewer disposition."
        )
        yield GaugeMetricFamily(
            "riskmgr_review_precision",
            "Live reviewer precision (confirmed fraud / disposed), by segment.",
            labels=["segment"],
        )
        yield GaugeMetricFamily(
            "riskmgr_llm_breaker_open", "1 if the LLM circuit breaker is open, else 0."
        )

    def collect(self):
        pending = GaugeMetricFamily(
            "riskmgr_review_queue_pending", "Items currently awaiting reviewer disposition."
        )
        pending.add_metric([], len(self._review_queue.list_pending()))
        yield pending

        metrics = self._review_queue.metrics()
        precision = GaugeMetricFamily(
            "riskmgr_review_precision",
            "Live reviewer precision (confirmed fraud / disposed), by segment.",
            labels=["segment"],
        )
        for segment, value in (
            ("overall", metrics["overall_precision"]),
            ("escalated", metrics["escalated_precision"]),
            ("non_escalated", metrics["non_escalated_precision"]),
        ):
            if value is not None:
                precision.add_metric([segment], value)
        yield precision

        breaker = GaugeMetricFamily(
            "riskmgr_llm_breaker_open", "1 if the LLM circuit breaker is open, else 0."
        )
        try:
            state = self._breaker_provider()
            breaker.add_metric([], 1.0 if state.get("state") == "open" else 0.0)
        except Exception:
            # Same tolerance GET /api/health already has for a broken
            # agent construction — a metrics scrape must never 500.
            logger.warning("Could not read LLM breaker state for /metrics", exc_info=True)
        yield breaker


def register_domain_state_collector(review_queue, breaker_provider) -> DomainStateCollector:
    collector = DomainStateCollector(review_queue, breaker_provider)
    REGISTRY.register(collector)
    return collector
