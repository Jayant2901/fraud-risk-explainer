"""
Every test class below runs TWICE — once against EntityRiskMemory
(in-process deque) and once against RedisEntityRiskMemory (backed by
fakeredis) — via the memory_factory fixture's parametrization. This is
what actually proves the two implementations are behaviorally
equivalent, not just "both pass their own separate tests."
"""
import math

import fakeredis
import pytest

from entity_memory import (
    EntityRiskMemory,
    RedisEntityRiskMemory,
    WINDOW_SIZE,
    VERDICT_WEIGHT,
    DEFAULT_WATCH_PRESSURE_THRESHOLD,
    DEFAULT_ELEVATED_PRESSURE_THRESHOLD,
    _compute_escalation_state,
    _risk_pressure,
)


@pytest.fixture(params=["in_process", "redis"])
def memory_factory(request):
    """Returns a callable(window_size=WINDOW_SIZE) -> memory instance,
    for the backend named by this fixture's current param."""
    if request.param == "in_process":
        return lambda window_size=WINDOW_SIZE: EntityRiskMemory(window_size=window_size)

    client = fakeredis.FakeRedis(decode_responses=True)
    return lambda window_size=WINDOW_SIZE: RedisEntityRiskMemory(client, window_size=window_size)


class TestEmptyHistory:
    def test_unknown_entity_defaults_to_normal(self, memory_factory):
        memory = memory_factory()
        state = memory.get_escalation_state("never-seen")
        assert state["state"] == "NORMAL"
        assert state["recent_verdict_count"] == 0
        assert state["recent_risky_count"] == 0
        assert state["avg_recent_risk_score"] == 0.0
        assert state["recent_verdicts"] == []


class TestRiskPressureFormula:
    """Hand-computable examples for _risk_pressure() — the core of the
    severity-weighted escalation formula (see VERDICT_WEIGHT in
    entity_memory.py)."""

    def test_single_block_at_full_score(self):
        history = [{"verdict": "BLOCK", "risk_score": 100.0}]
        # 2.0 * 1.0 = 2.0
        assert _risk_pressure(history) == pytest.approx(VERDICT_WEIGHT["BLOCK"] * 1.0)

    def test_single_review_at_half_score(self):
        history = [{"verdict": "REVIEW", "risk_score": 50.0}]
        # 1.0 * 0.5 = 0.5
        assert _risk_pressure(history) == pytest.approx(0.5)

    def test_block_and_review_sum(self):
        history = [
            {"verdict": "BLOCK", "risk_score": 90.0},   # 2.0 * 0.9 = 1.8
            {"verdict": "REVIEW", "risk_score": 50.0},  # 1.0 * 0.5 = 0.5
        ]
        assert _risk_pressure(history) == pytest.approx(2.3)

    def test_allow_contributes_nothing(self):
        history = [{"verdict": "ALLOW", "risk_score": 99.0}]
        assert _risk_pressure(history) == 0.0

    def test_block_weighs_more_than_review_at_the_same_score(self):
        block_pressure = _risk_pressure([{"verdict": "BLOCK", "risk_score": 60.0}])
        review_pressure = _risk_pressure([{"verdict": "REVIEW", "risk_score": 60.0}])
        assert block_pressure > review_pressure


class TestThresholdCrossingWithExplicitCutoffs:
    """Exercises the state boundary against an explicit, hand-picked
    cutoff pair — not the module's live-tuned defaults — so these stay
    exact no matter how DEFAULT_WATCH_PRESSURE_THRESHOLD/DEFAULT_
    ELEVATED_PRESSURE_THRESHOLD get re-tuned later. See TestLiveDefaults
    below for coverage against the real, current defaults."""

    WATCH = 1.0
    ELEVATED = 3.0

    def test_below_watch_threshold_stays_normal(self):
        # one REVIEW at risk_score=90 -> 1.0 * 0.9 = 0.9, just under 1.0
        history = [{"verdict": "REVIEW", "risk_score": 90.0}]
        state = _compute_escalation_state("e1", history, self.WATCH, self.ELEVATED)
        assert state["state"] == "NORMAL"
        assert state["risk_pressure"] == pytest.approx(0.9)

    def test_reaching_watch_threshold_crosses_to_watch(self):
        # one REVIEW at risk_score=100 -> 1.0 * 1.0 = 1.0, exactly at cutoff
        history = [{"verdict": "REVIEW", "risk_score": 100.0}]
        state = _compute_escalation_state("e1", history, self.WATCH, self.ELEVATED)
        assert state["state"] == "WATCH"

    def test_below_elevated_threshold_stays_watch(self):
        # one BLOCK at risk_score=100 -> 2.0 * 1.0 = 2.0, still under 3.0
        history = [{"verdict": "BLOCK", "risk_score": 100.0}]
        state = _compute_escalation_state("e1", history, self.WATCH, self.ELEVATED)
        assert state["state"] == "WATCH"

    def test_reaching_elevated_threshold_crosses_to_elevated(self):
        # two BLOCKs at risk_score=100 -> 4.0, over 3.0
        history = [
            {"verdict": "BLOCK", "risk_score": 100.0},
            {"verdict": "BLOCK", "risk_score": 100.0},
        ]
        state = _compute_escalation_state("e1", history, self.WATCH, self.ELEVATED)
        assert state["state"] == "ELEVATED"

    def test_allow_verdicts_never_count_as_risky_or_add_pressure(self):
        history = [{"verdict": "ALLOW", "risk_score": 5.0} for _ in range(WINDOW_SIZE)]
        state = _compute_escalation_state("e1", history, self.WATCH, self.ELEVATED)
        assert state["state"] == "NORMAL"
        assert state["recent_risky_count"] == 0
        assert state["risk_pressure"] == 0.0


class TestLiveDefaults:
    """Confirms EntityRiskMemory/RedisEntityRiskMemory actually escalate
    using the module's real, grid-chosen default cutoffs (see
    escalation_ablation.sweep_pressure_thresholds()) — not exact
    boundary crossings (those live in
    TestThresholdCrossingWithExplicitCutoffs above), just that enough
    severity clearly produces ELEVATED and an all-ALLOW history stays
    NORMAL, so this stays valid however the exact cutoffs get re-tuned."""

    def test_enough_full_strength_blocks_reaches_elevated(self, memory_factory):
        count = math.ceil(DEFAULT_ELEVATED_PRESSURE_THRESHOLD / VERDICT_WEIGHT["BLOCK"])
        memory = memory_factory(window_size=count)
        for _ in range(count):
            memory.record_verdict("e1", "BLOCK", 100.0)
        assert memory.get_escalation_state("e1")["state"] == "ELEVATED"

    def test_all_allow_history_stays_normal(self, memory_factory):
        memory = memory_factory()
        for _ in range(WINDOW_SIZE):
            memory.record_verdict("e1", "ALLOW", 5.0)
        state = memory.get_escalation_state("e1")
        assert state["state"] == "NORMAL"
        assert state["recent_risky_count"] == 0


class TestWindowEviction:
    def test_window_evicts_oldest_verdict_beyond_window_size(self, memory_factory):
        memory = memory_factory(window_size=3)
        memory.record_verdict("e1", "BLOCK", 90.0)   # will be evicted
        memory.record_verdict("e1", "ALLOW", 5.0)
        memory.record_verdict("e1", "ALLOW", 5.0)
        memory.record_verdict("e1", "ALLOW", 5.0)     # 4th push evicts the 1st

        state = memory.get_escalation_state("e1")
        assert state["recent_verdict_count"] == 3
        assert state["recent_risky_count"] == 0  # the one BLOCK aged out
        assert state["recent_verdicts"] == ["ALLOW", "ALLOW", "ALLOW"]

    def test_elevated_state_can_recover_as_risky_verdicts_age_out(self, memory_factory):
        count = math.ceil(DEFAULT_ELEVATED_PRESSURE_THRESHOLD / VERDICT_WEIGHT["BLOCK"])
        memory = memory_factory(window_size=count)
        for _ in range(count):
            memory.record_verdict("e1", "BLOCK", 100.0)
        assert memory.get_escalation_state("e1")["state"] == "ELEVATED"

        # Push enough clean verdicts to fully evict the risky ones.
        for _ in range(count):
            memory.record_verdict("e1", "ALLOW", 5.0)
        assert memory.get_escalation_state("e1")["state"] == "NORMAL"


class TestReset:
    def test_reset_single_entity_leaves_others_untouched(self, memory_factory):
        memory = memory_factory()
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.record_verdict("e2", "BLOCK", 90.0)

        memory.reset("e1")

        assert memory.get_escalation_state("e1")["recent_verdict_count"] == 0
        assert memory.get_escalation_state("e2")["recent_verdict_count"] == 1

    def test_reset_none_clears_every_entity(self, memory_factory):
        memory = memory_factory()
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.record_verdict("e2", "BLOCK", 90.0)

        memory.reset(None)

        assert memory.get_escalation_state("e1")["recent_verdict_count"] == 0
        assert memory.get_escalation_state("e2")["recent_verdict_count"] == 0

    def test_reset_unknown_entity_is_a_no_op(self, memory_factory):
        memory = memory_factory()
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.reset("never-seen")  # must not raise
        assert memory.get_escalation_state("e1")["recent_verdict_count"] == 1


class TestAvgRecentRiskScore:
    def test_average_is_rounded_to_one_decimal(self, memory_factory):
        memory = memory_factory()
        memory.record_verdict("e1", "ALLOW", 10.0)
        memory.record_verdict("e1", "ALLOW", 11.0)
        memory.record_verdict("e1", "ALLOW", 11.0)
        # mean = 10.666... -> rounds to 10.7
        assert memory.get_escalation_state("e1")["avg_recent_risk_score"] == 10.7


class TestRedisSpecific:
    """Behavior that only makes sense to assert for the Redis backend
    directly (not parametrized — EntityRiskMemory has no TTL/key-prefix
    concept to compare against)."""

    def test_uses_a_namespaced_key_per_entity(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        memory = RedisEntityRiskMemory(client)
        memory.record_verdict("some-entity", "BLOCK", 90.0)
        assert client.exists("riskmgr:entity_history:some-entity")

    def test_sets_a_ttl_on_write(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        memory = RedisEntityRiskMemory(client)
        memory.record_verdict("some-entity", "BLOCK", 90.0)
        ttl = client.ttl("riskmgr:entity_history:some-entity")
        assert ttl > 0

    def test_reset_none_only_touches_this_projects_keys(self):
        # reset(None) must not be a FLUSHALL — it should only delete keys
        # under this module's own prefix, leaving unrelated keys alone.
        client = fakeredis.FakeRedis(decode_responses=True)
        client.set("some_other_apps_key", "do-not-touch")

        memory = RedisEntityRiskMemory(client)
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.reset(None)

        assert client.get("some_other_apps_key") == "do-not-touch"
