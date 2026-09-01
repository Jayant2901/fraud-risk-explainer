from entity_memory import EntityRiskMemory, WATCH_THRESHOLD, ELEVATED_THRESHOLD, WINDOW_SIZE


class TestEmptyHistory:
    def test_unknown_entity_defaults_to_normal(self):
        memory = EntityRiskMemory()
        state = memory.get_escalation_state("never-seen")
        assert state["state"] == "NORMAL"
        assert state["recent_verdict_count"] == 0
        assert state["recent_risky_count"] == 0
        assert state["avg_recent_risk_score"] == 0.0
        assert state["recent_verdicts"] == []


class TestThresholdCrossing:
    def test_below_watch_threshold_stays_normal(self):
        memory = EntityRiskMemory()
        for _ in range(WATCH_THRESHOLD - 1):
            memory.record_verdict("e1", "REVIEW", 50.0)
        assert memory.get_escalation_state("e1")["state"] == "NORMAL"

    def test_reaching_watch_threshold_crosses_to_watch(self):
        memory = EntityRiskMemory()
        for _ in range(WATCH_THRESHOLD):
            memory.record_verdict("e1", "REVIEW", 50.0)
        assert memory.get_escalation_state("e1")["state"] == "WATCH"

    def test_below_elevated_threshold_stays_watch(self):
        memory = EntityRiskMemory()
        for _ in range(ELEVATED_THRESHOLD - 1):
            memory.record_verdict("e1", "BLOCK", 90.0)
        assert memory.get_escalation_state("e1")["state"] == "WATCH"

    def test_reaching_elevated_threshold_crosses_to_elevated(self):
        memory = EntityRiskMemory()
        for _ in range(ELEVATED_THRESHOLD):
            memory.record_verdict("e1", "BLOCK", 90.0)
        assert memory.get_escalation_state("e1")["state"] == "ELEVATED"

    def test_allow_verdicts_never_count_as_risky(self):
        memory = EntityRiskMemory()
        for _ in range(WINDOW_SIZE):
            memory.record_verdict("e1", "ALLOW", 5.0)
        state = memory.get_escalation_state("e1")
        assert state["state"] == "NORMAL"
        assert state["recent_risky_count"] == 0


class TestWindowEviction:
    def test_window_evicts_oldest_verdict_beyond_window_size(self):
        memory = EntityRiskMemory(window_size=3)
        memory.record_verdict("e1", "BLOCK", 90.0)   # will be evicted
        memory.record_verdict("e1", "ALLOW", 5.0)
        memory.record_verdict("e1", "ALLOW", 5.0)
        memory.record_verdict("e1", "ALLOW", 5.0)     # 4th push evicts the 1st

        state = memory.get_escalation_state("e1")
        assert state["recent_verdict_count"] == 3
        assert state["recent_risky_count"] == 0  # the one BLOCK aged out
        assert state["recent_verdicts"] == ["ALLOW", "ALLOW", "ALLOW"]

    def test_elevated_state_can_recover_as_risky_verdicts_age_out(self):
        memory = EntityRiskMemory(window_size=ELEVATED_THRESHOLD)
        for _ in range(ELEVATED_THRESHOLD):
            memory.record_verdict("e1", "BLOCK", 90.0)
        assert memory.get_escalation_state("e1")["state"] == "ELEVATED"

        # Push enough clean verdicts to fully evict the risky ones.
        for _ in range(ELEVATED_THRESHOLD):
            memory.record_verdict("e1", "ALLOW", 5.0)
        assert memory.get_escalation_state("e1")["state"] == "NORMAL"


class TestReset:
    def test_reset_single_entity_leaves_others_untouched(self):
        memory = EntityRiskMemory()
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.record_verdict("e2", "BLOCK", 90.0)

        memory.reset("e1")

        assert memory.get_escalation_state("e1")["recent_verdict_count"] == 0
        assert memory.get_escalation_state("e2")["recent_verdict_count"] == 1

    def test_reset_none_clears_every_entity(self):
        memory = EntityRiskMemory()
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.record_verdict("e2", "BLOCK", 90.0)

        memory.reset(None)

        assert memory.get_escalation_state("e1")["recent_verdict_count"] == 0
        assert memory.get_escalation_state("e2")["recent_verdict_count"] == 0

    def test_reset_unknown_entity_is_a_no_op(self):
        memory = EntityRiskMemory()
        memory.record_verdict("e1", "BLOCK", 90.0)
        memory.reset("never-seen")  # must not raise
        assert memory.get_escalation_state("e1")["recent_verdict_count"] == 1


class TestAvgRecentRiskScore:
    def test_average_is_rounded_to_one_decimal(self):
        memory = EntityRiskMemory()
        memory.record_verdict("e1", "ALLOW", 10.0)
        memory.record_verdict("e1", "ALLOW", 11.0)
        memory.record_verdict("e1", "ALLOW", 11.0)
        # mean = 10.666... -> rounds to 10.7
        assert memory.get_escalation_state("e1")["avg_recent_risk_score"] == 10.7
