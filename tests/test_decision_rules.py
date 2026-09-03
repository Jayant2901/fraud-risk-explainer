"""
Direct unit tests for decide_action() and load_decision_thresholds() —
the one function that actually gates a transaction, and the loader that
feeds it the real, cost-derived boundaries (see train_model.py). See
also test_api.py's TestDecideAction, which covers the *default*
40.0/80.0 behavior through the same function; this file focuses on
proving explicit thresholds are actually used, not just accepted.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_rules import (
    decide_action,
    load_decision_thresholds,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
)


class TestDecideActionWithExplicitThresholds:
    def test_uses_the_passed_in_thresholds_not_the_defaults(self):
        # With thresholds of 20/60, a score of 25 must land in the
        # REVIEW band even though it's well below the default 40 — if
        # decide_action() were silently still using 40/80 internally,
        # this would come back ALLOW instead.
        result = decide_action(25.0, {"state": "NORMAL"}, review_threshold=20.0, block_threshold=60.0)
        assert result["action"] == "REVIEW"

    def test_score_between_default_and_custom_review_threshold_moves_bands(self):
        # 25 is below the default review threshold (40) but above a
        # custom one (20) -- proves the boundary itself moved, not just
        # that some threshold is being applied.
        assert decide_action(25.0, {"state": "NORMAL"})["action"] == "ALLOW"
        assert decide_action(25.0, {"state": "NORMAL"}, review_threshold=20.0, block_threshold=60.0)["action"] == "REVIEW"

    def test_custom_block_threshold_moves_the_high_band_too(self):
        # 65 is below the default block threshold (80) but above a
        # custom one (60).
        assert decide_action(65.0, {"state": "NORMAL"})["action"] == "REVIEW"
        assert decide_action(65.0, {"state": "NORMAL"}, review_threshold=20.0, block_threshold=60.0)["action"] == "BLOCK"

    def test_elevated_escalation_still_applies_with_custom_thresholds(self):
        result = decide_action(25.0, {"state": "ELEVATED"}, review_threshold=20.0, block_threshold=60.0)
        assert result["action"] == "BLOCK"
        assert result["escalated_due_to_history"] is True

    def test_high_score_block_at_custom_threshold_is_never_reported_as_escalated(self):
        result = decide_action(70.0, {"state": "ELEVATED"}, review_threshold=20.0, block_threshold=60.0)
        assert result["action"] == "BLOCK"
        assert result["escalated_due_to_history"] is False

    def test_defaults_match_the_documented_fallback_constants(self):
        assert decide_action(45.0, {"state": "NORMAL"})["action"] == "REVIEW"
        assert decide_action(45.0, {"state": "NORMAL"}, DEFAULT_REVIEW_THRESHOLD, DEFAULT_BLOCK_THRESHOLD)["action"] == "REVIEW"


class TestLoadDecisionThresholds:
    def test_falls_back_to_documented_defaults_when_file_is_missing(self, tmp_path):
        missing_path = str(tmp_path / "does-not-exist.joblib")
        result = load_decision_thresholds(missing_path)
        assert result == {"review": DEFAULT_REVIEW_THRESHOLD, "block": DEFAULT_BLOCK_THRESHOLD}

    def test_loads_real_values_from_a_real_file(self, tmp_path):
        import joblib

        path = str(tmp_path / "decision_thresholds.joblib")
        joblib.dump({"review": 33.5, "block": 91.2}, path)

        result = load_decision_thresholds(path)
        assert result == {"review": 33.5, "block": 91.2}
