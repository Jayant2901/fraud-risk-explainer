import pandas as pd

from escalation_ablation import (
    compute_strategy_metrics,
    compute_escalation_flip_precision,
    build_report,
)


class TestComputeStrategyMetrics:
    def test_recall_and_false_flag_rate_on_known_sequence(self):
        # 4 fraud (2 flagged, 2 missed), 4 legit (1 wrongly flagged, 3 correctly allowed)
        is_fraud = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
        action = pd.Series(["BLOCK", "REVIEW", "ALLOW", "ALLOW", "REVIEW", "ALLOW", "ALLOW", "ALLOW"])

        result = compute_strategy_metrics(is_fraud, action)

        assert result["n_fraud"] == 4
        assert result["n_legit"] == 4
        assert result["flagged_fraud"] == 2
        assert result["flagged_legit"] == 1
        assert result["recall"] == 0.5
        assert result["false_flag_rate"] == 0.25

    def test_all_allowed_gives_zero_recall_and_zero_false_flag_rate(self):
        is_fraud = pd.Series([1, 1, 0, 0])
        action = pd.Series(["ALLOW", "ALLOW", "ALLOW", "ALLOW"])

        result = compute_strategy_metrics(is_fraud, action)

        assert result["recall"] == 0.0
        assert result["false_flag_rate"] == 0.0

    def test_empty_fraud_or_legit_class_does_not_divide_by_zero(self):
        is_fraud = pd.Series([1, 1])
        action = pd.Series(["BLOCK", "REVIEW"])

        result = compute_strategy_metrics(is_fraud, action)

        assert result["n_legit"] == 0
        assert result["false_flag_rate"] == 0.0
        assert result["recall"] == 1.0


class TestComputeEscalationFlipPrecision:
    def test_precision_among_flips_only(self):
        # 3 escalation-triggered flips: 2 fraud, 1 legit. 1 non-flip row
        # (fraud) should be excluded from the precision calc entirely.
        replay_df = pd.DataFrame([
            {"is_fraud": 1, "escalated_due_to_history": True},
            {"is_fraud": 1, "escalated_due_to_history": True},
            {"is_fraud": 0, "escalated_due_to_history": True},
            {"is_fraud": 1, "escalated_due_to_history": False},
        ])

        result = compute_escalation_flip_precision(replay_df)

        assert result["n_flips"] == 3
        assert result["n_flips_fraud"] == 2
        assert result["precision"] == 2 / 3

    def test_no_flips_gives_zero_precision_not_a_crash(self):
        replay_df = pd.DataFrame([
            {"is_fraud": 1, "escalated_due_to_history": False},
            {"is_fraud": 0, "escalated_due_to_history": False},
        ])

        result = compute_escalation_flip_precision(replay_df)

        assert result["n_flips"] == 0
        assert result["n_flips_fraud"] == 0
        assert result["precision"] == 0.0

    def test_all_flips_fraud_gives_perfect_precision(self):
        replay_df = pd.DataFrame([
            {"is_fraud": 1, "escalated_due_to_history": True},
            {"is_fraud": 1, "escalated_due_to_history": True},
        ])

        result = compute_escalation_flip_precision(replay_df)

        assert result["precision"] == 1.0


class TestBuildReport:
    def test_report_contains_real_numbers_not_placeholders(self):
        replay_df = pd.DataFrame([
            {"is_fraud": 1, "risk_score": 90.0, "baseline_action": "BLOCK",
             "adjusted_action": "BLOCK", "escalated_due_to_history": False},
            {"is_fraud": 0, "risk_score": 20.0, "baseline_action": "ALLOW",
             "adjusted_action": "REVIEW", "escalated_due_to_history": True},
        ])

        report = build_report(replay_df)

        assert "Escalation ablation study" in report
        assert "Baseline" in report
        assert "Escalation-adjusted" in report
        assert "Precision of escalation-triggered flips" in report
        # the one escalation flip in this fixture is the legit row -> 0 precision
        assert "Precision of escalation-triggered flips:  0.0000" in report
