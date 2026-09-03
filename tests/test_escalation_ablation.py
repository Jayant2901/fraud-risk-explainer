import pandas as pd

from escalation_ablation import (
    compute_strategy_metrics,
    compute_escalation_flip_precision,
    build_report,
    build_summary,
    compute_cost,
    replay_with_pressure_escalation,
    sweep_pressure_thresholds,
    build_sweep_report,
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


class TestComputeCost:
    def test_hand_computable_cost(self):
        is_fraud = pd.Series([1, 1, 0, 0])
        action = pd.Series(["ALLOW", "BLOCK", "REVIEW", "ALLOW"])
        # false negative: row0 (fraud, not flagged) -> 1 * avg_fraud_loss
        # false positive: row2 (legit, flagged) -> 1 * avg_fp_cost
        cost = compute_cost(is_fraud, action, avg_fraud_loss=5000.0, avg_fp_cost=150.0)
        assert cost == 5000.0 + 150.0

    def test_no_errors_costs_nothing(self):
        is_fraud = pd.Series([1, 0])
        action = pd.Series(["BLOCK", "ALLOW"])
        assert compute_cost(is_fraud, action) == 0.0


class TestSweepPressureThresholds:
    """entity e1: two BLOCKs at risk_score=85 (each contributes
    VERDICT_WEIGHT["BLOCK"] * 0.85 = 1.7 pressure -> 3.4 total after
    both), then a third, legit transaction at risk_score=30 (below the
    review threshold on its own). 3.4 clears the 2.0 and 2.8 ELEVATED
    candidates but not 3.6, so escalating that third transaction to
    REVIEW is a pure false positive under the two lower candidates —
    the 3.6 candidate should come out cheapest."""

    def _sweep(self):
        test_df = pd.DataFrame([
            {"entity_id": "e1", "isFraud": 0},
            {"entity_id": "e1", "isFraud": 0},
            {"entity_id": "e1", "isFraud": 0},
        ])
        risk_scores = pd.Series([85.0, 85.0, 30.0])
        thresholds = {"review": 40.0, "block": 80.0}
        return sweep_pressure_thresholds(test_df, risk_scores, thresholds)

    def test_picks_the_lowest_cost_elevated_cutoff(self):
        results = self._sweep()
        chosen = min(results, key=lambda r: r["cost"])
        assert chosen["elevated_threshold"] == 3.6

    def test_every_watch_candidate_ties_for_a_given_elevated_candidate(self):
        # decide_action() never branches on WATCH, only ELEVATED — so
        # varying the watch cutoff alone must never change the cost.
        results = self._sweep()
        by_elevated: dict[float, set[float]] = {}
        for r in results:
            by_elevated.setdefault(r["elevated_threshold"], set()).add(r["cost"])
        assert all(len(costs) == 1 for costs in by_elevated.values())

    def test_replay_with_pressure_escalation_matches_the_sweep_row_directly(self):
        test_df = pd.DataFrame([
            {"entity_id": "e1", "isFraud": 0},
            {"entity_id": "e1", "isFraud": 0},
            {"entity_id": "e1", "isFraud": 0},
        ])
        risk_scores = pd.Series([85.0, 85.0, 30.0])
        thresholds = {"review": 40.0, "block": 80.0}

        # elevated=2.0: 3.4 pressure clears it -> third txn escalates to REVIEW
        low_cutoff_df = replay_with_pressure_escalation(test_df, risk_scores, thresholds, 0.8, 2.0)
        assert list(low_cutoff_df["adjusted_action"]) == ["BLOCK", "BLOCK", "REVIEW"]

        # elevated=3.6: 3.4 pressure doesn't clear it -> third txn stays ALLOW
        high_cutoff_df = replay_with_pressure_escalation(test_df, risk_scores, thresholds, 0.8, 3.6)
        assert list(high_cutoff_df["adjusted_action"]) == ["BLOCK", "BLOCK", "ALLOW"]


class TestBuildSweepReport:
    def test_marks_the_lowest_cost_row_as_chosen(self):
        results = [
            {"watch_threshold": 0.8, "elevated_threshold": 2.0, "recall": 0.9, "false_flag_rate": 0.2, "cost": 500.0},
            {"watch_threshold": 0.8, "elevated_threshold": 3.6, "recall": 0.85, "false_flag_rate": 0.1, "cost": 300.0},
        ]
        report = build_sweep_report(results)
        chosen_lines = [line for line in report.splitlines() if "chosen (lowest cost)" in line]
        assert len(chosen_lines) == 1
        assert "3.6" in chosen_lines[0]


class TestBuildSummary:
    """The structured summary must never drift from the text report —
    both come from the same three computations over the same replay."""

    REPLAY = pd.DataFrame([
        {"is_fraud": 1, "risk_score": 90.0, "baseline_action": "BLOCK",
         "adjusted_action": "BLOCK", "escalated_due_to_history": False},
        {"is_fraud": 1, "risk_score": 30.0, "baseline_action": "ALLOW",
         "adjusted_action": "REVIEW", "escalated_due_to_history": True},
        {"is_fraud": 0, "risk_score": 20.0, "baseline_action": "ALLOW",
         "adjusted_action": "REVIEW", "escalated_due_to_history": True},
    ])

    def test_reports_the_same_numbers_the_text_report_does(self):
        summary = build_summary(self.REPLAY, [])
        report = build_report(self.REPLAY)

        # Baseline catches 1 of 2 frauds; escalation catches both.
        assert summary["baseline"]["recall"] == 0.5
        assert summary["adjusted"]["recall"] == 1.0
        assert f"{summary['baseline']['recall']:.4f}" in report
        assert f"{summary['adjusted']['recall']:.4f}" in report

    def test_carries_flip_precision_and_transaction_count(self):
        summary = build_summary(self.REPLAY, [])

        assert summary["n_transactions"] == 3
        assert summary["flips"]["n_flips"] == 2
        assert summary["flips"]["n_flips_fraud"] == 1
        assert summary["flips"]["precision"] == 0.5

    def test_passes_the_sweep_grid_through_untouched(self):
        sweep = [{"watch_threshold": 0.8, "elevated_threshold": 3.6, "recall": 0.9,
                  "false_flag_rate": 0.1, "cost": 100.0}]

        assert build_summary(self.REPLAY, sweep)["sweep"] == sweep


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
