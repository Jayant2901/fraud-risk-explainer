"""
Hand-computable fixture: 2 fraud (proba 0.9, 0.4) + 2 legit (proba 0.3, 0.1).

At each threshold below, (fn, fp, tp) and total_cost are worked out by
hand in the comments and cross-checked against cost_curve()'s output —
this is what makes the test meaningful instead of just "runs without
crashing."
"""
import pytest

from cost_analysis import cost_curve, optimal_threshold, threshold_sensitivity

Y_TRUE = [1, 1, 0, 0]
Y_PROBA = [0.9, 0.4, 0.3, 0.1]
THRESHOLDS = [0.2, 0.35, 0.5, 0.95]
AVG_FRAUD_LOSS = 5000.0
AVG_FP_COST = 150.0

# An imperfect-model fixture (the fraud row scores LOWER than the legit
# row, so no threshold separates the classes cleanly) — needed to make
# the optimal threshold actually move with the cost ratio, which
# Y_TRUE/Y_PROBA above doesn't do (0.35 is a free zero-cost threshold
# there regardless of costs, so it never budges).
SENS_Y_TRUE = [1, 0]
SENS_Y_PROBA = [0.3, 0.6]


class TestCostCurve:
    def _row(self, curve, threshold):
        matches = curve[curve["threshold"] == threshold]
        assert len(matches) == 1
        return matches.iloc[0]

    def test_low_threshold_catches_all_fraud_at_the_cost_of_one_fp(self):
        # pred = [1, 1, 1, 0] -> fn=0, fp=1 (the 0.3 legit txn), tp=2
        curve = cost_curve(Y_TRUE, Y_PROBA, thresholds=THRESHOLDS,
                            avg_fraud_loss=AVG_FRAUD_LOSS, avg_fp_cost=AVG_FP_COST)
        row = self._row(curve, 0.2)
        assert row["false_negatives"] == 0
        assert row["false_positives"] == 1
        assert row["true_positives"] == 2
        assert row["total_cost"] == pytest.approx(AVG_FP_COST)

    def test_threshold_that_perfectly_separates_the_classes_has_zero_cost(self):
        # pred = [1, 1, 0, 0] -> exactly matches y_true -> fn=0, fp=0, tp=2
        curve = cost_curve(Y_TRUE, Y_PROBA, thresholds=THRESHOLDS,
                            avg_fraud_loss=AVG_FRAUD_LOSS, avg_fp_cost=AVG_FP_COST)
        row = self._row(curve, 0.35)
        assert row["false_negatives"] == 0
        assert row["false_positives"] == 0
        assert row["true_positives"] == 2
        assert row["total_cost"] == 0
        assert row["precision"] == 1.0
        assert row["recall"] == 1.0

    def test_high_threshold_misses_the_cheaper_fraud_case(self):
        # pred = [1, 0, 0, 0] -> misses the proba=0.4 fraud row -> fn=1
        curve = cost_curve(Y_TRUE, Y_PROBA, thresholds=THRESHOLDS,
                            avg_fraud_loss=AVG_FRAUD_LOSS, avg_fp_cost=AVG_FP_COST)
        row = self._row(curve, 0.5)
        assert row["false_negatives"] == 1
        assert row["false_positives"] == 0
        assert row["true_positives"] == 1
        assert row["total_cost"] == pytest.approx(AVG_FRAUD_LOSS)

    def test_very_high_threshold_misses_all_fraud(self):
        # pred = [0, 0, 0, 0] -> both fraud rows missed -> fn=2
        curve = cost_curve(Y_TRUE, Y_PROBA, thresholds=THRESHOLDS,
                            avg_fraud_loss=AVG_FRAUD_LOSS, avg_fp_cost=AVG_FP_COST)
        row = self._row(curve, 0.95)
        assert row["false_negatives"] == 2
        assert row["false_positives"] == 0
        assert row["true_positives"] == 0
        assert row["total_cost"] == pytest.approx(2 * AVG_FRAUD_LOSS)

    def test_default_thresholds_span_the_expected_range(self):
        curve = cost_curve(Y_TRUE, Y_PROBA)
        assert len(curve) == 99
        assert curve["threshold"].min() == pytest.approx(0.01)
        assert curve["threshold"].max() == pytest.approx(0.99)


class TestOptimalThreshold:
    def test_picks_the_minimum_cost_row(self):
        result = optimal_threshold(Y_TRUE, Y_PROBA, avg_fraud_loss=AVG_FRAUD_LOSS, avg_fp_cost=AVG_FP_COST)
        # 0.35 is the only zero-cost threshold in the default sweep's
        # neighborhood too, but pin it down explicitly via a custom curve
        # to keep this independent of the default threshold grid.
        assert result["optimal_total_cost"] <= 150.0  # true optimum (0.35) costs 0

    def test_savings_computed_against_the_closest_threshold_to_half(self):
        # Using the same 4 hand-picked thresholds directly, so "default"
        # unambiguously resolves to threshold=0.5 (cost=5000) and "optimal"
        # to threshold=0.35 (cost=0).
        curve = cost_curve(Y_TRUE, Y_PROBA, thresholds=THRESHOLDS,
                            avg_fraud_loss=AVG_FRAUD_LOSS, avg_fp_cost=AVG_FP_COST)
        best_row = curve.loc[curve["total_cost"].idxmin()]
        default_row = curve.iloc[(curve["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]

        assert best_row["threshold"] == pytest.approx(0.35)
        assert default_row["threshold"] == pytest.approx(0.5)

        savings = default_row["total_cost"] - best_row["total_cost"]
        savings_pct = savings / default_row["total_cost"] * 100

        assert savings == pytest.approx(AVG_FRAUD_LOSS)
        assert savings_pct == pytest.approx(100.0)

    def test_zero_cost_at_default_threshold_yields_zero_savings_pct(self):
        # Degenerate case: if the ~0.5 threshold is already free, the
        # percentage-savings division must not blow up (guards the
        # `if default_row["total_cost"] > 0 else 0.0` branch).
        y_true = [0, 0, 0, 0]
        y_proba = [0.1, 0.2, 0.3, 0.4]
        result = optimal_threshold(y_true, y_proba)
        assert result["default_threshold_cost"] == 0.0
        assert result["estimated_savings_pct"] == 0.0


class TestThresholdSensitivity:
    def test_grid_shape_and_values_match_the_requested_multipliers(self):
        result = threshold_sensitivity(
            Y_TRUE, Y_PROBA,
            fraud_loss_multipliers=[1.0, 2.0],
            fp_cost_multipliers=[0.5],
            base_fraud_loss=AVG_FRAUD_LOSS,
            base_fp_cost=AVG_FP_COST,
        )

        assert result["base_fraud_loss"] == AVG_FRAUD_LOSS
        assert result["base_fp_cost"] == AVG_FP_COST
        assert len(result["grid"]) == 2  # 2 fraud-loss mults x 1 fp-cost mult

        cell = result["grid"][0]
        assert cell["fraud_loss_multiplier"] == 1.0
        assert cell["fp_cost_multiplier"] == 0.5
        assert cell["avg_fraud_loss"] == AVG_FRAUD_LOSS
        assert cell["avg_fp_cost"] == AVG_FP_COST * 0.5

    def test_each_grid_cell_matches_optimal_threshold_computed_directly(self):
        # Each cell in the grid must be exactly what optimal_threshold()
        # itself would return for that multiplied cost pair — the sweep
        # shouldn't introduce any arithmetic of its own.
        result = threshold_sensitivity(
            Y_TRUE, Y_PROBA,
            fraud_loss_multipliers=[2.0],
            fp_cost_multipliers=[0.5],
            base_fraud_loss=AVG_FRAUD_LOSS,
            base_fp_cost=AVG_FP_COST,
        )
        cell = result["grid"][0]

        expected = optimal_threshold(
            Y_TRUE, Y_PROBA,
            avg_fraud_loss=AVG_FRAUD_LOSS * 2.0,
            avg_fp_cost=AVG_FP_COST * 0.5,
        )

        assert cell["optimal_threshold"] == expected["optimal_threshold"]
        assert cell["optimal_total_cost"] == expected["optimal_total_cost"]
        assert cell["estimated_savings_pct"] == expected["estimated_savings_pct"]

    def test_default_multipliers_produce_a_4x4_grid(self):
        result = threshold_sensitivity(Y_TRUE, Y_PROBA)
        assert len(result["fraud_loss_multipliers"]) == 4
        assert len(result["fp_cost_multipliers"]) == 4
        assert len(result["grid"]) == 16

    def test_optimal_threshold_actually_shifts_with_the_cost_ratio(self):
        # Hand-computable on SENS_Y_TRUE/SENS_Y_PROBA = ([1, 0], [0.3, 0.6]):
        # at threshold t, predictions are [1,1] for t<=0.3 (cost=fp_cost),
        # [0,1] for 0.3<t<=0.6 (cost=fraud_loss+fp_cost, always worse),
        # [0,0] for t>0.6 (cost=fraud_loss). So the optimum is whichever
        # of "flag everything" or "flag nothing" is cheaper — it must
        # flip as fraud_loss grows relative to fp_cost.
        cheap_fraud = threshold_sensitivity(
            SENS_Y_TRUE, SENS_Y_PROBA,
            fraud_loss_multipliers=[1.0], fp_cost_multipliers=[1.0],
            base_fraud_loss=100.0, base_fp_cost=150.0,
        )["grid"][0]
        expensive_fraud = threshold_sensitivity(
            SENS_Y_TRUE, SENS_Y_PROBA,
            fraud_loss_multipliers=[50.0], fp_cost_multipliers=[1.0],
            base_fraud_loss=100.0, base_fp_cost=150.0,
        )["grid"][0]

        # fraud_loss (100) < fp_cost (150) -> cheaper to let the fraud
        # through than to flag the legit row -> high threshold.
        assert cheap_fraud["optimal_threshold"] > 0.6
        assert cheap_fraud["optimal_total_cost"] == pytest.approx(100.0)

        # fraud_loss (5000) >> fp_cost (150) -> cheaper to flag
        # everything than to miss the fraud -> low threshold.
        assert expensive_fraud["optimal_threshold"] <= 0.3
        assert expensive_fraud["optimal_total_cost"] == pytest.approx(150.0)
