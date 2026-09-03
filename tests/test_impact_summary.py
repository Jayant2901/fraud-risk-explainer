"""
Hand-computable examples for extrapolate_monthly_savings() — the single
headline number the Overview tab shows (Phase H).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from impact_summary import extrapolate_monthly_savings


class TestExtrapolateMonthlySavings:
    def test_hand_computable_scale_up(self):
        # Rs 1,000 saved over 1,000 real test transactions = Rs 1/txn.
        # Scaled to 500,000 assumed monthly transactions -> Rs 500,000/month.
        result = extrapolate_monthly_savings(
            estimated_savings=1000.0, n_test_transactions=1000, assumed_monthly_volume=500_000
        )
        assert result["headline_monthly_savings_estimate"] == 500_000.0
        assert result["assumed_monthly_volume"] == 500_000

    def test_fractional_per_transaction_rate(self):
        # Rs 421,850 over 118,108 real test transactions = Rs 3.571731.../txn.
        # Scaled to 10,000 assumed monthly transactions -> ~Rs 35,717.31.
        result = extrapolate_monthly_savings(
            estimated_savings=421850.0, n_test_transactions=118108, assumed_monthly_volume=10_000
        )
        assert result["headline_monthly_savings_estimate"] == pytest.approx(35717.31, abs=0.5)

    def test_zero_test_transactions_does_not_divide_by_zero(self):
        result = extrapolate_monthly_savings(
            estimated_savings=1000.0, n_test_transactions=0, assumed_monthly_volume=500_000
        )
        assert result["headline_monthly_savings_estimate"] == 0.0

    def test_basis_string_states_the_assumption_and_is_not_a_real_volume_claim(self):
        result = extrapolate_monthly_savings(
            estimated_savings=1000.0, n_test_transactions=1000, assumed_monthly_volume=500_000
        )
        assert "500,000" in result["basis"]
        assert "illustrative" in result["basis"].lower()
        assert "not a real razorpay" in result["basis"].lower()

    def test_uses_the_module_default_volume_when_not_overridden(self):
        from impact_summary import ASSUMED_MONTHLY_TRANSACTION_VOLUME

        result = extrapolate_monthly_savings(estimated_savings=1000.0, n_test_transactions=1000)
        assert result["assumed_monthly_volume"] == ASSUMED_MONTHLY_TRANSACTION_VOLUME
