"""
Feedback-loop tests.

Two things matter most here: only genuinely disposed items become labels,
and a feedback row from the TEST window can never be appended to
training. The second is a leakage guard — get it wrong and every number
this project reports is quietly inflated.
"""
import os

import pandas as pd
import pytest

from feedback_export import (
    build_feedback_rows,
    export_feedback,
    load_feedback,
    to_json_summary,
)
from review_queue import CONFIRMED_FRAUD, FALSE_POSITIVE, ReviewQueue
from train_model import select_feedback_for_training


def make_item(verdict_id="v1", disposition=None, *, escalated=False, transaction_dt=1000.0,
              action="REVIEW", risk_score=55.0):
    return {
        "verdict_id": verdict_id,
        "entity_id": "entity-a",
        "txn_index": 0,
        "risk_score": risk_score,
        "decision": {"action": action, "escalated_due_to_history": escalated},
        "baseline_decision": {"action": "REVIEW", "escalated_due_to_history": False},
        "escalated_due_to_history": escalated,
        "disposition": disposition,
        "disposed_at": "2026-01-01T00:00:00+00:00" if disposition else None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "notes": [],
        "transaction": {"TransactionAmt": 250.0, "ProductCD": "W", "TransactionDT": transaction_dt},
        "transaction_dt": transaction_dt,
        "escalation_state": "ELEVATED" if escalated else "NORMAL",
    }


class TestBuildFeedbackRows:
    def test_only_disposed_items_become_labels(self):
        rows = build_feedback_rows([
            make_item("v1", CONFIRMED_FRAUD),
            make_item("v2", None),  # still pending
            make_item("v3", FALSE_POSITIVE),
        ])

        assert {r["verdict_id"] for r in rows} == {"v1", "v3"}

    def test_dispositions_map_to_the_binary_label(self):
        rows = build_feedback_rows([
            make_item("v1", CONFIRMED_FRAUD),
            make_item("v2", FALSE_POSITIVE),
        ])

        assert {r["verdict_id"]: r["isFraud"] for r in rows} == {"v1": 1, "v2": 0}

    def test_carries_the_features_the_model_actually_saw(self):
        rows = build_feedback_rows([make_item("v1", CONFIRMED_FRAUD)])

        assert rows[0]["TransactionAmt"] == 250.0
        assert rows[0]["ProductCD"] == "W"

    def test_carries_the_decision_context_for_analysis(self):
        rows = build_feedback_rows([make_item("v1", CONFIRMED_FRAUD, escalated=True)])

        row = rows[0]
        assert row["escalated"] is True
        assert row["escalation_state"] == "ELEVATED"
        assert row["baseline_action"] == "REVIEW"
        assert row["risk_score"] == 55.0
        assert row["disposed_at"]

    def test_an_empty_queue_yields_no_rows(self):
        assert build_feedback_rows([]) == []


class TestChronologicalSplitGuard:
    """A feedback row belongs where its TRANSACTION sits in time, not
    where the reviewer's much-later click sits."""

    def _feedback(self, dts):
        return pd.DataFrame([{"transaction_dt": dt, "isFraud": 1} for dt in dts])

    def test_train_side_rows_are_selected(self):
        selected = select_feedback_for_training(self._feedback([100.0, 200.0]), split_dt=500.0)

        assert len(selected) == 2

    def test_a_test_side_row_is_never_trained_on(self):
        selected = select_feedback_for_training(self._feedback([600.0, 900.0]), split_dt=500.0)

        assert selected.empty

    def test_the_boundary_itself_belongs_to_test(self):
        # split_dt is the first test-set timestamp, so a row AT it is test.
        selected = select_feedback_for_training(self._feedback([500.0]), split_dt=500.0)

        assert selected.empty

    def test_mixed_rows_are_split_correctly(self):
        selected = select_feedback_for_training(
            self._feedback([100.0, 499.0, 500.0, 700.0]), split_dt=500.0
        )

        assert sorted(selected["transaction_dt"]) == [100.0, 499.0]

    def test_undated_rows_are_dropped_rather_than_guessed_at(self):
        feedback = pd.DataFrame([
            {"transaction_dt": None, "isFraud": 1},
            {"transaction_dt": 100.0, "isFraud": 0},
        ])

        selected = select_feedback_for_training(feedback, split_dt=500.0)

        assert len(selected) == 1
        assert selected.iloc[0]["transaction_dt"] == 100.0

    def test_empty_feedback_is_handled(self):
        assert select_feedback_for_training(pd.DataFrame(), split_dt=500.0).empty


class TestExportRoundTrip:
    def test_exports_disposed_items_to_a_readable_csv(self, tmp_path):
        queue = ReviewQueue()
        queue.add(make_item("v1"))
        queue.add(make_item("v2"))
        queue.dispose("v1", CONFIRMED_FRAUD)

        summary = export_feedback(queue, output_dir=str(tmp_path))

        assert summary["rows"] == 1
        assert summary["confirmed_fraud"] == 1
        assert os.path.exists(summary["path"])
        assert pd.read_csv(summary["path"]).iloc[0]["verdict_id"] == "v1"

    def test_load_feedback_reads_back_what_was_exported(self, tmp_path):
        queue = ReviewQueue()
        queue.add(make_item("v1"))
        queue.dispose("v1", FALSE_POSITIVE)
        export_feedback(queue, output_dir=str(tmp_path))

        loaded = load_feedback(str(tmp_path))

        assert len(loaded) == 1
        assert loaded.iloc[0]["isFraud"] == 0

    def test_a_verdict_exported_twice_counts_once(self, tmp_path):
        queue = ReviewQueue()
        queue.add(make_item("v1"))
        queue.dispose("v1", CONFIRMED_FRAUD)
        export_feedback(queue, output_dir=str(tmp_path))
        export_feedback(queue, output_dir=str(tmp_path))

        assert len(load_feedback(str(tmp_path))) == 1

    def test_loading_from_an_empty_directory_is_not_an_error(self, tmp_path):
        assert load_feedback(str(tmp_path)).empty

    def test_loading_from_a_missing_directory_is_not_an_error(self, tmp_path):
        assert load_feedback(str(tmp_path / "nope")).empty


class TestJsonSummary:
    def test_reports_counts_and_the_bias_caveat(self):
        queue = ReviewQueue()
        queue.add(make_item("v1"))
        queue.add(make_item("v2"))
        queue.dispose("v1", CONFIRMED_FRAUD)
        queue.dispose("v2", FALSE_POSITIVE)

        summary = to_json_summary(queue)

        assert summary["count"] == 2
        assert summary["confirmed_fraud"] == 1
        assert summary["false_positive"] == 1
        # The caveat travels with the data, not just in a docstring.
        assert "censored sample" in summary["bias_warning"]

    def test_undisposed_items_are_absent(self):
        queue = ReviewQueue()
        queue.add(make_item("v1"))

        assert to_json_summary(queue)["count"] == 0
