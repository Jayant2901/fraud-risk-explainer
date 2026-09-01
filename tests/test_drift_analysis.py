import numpy as np
import pandas as pd
import pytest

from drift_analysis import (
    choose_bucket_count,
    bucket_edges,
    assign_buckets,
    compute_bucket_metrics,
    build_report,
    MIN_BUCKETS,
    MAX_BUCKETS,
)


class TestChooseBucketCount:
    def test_long_span_uses_the_max_bucket_count(self):
        # 41.9-day span (roughly the real test set's span) is well over
        # MAX_BUCKETS days -> should use the finest-grained option.
        assert choose_bucket_count(41.9 * 86400) == MAX_BUCKETS

    def test_short_span_falls_back_toward_min_buckets(self):
        # A 2-day span can't support 6 one-day-or-longer buckets -> falls
        # back to MIN_BUCKETS rather than hardcoding a fixed unit.
        assert choose_bucket_count(2 * 86400) == MIN_BUCKETS

    def test_never_returns_zero_for_a_very_short_span(self):
        assert choose_bucket_count(1000) >= 1

    def test_span_exactly_at_max_buckets_days_uses_max(self):
        assert choose_bucket_count(MAX_BUCKETS * 86400) == MAX_BUCKETS


class TestBucketEdgesAndAssignment:
    def test_edges_span_from_min_to_max_in_equal_widths(self):
        edges = bucket_edges(0, 100, 4)
        assert list(edges) == [0, 25, 50, 75, 100]

    def test_minimum_timestamp_lands_in_bucket_zero(self):
        # pd.cut's bins are (left, right] by default, with include_lowest
        # extending the FIRST bin to [0, 25] — so a boundary value like
        # 25 falls into the bucket it's the upper edge of (bucket 0), and
        # 50 into the next one (25, 50] (bucket 1).
        edges = bucket_edges(0, 100, 4)
        dt = pd.Series([0, 25, 50, 99, 100])
        buckets = assign_buckets(dt, edges)
        assert list(buckets) == [0, 0, 1, 3, 3]

    def test_maximum_timestamp_lands_in_the_last_bucket(self):
        edges = bucket_edges(0, 100, 4)
        dt = pd.Series([100])
        buckets = assign_buckets(dt, edges)
        assert list(buckets) == [3]


class TestComputeBucketMetrics:
    def test_hand_computable_two_bucket_case(self):
        # Bucket 0: 2 fraud (proba .9, .8 -> both correctly flagged at
        # threshold .5) + 2 legit (proba .1, .2 -> both correctly
        # allowed) -> perfect separation, AUC=1.0, precision=1.0, recall=1.0
        # Bucket 1: 1 fraud (proba .3 -> missed) + 1 legit (proba .6 ->
        # wrongly flagged) -> AUC=0.0 (perfectly wrong ranking),
        # precision=0.0, recall=0.0
        y_true = [1, 1, 0, 0, 1, 0]
        y_proba = [0.9, 0.8, 0.1, 0.2, 0.3, 0.6]
        bucket_idx = [0, 0, 0, 0, 1, 1]

        rows = compute_bucket_metrics(y_true, y_proba, bucket_idx, threshold=0.5)

        assert len(rows) == 2
        b0, b1 = rows
        assert b0["bucket"] == 0
        assert b0["n"] == 4
        assert b0["n_fraud"] == 2
        assert b0["roc_auc"] == pytest.approx(1.0)
        assert b0["precision"] == pytest.approx(1.0)
        assert b0["recall"] == pytest.approx(1.0)

        assert b1["bucket"] == 1
        assert b1["n"] == 2
        assert b1["n_fraud"] == 1
        assert b1["roc_auc"] == pytest.approx(0.0)
        assert b1["precision"] == pytest.approx(0.0)
        assert b1["recall"] == pytest.approx(0.0)

    def test_bucket_with_only_one_class_has_no_auc(self):
        # A bucket with zero fraud makes AUC undefined -> None, not a crash.
        y_true = [0, 0, 0]
        y_proba = [0.1, 0.2, 0.3]
        bucket_idx = [0, 0, 0]

        rows = compute_bucket_metrics(y_true, y_proba, bucket_idx, threshold=0.5)
        assert rows[0]["roc_auc"] is None

    def test_buckets_are_returned_in_order(self):
        y_true = [1, 0, 1, 0]
        y_proba = [0.9, 0.1, 0.9, 0.1]
        bucket_idx = [2, 0, 1, 2]

        rows = compute_bucket_metrics(y_true, y_proba, bucket_idx, threshold=0.5)
        assert [r["bucket"] for r in rows] == [0, 1, 2]


class TestBuildReport:
    def test_report_includes_the_auc_spread_across_buckets(self):
        rows = [
            {"bucket": 0, "n": 10, "n_fraud": 2, "roc_auc": 0.95, "precision": 0.5, "recall": 0.8},
            {"bucket": 1, "n": 10, "n_fraud": 2, "roc_auc": 0.80, "precision": 0.4, "recall": 0.6},
        ]
        edges = np.array([0, 50, 100])
        report = build_report(rows, edges, span_seconds=100)

        assert "Temporal drift analysis" in report
        assert "0.8000 to 0.9500" in report or "0.95" in report
        assert "spread of 0.1500" in report

    def test_report_handles_a_bucket_with_no_auc_gracefully(self):
        rows = [
            {"bucket": 0, "n": 5, "n_fraud": 0, "roc_auc": None, "precision": 0.0, "recall": 0.0},
        ]
        edges = np.array([0, 100])
        report = build_report(rows, edges, span_seconds=100)
        assert "n/a" in report
