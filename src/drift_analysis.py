"""
Temporal drift analysis — is a STATIC model still good later in the test
window it hasn't seen? `models/eval_report.txt` reports one AUC number
from one train/test split and stops there, which quietly assumes the
model stays good forever. Fraud is adversarial and non-stationary, so
this checks that assumption directly: bucket the real chronological test
set by time and score the ALREADY-TRAINED model's performance in each
bucket separately (no retraining per bucket — the point is to see how a
static model's performance moves over time it hasn't seen).

Run:
    python src/drift_analysis.py

Output:
    models/drift_report.txt
"""
import sys
import os
import json

sys.path.append(os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from escalation_ablation import load_test_set

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
THRESHOLD_PATH = "models/optimal_threshold.joblib"
TEXT_REPORT_PATH = "models/drift_report.txt"
JSON_REPORT_PATH = "models/drift_report.json"

MIN_BUCKETS = 4
MAX_BUCKETS = 6


def choose_bucket_count(span_seconds: float, min_buckets: int = MIN_BUCKETS, max_buckets: int = MAX_BUCKETS) -> int:
    """Picks a bucket count in [min_buckets, max_buckets] from the ACTUAL
    span of the data, rather than assuming a fixed calendar unit (e.g.
    "weekly") that might not fit a shorter-or-longer test window. Aims
    for max_buckets (the finer-grained the better, as long as each
    bucket still covers at least a day); falls back toward min_buckets
    only if the window is too short for that.
    """
    span_days = span_seconds / 86400
    if span_days >= max_buckets:
        return max_buckets
    return max(min_buckets, int(span_days) or 1)


def bucket_edges(dt_min: float, dt_max: float, num_buckets: int) -> np.ndarray:
    return np.linspace(dt_min, dt_max, num_buckets + 1)


def assign_buckets(dt_series, edges: np.ndarray) -> pd.Series:
    """Returns a 0-indexed bucket number per row, using the given edges.
    include_lowest so the very first (minimum) timestamp lands in bucket
    0 rather than being dropped as NaN by pd.cut's default open-left bins."""
    return pd.cut(dt_series, bins=edges, labels=False, include_lowest=True)


def compute_bucket_metrics(y_true, y_proba, bucket_idx, threshold: float) -> list[dict]:
    """One row per bucket present in bucket_idx (0..max present), in
    order: n, n_fraud, roc_auc (None if the bucket has only one class —
    AUC is undefined then), precision/recall at the given threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bucket_idx = np.asarray(bucket_idx)

    rows = []
    for b in sorted(set(bucket_idx.tolist())):
        mask = bucket_idx == b
        yt = y_true[mask]
        yp = y_proba[mask]
        y_pred = (yp >= threshold).astype(int)

        n_fraud = int(yt.sum())
        roc_auc = float(roc_auc_score(yt, yp)) if n_fraud > 0 and n_fraud < len(yt) else None
        precision = float(precision_score(yt, y_pred, zero_division=0))
        recall = float(recall_score(yt, y_pred, zero_division=0))

        rows.append({
            "bucket": int(b),
            "n": int(mask.sum()),
            "n_fraud": n_fraud,
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    return rows


def _fmt_dt(seconds: float) -> str:
    days = seconds / 86400
    return f"{seconds:,.0f}s ({days:.1f}d)"


def build_report(bucket_rows: list[dict], edges: np.ndarray, span_seconds: float) -> str:
    lines = [
        "Temporal drift analysis",
        "========================",
        "",
        f"Test set span: {_fmt_dt(span_seconds)} — bucketed into {len(bucket_rows)} "
        f"equal-width windows (no retraining per bucket; same model, same threshold).",
        "",
        f"{'bucket':>6} {'window (days)':>16} {'n':>8} {'n_fraud':>8} {'roc_auc':>9} {'precision':>10} {'recall':>8}",
    ]
    for row in bucket_rows:
        b = row["bucket"]
        window_start_days = (edges[b] - edges[0]) / 86400
        window_end_days = (edges[b + 1] - edges[0]) / 86400
        window = f"{window_start_days:.1f}-{window_end_days:.1f}"
        auc_str = f"{row['roc_auc']:.4f}" if row["roc_auc"] is not None else "n/a"
        lines.append(
            f"{b:>6} {window:>16} {row['n']:>8,} {row['n_fraud']:>8,} "
            f"{auc_str:>9} {row['precision']:>10.4f} {row['recall']:>8.4f}"
        )

    aucs = [r["roc_auc"] for r in bucket_rows if r["roc_auc"] is not None]
    lines.append("")
    if len(aucs) >= 2:
        lines.append(f"ROC-AUC ranges from {min(aucs):.4f} to {max(aucs):.4f} across buckets "
                      f"(spread of {max(aucs) - min(aucs):.4f}).")
    return "\n".join(lines)


def run():
    print("Loading and feature-engineering the full dataset...")
    test_df = load_test_set()
    dt = test_df["TransactionDT"]
    span_seconds = float(dt.max() - dt.min())
    print(f"Test set: {len(test_df):,} transactions, span {_fmt_dt(span_seconds)}")

    num_buckets = choose_bucket_count(span_seconds)
    print(f"Using {num_buckets} buckets")
    edges = bucket_edges(float(dt.min()), float(dt.max()), num_buckets)
    bucket_idx = assign_buckets(dt, edges)

    print("Scoring test set with the trained model...")
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    y_proba = model.predict_proba(test_df[feature_cols])[:, 1]
    y_true = test_df["isFraud"]

    bucket_rows = compute_bucket_metrics(y_true, y_proba, bucket_idx, threshold)

    report = build_report(bucket_rows, edges, span_seconds)
    print(report)

    with open(TEXT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {TEXT_REPORT_PATH}")

    json_payload = {
        "span_seconds": span_seconds,
        "num_buckets": num_buckets,
        "edges": edges.tolist(),
        "buckets": bucket_rows,
    }
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved -> {JSON_REPORT_PATH}")


if __name__ == "__main__":
    run()
