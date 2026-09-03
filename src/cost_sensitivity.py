"""
Cost-assumption sensitivity sweep — how much does the "cost-optimal"
threshold actually move if the Rs 5,000/Rs 150 assumptions in
cost_analysis.py are wrong?

Scores the real chronological test set (the same split train_model.py
uses) with the already-trained model, then sweeps a grid of
avg_fraud_loss x avg_fp_cost values via cost_analysis.threshold_sensitivity().

Run:
    python src/cost_sensitivity.py

Output:
    models/cost_sensitivity_report.json  (grid, consumed by the API/frontend)
    models/cost_sensitivity_report.txt   (human-readable summary)
"""
import sys
import os
import json

sys.path.append(os.path.dirname(__file__))

import joblib

from escalation_ablation import load_test_set
from cost_analysis import threshold_sensitivity, DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
JSON_REPORT_PATH = "models/cost_sensitivity_report.json"
TEXT_REPORT_PATH = "models/cost_sensitivity_report.txt"


def score_test_set_proba(test_df):
    """Raw 0-1 probabilities (not the 0-100 risk_score scale) — what
    cost_analysis.py's threshold sweep expects."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    X_test = test_df[feature_cols]
    return model.predict_proba(X_test)[:, 1]


def build_text_report(result: dict) -> str:
    lines = [
        "Cost-assumption sensitivity sweep",
        "===================================",
        "",
        f"Base assumptions: avg_fraud_loss = Rs {result['base_fraud_loss']:,.0f}, "
        f"avg_fp_cost = Rs {result['base_fp_cost']:,.0f}",
        f"Grid: fraud_loss multipliers {result['fraud_loss_multipliers']} x "
        f"fp_cost multipliers {result['fp_cost_multipliers']}",
        "",
        f"{'avg_fraud_loss':>16} {'avg_fp_cost':>14} {'optimal_threshold':>18} {'savings_vs_0.5':>15}",
    ]
    for cell in result["grid"]:
        lines.append(
            f"{cell['avg_fraud_loss']:>16,.0f} {cell['avg_fp_cost']:>14,.0f} "
            f"{cell['optimal_threshold']:>18} {cell['estimated_savings_pct']:>14}%"
        )
    lines.append("")

    thresholds = [cell["optimal_threshold"] for cell in result["grid"]]
    lines.append(
        f"Optimal threshold ranges from {min(thresholds)} to {max(thresholds)} "
        f"across this grid (vs. the single default-assumption threshold)."
    )
    return "\n".join(lines)


def run():
    print("Loading and feature-engineering the full dataset...")
    test_df = load_test_set()
    print(f"Test set: {len(test_df):,} transactions (chronological, matches train_model.py's split)")

    print("Scoring test set with the trained model...")
    y_proba = score_test_set_proba(test_df)
    y_true = test_df["isFraud"]

    print("Sweeping cost-assumption grid...")
    result = threshold_sensitivity(
        y_true, y_proba,
        base_fraud_loss=DEFAULT_AVG_FRAUD_LOSS,
        base_fp_cost=DEFAULT_AVG_FP_COST,
    )

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {JSON_REPORT_PATH}")

    text_report = build_text_report(result)
    print(text_report)
    with open(TEXT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"Saved -> {TEXT_REPORT_PATH}")


if __name__ == "__main__":
    run()
