"""
Train the transaction risk scoring model on IEEE-CIS.

Run:
    python src/train_model.py

Outputs:
    models/risk_model.joblib          - trained XGBoost classifier
    models/feature_cols.joblib        - ordered list of feature names used
    models/optimal_threshold.joblib   - cost-optimal decision threshold (0-1
                                         probability scale) — the single
                                         binary "above/below cost-optimal"
                                         point RiskExplainer's above_threshold
                                         field is based on. Kept for that
                                         purpose alongside decision_thresholds
                                         below, which is a DIFFERENT, related
                                         concept: the live system's two-tier
                                         REVIEW/BLOCK boundaries.
    models/decision_thresholds.joblib - {"review": float, "block": float} on
                                         the 0-100 risk-score scale — what
                                         decision_rules.decide_action()
                                         actually gates transactions with.
                                         See BLOCK_FP_COST_MULTIPLIER below
                                         for exactly how "block" is derived.
    models/eval_report.txt            - AUC/PR-AUC + cost-based analysis
    models/cost_summary.json          - {"estimated_savings", "estimated_savings_pct",
                                         "n_test_transactions", "roc_auc"} from the
                                         cost-optimal threshold analysis above, in
                                         structured form — feeds GET /api/cost-analysis's
                                         headline_monthly_savings_estimate (see
                                         src/impact_summary.py) and its "At a glance"
                                         panel, since eval_report.txt is unstructured text.
    models/cost_curve.json            - per-threshold false_negatives/false_positives over
                                         the test set, so GET /api/cost-analysis can serve
                                         the cost curve (and recompute it for any cost
                                         assumption) without re-scoring 100k+ rows.
"""
import json

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from data_utils import load_raw_data, engineer_features, time_based_split, CATEGORICAL_COLS
from cost_analysis import cost_curve, optimal_threshold, DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
THRESHOLD_PATH = "models/optimal_threshold.joblib"
CATEGORIES_PATH = "models/categorical_categories.joblib"
REPORT_PATH = "models/eval_report.txt"
DECISION_THRESHOLDS_PATH = "models/decision_thresholds.joblib"
COST_SUMMARY_PATH = "models/cost_summary.json"
COST_CURVE_PATH = "models/cost_curve.json"

# The REVIEW threshold is just optimal_threshold()'s result under the
# default cost assumptions — the point where flagging first becomes
# worth it (see cost_analysis.DEFAULT_AVG_FRAUD_LOSS/DEFAULT_AVG_FP_COST).
#
# BLOCK needs a threshold that's actually HIGHER than that ("only act on
# high confidence"). Scaling avg_fraud_loss up — missing fraud made more
# expensive — moves optimal_threshold() DOWN, not up: it makes the model
# want to catch more fraud, i.e. flag MORE aggressively, the opposite of
# what a stricter tier needs. Scaling avg_fp_cost up instead does what we
# want: it says a false BLOCK (a legitimate transaction rejected
# outright, not just flagged for review) is assumed to cost far more
# than a false REVIEW — e.g. lost customer trust/churn on a hard
# rejection vs. mild friction on a review — which is exactly the "only
# act on high confidence" framing this tier needs. This multiplier is a
# documented assumption, the same way DEFAULT_SENSITIVITY_MULTIPLIERS is
# — revisit it if a retrain ever collapses the two tiers together.
BLOCK_FP_COST_MULTIPLIER = 6.0


def train(df=None):
    """df: pass an already-loaded-and-engineered DataFrame to skip
    re-running load_raw_data()/engineer_features() (used by
    src/graph_features_ablation.py, which needs the same engineered
    dataset for both the pre- and post-retrain comparison). Defaults to
    loading it fresh, exactly as before."""
    if df is None:
        df = load_raw_data()
        df = engineer_features(df)

    X_train, X_test, y_train, y_test, feature_cols = time_based_split(df)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")
    print("(chronological split — train is the earlier 80% of transactions,")
    print(" test is the later 20%, so entity-history features carry no leakage)")

    # Capture the exact category set for each categorical column as seen
    # across the FULL dataset (train+test), so inference-time single-row
    # DataFrames can be cast against this same fixed set. Without this,
    # astype('category') on a single row derives categories from just
    # that one row, which XGBoost's categorical predict path rejects.
    categories_map = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            categories_map[col] = list(df[col].cat.categories)

    fraud_rate = y_train.mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate
    print(f"Train fraud rate: {fraud_rate:.4%} | scale_pos_weight: {scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print("Training...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    default_report = classification_report(y_test, (y_proba >= 0.5).astype(int), digits=4)

    # --- Cost-optimal threshold analysis ---
    cost_result = optimal_threshold(y_test, y_proba)
    # The same curve optimal_threshold() minimizes over, persisted so the
    # API can serve it without re-scoring the test set per request. Only
    # the threshold and the two error counts are kept: total cost for any
    # other cost assumption is exactly fn * fraud_loss + fp * fp_cost, so
    # the frontend's chart can follow the user's inputs without needing
    # the model.
    curve_df = cost_curve(y_test, y_proba)
    optimal_t = cost_result["optimal_threshold"]
    optimal_report = classification_report(y_test, (y_proba >= optimal_t).astype(int), digits=4)

    # --- Live decision boundaries (0-100 risk-score scale), derived from
    # the same cost analysis rather than hardcoded — see
    # BLOCK_FP_COST_MULTIPLIER above for why BLOCK uses a scaled-up
    # avg_fp_cost rather than a scaled-up avg_fraud_loss.
    review_threshold = round(optimal_t * 100, 1)
    block_cost_result = optimal_threshold(
        y_test, y_proba,
        avg_fraud_loss=DEFAULT_AVG_FRAUD_LOSS,
        avg_fp_cost=DEFAULT_AVG_FP_COST * BLOCK_FP_COST_MULTIPLIER,
    )
    block_threshold = round(block_cost_result["optimal_threshold"] * 100, 1)
    if block_threshold <= review_threshold:
        # Report reality, don't force it — an honest "these collapsed"
        # is more useful than a silently hand-picked number.
        print(
            f"WARNING: derived block_threshold ({block_threshold}) did not come out "
            f"above review_threshold ({review_threshold}) — BLOCK_FP_COST_MULTIPLIER "
            f"({BLOCK_FP_COST_MULTIPLIER}) may need revisiting."
        )
    decision_thresholds = {"review": review_threshold, "block": block_threshold}

    summary = (
        f"ROC-AUC: {auc:.4f}\n"
        f"Average Precision (PR-AUC): {ap:.4f}\n\n"
        f"=== Default threshold (0.5) ===\n{default_report}\n"
        f"Estimated cost at threshold 0.5: Rs {cost_result['default_threshold_cost']:,.0f}\n"
        f"  (assumes avg fraud loss = Rs {DEFAULT_AVG_FRAUD_LOSS:,.0f}/missed fraud, "
        f"avg false-positive cost = Rs {DEFAULT_AVG_FP_COST:,.0f}/wrongly-flagged legit txn)\n\n"
        f"=== Cost-optimal threshold ({optimal_t}) ===\n{optimal_report}\n"
        f"Estimated cost at optimal threshold: Rs {cost_result['optimal_total_cost']:,.0f}\n"
        f"Estimated savings vs default: Rs {cost_result['estimated_savings']:,.0f} "
        f"({cost_result['estimated_savings_pct']}%)\n\n"
        f"=== Live decision boundaries (derived, 0-100 risk-score scale) ===\n"
        f"REVIEW threshold: {review_threshold} (= cost-optimal threshold under default "
        f"cost assumptions, x100)\n"
        f"BLOCK threshold:  {block_threshold} (= cost-optimal threshold with avg_fp_cost "
        f"scaled {BLOCK_FP_COST_MULTIPLIER}x, x100 — 'only act on high confidence')\n"
    )
    print(summary)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(optimal_t, THRESHOLD_PATH)
    joblib.dump(categories_map, CATEGORIES_PATH)
    joblib.dump(decision_thresholds, DECISION_THRESHOLDS_PATH)
    with open(COST_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "estimated_savings": cost_result["estimated_savings"],
            "estimated_savings_pct": cost_result["estimated_savings_pct"],
            "n_test_transactions": len(y_test),
            "roc_auc": round(float(auc), 4),
        }, f, indent=2)
    with open(COST_CURVE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            curve_df[["threshold", "false_negatives", "false_positives"]].to_dict("records"),
            f,
        )
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved feature list -> {FEATURES_PATH}")
    print(f"Saved cost-optimal threshold -> {THRESHOLD_PATH}")
    print(f"Saved categorical category sets -> {CATEGORIES_PATH}")
    print(f"Saved live decision thresholds -> {DECISION_THRESHOLDS_PATH}")
    print(f"Saved eval report -> {REPORT_PATH}")
    print(f"Saved cost summary -> {COST_SUMMARY_PATH}")
    print(f"Saved cost curve -> {COST_CURVE_PATH}")


if __name__ == "__main__":
    train()