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
from data_utils import TARGET_COL
from feedback_export import load_feedback

import pandas as pd

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


def select_feedback_for_training(feedback: "pd.DataFrame", split_dt: float) -> "pd.DataFrame":
    """Feedback rows that belong on the TRAIN side of the chronological
    boundary.

    A reviewer's label is attached to a transaction with its own
    timestamp, and that timestamp decides where the row belongs — not the
    (much later) moment the reviewer got to it. A row from the test
    window must never be appended to training: it would leak the test set
    into the model and quietly inflate every number this project reports.

    Rows with no timestamp (a custom transaction that never had one) are
    dropped rather than guessed at.
    """
    if feedback.empty or "transaction_dt" not in feedback.columns:
        return feedback.iloc[0:0]
    dated = feedback[feedback["transaction_dt"].notna()].copy()
    dated["transaction_dt"] = dated["transaction_dt"].astype(float)
    return dated[dated["transaction_dt"] < split_dt]


def _append_feedback(X_train, y_train, feedback_rows, feature_cols, weight: float):
    """Append labelled feedback to the training matrix and return sample
    weights. Feedback rows carry `weight`; original rows carry 1.0, so the
    loop's influence is an explicit, reportable number rather than an
    accident of how many items a reviewer happened to dispose."""
    import numpy as _np

    base_weights = _np.ones(len(X_train), dtype=float)
    if feedback_rows.empty:
        return X_train, y_train, base_weights

    # Only columns the model actually uses; anything the reviewer's export
    # carries that isn't a feature is ignored, and any feature the export
    # lacks stays missing (NaN), which is how this model already handles
    # missing features everywhere else.
    aligned = feedback_rows.reindex(columns=feature_cols)
    for col in feature_cols:
        if str(X_train[col].dtype) == "category":
            aligned[col] = aligned[col].astype("object").astype(
                pd.CategoricalDtype(categories=X_train[col].cat.categories)
            )
        else:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

    X_combined = pd.concat([X_train, aligned], ignore_index=True)
    y_combined = pd.concat(
        [y_train, feedback_rows[TARGET_COL].astype(int)], ignore_index=True
    )
    weights = _np.concatenate([base_weights, _np.full(len(aligned), float(weight))])
    return X_combined, y_combined, weights


def train(df=None, with_feedback: bool = False, feedback_weight: float = 1.0):
    """df: pass an already-loaded-and-engineered DataFrame to skip
    re-running load_raw_data()/engineer_features() (used by
    src/graph_features_ablation.py, which needs the same engineered
    dataset for both the pre- and post-retrain comparison). Defaults to
    loading it fresh, exactly as before.

    with_feedback: append reviewer-labelled rows from data/feedback/ to
    the training set. OFF by default — a retrain without it produces
    exactly the results the README reports. See src/feedback_export.py
    for why these labels are a censored sample and must be opt-in.
    """
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

    # --- Optional: reviewer feedback appended to the training set -------
    feedback_used = 0
    sample_weights = None
    if with_feedback:
        feedback = load_feedback()
        # The boundary is the first test-set timestamp: anything at or
        # after it is test data and must not be trained on.
        split_dt = float(df.sort_values("TransactionDT")["TransactionDT"]
                         .iloc[int(len(df) * 0.8)])
        selected = select_feedback_for_training(feedback, split_dt)
        dropped = len(feedback) - len(selected)
        X_train, y_train, sample_weights = _append_feedback(
            X_train, y_train, selected, feature_cols, feedback_weight
        )
        feedback_used = len(selected)
        print(f"Feedback: {feedback_used} rows appended at weight {feedback_weight} "
              f"({dropped} dropped as test-side or undated)")

    print("Training...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

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
        f"\nReviewer feedback rows used in training: {feedback_used}"
        f"{f' (weight {feedback_weight})' if feedback_used else ' (--with-feedback not used)'}\n"
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

    return {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "review_threshold": review_threshold,
        "block_threshold": block_threshold,
        "estimated_savings": cost_result["estimated_savings"],
        "feedback_rows_used": feedback_used,
    }


def _compare_with_baseline(feedback_weight: float) -> None:
    """Train twice — without feedback, then with — and print the delta.

    The interesting output of a feedback loop is whether human labels
    actually helped, so this reports the comparison rather than only the
    new numbers. A negative delta is a finding, not a failure: these
    labels are a censored sample (see src/feedback_export.py), and if
    they make the model worse on the held-out test set, that is worth
    knowing and saying — the same way the escalation ablation's honest
    finding is handled.
    """
    df = engineer_features(load_raw_data())

    print("\n=== Baseline: no reviewer feedback ===")
    baseline = train(df=df)

    print("\n=== With reviewer feedback ===")
    with_fb = train(df=df, with_feedback=True, feedback_weight=feedback_weight)

    print("\n=== Feedback delta (with - without) ===")
    print(f"Feedback rows used:   {with_fb['feedback_rows_used']}")
    for key in ("roc_auc", "average_precision", "review_threshold",
                "block_threshold", "estimated_savings"):
        delta = with_fb[key] - baseline[key]
        print(f"{key:22} {baseline[key]:>14.4f} -> {with_fb[key]:>14.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the transaction risk model.")
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help=("Append reviewer-labelled rows from data/feedback/ to the training set. "
              "Off by default; a plain run reproduces the numbers the README reports."),
    )
    parser.add_argument(
        "--feedback-weight",
        type=float,
        default=1.0,
        help="Sample weight for feedback rows (default 1.0, same as an original row).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train with and without feedback and print the delta between them.",
    )
    args = parser.parse_args()

    if args.compare:
        _compare_with_baseline(args.feedback_weight)
    else:
        train(with_feedback=args.with_feedback, feedback_weight=args.feedback_weight)