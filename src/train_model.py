"""
Train the transaction risk scoring model on IEEE-CIS.

Run:
    python src/train_model.py

Outputs:
    models/risk_model.joblib       - trained XGBoost classifier
    models/feature_cols.joblib     - ordered list of feature names used
    models/optimal_threshold.joblib - cost-optimal decision threshold
    models/eval_report.txt         - AUC/PR-AUC + cost-based analysis
"""
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from data_utils import load_raw_data, engineer_features, time_based_split, CATEGORICAL_COLS
from cost_analysis import optimal_threshold, DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
THRESHOLD_PATH = "models/optimal_threshold.joblib"
CATEGORIES_PATH = "models/categorical_categories.joblib"
REPORT_PATH = "models/eval_report.txt"


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
    optimal_t = cost_result["optimal_threshold"]
    optimal_report = classification_report(y_test, (y_proba >= optimal_t).astype(int), digits=4)

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
        f"({cost_result['estimated_savings_pct']}%)\n"
    )
    print(summary)

    with open(REPORT_PATH, "w") as f:
        f.write(summary)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(optimal_t, THRESHOLD_PATH)
    joblib.dump(categories_map, CATEGORIES_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved feature list -> {FEATURES_PATH}")
    print(f"Saved cost-optimal threshold -> {THRESHOLD_PATH}")
    print(f"Saved categorical category sets -> {CATEGORIES_PATH}")
    print(f"Saved eval report -> {REPORT_PATH}")


if __name__ == "__main__":
    train()