"""
Cold-start comparison — does the device/address graph signal
(src/graph_features.py) actually help on the hardest, highest-volume
real fraud case: a brand-new entity with NO history of its own
(entity_prior_txn_count == 0), which entity_prior_* features are
structurally blind to?

Captures the CURRENTLY-SAVED model's cold-start-subset recall as
"before" (trained without graph features), then retrains
(src/train_model.py's train()) with the graph features now wired into
data_utils.engineer_features()/get_feature_columns(), and reports
"after" on the exact same test rows — an overall-AUC bump would be much
less interesting than a recall improvement specifically on the
cold-start subset, which is the whole point of this comparison.

Run:
    python src/graph_features_ablation.py

Output:
    models/cold_start_report.txt

Note: this OVERWRITES models/risk_model.joblib and friends with the
retrained (graph-features-included) model — that's the point (the "real
retrain" the acceptance criteria calls for), not a side effect.
"""
import sys
import os

sys.path.append(os.path.dirname(__file__))

import joblib
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from data_utils import load_raw_data, engineer_features, TARGET_COL
from graph_features import GRAPH_FEATURE_COLS
from train_model import train, MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH

REPORT_PATH = "models/cold_start_report.txt"
TEST_FRAC = 0.2


def _cold_start_metrics(model, feature_cols: list, threshold: float, test_df) -> dict:
    X = test_df[feature_cols]
    y_true = test_df[TARGET_COL]
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    mask = (test_df["entity_prior_txn_count"] == 0).to_numpy()
    n_cold = int(mask.sum())
    n_cold_fraud = int(y_true[mask].sum())

    overall_auc = float(roc_auc_score(y_true, y_proba))
    cold_recall = (
        float(recall_score(y_true[mask], y_pred[mask], zero_division=0)) if n_cold_fraud > 0 else None
    )
    cold_precision = float(precision_score(y_true[mask], y_pred[mask], zero_division=0))

    return {
        "overall_auc": round(overall_auc, 4),
        "n_cold_start": n_cold,
        "n_cold_start_fraud": n_cold_fraud,
        "cold_start_recall": round(cold_recall, 4) if cold_recall is not None else None,
        "cold_start_precision": round(cold_precision, 4),
    }


def build_report(before: dict, after: dict, new_feature_cols: list) -> str:
    graph_wired = all(c in new_feature_cols for c in GRAPH_FEATURE_COLS)

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    lines = [
        "Cold-start comparison: device/address graph features",
        "========================================================",
        "",
        f"Cold-start subset: entity_prior_txn_count == 0 rows in the real "
        f"chronological test set ({before['n_cold_start']:,} rows, "
        f"{before['n_cold_start_fraud']:,} fraud) -- transactions from an "
        f"entity with no prior history of its own, which entity_prior_*",
        f"features cannot see anything about by construction.",
        "",
        f"Graph features present in the retrained feature set: {'yes' if graph_wired else 'NO -- wiring problem'}",
        "",
        f"{'':30}{'before':>12}{'after':>12}",
        f"{'Overall ROC-AUC':30}{fmt(before['overall_auc']):>12}{fmt(after['overall_auc']):>12}",
        f"{'Cold-start recall':30}{fmt(before['cold_start_recall']):>12}{fmt(after['cold_start_recall']):>12}",
        f"{'Cold-start precision':30}{fmt(before['cold_start_precision']):>12}{fmt(after['cold_start_precision']):>12}",
        "",
    ]

    if before["cold_start_recall"] is not None and after["cold_start_recall"] is not None:
        delta = after["cold_start_recall"] - before["cold_start_recall"]
        lines.append(f"Cold-start recall delta: {delta:+.4f}")

    return "\n".join(lines)


def run():
    print("Loading and feature-engineering the full dataset (shared by before/after)...")
    df = load_raw_data()
    df = engineer_features(df)

    # Same chronological split boundary train_model.py's time_based_split
    # uses -- unaffected by adding new feature COLUMNS, only the feature
    # SET differs between the old and new models.
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - TEST_FRAC))
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

    print("Loading the CURRENTLY-SAVED model (trained WITHOUT graph features) as the 'before' snapshot...")
    old_model = joblib.load(MODEL_PATH)
    old_feature_cols = joblib.load(FEATURES_PATH)
    old_threshold = joblib.load(THRESHOLD_PATH)
    before = _cold_start_metrics(old_model, old_feature_cols, old_threshold, test_df)
    print(f"Before: {before}")

    print("Retraining WITH graph features wired into engineer_features()/get_feature_columns()...")
    train(df=df)

    new_model = joblib.load(MODEL_PATH)
    new_feature_cols = joblib.load(FEATURES_PATH)
    new_threshold = joblib.load(THRESHOLD_PATH)
    after = _cold_start_metrics(new_model, new_feature_cols, new_threshold, test_df)
    print(f"After: {after}")

    report = build_report(before, after, new_feature_cols)
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    run()
