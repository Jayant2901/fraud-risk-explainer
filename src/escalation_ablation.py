"""
Escalation ablation study — does entity escalation memory actually help?

The README's headline claim is that watching an entity's recent verdict
history (src/entity_memory.py) and escalating borderline scores based on
it catches more fraud than the raw model score alone. This script is the
offline check of that claim, run against the real trained model and the
real chronological test split — not a synthetic example.

Method: replay the test set, in time order, through a fresh
EntityRiskMemory exactly the way api/main.py's /api/score does live.
For every transaction, compute the SAME risk score but TWO decisions:

  - baseline:            decide_action(risk_score, {"state": "NORMAL"})
                          i.e. what the system would do with no entity
                          memory at all.
  - escalation_adjusted:  decide_action(risk_score, escalation_before)
                          i.e. exactly what the live system does today,
                          using the real, accumulating escalation state.

Verdicts are recorded into the memory using the escalation-adjusted
action as they're replayed, so escalation state genuinely accumulates
the same way it would in production.

Run:
    python src/escalation_ablation.py

Output:
    models/escalation_ablation_report.txt
"""
import sys
import os

sys.path.append(os.path.dirname(__file__))

import pandas as pd
import joblib

from data_utils import load_raw_data, engineer_features
from decision_rules import decide_action
from entity_memory import EntityRiskMemory

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
REPORT_PATH = "models/escalation_ablation_report.txt"

TEST_FRAC = 0.2
FLAGGED_ACTIONS = {"REVIEW", "BLOCK"}


def load_test_set(test_frac: float = TEST_FRAC) -> pd.DataFrame:
    """Same chronological split as data_utils.time_based_split, but keeps
    the full DataFrame (entity_id, TransactionDT, isFraud included)
    instead of just X/y, since the replay needs all three."""
    df = load_raw_data()
    df = engineer_features(df)
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[split_idx:].reset_index(drop=True)


def score_test_set(test_df: pd.DataFrame) -> pd.Series:
    """Vectorized model scoring for the whole test set at once (the same
    model.predict_proba call train_model.py makes) — far faster than
    scoring one row at a time through RiskExplainer, and the replay loop
    below only needs the score, not the SHAP factors."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    X_test = test_df[feature_cols]
    proba = model.predict_proba(X_test)[:, 1]
    return pd.Series(proba * 100, index=test_df.index, name="risk_score").round(1)


def replay(test_df: pd.DataFrame, risk_scores: pd.Series) -> pd.DataFrame:
    """Walks the test set in time order through a fresh EntityRiskMemory,
    recording both decisions per transaction. Returns a DataFrame with
    one row per transaction: is_fraud, risk_score, baseline_action,
    adjusted_action, escalated_due_to_history."""
    memory = EntityRiskMemory()
    rows = []

    for idx, txn in test_df.iterrows():
        entity_id = txn["entity_id"]
        risk_score = float(risk_scores.loc[idx])

        escalation_before = memory.get_escalation_state(entity_id)

        baseline_decision = decide_action(risk_score, {"state": "NORMAL"})
        adjusted_decision = decide_action(risk_score, escalation_before)

        # Record the REAL (escalation-adjusted) action, exactly as
        # api/main.py's /api/score does — so escalation state accumulates
        # the same way it would in production, not against a hypothetical
        # no-escalation history.
        memory.record_verdict(entity_id, adjusted_decision["action"], risk_score)

        rows.append({
            "is_fraud": int(txn["isFraud"]),
            "risk_score": risk_score,
            "baseline_action": baseline_decision["action"],
            "adjusted_action": adjusted_decision["action"],
            "escalated_due_to_history": adjusted_decision["escalated_due_to_history"],
        })

    return pd.DataFrame(rows)


def compute_strategy_metrics(is_fraud: pd.Series, action: pd.Series) -> dict:
    """recall = of all true frauds, fraction flagged (REVIEW/BLOCK).
    false_flag_rate = of all legit transactions, fraction flagged."""
    flagged = action.isin(FLAGGED_ACTIONS)

    n_fraud = int(is_fraud.sum())
    n_legit = int((~is_fraud.astype(bool)).sum())

    flagged_fraud = int((flagged & (is_fraud == 1)).sum())
    flagged_legit = int((flagged & (is_fraud == 0)).sum())

    recall = flagged_fraud / n_fraud if n_fraud > 0 else 0.0
    false_flag_rate = flagged_legit / n_legit if n_legit > 0 else 0.0

    return {
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "flagged_fraud": flagged_fraud,
        "flagged_legit": flagged_legit,
        "recall": recall,
        "false_flag_rate": false_flag_rate,
    }


def compute_escalation_flip_precision(replay_df: pd.DataFrame) -> dict:
    """Among the transactions where escalation actually changed the
    action (escalated_due_to_history == True), what fraction were
    actually fraud? This is the direct answer to "is entity memory
    pulling its weight, or is it mostly false alarms."""
    flips = replay_df[replay_df["escalated_due_to_history"]]
    n_flips = len(flips)
    n_flips_fraud = int(flips["is_fraud"].sum())
    precision = n_flips_fraud / n_flips if n_flips > 0 else 0.0
    return {
        "n_flips": n_flips,
        "n_flips_fraud": n_flips_fraud,
        "precision": precision,
    }


def build_report(replay_df: pd.DataFrame) -> str:
    is_fraud = replay_df["is_fraud"]

    baseline = compute_strategy_metrics(is_fraud, replay_df["baseline_action"])
    adjusted = compute_strategy_metrics(is_fraud, replay_df["adjusted_action"])
    flips = compute_escalation_flip_precision(replay_df)

    lines = [
        "Escalation ablation study",
        "==========================",
        "",
        f"Test set: {len(replay_df):,} transactions "
        f"({baseline['n_fraud']:,} fraud, {baseline['n_legit']:,} legit), "
        "replayed in chronological order through a fresh EntityRiskMemory "
        "(same order/state a live deployment would see).",
        "",
        "=== Baseline: raw model score only, no entity escalation ===",
        f"Recall (frauds flagged REVIEW/BLOCK):    {baseline['recall']:.4f} "
        f"({baseline['flagged_fraud']:,} / {baseline['n_fraud']:,})",
        f"False-flag rate (legit txns flagged):    {baseline['false_flag_rate']:.4f} "
        f"({baseline['flagged_legit']:,} / {baseline['n_legit']:,})",
        "",
        "=== Escalation-adjusted: what the live system does today ===",
        f"Recall (frauds flagged REVIEW/BLOCK):    {adjusted['recall']:.4f} "
        f"({adjusted['flagged_fraud']:,} / {adjusted['n_fraud']:,})",
        f"False-flag rate (legit txns flagged):    {adjusted['false_flag_rate']:.4f} "
        f"({adjusted['flagged_legit']:,} / {adjusted['n_legit']:,})",
        "",
        f"Recall delta from escalation:  {adjusted['recall'] - baseline['recall']:+.4f}",
        f"False-flag-rate delta:         {adjusted['false_flag_rate'] - baseline['false_flag_rate']:+.4f}",
        "",
        "=== Precision of escalation-triggered flips ===",
        "Of the transactions where entity escalation history actually "
        "pushed the action higher than the raw score alone would have "
        "(escalated_due_to_history == True):",
        f"  Flip count:  {flips['n_flips']:,}",
        f"  Of those, actually fraud:  {flips['n_flips_fraud']:,}",
        f"  Precision of escalation-triggered flips:  {flips['precision']:.4f}",
        "",
    ]
    return "\n".join(lines)


def run():
    print("Loading and feature-engineering the full dataset...")
    test_df = load_test_set()
    print(f"Test set: {len(test_df):,} transactions (chronological, matches train_model.py's split)")

    print("Scoring test set with the trained model...")
    risk_scores = score_test_set(test_df)

    print("Replaying test set through EntityRiskMemory in time order...")
    replay_df = replay(test_df, risk_scores)

    report = build_report(replay_df)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    run()
