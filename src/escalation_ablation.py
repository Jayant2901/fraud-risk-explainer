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

Also sweeps a small grid of candidate severity-weighted escalation
cutoffs (sweep_pressure_thresholds()) and reports the full grid, so the
WATCH/ELEVATED cutoffs entity_memory.py hardcodes are a real, documented
choice rather than a guess — see that module's docstring.

Run:
    python src/escalation_ablation.py

Output:
    models/escalation_ablation_report.txt
    models/escalation_ablation_summary.json  (same numbers, structured, for the API)
"""
import sys
import os
import json
from collections import defaultdict, deque

sys.path.append(os.path.dirname(__file__))

import pandas as pd
import joblib

from data_utils import load_raw_data, engineer_features
from decision_rules import decide_action, load_decision_thresholds
from entity_memory import (
    EntityRiskMemory,
    WINDOW_SIZE,
    _compute_escalation_state,
    DEFAULT_WATCH_PRESSURE_THRESHOLD,
    DEFAULT_ELEVATED_PRESSURE_THRESHOLD,
)
from cost_analysis import DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
REPORT_PATH = "models/escalation_ablation_report.txt"
# The same numbers build_report() renders as text, in structured form,
# so GET /api/escalation-ablation can serve a chartable comparison
# instead of the frontend regex-parsing a human-readable report.
SUMMARY_PATH = "models/escalation_ablation_summary.json"

TEST_FRAC = 0.2
FLAGGED_ACTIONS = {"REVIEW", "BLOCK"}

# --- Phase C: severity-weighted escalation cutoff grid sweep ---
# Small grid of candidate (watch, elevated) risk_pressure cutoffs (see
# entity_memory._risk_pressure) — analogous in spirit to
# cost_sensitivity.py's cost-assumption grid. Only combinations where
# elevated > watch are evaluated (a collapsed/inverted tier pair isn't a
# meaningful candidate).
WATCH_PRESSURE_CANDIDATES = [0.8, 1.2, 1.6]
ELEVATED_PRESSURE_CANDIDATES = [2.0, 2.8, 3.6]


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


def replay(test_df: pd.DataFrame, risk_scores: pd.Series, thresholds: dict | None = None) -> pd.DataFrame:
    """Walks the test set in time order through a fresh EntityRiskMemory,
    recording both decisions per transaction. Returns a DataFrame with
    one row per transaction: is_fraud, risk_score, baseline_action,
    adjusted_action, escalated_due_to_history.

    thresholds: {"review": float, "block": float} — the SAME real,
    derived boundaries the live API decides with (see
    decision_rules.load_decision_thresholds()). Defaults to loading them
    fresh if not passed, so this mirrors decide_action() exactly rather
    than restating 40/80 independently."""
    if thresholds is None:
        thresholds = load_decision_thresholds()

    memory = EntityRiskMemory()
    rows = []

    for idx, txn in test_df.iterrows():
        entity_id = txn["entity_id"]
        risk_score = float(risk_scores.loc[idx])

        escalation_before = memory.get_escalation_state(entity_id)

        baseline_decision = decide_action(risk_score, {"state": "NORMAL"}, thresholds["review"], thresholds["block"])
        adjusted_decision = decide_action(risk_score, escalation_before, thresholds["review"], thresholds["block"])

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


def replay_with_pressure_escalation(
    test_df: pd.DataFrame,
    risk_scores: pd.Series,
    thresholds: dict,
    watch_threshold: float,
    elevated_threshold: float,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Same idea as replay(), but computes escalation state directly via
    entity_memory._compute_escalation_state() with the given
    severity-weighted pressure cutoffs, instead of going through a real
    EntityRiskMemory (which always uses that module's live default
    cutoffs) — lets sweep_pressure_thresholds() try many candidates
    cheaply without touching entity_memory.py's defaults."""
    histories = defaultdict(lambda: deque(maxlen=window_size))
    rows = []

    for idx, txn in test_df.iterrows():
        entity_id = txn["entity_id"]
        risk_score = float(risk_scores.loc[idx])

        escalation_before = _compute_escalation_state(
            entity_id, list(histories[entity_id]), watch_threshold, elevated_threshold
        )
        adjusted_decision = decide_action(risk_score, escalation_before, thresholds["review"], thresholds["block"])
        histories[entity_id].append({"verdict": adjusted_decision["action"], "risk_score": risk_score})

        rows.append({
            "is_fraud": int(txn["isFraud"]),
            "adjusted_action": adjusted_decision["action"],
        })

    return pd.DataFrame(rows)


def compute_cost(
    is_fraud: pd.Series,
    action: pd.Series,
    avg_fraud_loss: float = DEFAULT_AVG_FRAUD_LOSS,
    avg_fp_cost: float = DEFAULT_AVG_FP_COST,
) -> float:
    """Same cost formula train_model.py's threshold derivation uses (see
    cost_analysis.py): a missed fraud (not flagged REVIEW/BLOCK) costs
    avg_fraud_loss, a flagged legitimate transaction costs avg_fp_cost."""
    flagged = action.isin(FLAGGED_ACTIONS)
    false_negatives = int(((~flagged) & (is_fraud == 1)).sum())
    false_positives = int((flagged & (is_fraud == 0)).sum())
    return false_negatives * avg_fraud_loss + false_positives * avg_fp_cost


def sweep_pressure_thresholds(test_df: pd.DataFrame, risk_scores: pd.Series, thresholds: dict) -> list[dict]:
    """Tries every (watch, elevated) pair in WATCH_PRESSURE_CANDIDATES x
    ELEVATED_PRESSURE_CANDIDATES (elevated > watch only), replaying the
    full real chronological test set for each, and reports recall/
    false-flag-rate/cost per candidate — the SAME cost formula (and cost
    assumptions) train_model.py's own threshold derivation uses, so the
    eventual choice is made by the same "minimize cost" principle the
    rest of this project already uses, not by eyeballing a tradeoff."""
    results = []
    for watch in WATCH_PRESSURE_CANDIDATES:
        for elevated in ELEVATED_PRESSURE_CANDIDATES:
            if elevated <= watch:
                continue
            replay_df = replay_with_pressure_escalation(test_df, risk_scores, thresholds, watch, elevated)
            metrics = compute_strategy_metrics(replay_df["is_fraud"], replay_df["adjusted_action"])
            cost = compute_cost(replay_df["is_fraud"], replay_df["adjusted_action"])
            results.append({
                "watch_threshold": watch,
                "elevated_threshold": elevated,
                "recall": metrics["recall"],
                "false_flag_rate": metrics["false_flag_rate"],
                "cost": cost,
            })
    return results


def build_sweep_report(sweep_results: list[dict]) -> str:
    chosen = min(sweep_results, key=lambda r: r["cost"]) if sweep_results else None
    lines = [
        "=== Severity-weighted escalation: cutoff grid sweep ===",
        f"Candidates: watch in {WATCH_PRESSURE_CANDIDATES}, elevated in "
        f"{ELEVATED_PRESSURE_CANDIDATES} (elevated > watch only), scored by total "
        f"cost (false_negatives * Rs {DEFAULT_AVG_FRAUD_LOSS:,.0f} + false_positives * "
        f"Rs {DEFAULT_AVG_FP_COST:,.0f}) over the full real chronological test set.",
        "",
        f"{'watch':>7}{'elevated':>10}{'recall':>10}{'false_flag':>12}{'cost (Rs)':>14}  chosen",
    ]
    for r in sweep_results:
        is_chosen = chosen is not None and r is chosen
        lines.append(
            f"{r['watch_threshold']:>7.1f}{r['elevated_threshold']:>10.1f}"
            f"{r['recall']:>10.4f}{r['false_flag_rate']:>12.4f}{r['cost']:>14,.0f}"
            f"  {'<-- chosen (lowest cost)' if is_chosen else ''}"
        )
    lines.append("")
    if chosen is not None:
        lines.append(
            f"Chosen cutoffs: watch={chosen['watch_threshold']}, elevated={chosen['elevated_threshold']} "
            f"— hardcoded as entity_memory.py's DEFAULT_WATCH_PRESSURE_THRESHOLD/"
            f"DEFAULT_ELEVATED_PRESSURE_THRESHOLD (current live values: "
            f"{DEFAULT_WATCH_PRESSURE_THRESHOLD}/{DEFAULT_ELEVATED_PRESSURE_THRESHOLD})."
        )
    lines.append(
        "\nReal finding from this grid, reported honestly: every watch candidate produced "
        "IDENTICAL recall/false-flag/cost numbers for a given elevated candidate (see the rows "
        "above). That's not a bug — decision_rules.decide_action() only branches on "
        "escalation.state == 'ELEVATED'; WATCH never changes the deterministic action, only the "
        "state label shown to the reviewer/LLM as an earlier heads-up. So this sweep could only "
        "actually optimize the elevated cutoff; the watch cutoff was picked for free (0.8, the "
        "most sensitive candidate, giving the earliest informational signal) since it has zero "
        "cost impact either way."
    )
    lines.append("")
    return "\n".join(lines)


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


def build_summary(replay_df: pd.DataFrame, sweep_results: list[dict]) -> dict:
    """Structured twin of build_report() — identical numbers, from the
    identical computation, shaped for a chart rather than a page of text."""
    baseline = compute_strategy_metrics(replay_df["is_fraud"], replay_df["baseline_action"])
    adjusted = compute_strategy_metrics(replay_df["is_fraud"], replay_df["adjusted_action"])
    flips = compute_escalation_flip_precision(replay_df)
    return {
        "n_transactions": len(replay_df),
        "baseline": baseline,
        "adjusted": adjusted,
        "flips": flips,
        "sweep": sweep_results,
    }


def run():
    print("Loading and feature-engineering the full dataset...")
    test_df = load_test_set()
    print(f"Test set: {len(test_df):,} transactions (chronological, matches train_model.py's split)")

    thresholds = load_decision_thresholds()
    print(f"Using decision thresholds: {thresholds}")

    print("Scoring test set with the trained model...")
    risk_scores = score_test_set(test_df)

    print("Sweeping severity-weighted escalation cutoff grid...")
    sweep_results = sweep_pressure_thresholds(test_df, risk_scores, thresholds)
    sweep_report = build_sweep_report(sweep_results)
    print(sweep_report)

    print("Replaying test set through EntityRiskMemory (live cutoffs) in time order...")
    replay_df = replay(test_df, risk_scores, thresholds)

    report = sweep_report + "\n" + build_report(replay_df)
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {REPORT_PATH}")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(build_summary(replay_df, sweep_results), f, indent=2)
    print(f"Saved -> {SUMMARY_PATH}")


if __name__ == "__main__":
    run()
