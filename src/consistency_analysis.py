"""
Reviewer consistency analysis — does this system agree with itself?

Grounded in Razorpay's own stated pain point: ~12,000 merchant risk
review cases/month, ~20 minutes each, and different human analysts
reaching different conclusions on the identical case (see the README
section this feeds for the source posts). This project has no human
reviewers to test that with, but it does have an LLM agent
(src/llm_agent.py) that plays an equivalent role — recommending an
action for a case, the way a human reviewer would. Unlike a human, you
can cheaply ask it to look at the exact same case multiple times, which
gives a real, directly analogous measurement of the same problem:
case-level judgment consistency.

Two parts:
  A. Boundary fragility (free, full test set, no API calls): of the
     transactions the deterministic rules would flag REVIEW/BLOCK, what
     fraction sit within +/-2 points of a decision boundary (the real,
     cost-derived review/block thresholds — see decision_rules.py)
     — a cheap, statistically solid proxy for "how many of these
     decisions were close calls."
  B. LLM self-consistency and cross-agreement (COSTS REAL API QUOTA):
     for a small, deliberate 12-transaction x 2-escalation-context
     sample (24 pairs), calls RiskExplainerAgent.explain() 5 times each
     (up to 120 real Gemini calls, sequential with a delay between
     calls), and measures how often the LLM agrees with itself and with
     the deterministic decision. Fallback responses (rate-limited,
     unauthenticated, etc.) are detected and excluded from the count —
     see is_fallback_response() — since a rate-limit artifact would
     otherwise look identical to genuine model indecision.

Run (Part B makes real, quota-consuming Gemini API calls — needs a real
GEMINI_API_KEY; this is a manual, occasional analysis script, same as
train_model.py — it does NOT run in CI):
    python src/consistency_analysis.py

Output:
    models/consistency_report.txt
    models/consistency_report.json
"""
import sys
import os
import json
import math
import time
from collections import Counter

sys.path.append(os.path.dirname(__file__))

import joblib

from escalation_ablation import load_test_set
from decision_rules import decide_action, load_decision_thresholds
from entity_memory import _compute_escalation_state, DEFAULT_ELEVATED_PRESSURE_THRESHOLD, VERDICT_WEIGHT
from risk_explainer import RiskExplainer
from llm_agent import RiskExplainerAgent

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"

TEXT_REPORT_PATH = "models/consistency_report.txt"
JSON_REPORT_PATH = "models/consistency_report.json"

# --- Part A ---
# Default boundaries for is_near_boundary()/compute_boundary_fragility()
# when the caller doesn't pass real ones in — matches
# decision_rules.py's own 40.0/80.0 fallback. run() below always passes
# the real, loaded thresholds explicitly instead of relying on this
# default, so this exists purely as a documented, testable fallback (and
# what the unit tests exercise), not the operational source of truth.
BOUNDARIES = (40.0, 80.0)
BOUNDARY_TOLERANCE = 2.0

# --- Part B ---
# Half-width of the "near a boundary" sampling window, in risk-score
# points either side of the real review/block threshold.
BOUNDARY_BAND_HALF_WIDTH = 5.0


def score_bands(boundaries=BOUNDARIES) -> list:
    """Named (band_name, lo, hi) score ranges to sample from. The two
    boundary bands are centered on the REAL, cost-derived thresholds
    (not hardcoded 40/80) so this always samples close calls relative to
    what the live system actually decides with."""
    review, block = boundaries
    return [
        ("clear_allow", 0.0, 20.0),
        ("near_review_boundary", review - BOUNDARY_BAND_HALF_WIDTH, review + BOUNDARY_BAND_HALF_WIDTH),
        ("near_block_boundary", block - BOUNDARY_BAND_HALF_WIDTH, block + BOUNDARY_BAND_HALF_WIDTH),
        ("clear_block", 90.0, 100.0),
    ]


SAMPLES_PER_BAND = 3
CALLS_PER_PAIR = 5
CALL_DELAY_SECONDS = 4.5  # sequential, well under the free-tier rate limit
MIN_VALID_RESPONSES = 2
FALLBACK_MARKER = "Falling back to manual review"


# ============================================================ Part A ====

def is_near_boundary(risk_score: float, boundaries=BOUNDARIES, tolerance: float = BOUNDARY_TOLERANCE) -> bool:
    return any(abs(risk_score - b) <= tolerance for b in boundaries)


def compute_boundary_fragility(flagged_risk_scores: list, boundaries=BOUNDARIES) -> dict:
    """flagged_risk_scores: risk scores of transactions the deterministic,
    no-escalation decision already flagged REVIEW/BLOCK."""
    n = len(flagged_risk_scores)
    n_near = sum(1 for s in flagged_risk_scores if is_near_boundary(s, boundaries=boundaries))
    return {
        "n_flagged": int(n),
        "n_near_boundary": int(n_near),
        "fraction_near_boundary": float(round(n_near / n, 4)) if n else 0.0,
    }


# ============================================================ Part B ====

def is_fallback_response(verdict: dict) -> bool:
    """A fallback response (rate limit, auth failure, network error,
    invalid schema — see llm_agent._fallback_response) is NOT a real
    model opinion; every fallback rationale carries this distinctive
    marker string."""
    return FALLBACK_MARKER in verdict.get("rationale", "")


def modal_action(actions: list) -> str:
    """Most common action; ties broken by first-appearance order in the
    input list, so the result is deterministic given the same call
    order (rather than dict/Counter iteration order, which isn't
    guaranteed to match)."""
    counts = Counter(actions)
    best_count = max(counts.values())
    for a in actions:
        if counts[a] == best_count:
            return a
    raise ValueError("modal_action() called with an empty list")  # pragma: no cover


def aggregate_pair(verdicts: list, deterministic_action: str, min_valid: int = MIN_VALID_RESPONSES) -> dict:
    """One (transaction, escalation context) pair's worth of repeated
    explain() calls -> modal action, self-consistency rate, and whether
    that modal action agrees with decide_action()'s real, deterministic
    output for the same risk_score/escalation. Fallback responses are
    excluded from everything; if too few real responses remain, reports
    "insufficient_data" rather than fabricating a number from 0-1 valid
    responses."""
    n_calls = len(verdicts)
    valid = [v for v in verdicts if not is_fallback_response(v)]
    n_excluded = n_calls - len(valid)

    if len(valid) < min_valid:
        return {
            "status": "insufficient_data",
            "n_calls": n_calls,
            "n_excluded_fallback": n_excluded,
            "n_valid": len(valid),
            "modal_action": None,
            "self_consistency_rate": None,
            "cross_agreement": None,
        }

    actions = [v["action"] for v in valid]
    modal = modal_action(actions)
    consistency_rate = actions.count(modal) / len(actions)

    return {
        "status": "ok",
        "n_calls": n_calls,
        "n_excluded_fallback": n_excluded,
        "n_valid": len(valid),
        "modal_action": modal,
        "self_consistency_rate": float(round(consistency_rate, 4)),
        "cross_agreement": bool(modal == deterministic_action),
    }


def normal_escalation(entity_id: str) -> dict:
    return _compute_escalation_state(entity_id, [])


def elevated_escalation(entity_id: str, risk_score: float) -> dict:
    """A constructed ELEVATED state — exactly the "should history push
    this higher" judgment call the system prompt itself calls out as
    ambiguous, so it's tested deliberately rather than left to chance
    (most real sampled transactions' actual entities will be NORMAL).

    Uses BLOCK verdicts at risk_score=100 (max signal) for the
    synthetic history, not this pair's own (possibly very low) score —
    with the severity-weighted pressure formula (see entity_memory.py),
    reusing a low current-transaction score for the history too could
    fail to reach ELEVATED at all. enough_blocks is computed from the
    real, grid-chosen ELEVATED cutoff so this stays correct if that
    cutoff is ever re-tuned."""
    per_verdict_pressure = VERDICT_WEIGHT["BLOCK"] * 1.0
    enough_blocks = math.ceil(DEFAULT_ELEVATED_PRESSURE_THRESHOLD / per_verdict_pressure)
    history = [{"verdict": "BLOCK", "risk_score": 100.0} for _ in range(enough_blocks)]
    return _compute_escalation_state(entity_id, history)


def select_sample(test_df, boundaries=BOUNDARIES, samples_per_band: int = SAMPLES_PER_BAND) -> list:
    """Deterministic (dataset-order-based, no randomness) selection of
    samples_per_band real transactions per score band, so a re-run picks
    the identical sample."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    risk_scores = proba * 100

    picks = []
    for band_name, lo, hi in score_bands(boundaries):
        mask = (risk_scores >= lo) & (risk_scores <= hi)
        row_indices = test_df.index[mask][:samples_per_band]
        for i in row_indices:
            picks.append({"band": band_name, "row_index": int(i)})
    return picks


def build_report(part_a: dict, pair_results: list, boundaries=BOUNDARIES, bands=None) -> str:
    if bands is None:
        bands = score_bands(boundaries)
    lines = [
        "Reviewer consistency analysis",
        "===============================",
        "",
        "Part A -- Boundary fragility (deterministic rules, full test set, no API calls)",
        "----------------------------------------------------------------------------------",
        f"Flagged (REVIEW/BLOCK) transactions: {part_a['n_flagged']:,}",
        f"Within +/-{BOUNDARY_TOLERANCE:.0f} points of a decision boundary "
        f"({boundaries[0]:.1f} or {boundaries[1]:.1f}): "
        f"{part_a['n_near_boundary']:,} ({part_a['fraction_near_boundary']:.2%})",
        "",
        "Part B -- LLM self-consistency and cross-agreement (real Gemini API calls)",
        "----------------------------------------------------------------------------",
        f"{CALLS_PER_PAIR} calls per (transaction, escalation context) pair, "
        f"{len(pair_results)} pairs, up to {len(pair_results) * CALLS_PER_PAIR} total API calls.",
        "",
        f"{'band':<22}{'context':<10}{'risk':>7}  {'status':<18}{'valid':>6}{'excl':>6}"
        f"{'modal':>8}{'self-cons.':>11}{'cross-agree':>13}",
    ]
    for r in pair_results:
        modal_str = r["modal_action"] or "-"
        cons_str = "-" if r["self_consistency_rate"] is None else f"{r['self_consistency_rate']:.2f}"
        cross_str = "-" if r["cross_agreement"] is None else str(r["cross_agreement"])
        lines.append(
            f"{r['band']:<22}{r['escalation_context']:<10}{r['risk_score']:>7.1f}  {r['status']:<18}"
            f"{r['n_valid']:>6}{r['n_excluded_fallback']:>6}"
            f"{modal_str:>8}{cons_str:>11}{cross_str:>13}"
        )

    ok_pairs = [r for r in pair_results if r["status"] == "ok"]
    lines.append("")
    if ok_pairs:
        overall_consistency = sum(r["self_consistency_rate"] for r in ok_pairs) / len(ok_pairs)
        overall_cross = sum(1 for r in ok_pairs if r["cross_agreement"]) / len(ok_pairs)
        lines.append(
            f"Overall mean self-consistency rate ({len(ok_pairs)} pairs with sufficient data): "
            f"{overall_consistency:.4f}"
        )
        lines.append(f"Overall cross-agreement rate: {overall_cross:.4f}")
        lines.append("")

        for band_name, _, _ in bands:
            band_pairs = [r for r in ok_pairs if r["band"] == band_name]
            if band_pairs:
                bc = sum(r["self_consistency_rate"] for r in band_pairs) / len(band_pairs)
                lines.append(f"  band={band_name}: mean self-consistency {bc:.4f} ({len(band_pairs)} pairs)")

        for context_name in ("NORMAL", "ELEVATED"):
            ctx_pairs = [r for r in ok_pairs if r["escalation_context"] == context_name]
            if ctx_pairs:
                cc = sum(r["self_consistency_rate"] for r in ctx_pairs) / len(ctx_pairs)
                lines.append(f"  escalation={context_name}: mean self-consistency {cc:.4f} ({len(ctx_pairs)} pairs)")
    else:
        lines.append("No pairs had sufficient valid (non-fallback) data to compute consistency.")

    n_insufficient = len(pair_results) - len(ok_pairs)
    if n_insufficient:
        lines.append(
            f"\n{n_insufficient} of {len(pair_results)} pairs had insufficient valid data "
            f"(< {MIN_VALID_RESPONSES} non-fallback responses) and are excluded from the aggregates above."
        )

    return "\n".join(lines)


def run():
    print("Loading test set and trained model...")
    test_df = load_test_set()

    thresholds = load_decision_thresholds()
    boundaries = (thresholds["review"], thresholds["block"])
    print(f"Using decision thresholds: {thresholds}")

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    risk_scores = proba * 100

    print("=== Part A: boundary fragility (no API calls) ===")
    baseline_actions = [
        decide_action(float(s), {"state": "NORMAL"}, thresholds["review"], thresholds["block"])["action"]
        for s in risk_scores
    ]
    flagged_scores = [float(s) for s, a in zip(risk_scores, baseline_actions) if a != "ALLOW"]
    part_a = compute_boundary_fragility(flagged_scores, boundaries=boundaries)
    print(f"Part A: {part_a}")

    print("=== Part B: LLM self-consistency (REAL Gemini API calls) ===")
    bands = score_bands(boundaries)
    picks = select_sample(test_df, boundaries)
    print(f"Selected {len(picks)} sample transactions across {len(bands)} score bands")

    explainer = RiskExplainer()
    agent = RiskExplainerAgent(review_threshold=thresholds["review"], block_threshold=thresholds["block"])

    pair_results = []
    for pick in picks:
        row = test_df.iloc[pick["row_index"]]
        txn = row.to_dict()
        scored = explainer.score_transaction(txn)
        risk_score = scored["risk_score"]
        top_factors = scored["top_factors"]
        entity_id = row["entity_id"]

        contexts = [
            ("NORMAL", normal_escalation(entity_id)),
            ("ELEVATED", elevated_escalation(entity_id, risk_score)),
        ]
        for context_name, escalation in contexts:
            deterministic_action = decide_action(
                risk_score, escalation, thresholds["review"], thresholds["block"]
            )["action"]

            verdicts = []
            for call_i in range(CALLS_PER_PAIR):
                verdict = agent.explain(risk_score, top_factors, escalation)
                verdicts.append(verdict)
                time.sleep(CALL_DELAY_SECONDS)

            result = aggregate_pair(verdicts, deterministic_action)
            result.update({
                "band": pick["band"],
                "row_index": pick["row_index"],
                "risk_score": risk_score,
                "escalation_context": context_name,
                "deterministic_action": deterministic_action,
            })
            pair_results.append(result)
            print(
                f"  {pick['band']:<22} {context_name:<8} risk={risk_score:6.1f} -> "
                f"{result['status']} modal={result['modal_action']} "
                f"consistency={result['self_consistency_rate']} "
                f"cross_agree={result['cross_agreement']} excluded={result['n_excluded_fallback']}"
            )

    report = build_report(part_a, pair_results, boundaries=boundaries, bands=bands)
    print(report)

    with open(TEXT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {TEXT_REPORT_PATH}")

    json_payload = {"part_a_boundary_fragility": part_a, "part_b_pairs": pair_results}
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved -> {JSON_REPORT_PATH}")


if __name__ == "__main__":
    run()
