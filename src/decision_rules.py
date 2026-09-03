"""
Deterministic action rules — shared by the live API (api/main.py) and
offline analyses (e.g. src/escalation_ablation.py) so both always apply
the exact same thresholds and can never silently drift apart.

The two thresholds decide_action() uses are themselves DERIVED from the
real cost analysis (src/cost_analysis.py) by train_model.py, not chosen
independently — see train_model.py's module docstring for exactly how
review_threshold/block_threshold are computed and saved to
models/decision_thresholds.joblib. load_decision_thresholds() below is
the one place that file gets read, so every caller (the live API, the
offline ablation/consistency scripts, the LLM system prompt) is
guaranteed to see the same numbers. The 40.0/80.0 module defaults exist
ONLY as a fallback for when that file doesn't exist yet (e.g. before the
model has ever been trained) — once it exists, these are never used.
"""
import joblib

DECISION_THRESHOLDS_PATH = "models/decision_thresholds.joblib"
DEFAULT_REVIEW_THRESHOLD = 40.0
DEFAULT_BLOCK_THRESHOLD = 80.0


def load_decision_thresholds(path: str = DECISION_THRESHOLDS_PATH) -> dict:
    """Returns {"review": float, "block": float}. Falls back to the
    original hardcoded 40.0/80.0 ONLY if the file is missing — never a
    silent drift once train_model.py has produced a real one."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return {"review": DEFAULT_REVIEW_THRESHOLD, "block": DEFAULT_BLOCK_THRESHOLD}


def decide_action(
    risk_score: float,
    escalation: dict,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> dict:
    """Deterministic rule lookup — mirrors the thresholds the LLM agent's
    system prompt (src/llm_agent.py) is instructed to follow (both are
    built from the same review_threshold/block_threshold values — see
    llm_agent.build_system_prompt). This function itself is
    sub-millisecond; combined with the model score + SHAP call ahead of
    it (~100-130ms measured locally on CPU), the full decision path
    still lands orders of magnitude faster than the Gemini API
    round-trip it deliberately does not wait for.

    A real-time authorization path can't wait on an LLM call for that —
    the decision that actually gates the transaction has to return fast.
    The LLM is used afterward, asynchronously, only to produce a
    human-readable explanation for the reviewer queue; it never gets to
    change the action.

    review_threshold/block_threshold: callers should pass the real
    values from load_decision_thresholds() — the defaults here exist
    only so a direct unit test (or any caller that hasn't loaded real
    thresholds yet) still gets sane, documented behavior rather than a
    crash.
    """
    elevated = escalation.get("state") == "ELEVATED"
    if risk_score >= block_threshold:
        action = "BLOCK"
        escalated = False
    elif risk_score >= review_threshold:
        action = "BLOCK" if elevated else "REVIEW"
        escalated = elevated
    else:
        action = "REVIEW" if elevated else "ALLOW"
        escalated = elevated
    return {"action": action, "escalated_due_to_history": escalated}
