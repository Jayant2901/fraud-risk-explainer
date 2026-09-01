"""
Deterministic action rules — shared by the live API (api/main.py) and
offline analyses (e.g. src/escalation_ablation.py) so both always apply
the exact same thresholds and can never silently drift apart.
"""


def decide_action(risk_score: float, escalation: dict) -> dict:
    """Deterministic rule lookup — mirrors the thresholds the LLM agent's
    system prompt (src/llm_agent.py) is instructed to follow. This function
    itself is sub-millisecond; combined with the model score + SHAP call
    ahead of it (~100-130ms measured locally on CPU), the full decision
    path still lands orders of magnitude faster than the Gemini API
    round-trip it deliberately does not wait for.

    A real-time authorization path can't wait on an LLM call for that —
    the decision that actually gates the transaction has to return fast.
    The LLM is used afterward, asynchronously, only to produce a
    human-readable explanation for the reviewer queue; it never gets to
    change the action.
    """
    elevated = escalation.get("state") == "ELEVATED"
    if risk_score >= 80:
        action = "BLOCK"
        escalated = False
    elif risk_score >= 40:
        action = "BLOCK" if elevated else "REVIEW"
        escalated = elevated
    else:
        action = "REVIEW" if elevated else "ALLOW"
        escalated = elevated
    return {"action": action, "escalated_due_to_history": escalated}
