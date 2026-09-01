"""
Entity Risk Memory — the "agentic" piece of the system.

A static classifier scores every transaction as if the entity behind it
has no history. Real risk teams don't work that way: an entity (card/
account fingerprint) that's had several REVIEW/BLOCK verdicts recently
gets watched more closely on its NEXT transaction, even if that next
transaction looks borderline on its own.

This module maintains, per entity_id, a rolling window of recent
verdicts and derives an ESCALATION STATE: NORMAL, WATCH, or ELEVATED.
That state is then fed into the LLM agent's prompt so its recommended
action can be adjusted — e.g. a borderline REVIEW-level score from an
entity already in ELEVATED state may reasonably be escalated to BLOCK,
with the LLM asked to justify that escalation explicitly rather than
silently overriding the model.

This is in-memory and session-scoped (resets when the process restarts)
which is intentional for a demo — a production version would back this
with a real store (Redis/DB) keyed by entity_id with a TTL.
"""
from collections import defaultdict, deque

WINDOW_SIZE = 10           # how many recent verdicts we remember per entity
WATCH_THRESHOLD = 2        # >= this many REVIEW/BLOCK in window -> WATCH
ELEVATED_THRESHOLD = 4     # >= this many REVIEW/BLOCK in window -> ELEVATED

RISKY_VERDICTS = {"REVIEW", "BLOCK"}


class EntityRiskMemory:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._history = defaultdict(lambda: deque(maxlen=window_size))

    def record_verdict(self, entity_id: str, verdict: str, risk_score: float):
        self._history[entity_id].append({"verdict": verdict, "risk_score": risk_score})

    def get_escalation_state(self, entity_id: str) -> dict:
        history = list(self._history[entity_id])
        risky_count = sum(1 for h in history if h["verdict"] in RISKY_VERDICTS)

        if risky_count >= ELEVATED_THRESHOLD:
            state = "ELEVATED"
        elif risky_count >= WATCH_THRESHOLD:
            state = "WATCH"
        else:
            state = "NORMAL"

        avg_recent_score = (
            sum(h["risk_score"] for h in history) / len(history) if history else 0.0
        )

        return {
            "entity_id": entity_id,
            "state": state,
            "recent_verdict_count": len(history),
            "recent_risky_count": risky_count,
            "avg_recent_risk_score": round(avg_recent_score, 1),
            "recent_verdicts": [h["verdict"] for h in history],
        }

    def reset(self, entity_id: str | None = None):
        if entity_id is None:
            self._history.clear()
        else:
            self._history.pop(entity_id, None)
