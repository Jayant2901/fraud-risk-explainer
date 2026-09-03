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

The state is derived from a continuous, SEVERITY-WEIGHTED "risk
pressure" value rather than a raw count of risky verdicts (see
VERDICT_WEIGHT/_risk_pressure below) — a BLOCK contributes more than a
REVIEW, and a high-scoring REVIEW contributes more than a borderline
one, instead of every risky verdict counting identically. The
WATCH/ELEVATED cutoffs against that pressure value were chosen by a
real grid sweep in src/escalation_ablation.py (sweep_pressure_thresholds()),
picking whichever candidate minimizes the same cost formula
train_model.py's own threshold derivation uses (see DEFAULT_WATCH_
PRESSURE_THRESHOLD/DEFAULT_ELEVATED_PRESSURE_THRESHOLD below for the
exact grid and result) — not hand-picked to look good.

Two implementations, same behavioral contract (record_verdict /
get_escalation_state / reset), sharing the threshold math in
_compute_escalation_state so they can never silently disagree:

  - EntityRiskMemory: in-process deque, session-scoped (resets when the
    process restarts). The default — no setup required, which matters
    for local dev/demo.
  - RedisEntityRiskMemory: same rolling window, stored as a Redis list
    per entity (LPUSH + LTRIM), with a TTL. Survives restarts and is
    shared across workers. Used automatically when REDIS_URL is set —
    see create_entity_memory().
"""
import json
from collections import defaultdict, deque

WINDOW_SIZE = 10           # how many recent verdicts we remember per entity

RISKY_VERDICTS = {"REVIEW", "BLOCK"}

# A BLOCK is a stronger risk signal than a REVIEW — weight it higher —
# and within each tier, a verdict scored 95 says more than one scored 41
# that happened to cross the same review_threshold, so each verdict's
# own risk_score (normalized to 0-1) scales its contribution too. ALLOW
# verdicts contribute nothing (not in this dict -> weight 0).
VERDICT_WEIGHT = {"BLOCK": 2.0, "REVIEW": 1.0}

# Chosen by the real grid sweep in escalation_ablation.sweep_pressure_
# thresholds() — WATCH_PRESSURE_CANDIDATES x ELEVATED_PRESSURE_CANDIDATES,
# picking whichever (watch, elevated) pair minimizes total cost
# (false_negatives * avg_fraud_loss + false_positives * avg_fp_cost,
# cost_analysis.py's own default assumptions) over the full real
# chronological test set — see models/escalation_ablation_report.txt
# for the actual grid results this was picked from.
#
# A real finding from that sweep worth being explicit about: every
# WATCH candidate produced IDENTICAL cost/recall/false-flag numbers for
# a given ELEVATED candidate. That's not a sweep bug — decide_action()
# (src/decision_rules.py) only branches on escalation.get("state") ==
# "ELEVATED"; WATCH never changes the deterministic action, only the
# state label surfaced to the reviewer/LLM as an earlier informational
# heads-up. So DEFAULT_WATCH_PRESSURE_THRESHOLD only controls how early
# that label appears, not any decision — DEFAULT_ELEVATED_PRESSURE_
# THRESHOLD is the only cutoff the cost sweep could actually optimize.
DEFAULT_WATCH_PRESSURE_THRESHOLD = 0.8
DEFAULT_ELEVATED_PRESSURE_THRESHOLD = 3.6

# How long an entity's history survives with no new verdicts, for the
# Redis-backed implementation. Refreshed on every write, so an active
# entity never loses history — only a genuinely inactive one ages out.
REDIS_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = "riskmgr:entity_history"


def _risk_pressure(history: list[dict]) -> float:
    """Continuous severity-weighted signal: sum of each verdict's
    VERDICT_WEIGHT scaled by its own risk_score (0-1). E.g. one BLOCK at
    risk_score=90 contributes 2.0 * 0.90 = 1.8; one REVIEW at
    risk_score=50 contributes 1.0 * 0.50 = 0.5 — a hand-computable
    example exercised directly in test_entity_memory.py."""
    return sum(VERDICT_WEIGHT.get(h["verdict"], 0.0) * (h["risk_score"] / 100.0) for h in history)


def _compute_escalation_state(
    entity_id: str,
    history: list[dict],
    watch_threshold: float = DEFAULT_WATCH_PRESSURE_THRESHOLD,
    elevated_threshold: float = DEFAULT_ELEVATED_PRESSURE_THRESHOLD,
) -> dict:
    """history: list of {"verdict": str, "risk_score": float}, oldest
    first — the shared math both EntityRiskMemory and
    RedisEntityRiskMemory build their get_escalation_state() on.

    watch_threshold/elevated_threshold: cutoffs against _risk_pressure()
    (not a raw verdict count). Callers should rely on the module
    defaults, which are the real, grid-chosen cutoffs — these parameters
    exist so escalation_ablation.sweep_pressure_thresholds() can try
    other candidates without a real EntityRiskMemory per candidate."""
    risky_count = sum(1 for h in history if h["verdict"] in RISKY_VERDICTS)
    pressure = _risk_pressure(history)

    if pressure >= elevated_threshold:
        state = "ELEVATED"
    elif pressure >= watch_threshold:
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
        "risk_pressure": round(pressure, 3),
        "avg_recent_risk_score": round(avg_recent_score, 1),
        "recent_verdicts": [h["verdict"] for h in history],
    }


class EntityRiskMemory:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._history = defaultdict(lambda: deque(maxlen=window_size))

    def record_verdict(self, entity_id: str, verdict: str, risk_score: float):
        self._history[entity_id].append({"verdict": verdict, "risk_score": risk_score})

    def get_escalation_state(self, entity_id: str) -> dict:
        return _compute_escalation_state(entity_id, list(self._history[entity_id]))

    def reset(self, entity_id: str | None = None):
        if entity_id is None:
            self._history.clear()
        else:
            self._history.pop(entity_id, None)


class RedisEntityRiskMemory:
    """Same contract as EntityRiskMemory, backed by a Redis list per
    entity instead of an in-process deque. `redis_client` must be a
    redis.Redis (or fakeredis.FakeRedis for tests) constructed with
    decode_responses=True."""

    def __init__(self, redis_client, window_size: int = WINDOW_SIZE):
        self._redis = redis_client
        self.window_size = window_size

    def _key(self, entity_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}:{entity_id}"

    def record_verdict(self, entity_id: str, verdict: str, risk_score: float):
        key = self._key(entity_id)
        entry = json.dumps({"verdict": verdict, "risk_score": risk_score})
        pipe = self._redis.pipeline()
        pipe.lpush(key, entry)                       # newest at the head
        pipe.ltrim(key, 0, self.window_size - 1)      # keep only the window
        pipe.expire(key, REDIS_TTL_SECONDS)           # refresh TTL on activity
        pipe.execute()

    def get_escalation_state(self, entity_id: str) -> dict:
        raw = self._redis.lrange(self._key(entity_id), 0, -1)
        # Redis list is newest-first (LPUSH); flip to oldest-first so
        # this matches EntityRiskMemory's deque iteration order exactly
        # (recent_verdicts should read oldest -> newest either way).
        history = [json.loads(item) for item in reversed(raw)]
        return _compute_escalation_state(entity_id, history)

    def reset(self, entity_id: str | None = None):
        if entity_id is None:
            for key in self._redis.scan_iter(match=f"{REDIS_KEY_PREFIX}:*"):
                self._redis.delete(key)
        else:
            self._redis.delete(self._key(entity_id))


def create_entity_memory(redis_client=None, window_size: int = WINDOW_SIZE):
    """Factory used by api/main.py: Redis-backed if a client is given
    (i.e. REDIS_URL was configured), the original in-process behavior
    otherwise — so today's default (no Redis) is byte-for-byte unchanged."""
    if redis_client is not None:
        return RedisEntityRiskMemory(redis_client, window_size=window_size)
    return EntityRiskMemory(window_size=window_size)
