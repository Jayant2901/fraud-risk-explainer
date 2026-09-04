"""
The one scoring path.

Before this module existed, api/main.py's /api/score and /api/score-custom
each carried their own copy of "score the transaction, read escalation
state, decide, record the verdict, enqueue for review" — near-identical
but not identical. Phase 9 adds a second *process* (src/stream_consumer.py)
that scores transactions off a Redis Stream, and later phases hook online
features, escalation alerts, shadow scoring and the audit trail into
scoring. Every one of those has to see the same decision for the same
input, so the logic lives here once and every entry point calls it.

Deliberately free of FastAPI: this is called from request handlers and
from a plain worker process, so it takes its collaborators (explainer,
entity memory, review queue, explanation cache) as constructor arguments
rather than reaching for module-level singletons. api/main.py builds one
with the app's singletons; the consumer builds its own.
"""
import logging
import uuid
from datetime import datetime, timezone

from decision_rules import decide_action
from entity_memory import _compute_escalation_state
from feature_store import fingerprint_for

logger = logging.getLogger(__name__)

# Review-queue label for a custom transaction scored without being
# attached to any real entity — it has no entity of its own, but the
# queue item still needs a stable, non-null identifier.
UNATTACHED_ENTITY_LABEL = "custom-transaction"

FALLBACK_VERDICT = {
    "explanation": "An unexpected error occurred while generating the AI explanation.",
    "action": "REVIEW",
    "escalated_due_to_history": False,
    "rationale": "Falling back to manual review — explainer agent crashed.",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(txn: dict) -> dict:
    """Queue items are JSON-serialized into Redis, but a replayed
    historical transaction is a pandas row full of numpy scalars, NaNs and
    Timestamps. Coerce to plain JSON types, dropping anything that can't
    be represented rather than failing the scoring request over it."""
    out = {}
    for key, value in txn.items():
        if value is None:
            continue
        item = value.item() if hasattr(value, "item") else value
        if isinstance(item, float) and item != item:  # NaN
            continue
        if isinstance(item, (str, int, float, bool)):
            out[str(key)] = item
    return out


def _transaction_dt(txn: dict) -> float | None:
    """The transaction's own timestamp, used to place a feedback row on
    the correct side of the chronological train/test split. None for a
    custom transaction that never had one."""
    value = txn.get("TransactionDT")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ScoringService:
    """Scores a transaction and produces the decision that gates it.

    Everything here is synchronous and LLM-free by design: the action
    that actually gates a transaction must not wait on a Gemini call.
    Explanations are generated separately (generate_explanation below),
    after the decision is already final.
    """

    def __init__(self, explainer, memory, review_queue, explanations_cache, thresholds_provider,
                 feature_store=None, notifier=None):
        self._explainer = explainer
        self._memory = memory
        self._review_queue = review_queue
        self._explanations_cache = explanations_cache
        # Optional so a caller that only wants a decision (and every
        # existing test) behaves exactly as before. When present, the
        # transaction is scored against LIVE entity/device history rather
        # than the frozen training snapshot — see src/feature_store.py.
        self._feature_store = feature_store
        # Optional for the same reason. When present, notified with the
        # entity's escalation state before/after this verdict is recorded
        # — see src/notifications.py for what counts as alert-worthy.
        self._notifier = notifier
        # A callable rather than a dict: api/main.py resolves thresholds
        # through an lru_cache'd getter that tests monkeypatch, and this
        # has to keep seeing the patched value.
        self._thresholds_provider = thresholds_provider

    def score_and_decide(
        self,
        txn: dict,
        entity_id: str | None,
        *,
        record_verdict: bool = True,
        txn_index: int = 0,
        queue_entity_id: str | None = None,
    ) -> dict:
        """Score one transaction and return the full result.

        entity_id: the entity whose escalation history this transaction is
            judged against. None means no history at all — the cold-start
            NORMAL baseline (a custom transaction not attached to anyone).
        record_verdict: whether the resulting action is written back into
            that entity's history. False for a hypothetical "what if"
            that must never pollute a real entity's trajectory.
        """
        # --- Online features: READ, score, then RECORD ------------------
        # This ordering is the correctness argument for the whole feature
        # store. The features handed to the model must describe the
        # entity's history STRICTLY BEFORE this transaction; recording
        # first would let the transaction count itself, which is exactly
        # the leakage the offline causal functions use shift(1) to avoid.
        # tests/test_feature_store.py asserts the ordering directly.
        fingerprint = None
        if self._feature_store is not None:
            fingerprint = fingerprint_for(txn)
            txn = {**txn, **self._feature_store.features_for(entity_id, fingerprint)}

        result = self._explainer.score_transaction(txn)

        if self._feature_store is not None:
            # is_fraud is unknown at scoring time — the true label arrives
            # later, from a reviewer disposition or a chargeback. The
            # transaction counts toward volume now and toward fraud only
            # once labelled (FeatureStore.apply_label).
            self._feature_store.record(entity_id, fingerprint, is_fraud=None)

        if entity_id:
            escalation_before = self._memory.get_escalation_state(entity_id)
        else:
            escalation_before = _compute_escalation_state(None, [])

        thresholds = self._thresholds_provider()
        risk_score = result["risk_score"]

        # The decision that actually gates the transaction, made
        # synchronously from the score and the rules.
        decision = decide_action(
            risk_score, escalation_before, thresholds["review"], thresholds["block"]
        )
        # What the system would have done with no entity memory at all —
        # kept alongside the real decision so the review queue's metrics
        # can split reviewer precision by whether escalation is what
        # triggered the flag (the live counterpart of
        # src/escalation_ablation.py's offline comparison).
        baseline_decision = decide_action(
            risk_score, {"state": "NORMAL"}, thresholds["review"], thresholds["block"]
        )

        escalation_after = escalation_before
        if entity_id and record_verdict:
            self._memory.record_verdict(entity_id, decision["action"], risk_score)
            escalation_after = self._memory.get_escalation_state(entity_id)

        verdict_id = str(uuid.uuid4())

        if entity_id and record_verdict and self._notifier is not None:
            self._notifier.notify_transition(
                entity_id, escalation_before.get("state"), escalation_after.get("state"),
                risk_score, verdict_id,
            )
        self._explanations_cache.put(verdict_id, {"status": "pending"})

        if decision["action"] != "ALLOW":
            self._review_queue.add({
                "verdict_id": verdict_id,
                "entity_id": queue_entity_id or entity_id or UNATTACHED_ENTITY_LABEL,
                "txn_index": txn_index,
                "risk_score": risk_score,
                "decision": decision,
                "baseline_decision": baseline_decision,
                "escalated_due_to_history": decision["escalated_due_to_history"],
                "disposition": None,
                "disposed_at": None,
                "created_at": _now_iso(),
                "notes": [],
                # Kept so a disposed item can become a labelled training
                # row (src/feedback_export.py). Without the features the
                # model actually saw and the transaction's own timestamp,
                # a reviewer's verdict is a label with nothing to attach
                # it to, and can't be placed correctly relative to the
                # chronological train/test boundary.
                "transaction": _jsonable(txn),
                "transaction_dt": _transaction_dt(txn),
                "escalation_state": escalation_before.get("state"),
            })

        return {
            "risk_score": risk_score,
            "above_threshold": result["above_threshold"],
            "top_factors": result["top_factors"],
            "escalation_before": escalation_before,
            "decision": decision,
            "baseline_decision": baseline_decision,
            "verdict_id": verdict_id,
        }


def generate_explanation(agent_provider, explanations_cache, verdict_id, risk_score, top_factors, escalation):
    """Produce the LLM explanation for an already-final decision and write
    it into the cache the frontend reads.

    Runs after the response has gone out (a FastAPI background task) or
    inline in the stream consumer — either way an exception here must not
    propagate. RiskExplainerAgent.explain() already handles its own
    API-level failures; this is the backstop for anything else (e.g. a bug
    in agent construction), so a caller polling for this verdict always
    terminates one way or another instead of waiting forever.
    """
    try:
        verdict = agent_provider().explain(risk_score, top_factors, escalation)
    except Exception:
        logger.exception("Unhandled error generating explanation", extra={"verdict_id": verdict_id})
        verdict = dict(FALLBACK_VERDICT)
    explanations_cache.put(verdict_id, {"status": "ready", "verdict": verdict})
