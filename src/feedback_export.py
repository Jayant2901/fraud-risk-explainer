"""
Turns reviewer dispositions into a labelled dataset.

review_queue.py already captures CONFIRMED_FRAUD / FALSE_POSITIVE on
exactly the transactions the model was least certain about — human-
verified labels on the hardest cases, which is the most valuable training
data this system produces. Until now nothing consumed them.

Schema, one row per disposed review-queue item:

    verdict_id          str    the decision this label belongs to
    entity_id           str    card/account fingerprint
    transaction_dt      float  the transaction's own timestamp, used to
                               place the row correctly relative to the
                               chronological train/test boundary. None
                               for a custom transaction with no timestamp.
    risk_score          float  what the model scored it, at the time
    action              str    ALLOW/REVIEW/BLOCK actually taken
    baseline_action     str    what it would have been with no entity
                               memory — lets analysis separate
                               escalation-driven labels from score-driven
    escalated           bool   whether entity history changed the action
    escalation_state    str    NORMAL/WATCH/ELEVATED at decision time
    disposition         str    CONFIRMED_FRAUD or FALSE_POSITIVE
    disposed_at         str    ISO timestamp of the reviewer's decision
    isFraud             int    1 for CONFIRMED_FRAUD, 0 for FALSE_POSITIVE
    <feature columns>   any    the transaction fields the model saw

### The bias, stated plainly

These labels are collected ONLY on transactions the system flagged. The
model never receives ground truth on what it confidently allowed, so this
is a censored sample, not a random one: it over-represents the region
near the decision boundary and contains no information about false
negatives. Training on it makes the model better calibrated where
reviewers looked and says nothing about anywhere else.

That is why retraining with feedback is opt-in (train_model.py
--with-feedback), off by default, weighted explicitly, and reported as a
measured delta against the no-feedback baseline rather than assumed to be
an improvement. If the delta is negative, that is the finding.
"""
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from review_queue import CONFIRMED_FRAUD, FALSE_POSITIVE

logger = logging.getLogger(__name__)

FEEDBACK_DIR = "data/feedback"

# Item fields that describe the decision rather than the transaction —
# excluded from the feature columns so they can't leak into training as
# if they were inputs the model had at scoring time.
NON_FEATURE_KEYS = {
    "verdict_id", "entity_id", "txn_index", "risk_score", "decision",
    "baseline_decision", "escalated_due_to_history", "disposition",
    "disposed_at", "created_at", "notes", "transaction", "transaction_dt",
    "escalation_state",
}

LABEL_FOR_DISPOSITION = {CONFIRMED_FRAUD: 1, FALSE_POSITIVE: 0}


def build_feedback_rows(items: list[dict]) -> list[dict]:
    """One row per DISPOSED item. Undisposed items are not labels and are
    dropped — an item still sitting in the queue tells us nothing."""
    rows = []
    for item in items:
        disposition = item.get("disposition")
        if disposition not in LABEL_FOR_DISPOSITION:
            continue

        decision = item.get("decision") or {}
        baseline = item.get("baseline_decision") or {}
        row = {
            "verdict_id": item.get("verdict_id"),
            "entity_id": item.get("entity_id"),
            "transaction_dt": item.get("transaction_dt"),
            "risk_score": item.get("risk_score"),
            "action": decision.get("action"),
            "baseline_action": baseline.get("action"),
            "escalated": bool(item.get("escalated_due_to_history")),
            "escalation_state": item.get("escalation_state"),
            "disposition": disposition,
            "disposed_at": item.get("disposed_at"),
            "isFraud": LABEL_FOR_DISPOSITION[disposition],
        }
        # The features the model actually saw, flattened alongside.
        for key, value in (item.get("transaction") or {}).items():
            if key not in row and key not in NON_FEATURE_KEYS:
                row[key] = value
        rows.append(row)
    return rows


def export_feedback(review_queue, output_dir: str = FEEDBACK_DIR) -> dict:
    """Write every disposed item to data/feedback/ as CSV.

    CSV rather than parquet so the file is readable without pandas and
    diffable in review — at this project's scale (a reviewer's day, not a
    warehouse) the size argument for parquet doesn't apply.
    """
    items = _all_items(review_queue)
    rows = build_feedback_rows(items)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"feedback_{stamp}.csv")

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8")

    summary = {
        "path": path,
        "rows": len(rows),
        "confirmed_fraud": sum(1 for r in rows if r["isFraud"] == 1),
        "false_positive": sum(1 for r in rows if r["isFraud"] == 0),
        "escalation_driven": sum(1 for r in rows if r["escalated"]),
    }
    logger.info("Exported reviewer feedback", extra=summary)
    return summary


def load_feedback(feedback_dir: str = FEEDBACK_DIR) -> pd.DataFrame:
    """Every exported feedback file, concatenated, most recent last.
    Returns an empty frame when nothing has been exported."""
    if not os.path.isdir(feedback_dir):
        return pd.DataFrame()
    paths = sorted(
        os.path.join(feedback_dir, name)
        for name in os.listdir(feedback_dir)
        if name.endswith(".csv")
    )
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path, encoding="utf-8") for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    # A verdict exported twice (two export runs) is one label, not two.
    if "verdict_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["verdict_id"], keep="last")
    return combined.reset_index(drop=True)


def _all_items(review_queue) -> list[dict]:
    """Every item the queue knows about, disposed or not.

    ReviewQueue exposes list_pending() (undisposed only) and metrics()
    (aggregates), but no "everything" accessor — reaching into the
    backing store is deliberate and confined to this one function rather
    than widening the queue's public interface for an export concern.
    """
    if hasattr(review_queue, "_items"):
        return [dict(i) for i in review_queue._items.values()]
    ids = review_queue._redis.smembers(review_queue._all_key())
    items = [review_queue.get(i) for i in ids]
    return [i for i in items if i is not None]


def to_json_summary(review_queue) -> dict:
    """What GET /api/feedback/export returns: the rows themselves plus the
    counts, so the loop is inspectable without shell access."""
    rows = build_feedback_rows(_all_items(review_queue))
    return {
        "rows": rows,
        "count": len(rows),
        "confirmed_fraud": sum(1 for r in rows if r["isFraud"] == 1),
        "false_positive": sum(1 for r in rows if r["isFraud"] == 0),
        "bias_warning": (
            "These labels exist only for transactions the system flagged. "
            "There is no ground truth here for anything it confidently "
            "allowed, so this is a censored sample: useful for calibration "
            "near the decision boundary, silent about false negatives."
        ),
    }
