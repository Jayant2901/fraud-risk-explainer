"""
AI Risk Manager — JSON API (FastAPI)

Replaces the Streamlit prototype with a proper backend so the UI can be
a standalone, scalable frontend (see frontend/). Run from project root:

    uvicorn api.main:app --reload --port 8000

Requires:
  - models/ populated by `python src/train_model.py`
  - API_KEY set in the environment — every /api/* route except
    /api/health requires header `X-API-Key: <API_KEY>`. There is no
    fallback or generated default: if API_KEY is unset, every protected
    route returns 401 (fail closed) rather than being left open.
  - GEMINI_API_KEY set in the environment (see src/llm_agent.py) — free
    tier, no billing required (https://aistudio.google.com/apikey).
    Without it, the score/decision endpoint still works; only the AI
    explanation panel falls back to a "credentials missing" message.
"""
import logging
import os
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import OrderedDict
from functools import lru_cache

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from risk_explainer import RiskExplainer
from llm_agent import RiskExplainerAgent
from entity_memory import EntityRiskMemory
from data_utils import load_raw_data, engineer_features
from cost_analysis import cost_curve, DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Risk Manager API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str | None = Security(_api_key_header)) -> None:
    """Every /api/* route except /api/health depends on this (via `router`
    below). Reads API_KEY from the environment on every call, rather than
    caching it at import time, so there's exactly one source of truth and
    no risk of serving stale auth after a config change picked up by a
    process restart.

    Fails closed: an unset API_KEY is NOT "auth disabled" — every request
    is rejected. A generated/default key would be worse than no key at
    all (false sense of security), so this refuses instead.
    """
    expected_key = os.environ.get("API_KEY")
    if not expected_key:
        raise HTTPException(status_code=401, detail="API_KEY is not configured on the server")
    if provided_key != expected_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# All routes except /api/health require a valid API key. Grouped on a
# router (rather than a per-route `dependencies=[...]`) so a new route
# added later is protected by default instead of by remembering to
# annotate it.
router = APIRouter(dependencies=[Depends(verify_api_key)])

EVAL_REPORT_PATH = "models/eval_report.txt"

# --- Singletons (model, explainer, agent, entity memory) ---------------
# Entity memory is in-process, matching the original demo's documented
# limitation (README: "persist entity memory in Redis... so it survives
# restarts and scales across workers"). Fine for a single-worker deploy.
_memory = EntityRiskMemory()

# LLM-generated explanations, keyed by verdict_id, filled in by a
# background task after the score/decision response has already gone
# out. In-memory + unbounded, matching entity memory's documented
# single-worker-demo limitation above (a real deploy would use Redis
# with a TTL instead).
_explanations: dict[str, dict] = {}

# Idempotency cache for POST /api/score, keyed by the caller-supplied
# Idempotency-Key header. Without this, a retried request (network
# blip, client double-click/double-submit) would score and record the
# SAME transaction twice — double-counting it in the entity's
# escalation history and pushing them toward WATCH/ELEVATED on a
# duplicate, not a second real transaction. Bounded + LRU-evicted since
# this is in-process (a real deploy would use Redis with a TTL, same as
# entity memory above).
_IDEMPOTENCY_CACHE_MAX = 1000
_idempotency_cache: "OrderedDict[str, dict]" = OrderedDict()


def _idempotency_cache_get(key: str) -> dict | None:
    if key not in _idempotency_cache:
        return None
    _idempotency_cache.move_to_end(key)
    return _idempotency_cache[key]


def _idempotency_cache_put(key: str, response: dict) -> None:
    _idempotency_cache[key] = response
    _idempotency_cache.move_to_end(key)
    if len(_idempotency_cache) > _IDEMPOTENCY_CACHE_MAX:
        _idempotency_cache.popitem(last=False)


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


def _generate_explanation(verdict_id: str, risk_score: float, top_factors: list, escalation: dict):
    # This runs in a background task after the response has already gone
    # out — an uncaught exception here doesn't fail the request, it just
    # logs silently and leaves the frontend polling forever, since nothing
    # ever writes a terminal status for verdict_id. RiskExplainerAgent.explain()
    # already catches its own API-level failures, but this is the backstop
    # for anything else (e.g. a bug in agent construction) so polling always
    # terminates one way or another.
    try:
        agent = get_agent()
        verdict = agent.explain(risk_score, top_factors, escalation)
    except Exception:
        logger.exception("Unhandled error generating explanation for verdict_id=%s", verdict_id)
        verdict = {
            "explanation": "An unexpected error occurred while generating the AI explanation.",
            "action": "REVIEW",
            "escalated_due_to_history": False,
            "rationale": "Falling back to manual review — explainer agent crashed.",
        }
    _explanations[verdict_id] = {"status": "ready", "verdict": verdict}


@lru_cache(maxsize=1)
def get_explainer() -> RiskExplainer:
    return RiskExplainer()


@lru_cache(maxsize=1)
def get_agent() -> RiskExplainerAgent:
    return RiskExplainerAgent()


SAMPLE_DATA_CACHE_PATH = "models/sample_data_cache.pkl"


@lru_cache(maxsize=1)
def get_sample_data():
    # Only ~30 entities' worth of rows ever get served to the frontend, but
    # computing that slice requires loading + feature-engineering the FULL
    # 590K-row dataset first — multiple minutes. lru_cache only helps within
    # one process; `uvicorn --reload` starts a fresh process on every file
    # save, which was silently repeating that multi-minute load on every
    # reload (surfacing as the frontend's request timing out / 502ing while
    # waiting). Caching the small final slice to disk fixes that: delete
    # models/sample_data_cache.pkl if you ever regenerate data/ and need a
    # fresh sample.
    if os.path.exists(SAMPLE_DATA_CACHE_PATH):
        return pd.read_pickle(SAMPLE_DATA_CACHE_PATH)

    df = load_raw_data()
    df = engineer_features(df)
    counts = df["entity_id"].value_counts()
    active_entities = counts[counts >= 5].index[:30]
    sample = (
        df[df["entity_id"].isin(active_entities)]
        .sort_values(["entity_id", "TransactionDT"])
        .reset_index(drop=True)
    )
    sample.to_pickle(SAMPLE_DATA_CACHE_PATH)
    return sample


class ScoreRequest(BaseModel):
    entity_id: str
    txn_index: int


class ResetRequest(BaseModel):
    entity_id: str


@router.get("/api/entities")
def list_entities():
    samples = get_sample_data()
    entity_ids = samples["entity_id"].unique().tolist()
    return {"entities": entity_ids}


@router.get("/api/entities/{entity_id}/transactions")
def list_transactions(entity_id: str):
    samples = get_sample_data()
    txns = samples[samples["entity_id"] == entity_id]
    if txns.empty:
        raise HTTPException(status_code=404, detail="Unknown entity_id")
    return {
        "entity_id": entity_id,
        "count": len(txns),
        "transactions": [
            {
                "index": i,
                "TransactionAmt": float(row["TransactionAmt"]),
                "TransactionDT": int(row["TransactionDT"]),
                "ProductCD": str(row["ProductCD"]),
            }
            for i, (_, row) in enumerate(txns.iterrows())
        ],
    }


@router.get("/api/entities/{entity_id}/escalation")
def get_escalation(entity_id: str):
    return _memory.get_escalation_state(entity_id)


@router.post("/api/entities/reset")
def reset_entity(req: ResetRequest):
    _memory.reset(req.entity_id)
    return {"status": "ok"}


@router.post("/api/score")
def score(
    req: ScoreRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key is not None:
        cached = _idempotency_cache_get(idempotency_key)
        if cached is not None:
            return cached

    samples = get_sample_data()
    entity_txns = samples[samples["entity_id"] == req.entity_id].reset_index(drop=True)
    if entity_txns.empty:
        raise HTTPException(status_code=404, detail="Unknown entity_id")
    if not (0 <= req.txn_index < len(entity_txns)):
        raise HTTPException(status_code=400, detail="txn_index out of range")

    txn = entity_txns.iloc[req.txn_index].to_dict()

    explainer = get_explainer()
    result = explainer.score_transaction(txn)

    escalation_before = _memory.get_escalation_state(req.entity_id)

    # The decision that actually gates the transaction is made right here,
    # synchronously (score + SHAP + rules, ~100-130ms measured locally) —
    # it does not wait on the LLM. record_verdict() runs immediately too,
    # so escalation state for this entity's NEXT transaction is already
    # correct.
    decision = decide_action(result["risk_score"], escalation_before)
    _memory.record_verdict(req.entity_id, decision["action"], result["risk_score"])

    verdict_id = str(uuid.uuid4())
    _explanations[verdict_id] = {"status": "pending"}
    background_tasks.add_task(
        _generate_explanation, verdict_id, result["risk_score"], result["top_factors"], escalation_before
    )

    response = {
        "risk_score": result["risk_score"],
        "above_threshold": result["above_threshold"],
        "top_factors": result["top_factors"],
        "escalation_before": escalation_before,
        "decision": decision,
        "verdict_id": verdict_id,
    }
    if idempotency_key is not None:
        _idempotency_cache_put(idempotency_key, response)
    return response


@router.get("/api/explanations/{verdict_id}")
def get_explanation(verdict_id: str):
    explanation = _explanations.get(verdict_id)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Unknown verdict_id")
    return explanation


@router.get("/api/cost-analysis")
def get_cost_analysis(
    fraud_loss: float = DEFAULT_AVG_FRAUD_LOSS,
    fp_cost: float = DEFAULT_AVG_FP_COST,
):
    eval_report = None
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH) as f:
            eval_report = f.read()

    return {
        "eval_report": eval_report,
        "defaults": {
            "avg_fraud_loss": DEFAULT_AVG_FRAUD_LOSS,
            "avg_fp_cost": DEFAULT_AVG_FP_COST,
        },
        "params": {"fraud_loss": fraud_loss, "fp_cost": fp_cost},
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(router)
