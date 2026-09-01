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
  - REDIS_URL, optional. Unset: entity memory / idempotency cache /
    explanation cache are in-process (default, zero setup). Set: the
    same state is backed by Redis instead — survives restarts, shared
    across workers. See src/entity_memory.py and src/redis_utils.py.

POST /api/score is rate-limited to 30/minute per caller (X-API-Key if
present, else source IP) — see the `limiter` set up below.
"""
import logging
import os
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))

from functools import lru_cache

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from logging_utils import configure_logging, RequestIDMiddleware
from risk_explainer import RiskExplainer
from llm_agent import RiskExplainerAgent
from entity_memory import create_entity_memory
from redis_utils import get_redis_client, KeyedCache
from data_utils import load_raw_data, engineer_features
from cost_analysis import cost_curve, DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Risk Manager API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Outermost middleware, so every request — including ones that never
# reach a route (rate-limited, auth-rejected) — gets a request id logs
# can be correlated by. See logging_utils.py.
app.add_middleware(RequestIDMiddleware)

# /metrics (Prometheus text format): request count/latency per route,
# unauthenticated by design — Prometheus scraping conventions expect
# network-level access control (e.g. not publicly exposed), not an
# application API key, and it's operational data, not customer data.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _rate_limit_key(request: Request) -> str:
    """Rate limit per caller identity: the X-API-Key if one was sent,
    falling back to source IP for requests that never got that far
    (e.g. no key sent at all — those already 401 via verify_api_key, but
    the rate limiter's key function runs independently of that
    dependency, so it still needs a sane fallback)."""
    api_key = request.headers.get("X-API-Key")
    return f"apikey:{api_key}" if api_key else get_remote_address(request)


# Backed by Redis (shared across restarts/workers) when REDIS_URL is set,
# in-process memory otherwise — same optional-Redis pattern as entity
# memory / idempotency / explanations below.
#
# headers_enabled is deliberately False: slowapi's per-route header
# injection expects the endpoint to return a starlette Response, but
# these routes return plain dicts (FastAPI serializes them) — with it
# True, a successful request crashes trying to inject headers into a
# dict. The 429 response itself (via _rate_limit_exceeded_handler below)
# is unaffected either way, since that one already returns a real
# JSONResponse.
limiter = Limiter(key_func=_rate_limit_key, storage_uri=os.environ.get("REDIS_URL") or None, headers_enabled=False)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
# REDIS_URL is read once here, at process startup — same as any other
# deployment config — not re-checked per request. Unset (the default):
# every store below behaves exactly as it did before this file supported
# Redis at all. Set: entity escalation history, the idempotency cache,
# and pending/ready explanations all persist in Redis instead, surviving
# restarts and shared across workers.
_redis_client = get_redis_client()

_memory = create_entity_memory(_redis_client)

# LLM-generated explanations, keyed by verdict_id, filled in by a
# background task after the score/decision response has already gone out.
_explanations_cache = KeyedCache(_redis_client, prefix="riskmgr:explanations", ttl_seconds=60 * 60)

# Idempotency cache for POST /api/score, keyed by the caller-supplied
# Idempotency-Key header. Without this, a retried request (network
# blip, client double-click/double-submit) would score and record the
# SAME transaction twice — double-counting it in the entity's
# escalation history and pushing them toward WATCH/ELEVATED on a
# duplicate, not a second real transaction.
_idempotency_cache = KeyedCache(_redis_client, prefix="riskmgr:idempotency", ttl_seconds=24 * 60 * 60)


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
        logger.exception("Unhandled error generating explanation", extra={"verdict_id": verdict_id})
        verdict = {
            "explanation": "An unexpected error occurred while generating the AI explanation.",
            "action": "REVIEW",
            "escalated_due_to_history": False,
            "rationale": "Falling back to manual review — explainer agent crashed.",
        }
    _explanations_cache.put(verdict_id, {"status": "ready", "verdict": verdict})


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
@limiter.limit("30/minute")
def score(
    request: Request,
    req: ScoreRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key is not None:
        cached = _idempotency_cache.get(idempotency_key)
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
    _explanations_cache.put(verdict_id, {"status": "pending"})
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
        _idempotency_cache.put(idempotency_key, response)
    return response


@router.get("/api/explanations/{verdict_id}")
def get_explanation(verdict_id: str):
    explanation = _explanations_cache.get(verdict_id)
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
