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

POST /api/score and POST /api/score-custom are each rate-limited to
30/minute per caller (X-API-Key if present, else source IP) — see the
`limiter` set up below.
"""
import json
import logging
import os
import sys
import uuid
from typing import Literal

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))

from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from logging_utils import configure_logging, RequestIDMiddleware
from decision_rules import load_decision_thresholds
from risk_explainer import RiskExplainer
from llm_agent import RiskExplainerAgent
from entity_memory import (
    create_entity_memory,
    DEFAULT_WATCH_PRESSURE_THRESHOLD,
    DEFAULT_ELEVATED_PRESSURE_THRESHOLD,
)
from redis_utils import get_redis_client, KeyedCache
from review_queue import create_review_queue, UnknownVerdictError, AlreadyDisposedError, NOTE_MAX_LEN
from data_utils import load_raw_data, engineer_features
from cost_analysis import cost_curve, DEFAULT_AVG_FRAUD_LOSS, DEFAULT_AVG_FP_COST
from impact_summary import extrapolate_monthly_savings
from scoring_service import FALLBACK_VERDICT, ScoringService, generate_explanation
from explanation_bus import ExplanationBus
from circuit_breaker import CircuitBreaker
from feature_store import create_feature_store, seed_from_history
from notifications import create_notifier
from domain_metrics import register_domain_state_collector
from shadow_scoring import create_shadow_scorer, create_shadow_comparison
from audit_log import create_audit_log
from feedback_export import export_feedback, to_json_summary
import event_stream

configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the online feature store before the first request. Deliberately
    # a lifespan handler rather than the deprecated @app.on_event.
    seed_feature_store()
    yield


app = FastAPI(title="AI Risk Manager API", lifespan=lifespan)

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
ESCALATION_ABLATION_REPORT_PATH = "models/escalation_ablation_report.txt"
ESCALATION_ABLATION_SUMMARY_PATH = "models/escalation_ablation_summary.json"
COST_SENSITIVITY_REPORT_PATH = "models/cost_sensitivity_report.json"
DRIFT_REPORT_PATH = "models/drift_report.json"
CONSISTENCY_REPORT_PATH = "models/consistency_report.json"
COLD_START_REPORT_PATH = "models/cold_start_report.txt"
COST_SUMMARY_PATH = "models/cost_summary.json"
COST_CURVE_PATH = "models/cost_curve.json"

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

# Human review queue: every flagged (non-ALLOW) verdict lands here for a
# reviewer to dispose as CONFIRMED_FRAUD/FALSE_POSITIVE. See
# src/review_queue.py.
_review_queue = create_review_queue(_redis_client)

# Carries explanation deltas from the worker running the LLM call to the
# worker holding the client's SSE connection — see src/explanation_bus.py.
_explanation_bus = ExplanationBus(_redis_client)

# Live entity/device aggregates, so a transaction is scored against the
# history that exists NOW rather than the frozen training snapshot — see
# src/feature_store.py.
_feature_store = create_feature_store(_redis_client)
_feature_store_seed = {"seeded_rows": 0}

# Alerts a fraud team when a scored transaction pushes an entity into a
# worse escalation state. Opt-in: no-op until ESCALATION_WEBHOOK_URL is
# set — see src/notifications.py.
_notifier = create_notifier(_redis_client)

# Fraud-shaped metrics (queue depth, live precision, LLM breaker state)
# on the same GET /metrics text-format endpoint the HTTP-level metrics
# already use — see src/domain_metrics.py. get_agent is defined further
# down; the lambda defers the lookup to scrape time, by which point the
# module is fully loaded.
register_domain_state_collector(_review_queue, lambda: get_agent().breaker.state())

# A candidate model (SHADOW_MODEL_PATH) scores every transaction silently
# alongside the live one, so its decisions can be compared before anyone
# promotes it. None/no-op until that env var is set — see
# src/shadow_scoring.py and GET /api/shadow-comparison below.
_shadow_scorer = create_shadow_scorer()
_shadow_comparison = create_shadow_comparison(_redis_client)

# Immutable, hash-chained record of every verdict — always on, local-file
# backed by default (data/audit_log.jsonl), Redis-backed (shared with
# src/stream_consumer.py, compare-and-swap under concurrent writers) when
# configured. See src/audit_log.py and `python -m src.audit verify`.
_audit_log = create_audit_log(_redis_client)


def _generate_explanation(verdict_id: str, risk_score: float, top_factors: list, escalation: dict):
    """Background task. Streams the LLM response, publishing each delta to
    subscribers of GET /api/verdicts/{verdict_id}/stream, and writes the
    finished verdict into the same cache
    GET /api/explanations/{verdict_id} has always served — so a client on
    either transport ends up with the identical result.

    Falls back to the batch path if streaming itself fails, so an SDK or
    transport problem degrades to today's behavior rather than leaving a
    verdict with no explanation at all.
    """
    try:
        verdict = None
        for message in get_agent().explain_stream(risk_score, top_factors, escalation):
            _explanation_bus.publish(verdict_id, message)
            if message["type"] == "complete":
                verdict = message["verdict"]
        if verdict is not None:
            _explanations_cache.put(verdict_id, {"status": "ready", "verdict": verdict})
            return
        logger.warning("Explanation stream ended with no verdict", extra={"verdict_id": verdict_id})
    except Exception:
        logger.exception("Streaming explanation failed; using batch path", extra={"verdict_id": verdict_id})

    generate_explanation(
        get_agent, _explanations_cache, verdict_id, risk_score, top_factors, escalation
    )
    ready = _explanations_cache.get(verdict_id) or {}
    if ready.get("status") == "ready":
        _explanation_bus.publish(verdict_id, {"type": "complete", "verdict": ready["verdict"]})


# Seeding walks the cached sample chronologically; this bounds the work
# at startup. The sample is ~30 entities' worth of rows, so in practice
# this is the whole thing.
FEATURE_SEED_LIMIT = int(os.environ.get("FEATURE_SEED_LIMIT", "20000"))


def seed_feature_store() -> None:
    """Warm the online feature store from the historical sample so
    entities aren't cold on first run.

    Best-effort: a failure here (no dataset on disk, say) must not stop
    the API from serving — scoring still works, entities just start with
    empty history, and /api/health reports seeded_rows as 0 so that is
    visible rather than silent.
    """
    if os.environ.get("SKIP_FEATURE_SEED"):
        return
    try:
        _feature_store_seed.update(
            seed_from_history(_feature_store, get_sample_data(), limit=FEATURE_SEED_LIMIT)
        )
    except Exception:
        logger.warning("Could not seed the feature store from history", exc_info=True)


def _score_response(scored: dict) -> dict:
    """The public shape of a scoring response. baseline_decision is used
    internally (review-queue metrics) but has never been part of this
    response, so it's dropped here rather than silently widening the
    documented API."""
    return {
        "risk_score": scored["risk_score"],
        "above_threshold": scored["above_threshold"],
        "top_factors": scored["top_factors"],
        "escalation_before": scored["escalation_before"],
        "decision": scored["decision"],
        "verdict_id": scored["verdict_id"],
    }


def _scoring_service() -> ScoringService:
    """Built per call rather than once at import: the singletons it wraps
    (get_explainer/get_decision_thresholds) are lru_cache'd getters that
    tests monkeypatch, and a service captured at import time would pin
    the unpatched originals."""
    return ScoringService(
        explainer=get_explainer(),
        memory=_memory,
        review_queue=_review_queue,
        explanations_cache=_explanations_cache,
        thresholds_provider=get_decision_thresholds,
        feature_store=_feature_store,
        notifier=_notifier,
        shadow_scorer=_shadow_scorer,
        shadow_comparison=_shadow_comparison,
        audit_log=_audit_log,
    )


@lru_cache(maxsize=1)
def get_explainer() -> RiskExplainer:
    return RiskExplainer()


@lru_cache(maxsize=1)
def get_decision_thresholds() -> dict:
    """{"review": float, "block": float} — loaded once at process
    startup, same singleton pattern as get_explainer()/get_agent(), from
    the real values train_model.py derived from the cost analysis. See
    decision_rules.load_decision_thresholds()."""
    return load_decision_thresholds()


@lru_cache(maxsize=1)
def get_agent() -> RiskExplainerAgent:
    thresholds = get_decision_thresholds()
    return RiskExplainerAgent(
        review_threshold=thresholds["review"],
        block_threshold=thresholds["block"],
        # Redis-backed when configured, so every worker sees the same
        # failure count instead of each independently rediscovering that
        # the LLM is down.
        breaker=CircuitBreaker("llm", redis_client=_redis_client),
    )


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


class CustomTransactionRequest(BaseModel):
    """Fields a person could reasonably fill in by hand for a transaction
    that isn't one of the ~30 cached historical entities. Everything
    RiskExplainer.score_transaction expects beyond these (C1-C14,
    D-features, V-features, M-features, entity_prior_* features, ...) is
    left missing and handled exactly like any other missing feature —
    see RiskExplainer.score_transaction's docstring. TransactionAmt is
    the only required field.
    """
    TransactionAmt: float
    ProductCD: str | None = None
    card4: str | None = None
    card6: str | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    hour_of_day: int | None = Field(default=None, ge=0, le=23)
    # When set to an existing entity id, the custom transaction is scored
    # against that entity's *current* real escalation state, and — since
    # this is an explicit opt-in — the resulting verdict is recorded into
    # that entity's real history. When unset, escalation is the NORMAL
    # baseline (this dataset's actual cold-start case) and nothing is
    # recorded anywhere, so a hypothetical "what if" can never silently
    # pollute a real entity's trajectory.
    attach_to_entity_id: str | None = None


class ResetRequest(BaseModel):
    entity_id: str


class DispositionRequest(BaseModel):
    disposition: Literal["CONFIRMED_FRAUD", "FALSE_POSITIVE"]


class AddNoteRequest(BaseModel):
    author: str = Field(default="Reviewer", max_length=100)
    text: str = Field(min_length=1, max_length=NOTE_MAX_LEN)


class TransactionEventRequest(CustomTransactionRequest):
    """Webhook-shaped ingestion payload. Deliberately extends the existing
    CustomTransactionRequest rather than defining a second transaction
    schema that could drift from it — the transaction fields are the same
    fields, plus the two things an event carries that a synchronous
    request doesn't."""
    # Sender-supplied and stable across retries: real payment processors
    # redeliver webhooks, and a redelivery must not produce a second
    # verdict or a second entity-memory record.
    event_id: str = Field(min_length=1, max_length=200)
    entity_id: str | None = None


# /api/score is 30/minute because it's driven by a human clicking in the
# UI. This endpoint is machine-facing — a processor's webhook sender, not
# a person — so the limit reflects intended ingestion throughput instead.
# 600/minute is 10/second sustained, comfortably above what the consumer
# needs to keep up with on this hardware and still low enough to be a
# real ceiling rather than an open door.
INGEST_RATE_LIMIT = "600/minute"

# Dedup window for event_id. Longer than any sane webhook retry schedule,
# short enough that the key space stays bounded.
EVENT_DEDUP_TTL_SECONDS = 24 * 60 * 60

_event_dedup_cache = KeyedCache(
    _redis_client, prefix="riskmgr:events:seen", ttl_seconds=EVENT_DEDUP_TTL_SECONDS
)


@router.post("/api/events/transaction", status_code=202)
@limiter.limit(INGEST_RATE_LIMIT)
def ingest_transaction_event(request: Request, req: TransactionEventRequest):
    """Accept a transaction event for asynchronous scoring.

    Returns 202 immediately without scoring: a webhook sender times out,
    it does not wait for a model — let alone an LLM. The event is queued
    onto a Redis Stream and scored by src/stream_consumer.py, which runs
    the same scoring path this API's synchronous endpoints use.

    Requires REDIS_URL. Everywhere else in this codebase Redis is
    optional and absence falls back to in-process state, but durable
    ingestion is the one place that would be a lie: an in-process queue
    would accept events and lose them on restart while looking exactly
    like a working pipeline. So this returns 503 instead.
    """
    if _redis_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Streaming ingestion requires Redis. Set REDIS_URL and run the stream "
                "consumer (python -m src.stream_consumer); the synchronous /api/score "
                "and /api/score-custom endpoints work without it."
            ),
        )

    existing = _event_dedup_cache.get(req.event_id)
    if existing is not None:
        # Idempotent by design: the same answer, and nothing enqueued a
        # second time.
        return {**existing, "duplicate": True}

    verdict_id = str(uuid.uuid4())
    event = {
        "event_id": req.event_id,
        "verdict_id": verdict_id,
        "entity_id": req.entity_id,
        "transaction": req.model_dump(
            exclude={"event_id", "entity_id", "attach_to_entity_id"}, exclude_none=True
        ),
    }

    event_stream.ensure_group(_redis_client)
    event_stream.publish_event(_redis_client, event)

    accepted = {"event_id": req.event_id, "verdict_id": verdict_id, "status": "accepted"}
    _event_dedup_cache.put(req.event_id, accepted)
    logger.info("Transaction event accepted", extra={"event_id": req.event_id, "verdict_id": verdict_id})
    return {**accepted, "duplicate": False}


@router.get("/api/feedback/export")
def export_feedback_dataset(write_file: bool = False):
    """Reviewer dispositions as a labelled dataset.

    These are human-verified labels on exactly the transactions the model
    was least certain about — and a censored sample: they exist only for
    transactions the system flagged, so they say nothing about what it
    confidently allowed. The response carries that caveat with the data
    rather than leaving it to be rediscovered.

    write_file=true also writes a CSV under data/feedback/ for
    train_model.py --with-feedback to pick up.
    """
    summary = to_json_summary(_review_queue)
    if write_file:
        summary["written"] = export_feedback(_review_queue)
    return summary


@router.get("/api/events/dead-letter")
def list_dead_letter_events(limit: int = 50):
    """Events the consumer failed to score after every retry. Visible
    here so a failed event doesn't require a Redis CLI to find."""
    if _redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Streaming ingestion requires Redis. Set REDIS_URL to use this endpoint.",
        )
    return {"items": event_stream.list_dead_letter(_redis_client, limit=limit)}


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

    # The decision that actually gates the transaction is made right here,
    # synchronously (score + SHAP + rules, ~100-130ms measured locally) —
    # it does not wait on the LLM. The verdict is recorded immediately
    # too, so escalation state for this entity's NEXT transaction is
    # already correct. See src/scoring_service.py.
    scored = _scoring_service().score_and_decide(
        txn, req.entity_id, txn_index=req.txn_index
    )

    background_tasks.add_task(
        _generate_explanation,
        scored["verdict_id"],
        scored["risk_score"],
        scored["top_factors"],
        scored["escalation_before"],
    )

    response = _score_response(scored)
    if idempotency_key is not None:
        _idempotency_cache.put(idempotency_key, response)
    return response


@router.post("/api/score-custom")
@limiter.limit("30/minute")
def score_custom(
    request: Request,
    req: CustomTransactionRequest,
    background_tasks: BackgroundTasks,
):
    # Only the fields the caller actually provided go into the model
    # input — everything else stays missing (-> NaN), exactly the same
    # path RiskExplainer.score_transaction already takes for any missing
    # feature on a replayed historical transaction.
    txn = req.model_dump(exclude={"attach_to_entity_id"}, exclude_none=True)

    # Same scoring path /api/score uses — a custom transaction can never
    # gate itself differently than a replayed one.
    scored = _scoring_service().score_and_decide(txn, req.attach_to_entity_id)

    background_tasks.add_task(
        _generate_explanation,
        scored["verdict_id"],
        scored["risk_score"],
        scored["top_factors"],
        scored["escalation_before"],
    )

    return _score_response(scored)


# An explanation stream that has produced nothing for this many idle
# ticks (see explanation_bus.POLL_TIMEOUT_SECONDS) is assumed dead — the
# worker running the LLM call crashed, or the verdict was never generated
# — and is closed with a terminal error rather than held open forever.
# 8 x 15s = two minutes, comfortably longer than any LLM call this
# project makes (see RT-3's LLM_TIMEOUT_SECONDS).
STREAM_MAX_IDLE_TICKS = 8


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def verify_api_key_or_query(
    provided_key: str | None = Security(_api_key_header),
    api_key: str | None = None,
) -> None:
    """Auth for the SSE route only.

    The browser's EventSource cannot set request headers, so it has no way
    to send X-API-Key. This accepts the key as a query parameter as well —
    narrowly, on this one read-only endpoint, rather than weakening
    verify_api_key for every route.

    The tradeoff is real: query strings are likelier to end up in access
    logs and proxy logs than headers are. It is accepted here because this
    endpoint is read-only, returns nothing that isn't already available
    from GET /api/explanations/{id}, and the alternative (hand-rolling SSE
    parsing over fetch to get a header) buys nothing for a demo-scale
    deployment. A production deployment should hand the browser a
    short-lived stream token instead.
    """
    verify_api_key(provided_key if provided_key is not None else api_key)


@app.get("/api/verdicts/{verdict_id}/stream", dependencies=[Depends(verify_api_key_or_query)])
def stream_verdict(verdict_id: str):
    """Server-Sent Events for one verdict: the decision immediately, then
    the explanation as the model produces it.

    SSE rather than WebSockets because this is strictly one-directional
    server->client: SSE reconnects natively, survives proxies better, and
    needs no new dependency.

    Emits:
      decision              — score/action/escalation, available at once
      explanation_delta     — incremental text as the LLM produces it
      explanation_complete  — the final verdict, same shape as
                              GET /api/explanations/{verdict_id}
      error                 — terminal, carrying the same fallback verdict
                              llm_agent already produces on failure

    GET /api/explanations/{verdict_id} remains as the polling fallback for
    clients behind a proxy that buffers event streams.
    """
    cached = _explanations_cache.get(verdict_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Unknown verdict_id")

    def event_source():
        # The decision itself is already final by the time any client can
        # subscribe — scoring never waits for this stream.
        yield _sse("decision", {"verdict_id": verdict_id, "status": cached.get("status", "pending")})

        # Already finished (a reconnect, or a fast model): replay the
        # terminal event rather than waiting for a delta that will never
        # come.
        if cached.get("status") == "ready":
            yield _sse("explanation_complete", cached["verdict"])
            return

        # Attach BEFORE re-reading the cache. The LLM call starts as soon
        # as the score response goes out, so its first deltas are often
        # published in the milliseconds before this connection exists —
        # and pub/sub has no retention. Subscribing first, then checking
        # the cache, leaves no window where a result is in neither place.
        subscription = _explanation_bus.subscription(verdict_id)
        settled = _explanations_cache.get(verdict_id) or {}
        if settled.get("status") == "ready":
            subscription.close()
            yield _sse("explanation_complete", settled["verdict"])
            return

        ticks = 0
        for message in subscription.messages():
            if message is None:
                # Idle tick. Pub/sub has no retention, so a verdict that
                # completed in the window between the cache check above
                # and the subscription would otherwise never reach this
                # client. Re-reading the cache here closes that race
                # instead of leaving the connection hanging.
                latest = _explanations_cache.get(verdict_id) or {}
                if latest.get("status") == "ready":
                    yield _sse("explanation_complete", latest["verdict"])
                    return
                ticks += 1
                if ticks >= STREAM_MAX_IDLE_TICKS:
                    # The producer is gone (worker died mid-call). Close
                    # with a terminal error rather than holding a
                    # connection open forever.
                    yield _sse("error", dict(FALLBACK_VERDICT))
                    return
                # Keepalive comment — not an event, just bytes on the wire
                # so an idle proxy doesn't close the connection.
                yield ": keepalive\n\n"
                continue
            if message["type"] == "delta":
                yield _sse("explanation_delta", {"text": message["text"]})
            elif message["type"] == "complete":
                yield _sse("explanation_complete", message["verdict"])
                return

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx buffers proxied responses by default, which would hold
            # every delta until the stream closed.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/explanations/{verdict_id}")
def get_explanation(verdict_id: str):
    explanation = _explanations_cache.get(verdict_id)
    if explanation is None:
        raise HTTPException(status_code=404, detail="Unknown verdict_id")
    return explanation


@router.get("/api/review-queue")
def list_review_queue(status: str = "pending"):
    if status != "pending":
        raise HTTPException(status_code=400, detail="Only status=pending is currently supported")
    return {"items": _review_queue.list_pending()}


@router.post("/api/review-queue/{verdict_id}/disposition")
def dispose_review_item(verdict_id: str, req: DispositionRequest):
    try:
        return _review_queue.dispose(verdict_id, req.disposition)
    except UnknownVerdictError:
        raise HTTPException(status_code=404, detail="Unknown verdict_id")
    except AlreadyDisposedError:
        raise HTTPException(status_code=409, detail="This item has already been disposed")


@router.get("/api/review-queue/metrics")
def get_review_queue_metrics():
    return _review_queue.metrics()


@router.post("/api/review-queue/{verdict_id}/notes")
def add_review_note(verdict_id: str, req: AddNoteRequest):
    try:
        return _review_queue.add_note(verdict_id, req.author, req.text)
    except UnknownVerdictError:
        raise HTTPException(status_code=404, detail="Unknown verdict_id")


@router.get("/api/review-queue/{verdict_id}/related")
def get_related_review_items(verdict_id: str):
    try:
        return {"items": _review_queue.related(verdict_id)}
    except UnknownVerdictError:
        raise HTTPException(status_code=404, detail="Unknown verdict_id")


@router.get("/api/cost-analysis")
def get_cost_analysis(
    fraud_loss: float = DEFAULT_AVG_FRAUD_LOSS,
    fp_cost: float = DEFAULT_AVG_FP_COST,
):
    eval_report = None
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH, encoding="utf-8") as f:
            eval_report = f.read()

    # Phase H's headline number — a linear extrapolation of the real,
    # measured cost-optimal-threshold savings to an assumed monthly
    # volume (see src/impact_summary.py). Only computable once
    # train_model.py has produced models/cost_summary.json; None (with
    # no basis) otherwise, same "generate one first" pattern as every
    # other report field on this endpoint.
    cost_curve_rows = []
    if os.path.exists(COST_CURVE_PATH):
        with open(COST_CURVE_PATH, encoding="utf-8") as f:
            cost_curve_rows = json.load(f)

    headline = None
    cost_summary = {}
    if os.path.exists(COST_SUMMARY_PATH):
        with open(COST_SUMMARY_PATH, encoding="utf-8") as f:
            cost_summary = json.load(f)
        headline = extrapolate_monthly_savings(
            cost_summary["estimated_savings"], cost_summary["n_test_transactions"]
        )

    return {
        "eval_report": eval_report,
        "defaults": {
            "avg_fraud_loss": DEFAULT_AVG_FRAUD_LOSS,
            "avg_fp_cost": DEFAULT_AVG_FP_COST,
        },
        "params": {"fraud_loss": fraud_loss, "fp_cost": fp_cost},
        "headline_monthly_savings_estimate": headline["headline_monthly_savings_estimate"] if headline else None,
        "headline_basis": headline["basis"] if headline else None,
        # The live decision boundary and escalation cutoffs, so the
        # frontend's "At a glance" panel reads the real numbers this
        # process is actually deciding with rather than restating
        # hardcoded copies of them.
        # Total cost is a pure function of the persisted error counts and
        # the two cost assumptions (see src/cost_analysis.py's cost_curve),
        # so the curve follows the caller's fraud_loss/fp_cost without
        # re-scoring the test set.
        "cost_curve": [
            {
                "threshold": row["threshold"],
                "total_cost": row["false_negatives"] * fraud_loss + row["false_positives"] * fp_cost,
            }
            for row in cost_curve_rows
        ],
        "decision_thresholds": get_decision_thresholds(),
        "escalation_cutoffs": {
            "watch": DEFAULT_WATCH_PRESSURE_THRESHOLD,
            "elevated": DEFAULT_ELEVATED_PRESSURE_THRESHOLD,
        },
        "roc_auc": cost_summary.get("roc_auc"),
    }


@router.get("/api/cost-analysis/sensitivity")
def get_cost_sensitivity():
    if not os.path.exists(COST_SENSITIVITY_REPORT_PATH):
        return {
            "sensitivity": None,
            "message": "No sensitivity sweep yet — run `python src/cost_sensitivity.py` to generate one.",
        }
    with open(COST_SENSITIVITY_REPORT_PATH, encoding="utf-8") as f:
        return {"sensitivity": json.load(f), "message": None}


@router.get("/api/escalation-ablation")
def get_escalation_ablation():
    if not os.path.exists(ESCALATION_ABLATION_REPORT_PATH):
        return {
            "report": None,
            "summary": None,
            "message": "No ablation report yet — run `python src/escalation_ablation.py` to generate one.",
        }
    summary = None
    if os.path.exists(ESCALATION_ABLATION_SUMMARY_PATH):
        with open(ESCALATION_ABLATION_SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)
    with open(ESCALATION_ABLATION_REPORT_PATH, encoding="utf-8") as f:
        return {"report": f.read(), "summary": summary, "message": None}


@router.get("/api/drift-analysis")
def get_drift_analysis():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return {
            "drift": None,
            "message": "No drift report yet — run `python src/drift_analysis.py` to generate one.",
        }
    with open(DRIFT_REPORT_PATH, encoding="utf-8") as f:
        return {"drift": json.load(f), "message": None}


@router.get("/api/consistency-analysis")
def get_consistency_analysis():
    if not os.path.exists(CONSISTENCY_REPORT_PATH):
        return {
            "consistency": None,
            "message": "No consistency report yet — run `python src/consistency_analysis.py` to generate one.",
        }
    with open(CONSISTENCY_REPORT_PATH, encoding="utf-8") as f:
        return {"consistency": json.load(f), "message": None}


@router.get("/api/cold-start-analysis")
def get_cold_start_analysis():
    if not os.path.exists(COLD_START_REPORT_PATH):
        return {
            "report": None,
            "message": "No cold-start report yet — run `python src/graph_features_ablation.py` to generate one.",
        }
    with open(COLD_START_REPORT_PATH, encoding="utf-8") as f:
        return {"report": f.read(), "message": None}


@router.get("/api/shadow-comparison")
def get_shadow_comparison():
    """How often would SHADOW_MODEL_PATH's decision have differed from
    the one that actually shipped, on the transactions scored since this
    process started (or since Redis-backed counts were last reset)."""
    if _shadow_scorer is None:
        return {
            "configured": False,
            "total_scored": 0,
            "agreement_rate": None,
            "action_pairs": [],
            "message": "No shadow model configured — set SHADOW_MODEL_PATH to compare a "
                       "candidate model against live decisions before promoting it.",
        }
    return {**_shadow_comparison.summary(), "message": None}


@router.get("/api/audit/{verdict_id}")
def get_audit_entry(verdict_id: str):
    """The immutable record ScoringService wrote for this verdict — see
    src/audit_log.py. Distinct from GET /api/explanations/{verdict_id}
    (the LLM's explanation, best-effort and re-generatable) and GET
    /api/review-queue/{verdict_id} (the reviewer workflow state, which
    changes as a human disposes it): this is what the system actually
    decided, at the moment it decided it, and its hash chain is what
    makes that claim checkable rather than just asserted."""
    entry = _audit_log.get(verdict_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="No audit entry for this verdict_id")
    return entry


@app.get("/api/health")
def health():
    """Liveness plus the state of the one dependency that can degrade
    without any HTTP metric noticing.

    "status" stays "ok" even with the LLM breaker open: explanations are
    best-effort and their failure is deliberately not a failure of this
    service — scoring is unaffected. The llm block is what turns "the
    explanations stopped working" into something diagnosable.
    """
    try:
        llm = get_agent().breaker.state()
    except Exception:
        # Agent construction itself failed (e.g. no credentials). Report
        # it rather than 500-ing the health check.
        logger.warning("Could not read LLM breaker state", exc_info=True)
        llm = {"state": "unknown", "consecutive_failures": 0, "seconds_until_retry": 0.0}
    return {
        "status": "ok",
        "llm": llm,
        # Seeded vs. live counts: a store that looks populated but was
        # never seeded would score early transactions against empty
        # history, and nothing else would show that.
        "feature_store": {**_feature_store.stats(), **_feature_store_seed},
        "escalation_alerts": {
            "webhook_configured": bool(os.environ.get("ESCALATION_WEBHOOK_URL")),
        },
        "shadow_scoring": {"configured": _shadow_scorer is not None},
        # Not an entry count: entries() reads the whole log (a file scan
        # locally, an LRANGE 0 -1 in Redis) — fine for GET
        # /api/audit/{verdict_id} and `python -m src.audit verify`, too
        # expensive to pay on every liveness probe.
        "audit_log": {"backend": "redis" if _redis_client is not None else "local_file"},
    }


app.include_router(router)
