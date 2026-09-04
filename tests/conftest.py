import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

TEST_API_KEY = "test-secret-key-for-tests"


class FakeExplainer:
    """Stands in for RiskExplainer — returns a fixed score/factors shape
    without needing a real trained model or SHAP explainer."""

    def __init__(self, risk_score: float = 42.0, above_threshold: bool = False):
        self.risk_score = risk_score
        self.above_threshold = above_threshold

    def score_transaction(self, txn: dict) -> dict:
        return {
            "risk_score": self.risk_score,
            "raw_proba": self.risk_score / 100,
            "above_threshold": self.above_threshold,
            "top_factors": [
                {"feature": "TransactionAmt", "label": "transaction amount", "value": "100.0", "contribution": 0.5},
            ],
        }


class FakeAgent:
    """Stands in for RiskExplainerAgent — returns a fixed verdict without
    needing a real Gemini API key or network access."""

    def explain(self, risk_score: float, top_factors: list, escalation: dict | None = None) -> dict:
        return {
            "explanation": "fake explanation",
            "action": "REVIEW",
            "escalated_due_to_history": False,
            "rationale": "fake rationale",
        }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal stand-in for get_sample_data()'s output — only the columns
    api/main.py's route bodies actually read directly."""
    return pd.DataFrame([
        {"entity_id": "entity-a", "TransactionAmt": 100.0, "TransactionDT": 1000, "ProductCD": "W"},
        {"entity_id": "entity-a", "TransactionAmt": 250.0, "TransactionDT": 2000, "ProductCD": "W"},
        {"entity_id": "entity-b", "TransactionAmt": 50.0, "TransactionDT": 1500, "ProductCD": "C"},
    ])


@pytest.fixture
def client(monkeypatch, sample_df, tmp_path):
    """A TestClient wired to a fully-faked app: real API_KEY auth (set to
    TEST_API_KEY), fake data/model/LLM so no trained model, dataset, or
    GEMINI_API_KEY is needed, and fresh shared state per test."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    import api.main as main
    from audit_log import AuditLog

    # The audit log is always-on and local-file-backed by default
    # (src/audit_log.py) — every scored transaction in every test would
    # otherwise append to the real data/audit_log.jsonl in the repo.
    # Redirected to a per-test temp file so the suite still exercises the
    # real local-file backend, just without polluting the working tree.
    monkeypatch.setattr(main, "_audit_log", AuditLog(redis_client=None, log_path=str(tmp_path / "audit_log.jsonl")))

    main.get_sample_data.cache_clear()
    main.get_explainer.cache_clear()
    main.get_agent.cache_clear()
    main.get_decision_thresholds.cache_clear()
    monkeypatch.setattr(main, "get_sample_data", lambda: sample_df)
    monkeypatch.setattr(main, "get_explainer", lambda: FakeExplainer())
    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent())
    # Inject fixed, explicit thresholds rather than letting tests depend
    # on whatever models/decision_thresholds.joblib happens to contain
    # (or not) on the machine running them — every existing test's
    # assertions (e.g. FakeExplainer's fixed risk_score=42.0 -> REVIEW)
    # were written against the original 40.0/80.0 boundaries.
    monkeypatch.setattr(main, "get_decision_thresholds", lambda: {"review": 40.0, "block": 80.0})

    # Module-level singletons persist across tests otherwise (escalation
    # history, idempotency cache, pending explanations, and the rate
    # limiter's request counts — all tests share TEST_API_KEY, so without
    # this every test's /api/score calls would count against the SAME
    # 30/minute bucket and tests could start flakily 429ing each other).
    main._memory.reset()
    main._explanations_cache.clear()
    main._idempotency_cache.clear()
    main._review_queue.reset()
    main.limiter._storage.reset()

    from fastapi.testclient import TestClient

    return TestClient(main.app)


@pytest.fixture
def auth_headers() -> dict:
    return {"X-API-Key": TEST_API_KEY}
