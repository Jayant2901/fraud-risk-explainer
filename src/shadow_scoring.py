"""
Shadow scoring: a candidate model scores every transaction alongside the
live one, silently, so its decisions can be compared against what
actually shipped before anyone decides to promote it. SHADOW_MODEL_PATH
unset (the default) means this module does nothing at all.

Predict-only, deliberately. RiskExplainer's SHAP computation (a
TreeExplainer construction plus a shap_values() call per transaction) is
the expensive part of scoring, and a shadow model's whole point is "would
this transaction's decision differ" — a risk score and the decision it
implies, not an explanation nobody will ever read. Duplicating SHAP's
cost on every real request for a feature that returns nothing to that
request would spend exactly the latency budget the "Latency budget: the
LLM can never degrade the decision" section of the README holds the LLM
to, for no visible benefit.

Comparison counts follow the same REDIS_URL-optional dual-backend
pattern as circuit_breaker.py: shared across workers when Redis is
configured, in-process (and reset on restart) otherwise.
"""
import os

import joblib
import pandas as pd

REDIS_KEY_PREFIX = "riskmgr:shadow"

SHADOW_MODEL_PATH = os.environ.get("SHADOW_MODEL_PATH")
FEATURES_PATH = "models/feature_cols.joblib"
CATEGORIES_PATH = "models/categorical_categories.joblib"


class ShadowScorer:
    """Assumes the shadow model was trained on the SAME feature schema as
    the live one (models/feature_cols.joblib, models/categorical_categories
    .joblib) — the intended use is comparing model versions (e.g. with vs.
    without the feedback loop, src/train_model.py --with-feedback), not
    models with a different feature set."""

    def __init__(self, model_path: str, features_path: str = FEATURES_PATH,
                 categories_path: str = CATEGORIES_PATH):
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)
        self.categories_map = joblib.load(categories_path)

    def score(self, txn: dict) -> float:
        row = {c: txn.get(c, None) for c in self.feature_cols}
        X = pd.DataFrame([row])[self.feature_cols]
        # Same category-dtype rebuild RiskExplainer.score_transaction does
        # — a single-row DataFrame loses pandas' 'category' dtype, and
        # XGBoost's categorical predict path needs the training-time
        # category set restored explicitly rather than re-derived from
        # this one row.
        for col, categories in self.categories_map.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=categories)
        # Same None -> object-dtype trap RiskExplainer.score_transaction
        # fixes and documents in full: a missing numeric column becomes
        # pandas 'object' dtype in a single-row DataFrame, which XGBoost
        # rejects outright. Coerce to numeric so None becomes NaN.
        numeric_cols = [c for c in X.columns if c not in self.categories_map]
        X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
        proba = float(self.model.predict_proba(X)[0, 1])
        return round(proba * 100, 1)


class ShadowComparison:
    """Running agreement counts between the live decision and the shadow
    model's decision on the same transaction, under the same escalation
    state and thresholds — the only two things allowed to differ are the
    two models' risk scores."""

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._total = 0
        self._agree = 0
        self._pairs: dict[str, int] = {}

    def record(self, live_action: str, shadow_action: str) -> None:
        pair_key = f"{live_action}:{shadow_action}"
        if self._redis is None:
            self._total += 1
            if live_action == shadow_action:
                self._agree += 1
            self._pairs[pair_key] = self._pairs.get(pair_key, 0) + 1
            return
        pipe = self._redis.pipeline()
        pipe.incr(f"{REDIS_KEY_PREFIX}:total")
        if live_action == shadow_action:
            pipe.incr(f"{REDIS_KEY_PREFIX}:agree")
        pipe.hincrby(f"{REDIS_KEY_PREFIX}:pairs", pair_key, 1)
        pipe.execute()

    def summary(self) -> dict:
        if self._redis is None:
            total, agree, pairs = self._total, self._agree, dict(self._pairs)
        else:
            total = int(self._redis.get(f"{REDIS_KEY_PREFIX}:total") or 0)
            agree = int(self._redis.get(f"{REDIS_KEY_PREFIX}:agree") or 0)
            pairs = {k: int(v) for k, v in (self._redis.hgetall(f"{REDIS_KEY_PREFIX}:pairs") or {}).items()}

        action_pairs = [
            {"live_action": key.split(":", 1)[0], "shadow_action": key.split(":", 1)[1], "count": count}
            for key, count in sorted(pairs.items())
        ]
        return {
            "configured": True,
            "total_scored": total,
            "agreement_rate": round(agree / total, 4) if total else None,
            "action_pairs": action_pairs,
        }

    def reset(self) -> None:
        """Test-only."""
        if self._redis is None:
            self._total = 0
            self._agree = 0
            self._pairs = {}
            return
        self._redis.delete(
            f"{REDIS_KEY_PREFIX}:total", f"{REDIS_KEY_PREFIX}:agree", f"{REDIS_KEY_PREFIX}:pairs"
        )


def create_shadow_scorer() -> ShadowScorer | None:
    """None when SHADOW_MODEL_PATH is unset — shadow scoring is opt-in,
    same as the feedback loop and escalation alerting."""
    if not SHADOW_MODEL_PATH:
        return None
    return ShadowScorer(SHADOW_MODEL_PATH)


def create_shadow_comparison(redis_client=None) -> ShadowComparison:
    return ShadowComparison(redis_client)
