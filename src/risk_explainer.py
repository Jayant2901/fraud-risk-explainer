"""
Wraps the trained model + SHAP so we can, for a single transaction:
  1. Get a 0-100 risk score
  2. Get the top N features driving that score, in plain terms

We never hand raw SHAP arrays to the LLM — we hand it a small, clean
list of (feature, value, contribution, human_label) tuples.
"""
import joblib
import shap
import pandas as pd

MODEL_PATH = "models/risk_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"
THRESHOLD_PATH = "models/optimal_threshold.joblib"
CATEGORIES_PATH = "models/categorical_categories.joblib"

FEATURE_LABELS = {
    "TransactionAmt": "transaction amount",
    "amount_log": "transaction amount (log-scaled)",
    "hour_of_day": "hour of day",
    "is_night_txn": "late-night transaction flag",
    "card1": "card fingerprint (primary)",
    "card2": "card fingerprint (secondary)",
    "card3": "card issuing bank code",
    "card4": "card network (Visa/Mastercard/etc)",
    "card5": "card fingerprint (tertiary)",
    "card6": "card type (debit/credit)",
    "addr1": "billing region code",
    "addr2": "billing country code",
    "dist1": "distance from billing address",
    "dist2": "distance from shipping address",
    "ProductCD": "product category code",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "DeviceType": "device type used",
    "entity_prior_txn_count": "this entity's prior transaction count",
    "entity_prior_fraud_count": "this entity's prior flagged-fraud count",
    "entity_prior_fraud_rate": "this entity's historical fraud rate",
    "shared_device_prior_entity_count": "distinct entities previously seen on this device/address",
    "shared_device_prior_fraud_rate": "fraud rate previously seen on this device/address (any entity)",
}


def human_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("C") and feature[1:].isdigit():
        return f"related-transaction count signal ({feature})"
    if feature.startswith("D") and feature[1:].isdigit():
        return f"time-since-last-event signal ({feature})"
    if feature.startswith("V") and feature[1:].isdigit():
        return f"anonymized behavioral signal ({feature})"
    if feature.startswith("M") and feature[1:].isdigit():
        return f"identity/address match flag ({feature})"
    return feature


class RiskExplainer:
    def __init__(self, model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH,
                 threshold_path: str = THRESHOLD_PATH, categories_path: str = CATEGORIES_PATH):
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)
        try:
            self.threshold = joblib.load(threshold_path)
        except FileNotFoundError:
            self.threshold = 0.5
        self.categories_map = joblib.load(categories_path)
        self.explainer = shap.TreeExplainer(self.model)

    def score_transaction(self, txn: dict) -> dict:
        """
        txn: dict of feature_name -> value (missing ones filled with NaN,
             which XGBoost's native categorical/missing-value handling
             deals with directly — no manual imputation needed).
        """
        row = {c: txn.get(c, None) for c in self.feature_cols}
        X = pd.DataFrame([row])[self.feature_cols]

        # Rebuilding a DataFrame from a plain dict loses pandas' 'category'
        # dtype, and naively re-casting with astype('category') on a
        # SINGLE row derives the category set from just that one value —
        # XGBoost's categorical predict path rejects that (it needs the
        # same category set seen at training time). So we explicitly
        # apply the fixed training-time category list here instead.
        for col, categories in self.categories_map.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=categories)

        proba = float(self.model.predict_proba(X)[0, 1])
        risk_score = round(proba * 100, 1)

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = shap_values[0]

        factors = []
        for feat, val, contrib in zip(self.feature_cols, X.iloc[0].values, contributions):
            factors.append({
                "feature": feat,
                "label": human_label(feat),
                "value": str(val),
                "contribution": round(float(contrib), 4),
            })

        factors_sorted = sorted(factors, key=lambda f: f["contribution"], reverse=True)
        top_factors = [f for f in factors_sorted if f["contribution"] > 0][:4]
        if not top_factors:
            top_factors = factors_sorted[:4]

        return {
            "risk_score": risk_score,
            "raw_proba": proba,
            "above_threshold": proba >= self.threshold,
            "top_factors": top_factors,
        }