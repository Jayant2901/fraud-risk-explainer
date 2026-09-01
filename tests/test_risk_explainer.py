import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from risk_explainer import RiskExplainer, human_label

FEATURE_COLS = ["amount", "hour", "category_col"]


@pytest.fixture
def trained_artifacts(tmp_path):
    """A tiny real XGBoost model (not the 590K-row IEEE-CIS CSVs) with one
    categorical column, saved to disk exactly like train_model.py does,
    so RiskExplainer's actual joblib.load() + SHAP path gets exercised."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "amount": rng.uniform(10, 500, n),
        "hour": rng.integers(0, 24, n),
        "category_col": rng.choice(["A", "B", "C"], n),
    })
    df["category_col"] = df["category_col"].astype("category")
    y = (df["amount"] > 250).astype(int)  # simple learnable signal

    model = XGBClassifier(
        n_estimators=15, max_depth=2, enable_categorical=True,
        tree_method="hist", random_state=0,
    )
    model.fit(df, y)

    categories_map = {"category_col": list(df["category_col"].cat.categories)}

    paths = {
        "model_path": str(tmp_path / "model.joblib"),
        "features_path": str(tmp_path / "features.joblib"),
        "threshold_path": str(tmp_path / "threshold.joblib"),
        "categories_path": str(tmp_path / "categories.joblib"),
    }
    joblib.dump(model, paths["model_path"])
    joblib.dump(FEATURE_COLS, paths["features_path"])
    joblib.dump(0.5, paths["threshold_path"])
    joblib.dump(categories_map, paths["categories_path"])
    return paths


class TestScoreTransaction:
    def test_returns_a_score_in_range(self, trained_artifacts):
        explainer = RiskExplainer(**trained_artifacts)
        result = explainer.score_transaction({"amount": 300.0, "hour": 14, "category_col": "A"})
        assert 0 <= result["risk_score"] <= 100
        assert 0 <= result["raw_proba"] <= 1
        assert isinstance(result["above_threshold"], (bool, np.bool_))

    def test_top_factors_capped_at_four(self, trained_artifacts):
        explainer = RiskExplainer(**trained_artifacts)
        result = explainer.score_transaction({"amount": 300.0, "hour": 14, "category_col": "A"})
        assert len(result["top_factors"]) <= 4
        for factor in result["top_factors"]:
            assert set(factor.keys()) == {"feature", "label", "value", "contribution"}

    def test_nan_valued_numeric_feature_does_not_crash(self, trained_artifacts):
        # In real usage txn always comes from a DataFrame row (df.iloc[idx]
        # .to_dict()) — every column present, real missingness is NaN, not
        # an absent key. XGBoost's native missing-value handling (see the
        # module docstring) is what's actually being exercised here.
        explainer = RiskExplainer(**trained_artifacts)
        result = explainer.score_transaction({"amount": float("nan"), "hour": 10, "category_col": "A"})
        assert 0 <= result["risk_score"] <= 100

    def test_single_row_categorical_dtype_regression(self, trained_artifacts):
        """
        Regression test for the bug described in this module's own
        docstring: rebuilding a single-row DataFrame and naively calling
        astype('category') derives the category set from just that ONE
        value, which XGBoost's categorical predict path rejects (the
        codes don't match what the model was trained on). RiskExplainer
        must instead re-cast against the saved categories_map. If that
        re-cast were ever removed, this raises inside score_transaction
        instead of silently returning a wrong score.
        """
        explainer = RiskExplainer(**trained_artifacts)
        for category_value in ["A", "B", "C"]:
            result = explainer.score_transaction({"amount": 100.0, "hour": 5, "category_col": category_value})
            assert 0 <= result["risk_score"] <= 100

    def test_uses_default_threshold_when_threshold_file_is_missing(self, trained_artifacts, tmp_path):
        explainer = RiskExplainer(
            model_path=trained_artifacts["model_path"],
            features_path=trained_artifacts["features_path"],
            threshold_path=str(tmp_path / "does_not_exist.joblib"),
            categories_path=trained_artifacts["categories_path"],
        )
        assert explainer.threshold == 0.5

    def test_higher_amount_scores_at_least_as_risky(self, trained_artifacts):
        # Not a strict monotonicity guarantee for an arbitrary model, but
        # for this fixture's obviously-learnable signal (amount > 250 ->
        # fraud), a clearly-above-threshold amount should score >= a
        # clearly-below-threshold one.
        explainer = RiskExplainer(**trained_artifacts)
        low = explainer.score_transaction({"amount": 20.0, "hour": 10, "category_col": "A"})
        high = explainer.score_transaction({"amount": 480.0, "hour": 10, "category_col": "A"})
        assert high["risk_score"] >= low["risk_score"]


class TestHumanLabel:
    def test_known_feature_names_use_the_explicit_mapping(self):
        assert human_label("TransactionAmt") == "transaction amount"
        assert human_label("P_emaildomain") == "purchaser email domain"

    def test_c_columns_get_the_count_signal_label(self):
        assert human_label("C5") == "related-transaction count signal (C5)"

    def test_d_columns_get_the_time_since_event_label(self):
        assert human_label("D3") == "time-since-last-event signal (D3)"

    def test_v_columns_get_the_behavioral_signal_label(self):
        assert human_label("V10") == "anonymized behavioral signal (V10)"

    def test_m_columns_get_the_match_flag_label(self):
        assert human_label("M2") == "identity/address match flag (M2)"

    def test_unrecognized_feature_falls_back_to_the_raw_name(self):
        assert human_label("some_unmapped_feature") == "some_unmapped_feature"
