"""
Online feature store tests.

The one that matters is TestOfflineOnlineEquivalence: replay historical
rows through the online store one at a time and assert the features match
what the offline causal functions produce for the same rows in batch. If
they disagree, the live system is serving the model features it was not
trained on — which is the whole reason this module exists.
"""
import fakeredis
import numpy as np
import pandas as pd
import pytest

from data_utils import add_causal_entity_history
from feature_store import (
    FeatureStore,
    RedisFeatureStore,
    create_feature_store,
    fingerprint_for,
    seed_from_history,
)
from graph_features import GRAPH_FEATURE_COLS, add_causal_device_graph_features

ENTITY_FEATURE_COLS = [
    "entity_prior_txn_count",
    "entity_prior_fraud_count",
    "entity_prior_fraud_rate",
]
ALL_FEATURE_COLS = ENTITY_FEATURE_COLS + GRAPH_FEATURE_COLS


@pytest.fixture(params=["in-process", "redis"])
def store(request):
    if request.param == "redis":
        return RedisFeatureStore(fakeredis.FakeRedis(decode_responses=True))
    return FeatureStore()


def synthetic_transactions(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Entities and devices deliberately overlap — several entities share
    a device, several devices serve one entity — so the distinct-entity
    graph feature is actually exercised rather than trivially 1."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionDT": np.arange(n) * 60 + rng.integers(0, 30, n),
        "entity_id": [f"entity-{i % 37}" for i in rng.integers(0, 200, n)],
        "DeviceInfo": [
            None if i % 5 == 0 else f"device-{i % 23}" for i in rng.integers(0, 200, n)
        ],
        "addr1": rng.integers(100, 110, n).astype(float),
        "addr2": rng.integers(80, 82, n).astype(float),
        "isFraud": (rng.random(n) < 0.15).astype(int),
        "TransactionAmt": rng.random(n) * 500,
    })


def offline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly what the training pipeline computes, in batch."""
    with_history = add_causal_entity_history(df.copy())
    return add_causal_device_graph_features(with_history)


def replay_online(df: pd.DataFrame, store) -> pd.DataFrame:
    """Feed rows through the store one at a time, in the order a live
    stream would deliver them: read features, then record."""
    rows = []
    for row in df.sort_values("TransactionDT").to_dict("records"):
        entity_id = row["entity_id"]
        fingerprint = fingerprint_for(row)
        features = store.features_for(entity_id, fingerprint)
        features["TransactionDT"] = row["TransactionDT"]
        features["entity_id"] = entity_id
        rows.append(features)
        # AFTER the read — recording first would make the transaction
        # count itself.
        store.record(entity_id, fingerprint, int(row["isFraud"]))
    return pd.DataFrame(rows)


class TestOfflineOnlineEquivalence:
    """The acceptance criterion for RT-4."""

    def _compare(self, n, store):
        df = synthetic_transactions(n)
        # Distinct timestamps only: with ties, "strictly earlier" is
        # ambiguous and batch/stream orderings can legitimately differ.
        df = df.drop_duplicates(subset=["TransactionDT"]).sort_values("TransactionDT")

        offline = offline_features(df).sort_values("TransactionDT").reset_index(drop=True)
        online = replay_online(df, store).sort_values("TransactionDT").reset_index(drop=True)

        deltas = {
            col: float((offline[col].astype(float) - online[col].astype(float)).abs().max())
            for col in ALL_FEATURE_COLS
        }
        return deltas

    def test_online_matches_offline_across_a_thousand_transactions(self, store):
        deltas = self._compare(1000, store)

        # Reported in the summary as a number, per the phase's own
        # instruction — not "they match".
        print("\nmax |offline - online| per feature:", deltas)
        for col, delta in deltas.items():
            assert delta < 1e-9, f"{col} diverged by {delta}"

    def test_entity_history_features_match(self, store):
        deltas = self._compare(300, store)
        assert all(deltas[col] < 1e-9 for col in ENTITY_FEATURE_COLS)

    def test_device_graph_features_match(self, store):
        deltas = self._compare(300, store)
        assert all(deltas[col] < 1e-9 for col in GRAPH_FEATURE_COLS)


class TestUpdateAfterReadOrdering:
    """Recording before reading reintroduces exactly the leakage the
    offline functions' shift(1) exists to prevent."""

    def test_a_transaction_never_counts_itself(self, store):
        first = store.features_for("e1", "device-1")
        store.record("e1", "device-1", is_fraud=1)
        second = store.features_for("e1", "device-1")

        assert first["entity_prior_txn_count"] == 0.0
        assert first["entity_prior_fraud_count"] == 0.0
        assert second["entity_prior_txn_count"] == 1.0
        assert second["entity_prior_fraud_count"] == 1.0

    def test_the_first_transaction_of_an_entity_has_no_history(self, store):
        assert store.features_for("brand-new", "device-x") == {
            "entity_prior_txn_count": 0.0,
            "entity_prior_fraud_count": 0.0,
            "entity_prior_fraud_rate": 0.0,
            "shared_device_prior_entity_count": 0.0,
            "shared_device_prior_fraud_rate": 0.0,
        }

    def test_fraud_rate_is_zero_rather_than_undefined_with_no_priors(self, store):
        assert store.features_for("e", "d")["entity_prior_fraud_rate"] == 0.0


class TestDeviceGraphSemantics:
    def test_counts_distinct_entities_not_transactions(self, store):
        for _ in range(3):
            store.record("entity-a", "shared-device", is_fraud=0)
        store.record("entity-b", "shared-device", is_fraud=0)

        features = store.features_for("entity-c", "shared-device")

        # Two distinct entities have used it, across four transactions.
        assert features["shared_device_prior_entity_count"] == 2.0

    def test_fraud_rate_spans_every_entity_on_that_device(self, store):
        store.record("entity-a", "shared-device", is_fraud=1)
        store.record("entity-b", "shared-device", is_fraud=0)

        # A brand-new entity with no history of its own still inherits the
        # device's signal — the entire point of the cold-start feature.
        features = store.features_for("brand-new-entity", "shared-device")

        assert features["entity_prior_txn_count"] == 0.0
        assert features["shared_device_prior_fraud_rate"] == 0.5

    def test_a_transaction_with_no_fingerprint_gets_zero_not_a_shared_bucket(self, store):
        store.record("entity-a", None, is_fraud=1)

        features = store.features_for("entity-b", None)

        assert features["shared_device_prior_entity_count"] == 0.0
        assert features["shared_device_prior_fraud_rate"] == 0.0


class TestFingerprintDerivation:
    def test_prefers_device_info_when_present(self):
        assert fingerprint_for({"DeviceInfo": "iPhone", "addr1": 100.0, "addr2": 87.0}) == "iPhone"

    def test_falls_back_to_the_address_pair(self):
        assert fingerprint_for({"addr1": 100.0, "addr2": 87.0}) == "ADDR_100.0_87.0"

    def test_returns_none_when_neither_is_available(self):
        assert fingerprint_for({"TransactionAmt": 10.0}) is None

    def test_a_partial_address_is_not_a_fingerprint(self):
        # Half an address would falsely link unrelated entities.
        assert fingerprint_for({"addr1": 100.0}) is None


class TestLateArrivingLabels:
    """Online, a transaction's true label arrives days later — from a
    reviewer disposition or a chargeback — not at scoring time."""

    def test_an_unlabelled_transaction_counts_toward_volume_only(self, store):
        store.record("e1", "d1", is_fraud=None)

        features = store.features_for("e2", "d1")
        assert store.features_for("e1", "d1")["entity_prior_txn_count"] == 1.0
        assert features["shared_device_prior_fraud_rate"] == 0.0

    def test_a_later_label_updates_the_fraud_side_without_double_counting_volume(self, store):
        store.record("e1", "d1", is_fraud=None)
        store.apply_label("e1", "d1", is_fraud=True)

        features = store.features_for("e1", "d1")

        assert features["entity_prior_txn_count"] == 1.0  # not 2
        assert features["entity_prior_fraud_count"] == 1.0
        assert features["entity_prior_fraud_rate"] == 1.0

    def test_a_false_positive_disposition_changes_nothing(self, store):
        store.record("e1", "d1", is_fraud=None)
        store.apply_label("e1", "d1", is_fraud=False)

        assert store.features_for("e1", "d1")["entity_prior_fraud_count"] == 0.0


class TestBackendParity:
    def test_both_backends_produce_identical_features(self):
        df = synthetic_transactions(200).drop_duplicates(subset=["TransactionDT"])

        in_process = replay_online(df, FeatureStore())
        redis_backed = replay_online(
            df, RedisFeatureStore(fakeredis.FakeRedis(decode_responses=True))
        )

        for col in ALL_FEATURE_COLS:
            assert (in_process[col] - redis_backed[col]).abs().max() < 1e-9

    def test_the_factory_picks_the_backend_from_the_client(self):
        assert isinstance(create_feature_store(None), FeatureStore)
        assert isinstance(
            create_feature_store(fakeredis.FakeRedis(decode_responses=True)), RedisFeatureStore
        )


class TestSeeding:
    def test_seeding_warms_the_store_from_history(self, store):
        df = synthetic_transactions(50)

        result = seed_from_history(store, df)

        assert result["seeded_rows"] == 50
        # An entity present in the seed data is no longer cold.
        seeded_entity = df.sort_values("TransactionDT").iloc[0]["entity_id"]
        assert store.features_for(seeded_entity, None)["entity_prior_txn_count"] > 0

    def test_seeding_respects_a_row_limit(self, store):
        assert seed_from_history(store, synthetic_transactions(50), limit=10)["seeded_rows"] == 10

    def test_stats_report_what_is_tracked(self, store):
        seed_from_history(store, synthetic_transactions(50))

        stats = store.stats()

        assert stats["entities_tracked"] > 0
        assert stats["fingerprints_tracked"] > 0


class TestScoringServiceIntegration:
    """The store is only useful if the one scoring path reads it before
    scoring and records after — and if every entry point does so."""

    def _service(self, feature_store, captured):
        from redis_utils import KeyedCache
        from review_queue import ReviewQueue
        from entity_memory import EntityRiskMemory
        from scoring_service import ScoringService

        class CapturingExplainer:
            def score_transaction(self, txn):
                captured.append(dict(txn))
                return {
                    "risk_score": 10.0,
                    "raw_proba": 0.1,
                    "above_threshold": False,
                    "top_factors": [],
                }

        return ScoringService(
            explainer=CapturingExplainer(),
            memory=EntityRiskMemory(),
            review_queue=ReviewQueue(),
            explanations_cache=KeyedCache(None, prefix="t", ttl_seconds=60),
            thresholds_provider=lambda: {"review": 40.0, "block": 80.0},
            feature_store=feature_store,
        )

    def test_the_model_is_given_live_features_not_the_frozen_snapshot(self, store):
        captured = []
        service = self._service(store, captured)
        txn = {
            "TransactionAmt": 100.0,
            "DeviceInfo": "device-1",
            # A stale value from the training snapshot — the live store
            # must override it.
            "entity_prior_txn_count": 999.0,
        }

        service.score_and_decide(txn, "entity-a")

        assert captured[0]["entity_prior_txn_count"] == 0.0

    def test_features_are_read_before_the_transaction_is_recorded(self, store):
        captured = []
        service = self._service(store, captured)
        txn = {"TransactionAmt": 100.0, "DeviceInfo": "device-1"}

        service.score_and_decide(txn, "entity-a")
        service.score_and_decide(dict(txn), "entity-a")

        # First score saw no history; second saw exactly the first.
        assert captured[0]["entity_prior_txn_count"] == 0.0
        assert captured[1]["entity_prior_txn_count"] == 1.0

    def test_the_device_signal_reaches_a_brand_new_entity(self, store):
        captured = []
        service = self._service(store, captured)
        shared = {"TransactionAmt": 100.0, "DeviceInfo": "shared-device"}

        service.score_and_decide(dict(shared), "entity-a")
        service.score_and_decide(dict(shared), "entity-b")

        # entity-b is cold, but the device has been seen before.
        assert captured[1]["entity_prior_txn_count"] == 0.0
        assert captured[1]["shared_device_prior_entity_count"] == 1.0

    def test_a_service_without_a_store_is_unchanged(self):
        captured = []
        service = self._service(None, captured)
        txn = {"TransactionAmt": 100.0, "entity_prior_txn_count": 999.0}

        service.score_and_decide(txn, "entity-a")

        # No store configured: the transaction passes through untouched.
        assert captured[0]["entity_prior_txn_count"] == 999.0

    def test_scoring_leaves_the_fraud_count_alone_until_a_label_arrives(self, store):
        captured = []
        service = self._service(store, captured)

        service.score_and_decide({"TransactionAmt": 100.0, "DeviceInfo": "d"}, "entity-a")

        # Volume counted, fraud not — the label isn't known yet.
        features = store.features_for("entity-a", "d")
        assert features["entity_prior_txn_count"] == 1.0
        assert features["entity_prior_fraud_count"] == 0.0
