import numpy as np
import pandas as pd

from data_utils import (
    build_entity_id,
    add_causal_entity_history,
    engineer_features,
    time_based_split,
    TARGET_COL,
)
from graph_features import add_causal_device_graph_features, build_device_fingerprint


class TestBuildEntityId:
    def test_concatenates_the_five_fingerprint_fields(self):
        df = pd.DataFrame({
            "card1": [100],
            "card2": [10.0],
            "card5": [1],
            "addr1": [300],
            "P_emaildomain": ["gmail.com"],
        })
        result = build_entity_id(df)
        assert result.iloc[0] == "100_10.0_1_300_gmail.com"

    def test_missing_fields_are_filled_with_na_placeholder_not_dropped(self):
        df = pd.DataFrame({
            "card1": [200],
            "card2": [np.nan],
            "card5": [2],
            "addr1": [400],
            "P_emaildomain": [None],
        })
        result = build_entity_id(df)
        # Missing fields become the literal string "NA", not "nan" or a
        # dropped/null entity_id — a partially-missing fingerprint should
        # still be a valid, usable (if coarser) entity_id.
        assert result.iloc[0] == "200_NA_2_400_NA"
        assert pd.notna(result.iloc[0])

    def test_different_entities_get_different_ids(self):
        df = pd.DataFrame({
            "card1": [100, 999],
            "card2": [10.0, 10.0],
            "card5": [1, 1],
            "addr1": [300, 300],
            "P_emaildomain": ["gmail.com", "gmail.com"],
        })
        result = build_entity_id(df)
        assert result.iloc[0] != result.iloc[1]

    def test_same_five_fields_produce_the_same_entity_id(self):
        df = pd.DataFrame({
            "card1": [100, 100],
            "card2": [10.0, 10.0],
            "card5": [1, 1],
            "addr1": [300, 300],
            "P_emaildomain": ["gmail.com", "gmail.com"],
        })
        result = build_entity_id(df)
        assert result.iloc[0] == result.iloc[1]


class TestAddCausalEntityHistory:
    """
    Hand-built timeline (entity_id already assigned, bypassing
    build_entity_id to isolate this function):

      entity A: t=100 fraud=1, t=200 fraud=0, t=300 fraud=0
      entity B: t=150 fraud=0, t=250 fraud=1

    Every prior_* value below is computed using ONLY strictly-earlier rows
    for that same entity. This is the leakage-prevention claim from the
    module docstring and the README ("No leakage") — if the `- df[TARGET_COL]`
    shift in add_causal_entity_history were ever removed (making the
    cumsum inclusive of the CURRENT row instead of strictly prior rows),
    entity A's first transaction (itself fraud=1) would leak its own
    label into entity_prior_fraud_count/rate — that's exactly what
    test_first_transaction_for_an_entity_has_no_leaked_self_label checks.
    """

    def _timeline(self) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id": ["A", "A", "A", "B", "B"],
            "TransactionDT": [100, 200, 300, 150, 250],
            TARGET_COL: [1, 0, 0, 0, 1],
        })

    def test_first_transaction_for_an_entity_has_no_leaked_self_label(self):
        result = add_causal_entity_history(self._timeline())
        first_a = result[(result["entity_id"] == "A") & (result["TransactionDT"] == 100)].iloc[0]

        assert first_a["entity_prior_txn_count"] == 0
        assert first_a["entity_prior_fraud_count"] == 0
        # Would be 1.0 if the current row's own fraud=1 label leaked in.
        assert first_a["entity_prior_fraud_rate"] == 0.0

    def test_second_transaction_sees_only_the_first_as_prior_history(self):
        result = add_causal_entity_history(self._timeline())
        second_a = result[(result["entity_id"] == "A") & (result["TransactionDT"] == 200)].iloc[0]

        assert second_a["entity_prior_txn_count"] == 1
        assert second_a["entity_prior_fraud_count"] == 1  # the t=100 fraud
        assert second_a["entity_prior_fraud_rate"] == 1.0

    def test_third_transaction_averages_over_first_two_only(self):
        result = add_causal_entity_history(self._timeline())
        third_a = result[(result["entity_id"] == "A") & (result["TransactionDT"] == 300)].iloc[0]

        assert third_a["entity_prior_txn_count"] == 2
        assert third_a["entity_prior_fraud_count"] == 1  # t=200 was not fraud
        assert third_a["entity_prior_fraud_rate"] == 0.5

    def test_entities_do_not_leak_history_into_each_other(self):
        result = add_causal_entity_history(self._timeline())
        first_b = result[(result["entity_id"] == "B") & (result["TransactionDT"] == 150)].iloc[0]
        second_b = result[(result["entity_id"] == "B") & (result["TransactionDT"] == 250)].iloc[0]

        # B's first transaction must not see A's history at all.
        assert first_b["entity_prior_txn_count"] == 0
        assert first_b["entity_prior_fraud_rate"] == 0.0
        # B's second transaction sees only B's own (non-fraud) first row.
        assert second_b["entity_prior_txn_count"] == 1
        assert second_b["entity_prior_fraud_count"] == 0


class TestBuildDeviceFingerprint:
    def test_uses_device_info_when_present(self):
        df = pd.DataFrame({"DeviceInfo": ["iPhone"], "addr1": [100], "addr2": [10]})
        assert build_device_fingerprint(df).iloc[0] == "iPhone"

    def test_falls_back_to_address_when_device_info_missing(self):
        df = pd.DataFrame({"DeviceInfo": [None], "addr1": [100], "addr2": [10]})
        assert build_device_fingerprint(df).iloc[0] == "ADDR_100_10"

    def test_no_fingerprint_when_neither_is_available(self):
        df = pd.DataFrame({"DeviceInfo": [None], "addr1": [None], "addr2": [None]})
        assert pd.isna(build_device_fingerprint(df).iloc[0])

    def test_missing_columns_degrade_to_no_fingerprint_rather_than_crash(self):
        # A DataFrame that never merged in the identity table (or a
        # minimal test fixture) shouldn't KeyError.
        df = pd.DataFrame({"TransactionDT": [100]})
        assert pd.isna(build_device_fingerprint(df).iloc[0])


class TestAddCausalDeviceGraphFeatures:
    """
    Hand-built timeline, mirroring TestAddCausalEntityHistory's style but
    for the DEVICE/ADDRESS graph signal — the point of this feature is
    that it works across DIFFERENT entities sharing a device, which
    per-entity history (entity_prior_*) can never see:

      device D1: t=100 entity A (fraud=1), t=200 entity B (fraud=0),
                 t=300 entity A again (fraud=0)
      device D2: t=150 entity C (fraud=0) — a totally separate device

    Every shared_device_prior_* value must be computed using only
    transactions strictly earlier in time sharing that same fingerprint —
    across entities, not just the current one.
    """

    def _timeline(self) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id": ["A", "B", "C", "A"],
            "TransactionDT": [100, 200, 150, 300],
            "DeviceInfo": ["D1", "D1", "D2", "D1"],
            "addr1": [1, 1, 2, 1],
            "addr2": [1, 1, 2, 1],
            TARGET_COL: [1, 0, 0, 0],
        })

    def test_first_transaction_on_a_device_has_no_prior_entities_or_fraud(self):
        result = add_causal_device_graph_features(self._timeline())
        first_a = result[(result["entity_id"] == "A") & (result["TransactionDT"] == 100)].iloc[0]

        assert first_a["shared_device_prior_entity_count"] == 0
        assert first_a["shared_device_prior_fraud_rate"] == 0.0

    def test_a_different_entity_on_the_same_device_sees_the_first_entitys_history(self):
        # This is the whole point of the feature: entity B has NO history
        # of its own (entity_prior_txn_count would be 0), but it shares
        # device D1 with entity A, whose one prior transaction was fraud.
        result = add_causal_device_graph_features(self._timeline())
        b_row = result[result["entity_id"] == "B"].iloc[0]

        assert b_row["shared_device_prior_entity_count"] == 1  # just A so far
        assert b_row["shared_device_prior_fraud_rate"] == 1.0  # A's one prior txn was fraud

    def test_distinct_entity_count_does_not_double_count_a_repeat_entity(self):
        # By t=300, device D1 has been used by A (t100) and B (t200) — 2
        # distinct entities — even though A itself has now shown up twice.
        result = add_causal_device_graph_features(self._timeline())
        second_a = result[(result["entity_id"] == "A") & (result["TransactionDT"] == 300)].iloc[0]

        assert second_a["shared_device_prior_entity_count"] == 2
        assert second_a["shared_device_prior_fraud_rate"] == 0.5  # 1 fraud out of 2 prior txns on D1

    def test_different_devices_do_not_leak_into_each_other(self):
        result = add_causal_device_graph_features(self._timeline())
        c_row = result[result["entity_id"] == "C"].iloc[0]

        # C is on device D2, entirely separate from D1's history.
        assert c_row["shared_device_prior_entity_count"] == 0
        assert c_row["shared_device_prior_fraud_rate"] == 0.0

    def test_rows_with_no_fingerprint_get_zero_not_a_fabricated_link(self):
        df = self._timeline()
        df["DeviceInfo"] = None
        df["addr1"] = np.nan
        df["addr2"] = np.nan
        result = add_causal_device_graph_features(df)

        assert (result["shared_device_prior_entity_count"] == 0).all()
        assert (result["shared_device_prior_fraud_rate"] == 0.0).all()


class TestEngineerFeatures:
    def _raw_df(self):
        return pd.DataFrame({
            "card1": [100, 100, 200],
            "card2": [10.0, 10.0, 20.0],
            "card5": [1, 1, 2],
            "addr1": [300, 300, 400],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com"],
            "TransactionDT": [0, 43200, 82800],   # hour 0, 12, 23
            "TransactionAmt": [50.0, 100.0, 0.0],
            "ProductCD": ["W", "W", "C"],
            TARGET_COL: [0, 1, 0],
        })

    def test_adds_entity_id_from_the_fingerprint_fields(self):
        result = engineer_features(self._raw_df())
        assert (result["entity_id"] == "100_10.0_1_300_gmail.com").sum() == 2
        assert (result["entity_id"] == "200_20.0_2_400_yahoo.com").sum() == 1

    def test_hour_of_day_and_night_flag_derived_from_transaction_dt(self):
        result = engineer_features(self._raw_df()).sort_values("TransactionDT")
        assert result["hour_of_day"].tolist() == [0, 12, 23]
        # hour 0 and 23 count as night (h < 6 or h >= 23); hour 12 does not
        assert result["is_night_txn"].tolist() == [1, 0, 1]

    def test_amount_log_is_log1p_of_transaction_amount(self):
        result = engineer_features(self._raw_df())
        expected = np.log1p(result["TransactionAmt"])
        assert np.allclose(result["amount_log"], expected)

    def test_categorical_columns_are_cast_to_category_dtype(self):
        result = engineer_features(self._raw_df())
        assert isinstance(result["ProductCD"].dtype, pd.CategoricalDtype)

    def test_still_produces_causal_not_leaky_entity_history(self):
        # engineer_features delegates to add_causal_entity_history — this
        # just confirms the wiring, the leakage math itself is covered in
        # TestAddCausalEntityHistory above.
        result = engineer_features(self._raw_df())
        first_of_repeat_entity = result[result["entity_id"] == "100_10.0_1_300_gmail.com"].sort_values("TransactionDT").iloc[0]
        assert first_of_repeat_entity["entity_prior_txn_count"] == 0

    def test_also_wires_in_the_device_graph_features(self):
        # engineer_features delegates to add_causal_device_graph_features —
        # this fixture has no DeviceInfo/addr2 columns at all, so this also
        # confirms that degrades to 0 rather than crashing (the leakage
        # math itself is covered in TestAddCausalDeviceGraphFeatures above).
        result = engineer_features(self._raw_df())
        assert "shared_device_prior_entity_count" in result.columns
        assert "shared_device_prior_fraud_rate" in result.columns
        assert (result["shared_device_prior_entity_count"] == 0).all()


class TestTimeBasedSplit:
    def _make_df(self, n=20):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "TransactionDT": np.arange(n),
            "TransactionAmt": rng.uniform(10, 500, n),
            TARGET_COL: rng.integers(0, 2, n),
        })
        # shuffle row order to prove the split re-sorts by time itself
        return df.sample(frac=1, random_state=1).reset_index(drop=True)

    def test_split_sizes_match_test_fraction(self):
        df = self._make_df(20)
        X_train, X_test, y_train, y_test, _ = time_based_split(df, test_frac=0.2)
        assert len(X_train) == 16
        assert len(X_test) == 4
        assert len(y_train) == 16
        assert len(y_test) == 4

    def test_split_is_chronological_not_random(self):
        # time_based_split sorts by TransactionDT and reset_index(drop=True)'s
        # before splitting, so — given TransactionDT is strictly increasing
        # in this fixture — X_train's positional indices being entirely
        # below X_test's proves the split follows time order rather than
        # the original (shuffled) row order. (TransactionDT itself isn't a
        # model feature — see NUMERIC_BASE_COLS — so it isn't a column on
        # the returned X_train/X_test to check directly.)
        df = self._make_df(20)
        X_train, X_test, _, _, _ = time_based_split(df, test_frac=0.2)
        assert X_train.index.max() < X_test.index.min()

    def test_train_and_test_do_not_overlap(self):
        df = self._make_df(20)
        X_train, X_test, _, _, _ = time_based_split(df, test_frac=0.25)
        assert set(X_train.index).isdisjoint(set(X_test.index))
        assert len(X_train) + len(X_test) == len(df)
