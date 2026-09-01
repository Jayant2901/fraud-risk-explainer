"""
Device/address graph features — a signal for the COLD-START case.

`entity_prior_txn_count`/`entity_prior_fraud_rate` (data_utils.py) are
blind to a brand-new entity by construction: a first-ever transaction
from any entity has entity_prior_txn_count == 0, fraud or not — there's
no per-entity history to draw on. Real fraud rings get caught anyway
because the underlying DEVICE or ADDRESS is often reused across several
"new" card/account fingerprints even when the card itself has never been
seen. This module builds that signal causally, from IEEE-CIS's
`DeviceInfo`/`addr1`/`addr2` columns:

  - shared_device_prior_entity_count: how many DISTINCT entities have
    used this same device/address before, as of this transaction.
  - shared_device_prior_fraud_rate: the fraud rate among ALL prior
    transactions sharing that device/address — across every entity, not
    just this one — which is exactly what a cold-start entity (no
    history of its own) can still benefit from.

Coverage matters here: `DeviceInfo` is populated for only ~20% of all
transactions in this dataset (most rows have no identity-table match at
all — see engineer_features()'s own coverage log). `addr1`/`addr2` are
far more complete (~89%), so this falls back to an addr1+addr2
fingerprint when DeviceInfo is missing, rather than leaving most rows
with no signal at all. A transaction with NEITHER available gets 0 for
both features — not lumped into a fake "unknown device" bucket, which
would falsely link unrelated entities together.

Same causal-windowing discipline as add_causal_entity_history
(data_utils.py): every value uses only transactions strictly earlier in
time for that same fingerprint, or it's leakage.
"""
import numpy as np
import pandas as pd

TARGET_COL = "isFraud"

FINGERPRINT_COL = "device_fingerprint"
GRAPH_FEATURE_COLS = ["shared_device_prior_entity_count", "shared_device_prior_fraud_rate"]


def _col_or_all_missing(df: pd.DataFrame, col: str) -> pd.Series:
    """df[col] if it exists, else an all-NaN Series of the right length —
    so callers on a DataFrame that never merged in the identity table (or
    a minimal test fixture) degrade to "no fingerprint available" instead
    of a KeyError."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def build_device_fingerprint(df: pd.DataFrame) -> pd.Series:
    """DeviceInfo when present, else "ADDR_<addr1>_<addr2>" when both
    address fields are present, else NaN (no usable fingerprint)."""
    device = _col_or_all_missing(df, "DeviceInfo")
    addr1 = _col_or_all_missing(df, "addr1")
    addr2 = _col_or_all_missing(df, "addr2")
    has_addr = addr1.notna() & addr2.notna()
    addr_fallback = "ADDR_" + addr1.astype(str) + "_" + addr2.astype(str)
    return device.where(device.notna(), addr_fallback.where(has_addr))


def add_causal_device_graph_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Adds `device_fingerprint`, `shared_device_prior_entity_count`, and
    `shared_device_prior_fraud_rate` to df. Requires `entity_id` and
    `target_col` to already be present (run after
    data_utils.add_causal_entity_history). Does not reorder df — values
    are computed on a sorted-by-fingerprint working copy and assigned
    back by the original index.
    """
    df = df.copy()
    df[FINGERPRINT_COL] = build_device_fingerprint(df)

    df["shared_device_prior_entity_count"] = 0.0
    df["shared_device_prior_fraud_rate"] = 0.0

    has_fp = df[FINGERPRINT_COL].notna()
    if not has_fp.any():
        return df

    fp_df = df[has_fp].sort_values([FINGERPRINT_COL, "TransactionDT"])
    fp_group = fp_df[FINGERPRINT_COL]

    # Distinct-entity count: mark each row where this is the first time
    # THIS entity has been seen under THIS fingerprint, running-sum those
    # flags within the fingerprint group (so the count only ever grows
    # when a genuinely new entity shows up under that device/address),
    # then shift by one row so the current transaction never counts
    # itself — matching add_causal_entity_history's "strictly before"
    # discipline.
    is_first_for_entity = ~fp_df.duplicated(subset=[FINGERPRINT_COL, "entity_id"], keep="first")
    running_distinct = is_first_for_entity.groupby(fp_group).cumsum()
    prior_distinct = running_distinct.groupby(fp_group).shift(1).fillna(0)

    # Prior fraud rate across ALL entities sharing this fingerprint — the
    # same cumcount/cumsum-shift trick add_causal_entity_history uses,
    # just grouped by fingerprint instead of by entity.
    prior_txn_count = fp_df.groupby(fp_group).cumcount()
    cum_fraud_inclusive = fp_df.groupby(fp_group)[target_col].cumsum()
    prior_fraud_count = cum_fraud_inclusive - fp_df[target_col]
    prior_fraud_rate = (prior_fraud_count / prior_txn_count.replace(0, np.nan)).fillna(0.0)

    df.loc[fp_df.index, "shared_device_prior_entity_count"] = prior_distinct.astype(float).values
    df.loc[fp_df.index, "shared_device_prior_fraud_rate"] = prior_fraud_rate.astype(float).values

    return df
