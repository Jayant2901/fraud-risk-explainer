"""
Data loading & feature engineering for the IEEE-CIS Fraud Detection dataset.

Key design decisions (documented here so the reasoning is traceable):

1. ENTITY FINGERPRINTING
   IEEE-CIS deliberately does not provide a raw account/merchant ID
   (it's a real anonymized dataset). Following the well-known "UID"
   technique from top-performing solutions in the original Kaggle
   competition, we construct a proxy entity fingerprint from
   card1 + card2 + card5 + addr1 + P_emaildomain. Transactions sharing
   all five values are, with high probability, the same underlying
   card/account. This is a legitimate feature-engineering technique,
   not a fabrication — it mirrors how real card networks and payment
   processors do entity resolution when no explicit account ID exists
   in the raw event stream (device + card + address fingerprinting).

2. NO LEAKAGE ON ENTITY HISTORY FEATURES
   Any feature describing an entity's past behavior (fraud rate,
   transaction count) is computed using ONLY transactions strictly
   before the current one in time (via TransactionDT). This matters:
   naively computing "this entity's fraud rate" using the full dataset
   (past AND future transactions) would leak the label and produce an
   unrealistically strong, useless-in-production model.

3. TIME-BASED SPLIT
   Transactions are time-ordered. We split train/test chronologically
   (train on earlier transactions, test on later ones) rather than
   randomly, because random splits let the model "see the future" via
   entity history features computed elsewhere in the dataset. A
   time-based split is the honest way to estimate real-world
   performance for a fraud model that only ever sees the past.
"""
import pandas as pd
import numpy as np

from graph_features import add_causal_device_graph_features, GRAPH_FEATURE_COLS

TARGET_COL = "isFraud"
TXN_PATH = "data/train_transaction.csv"
IDENTITY_PATH = "data/train_identity.csv"

# Kept subset of V-columns to avoid an unwieldy 339-column block; these
# are commonly used in public IEEE-CIS analyses as a solid subset.
V_COLS_KEEP = [f"V{i}" for i in [1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30,
                                   36, 37, 40, 41, 44, 47, 48, 54, 56, 59, 62, 65,
                                   67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91]]

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
]

NUMERIC_BASE_COLS = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]


def load_raw_data(txn_path: str = TXN_PATH, identity_path: str = IDENTITY_PATH) -> pd.DataFrame:
    print("Loading train_transaction.csv...")
    txn = pd.read_csv(txn_path)
    print(f"  {len(txn):,} transactions")

    print("Loading train_identity.csv...")
    identity = pd.read_csv(identity_path)
    print(f"  {len(identity):,} identity records")

    df = txn.merge(identity, on="TransactionID", how="left")
    print(f"Merged: {len(df):,} rows, identity coverage: {df['DeviceType'].notna().mean():.1%}")
    return df


def build_entity_id(df: pd.DataFrame) -> pd.Series:
    """
    Proxy entity fingerprint: card1 + card2 + card5 + addr1 + P_emaildomain.
    Any of these being missing still produces a valid (if coarser) fingerprint —
    we fill NaNs with a placeholder string before concatenation so entities
    with partially-missing fields don't just become NaN.
    """
    parts = df[["card1", "card2", "card5", "addr1", "P_emaildomain"]].astype(str).fillna("NA")
    entity_id = parts.agg("_".join, axis=1)
    return entity_id


def add_causal_entity_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, compute the entity's PRIOR transaction count and
    PRIOR fraud rate using only transactions strictly earlier in time
    (TransactionDT). This is what makes these features leakage-free.

    Implementation: sort by (entity_id, TransactionDT), then use an
    expanding window shifted by 1 row per entity group.
    """
    df = df.sort_values(["entity_id", "TransactionDT"]).reset_index(drop=True)

    grouped = df.groupby("entity_id")[TARGET_COL]
    # expanding count/mean up to and including current row, then shift by 1
    # so the current row's own label is excluded (strictly causal)
    df["entity_prior_txn_count"] = grouped.cumcount()
    cum_fraud = grouped.cumsum() - df[TARGET_COL]  # fraud count before this row
    df["entity_prior_fraud_count"] = cum_fraud
    df["entity_prior_fraud_rate"] = (
        df["entity_prior_fraud_count"] / df["entity_prior_txn_count"].replace(0, np.nan)
    ).fillna(0.0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Entity fingerprint + causal history ---
    df["entity_id"] = build_entity_id(df)
    df = add_causal_entity_history(df)

    # --- Device/address graph signal (cold-start case — see graph_features.py) ---
    df = add_causal_device_graph_features(df, target_col=TARGET_COL)

    # --- Time-of-day (TransactionDT is seconds from an arbitrary reference) ---
    seconds_in_day = 24 * 60 * 60
    df["hour_of_day"] = (df["TransactionDT"] % seconds_in_day) // 3600
    df["is_night_txn"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 23) else 0)

    # --- Amount signals ---
    df["amount_log"] = np.log1p(df["TransactionAmt"])

    # --- Categorical dtype for XGBoost's native categorical support ---
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    engineered = [
        "hour_of_day", "is_night_txn", "amount_log",
        "entity_prior_txn_count", "entity_prior_fraud_count", "entity_prior_fraud_rate",
    ] + GRAPH_FEATURE_COLS
    v_cols = [c for c in V_COLS_KEEP if c in df.columns]
    feature_cols = NUMERIC_BASE_COLS + CATEGORICAL_COLS + v_cols + engineered
    return [c for c in feature_cols if c in df.columns]


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Chronological split: earliest (1 - test_frac) of transactions -> train,
    latest test_frac -> test. This avoids the leakage a random split would
    introduce via entity history features.
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    feature_cols = get_feature_columns(df_sorted)
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test, feature_cols
