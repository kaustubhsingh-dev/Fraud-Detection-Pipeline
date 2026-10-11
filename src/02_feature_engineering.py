"""
Phase 2 — Feature Engineering (Fixed Pipeline)
==============================================
Fixes:
1. Retains top high-signal Vesta features while dropping noise.
2. Adds card and address transaction amount ratio features.
3. Stable UID reconstruction via constant card start day.
4. Continuous timeline expanding history (prevents split resets).
5. Explicit numeric vs. categorical missing value filling (fixes PyArrow bug).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATE = pd.Timestamp("2017-12-01")
TRAIN_END = pd.Timestamp("2018-04-01")
VAL_END   = pd.Timestamp("2018-05-01")

TOP_VESTA_COLS = [
    "V258", "V294", "V307", "V317", "V188", "V201", "V243", 
    "V257", "V283", "V130", "V45", "V87", "V86"
]


def add_customer_uid(df: pd.DataFrame) -> pd.DataFrame:
    print("[1/6] Reconstructing stable customer UIDs...")
    day = df["TransactionDT"] / (3600 * 24)
    df["card_start_day"] = (day - df["D1"]).round(1).astype(str)

    df["uid"] = (
        df["card1"].astype(str) + "_"
        + df["card2"].fillna(-999).astype(str) + "_"
        + df["addr1"].fillna(-999).astype(str) + "_"
        + df["card_start_day"]
    )

    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)
    df["card1_card2"] = df["card1"].astype(str) + "_" + df["card2"].fillna(-999).astype(str)
    
    print(f"      Unique stable UIDs: {df['uid'].nunique():,}")
    return df


def add_time_and_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Adding time and amount features...")
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["is_weekend"] = df["day"].isin([5, 6]).astype(int)
    
    df["amount_log"] = np.log1p(df["TransactionAmt"])
    df["amount_decimal"] = (df["TransactionAmt"] - df["TransactionAmt"].astype(int))
    df["amount_is_round"] = (df["amount_decimal"] == 0).astype(int)
    
    return df


def add_amount_group_ratios(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Adding card & address amount group ratios...")
    
    card1_amt_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    df["amt_to_mean_card1"] = df["TransactionAmt"] / (card1_amt_mean + 1e-5)
    
    card1_addr1_amt_mean = df.groupby(["card1", "addr1"])["TransactionAmt"].transform("mean")
    df["amt_to_mean_card1_addr1"] = df["TransactionAmt"] / (card1_addr1_amt_mean + 1e-5)
    
    return df


def add_temporal_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] Computing continuous temporal aggregations across timeline...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    g = df.groupby("uid", sort=False)

    df["uid_txn_count_prior"] = g.cumcount()
    df["uid_avg_amt_prior"] = g["TransactionAmt"].transform(lambda x: x.expanding().mean().shift(1))
    df["amt_dev_from_user_mean"] = df["TransactionAmt"] - df["uid_avg_amt_prior"]
    df["uid_std_amt_prior"] = g["TransactionAmt"].transform(lambda x: x.expanding().std().shift(1))
    df["amt_zscore_user"] = df["amt_dev_from_user_mean"] / (df["uid_std_amt_prior"] + 1)
    df["time_since_prev_uid"] = g["TransactionDT"].diff().fillna(-1)

    return df


def add_frequency_encoding(train, val, test):
    print("[5/6] Applying frequency encoding (Train fit only)...")
    freq_cols = ["card1", "card2", "addr1", "P_emaildomain", "card1_card2"]

    for col in freq_cols:
        freq_map = train[col].value_counts()
        train[f"{col}_freq"] = train[col].map(freq_map).fillna(0).astype(int)
        val[f"{col}_freq"]   = val[col].map(freq_map).fillna(0).astype(int)
        test[f"{col}_freq"]  = test[col].map(freq_map).fillna(0).astype(int)

    return train, val, test


def assemble_features(train, val, test):
    print("[6/6] Assembling final feature matrices...")
    drop_cols = [
        "TransactionID", "TransactionDT", "isFraud", "timestamp",
        "uid", "card_start_day", "card1_card2", "P_emaildomain", "R_emaildomain"
    ]

    y_train = train["isFraud"].astype(int).reset_index(drop=True)
    y_val   = val["isFraud"].astype(int).reset_index(drop=True)
    y_test  = test["isFraud"].astype(int).reset_index(drop=True)

    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_val   = val.drop(columns=[c for c in drop_cols if c in val.columns])
    X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])

    # Drop raw Vesta V-columns EXCEPT top signal features
    vesta_cols = [c for c in X_train.columns if c.startswith("V") and c[1:].isdigit()]
    vesta_to_drop = [c for c in vesta_cols if c not in TOP_VESTA_COLS]
    
    X_train = X_train.drop(columns=vesta_to_drop)
    X_val   = X_val.drop(columns=vesta_to_drop)
    X_test  = X_test.drop(columns=vesta_to_drop)

    # Separate numeric vs categorical filling to prevent PyArrow conversion errors
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    for df in [X_train, X_val, X_test]:
        df[num_cols] = df[num_cols].fillna(-999)
        df[cat_cols] = df[cat_cols].astype(str).fillna("missing")

    # Standardize category levels based on training set
    for col in cat_cols:
        categories = X_train[col].unique()
        cat_type = pd.CategoricalDtype(categories=categories)
        X_train[col] = X_train[col].astype(cat_type)
        X_val[col]   = X_val[col].astype(cat_type)
        X_test[col]  = X_test[col].astype(cat_type)

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print("Loading Phase 1 output...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    df_full = pd.concat([train, val, test], ignore_index=True)
    df_full = add_customer_uid(df_full)
    df_full = add_time_and_amount_features(df_full)
    df_full = add_amount_group_ratios(df_full)
    df_full = add_temporal_aggregations(df_full)

    train = df_full[df_full["timestamp"] < TRAIN_END].copy()
    val   = df_full[(df_full["timestamp"] >= TRAIN_END) & (df_full["timestamp"] < VAL_END)].copy()
    test  = df_full[df_full["timestamp"] >= VAL_END].copy()

    train, val, test = add_frequency_encoding(train, val, test)
    X_train, y_train, X_val, y_val, X_test, y_test = assemble_features(train, val, test)

    print("\nSaving feature matrices...")
    X_train.to_parquet(FEATURES_DIR / "X_train.parquet", index=False)
    y_train.to_frame("isFraud").to_parquet(FEATURES_DIR / "y_train.parquet", index=False)
    X_val.to_parquet(FEATURES_DIR / "X_val.parquet", index=False)
    y_val.to_frame("isFraud").to_parquet(FEATURES_DIR / "y_val.parquet", index=False)
    X_test.to_parquet(FEATURES_DIR / "X_test.parquet", index=False)
    y_test.to_frame("isFraud").to_parquet(FEATURES_DIR / "y_test.parquet", index=False)

    print(f"✅ Features updated in {FEATURES_DIR}/ (Feature count: {X_train.shape[1]})")

if __name__ == "__main__":
    main()