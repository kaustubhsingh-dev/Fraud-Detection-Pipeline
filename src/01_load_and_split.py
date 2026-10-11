"""
Phase 1 — Data Loading & Temporal Split
========================================
Loads IEEE-CIS transaction + identity tables, merges them, converts
TransactionDT to real timestamps, and performs a STRICT chronological
train/val/test split to prevent future-data leakage.

Output:
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-CIS reference date (documented by Kaggle community)
REFERENCE_DATE = pd.Timestamp("2017-12-01")

# Temporal split boundaries (approximate — IEEE-CIS train spans ~6 months)
# Train: first 4 months | Val: month 5 | Test: month 6
TRAIN_END = pd.Timestamp("2018-04-01")
VAL_END   = pd.Timestamp("2018-05-01")


def load_merged_data() -> pd.DataFrame:
    """Load and merge the transaction + identity tables."""
    print("[1/4] Loading raw CSVs...")
    txn = pd.read_csv(RAW_DIR / "train_transaction.csv")
    ident = pd.read_csv(RAW_DIR / "train_identity.csv")
    print(f"      Transactions: {len(txn):,} rows, {txn.shape[1]} cols")
    print(f"      Identity:     {len(ident):,} rows, {ident.shape[1]} cols")

    print("[2/4] Merging on TransactionID (left join)...")
    df = txn.merge(ident, on="TransactionID", how="left")
    print(f"      Merged shape: {df.shape}")
    print(f"      Rows with identity info: {df['DeviceType'].notna().sum():,} "
          f"({df['DeviceType'].notna().mean() * 100:.1f}%)")

    return df


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TransactionDT (seconds offset) to real datetime."""
    print("[3/4] Converting TransactionDT to real timestamps...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["timestamp"] = REFERENCE_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    print(f"      Time range: {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    return df


def temporal_split(df: pd.DataFrame):
    """Strict chronological split. No row leaks across boundaries."""
    print("[4/4] Performing temporal split...")
    train = df[df["timestamp"] < TRAIN_END].copy()
    val   = df[(df["timestamp"] >= TRAIN_END) & (df["timestamp"] < VAL_END)].copy()
    test  = df[df["timestamp"] >= VAL_END].copy()

    # Sanity check — confirm no overlap in time
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max()   < test["timestamp"].min()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        fraud_rate = split["isFraud"].mean() * 100
        print(f"      {name:<6}: {len(split):>7,} rows | "
              f"fraud rate: {fraud_rate:>5.2f}% | "
              f"{split['timestamp'].min().date()} → {split['timestamp'].max().date()}")

    return train, val, test


def save_splits(train, val, test):
    """Save as parquet — 5-10x faster to reload than CSV."""
    print("\nSaving parquet files...")
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)
    print(f"  ✅ Saved to {OUT_DIR}/")


def main():
    df = load_merged_data()
    df = add_timestamps(df)
    train, val, test = temporal_split(df)
    save_splits(train, val, test)

    print("\n" + "=" * 60)
    print("Phase 1 complete. Ready for feature engineering.")
    print("=" * 60)


if __name__ == "__main__":
    main()