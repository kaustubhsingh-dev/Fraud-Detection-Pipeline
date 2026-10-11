"""
Phase 3 — Model Training & Calibration
=======================================
Fixes:
1. Excludes high-cardinality categorical features to prevent memorization.
2. Calculates ranking metrics (PR-AUC / ROC-AUC) on raw predictions.
3. Fits Isotonic Regression on validation set solely for probability calibration.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_lgbm(X_train, y_train, X_val, y_val):
    print(f"\n[Train] Training LightGBM on {X_train.shape[1]} features...")

    high_cardinality_cols = {"card1", "card2", "card3", "card5", "addr1", "addr2"}
    cat_features = [
        c for c in X_train.columns
        if str(X_train[c].dtype) == "category" and c not in high_cardinality_cols
    ]
    print(f"      Active categorical features: {len(cat_features)}")

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 300,
        "learning_rate": 0.03,
        "feature_fraction": 0.4,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "scale_pos_weight": 3.0,
        "device": "cpu",
        "verbose": -1,
        "seed": 42,
        "num_threads": 0,
    }

    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
    dval   = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature=cat_features, free_raw_data=False)

    evals_result = {}
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(150, verbose=True),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result),
        ],
    )
    return model, evals_result


def calibrate_model(model, X_val, y_val):
    print("\n[Calibrate] Isotonic regression on validation set...")
    val_raw = model.predict(X_val, num_iteration=model.best_iteration)
    
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, y_val)
    
    val_cal = iso.predict(val_raw)
    print(f"      Val Brier (raw):        {brier_score_loss(y_val, val_raw):.5f}")
    print(f"      Val Brier (calibrated): {brier_score_loss(y_val, val_cal):.5f}")
    
    return iso


def evaluate(model, iso, X, y, split_name):
    raw_probs = model.predict(X, num_iteration=model.best_iteration)
    cal_probs = iso.predict(raw_probs)

    metrics = {
        "split":   split_name,
        "PR_AUC":  average_precision_score(y, raw_probs),
        "ROC_AUC": roc_auc_score(y, raw_probs),
        "Brier":   brier_score_loss(y, cal_probs),
        "LogLoss": log_loss(y, cal_probs),
    }

    print(f"\n      {split_name} metrics:")
    print(f"        PR-AUC   : {metrics['PR_AUC']:.4f}")
    print(f"        ROC-AUC  : {metrics['ROC_AUC']:.4f}")
    print(f"        Brier    : {metrics['Brier']:.5f}")
    print(f"        LogLoss  : {metrics['LogLoss']:.5f}")

    return metrics, cal_probs


def main():
    print("Loading feature matrices...")
    X_train = pd.read_parquet(FEATURES_DIR / "X_train.parquet")
    y_train = pd.read_parquet(FEATURES_DIR / "y_train.parquet")["isFraud"].values
    X_val   = pd.read_parquet(FEATURES_DIR / "X_val.parquet")
    y_val   = pd.read_parquet(FEATURES_DIR / "y_val.parquet")["isFraud"].values
    X_test  = pd.read_parquet(FEATURES_DIR / "X_test.parquet")
    y_test  = pd.read_parquet(FEATURES_DIR / "y_test.parquet")["isFraud"].values

    model, _ = train_lgbm(X_train, y_train, X_val, y_val)
    iso = calibrate_model(model, X_val, y_val)

    print("\n--- Model Evaluation ---")
    evaluate(model, iso, X_train, y_train, "Train")
    evaluate(model, iso, X_val, y_val, "Validation")
    evaluate(model, iso, X_test, y_test, "Test")

    print("\n[Saving] Writing models and logs...")
    with open(MODELS_DIR / "lgbm_raw.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "lgbm_calibrated.pkl", "wb") as f:
        pickle.dump({"model": model, "isotonic": iso, "features": list(X_train.columns)}, f)
    with open(MODELS_DIR / "selected_features.pkl", "wb") as f:
        pickle.dump(list(X_train.columns), f)

    print("  ✅ models/lgbm_raw.pkl")
    print("  ✅ models/lgbm_calibrated.pkl")
    print("  ✅ models/selected_features.pkl")


if __name__ == "__main__":
    main()