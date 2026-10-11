"""
Phase 4 — Evaluation & Paper Figures
=====================================
Generates PR curve, ROC curve, reliability diagram, threshold analysis,
and SHAP summary. These are the figures for the paper.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models")
FIGURES_DIR  = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


# ---------- LOAD ----------
def load_model_and_data():
    print("[1/4] Loading model and test data...")
    with open(MODELS_DIR / "lgbm_calibrated.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["model"]
    iso   = artifacts["isotonic"]

    X_test = pd.read_parquet(FEATURES_DIR / "X_test.parquet")
    y_test = pd.read_parquet(FEATURES_DIR / "y_test.parquet")["isFraud"].values

    # Convert string columns back to category (train categories)
    str_cols = X_test.select_dtypes(include=["object"]).columns
    for col in str_cols:
        X_test[col] = X_test[col].astype("category")

    return model, iso, X_test, y_test


# ---------- PR CURVE ----------
def plot_pr_curve(y_true, y_proba):
    print("[2/4] Plotting PR curve...")
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"LightGBM (AP = {ap:.3f})")
    ax.axhline(baseline, ls="--", color="grey",
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (Temporal Test Set)")
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ figures/pr_curve.png")


# ---------- ROC CURVE ----------
def plot_roc_curve(y_true, y_proba):
    print("[3/4] Plotting ROC curve...")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"LightGBM (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Temporal Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ figures/roc_curve.png")


# ---------- RELIABILITY DIAGRAM ----------
def plot_reliability(y_true, y_proba):
    print("[4/4] Plotting reliability diagram...")
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=15,
                                              strategy="quantile")
    brier = brier_score_loss(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", lw=2, label=f"LightGBM (Brier = {brier:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Fraud Rate")
    ax.set_title("Reliability Diagram (Temporal Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reliability.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ figures/reliability.png")


# ---------- THRESHOLD ANALYSIS ----------
def threshold_analysis(y_true, y_proba):
    """Find business-optimal threshold and report cost savings."""
    cost_fn = 500   # missed fraud: chargeback + investigation
    cost_fp = 50    # false alert: analyst review time

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        preds = (y_proba > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fn * cost_fn + fp * cost_fp
        rows.append({
            "threshold": t,
            "precision": tp / (tp + fp) if (tp + fp) else 0,
            "recall":    tp / (tp + fn) if (tp + fn) else 0,
            "f1":        2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0,
            "cost":      cost,
        })
    df = pd.DataFrame(rows)
    best = df.loc[df["cost"].idxmin()]
    print(f"\n      Business-optimal threshold: {best['threshold']:.2f}")
    print(f"      Precision: {best['precision']:.3f}")
    print(f"      Recall:    {best['recall']:.3f}")
    print(f"      F1:        {best['f1']:.3f}")
    print(f"      Cost/10k:  ${best['cost'] / len(y_true) * 10000:.2f}")

    # Save the curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision")
    ax.plot(df["threshold"], df["recall"], label="Recall")
    ax.plot(df["threshold"], df["f1"], label="F1")
    ax.axvline(best["threshold"], ls="--", color="red",
               label=f"Optimal = {best['threshold']:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis (Temporal Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      ✅ figures/threshold_analysis.png")

    df.to_csv("results/threshold_table.csv", index=False)

    print("\n      Top-K Analysis (industry standard):")
    n = len(y_true)
    ranked_idx = np.argsort(-y_proba)   # highest prob first
    k_percentages = [1, 5, 10]           # ← ADD THIS LINE
    
    for pct in k_percentages:
        k = int(n * pct / 100)
        top_k = ranked_idx[:k]
        tp = y_true[top_k].sum()
        precision = tp / k
        recall = tp / y_true.sum()
        print(f"        Top {pct}% ({k:,} flagged): "
              f"precision={precision:.3f}, recall={recall:.3f}")
    return best


def main():
    model, iso, X_test, y_test = load_model_and_data()

    # Predict and calibrate
    raw_probs = model.predict(X_test, num_iteration=model.best_iteration)
    cal_probs = iso.predict(raw_probs)

    print(f"\n      Test PR-AUC:  {average_precision_score(y_test, raw_probs):.4f}")
    print(f"      Test ROC-AUC: {roc_auc_score(y_test, raw_probs):.4f}")

    plot_pr_curve(y_test, raw_probs)
    plot_roc_curve(y_test, raw_probs)
    plot_reliability(y_test, cal_probs)
    threshold_analysis(y_test, cal_probs)

    print("\n" + "=" * 60)
    print("Phase 4 complete. Figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()