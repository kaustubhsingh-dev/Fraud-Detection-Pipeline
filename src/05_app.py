"""
Phase 5 — Interactive Fraud Detection Dashboard (Streamlit)
============================================================
Fixed:
1. load_models() correctly unpacks the dict saved by 03_train_model.py
2. Real-time scoring tab replaced with production deployment info
   (single transactions lack historical context needed for scoring)
3. Tabs 1 & 2 (Performance + Threshold Optimizer) retained as-is
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="ProcureAI | Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------- PATHS ----------
MODELS_DIR   = Path("models")
FIGURES_DIR  = Path("figures")
FEATURES_DIR = Path("data/features")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.block-container { max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; }
h1 { color: #0f172a; font-weight: 800; letter-spacing: -0.5px; }
h2 { color: #0f172a; font-weight: 700; }
h3 { color: #0f172a; font-weight: 700; font-size: 1.05rem; }
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.15rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    background: #dcfce7;
    color: #166534;
    font-size: 0.75rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODELS (FIXED) ----------
@st.cache_resource
def load_models():
    """Load the calibrated model + isotonic regression + feature list.

    NOTE: 03_train_model.py saves a single dict with keys:
      { 'model': LGBMRegressor, 'isotonic': IsotonicRegression, 'features': [...] }
    """
    cal_path = MODELS_DIR / "lgbm_calibrated.pkl"
    if not cal_path.exists():
        return None, None, None

    with open(cal_path, "rb") as f:
        artifacts = pickle.load(f)

    # Handle both the new dict format and any legacy single-object format
    if isinstance(artifacts, dict):
        model = artifacts.get("model")
        iso   = artifacts.get("isotonic")
        features = artifacts.get("features", [])
    else:
        # Legacy: the pickle was just the isotonic model
        model = None
        iso = artifacts
        features = []

    return model, iso, features


model, iso, feature_names = load_models()
models_loaded = model is not None and iso is not None


# ---------- HEADER ----------
st.title("🛡️ Enterprise Fraud Detection Dashboard")
st.markdown(
    "Out-of-time evaluation, calibrated probability estimation, "
    "and cost-optimal decision thresholding on the IEEE-CIS dataset."
)

if not models_loaded:
    st.warning(
        "⚠️ Trained models not found in `models/`. "
        "Run `python src/03_train_model.py` first to generate them."
    )

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs([
    "📊 Performance & Evaluation",
    "💰 Threshold & Cost Optimizer",
    "🚀 Production Deployment"
])


# ============================================================
# TAB 1 — MODEL PERFORMANCE
# ============================================================
with tab1:
    st.header("Out-of-Time Model Performance")
    st.caption(
        "Metrics computed on a strict temporal test set "
        "(May 2018), with no data leakage from training."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test PR-AUC",  "0.5177", "+0.0287 vs base")
    col2.metric("Test ROC-AUC", "0.8984", "Stable OOT")
    col3.metric("Brier Score",  "0.0227", "Calibrated")
    col4.metric("LogLoss",      "0.0935", "Optimal")

    st.markdown("---")

    fig_col1, fig_col2, fig_col3 = st.columns(3)

    with fig_col1:
        st.subheader("Precision-Recall Curve")
        pr_path = FIGURES_DIR / "pr_curve.png"
        if pr_path.exists():
            st.image(Image.open(pr_path), use_container_width=True)
        else:
            st.info("Run Phase 4 to generate figures.")

    with fig_col2:
        st.subheader("ROC Curve")
        roc_path = FIGURES_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(Image.open(roc_path), use_container_width=True)
        else:
            st.info("Run Phase 4 to generate figures.")

    with fig_col3:
        st.subheader("Reliability Diagram")
        rel_path = FIGURES_DIR / "reliability.png"
        if rel_path.exists():
            st.image(Image.open(rel_path), use_container_width=True)
        else:
            st.info("Run Phase 4 to generate figures.")

    st.markdown("---")

    # Top-K Deployment table (the paper's headline)
    st.subheader("Top-K Deployment Analysis")
    st.caption(
        "Fraud operations deploy by reviewing the top-K% "
        "of transactions by risk score — not by thresholding."
    )

    topk_df = pd.DataFrame({
        "Top-K": ["Top 1%", "Top 5%", "Top 10%"],
        "Flagged": ["924", "4,621", "9,242"],
        "Precision": ["0.848", "0.401", "0.248"],
        "Recall": ["0.244", "0.577", "0.712"],
        "Lift over base (3.5%)": ["24.2×", "11.5×", "7.1×"],
    })
    st.dataframe(topk_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — THRESHOLD & COST OPTIMIZER
# ============================================================
with tab2:
    st.header("Business Cost & Threshold Optimization")
    st.caption(
        "Adjust the relative cost of missed fraud vs false alerts to see how "
        "the business-optimal decision threshold shifts."
    )

    thresh_path = FIGURES_DIR / "threshold_analysis.png"
    if thresh_path.exists():
        st.image(Image.open(thresh_path), use_container_width=True)
    else:
        st.info("Run Phase 4 to generate threshold_analysis.png.")

    st.markdown("---")
    st.subheader("Interactive Cost Simulator")

    col_a, col_b = st.columns(2)
    with col_a:
        cost_fn = st.slider(
            "False Negative Cost — $ per missed fraud",
            min_value=100, max_value=2000, value=500, step=50,
        )
    with col_b:
        cost_fp = st.slider(
            "False Positive Cost — $ per analyst review",
            min_value=1, max_value=200, value=50, step=5,
        )

    ratio = cost_fn / cost_fp
    st.info(
        f"With FN cost = **${cost_fn}** and FP cost = **${cost_fp}**, "
        f"the cost ratio is **{ratio:.1f} : 1**. "
        f"Higher ratios push the optimal threshold **down** "
        f"(catch more fraud at the cost of more alerts)."
    )

    # Quick reference table
    st.markdown("**Reference Operating Points (from Temporal Test Set)**")
    ref_df = pd.DataFrame({
        "Threshold": ["0.05", "0.09", "0.15", "0.30"],
        "Precision": ["~0.24", "0.298", "~0.42", "~0.61"],
        "Recall":    ["~0.75", "0.670", "~0.51", "~0.28"],
        "Notes": [
            "High-recall triage mode",
            "Cost-optimal ($500 / $50)",
            "Balanced mode",
            "High-precision blocking mode",
        ],
    })
    st.dataframe(ref_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 — PRODUCTION DEPLOYMENT (replaces broken scoring tab)
# ============================================================
with tab3:
    st.header("Production Deployment Architecture")
    st.caption(
        "How this model would be deployed in a real fraud operations pipeline."
    )

    st.markdown("""
    ### Why Not Just Score a Single Transaction?

    The model uses **contextual features** that require a user's *history*, not just
    the current transaction. These are computed by an upstream **feature store**:

    | Feature | Source | Purpose |
    | :--- | :--- | :--- |
    | `uid_txn_count_prior` | Feature store | How many prior txns from this account |
    | `uid_avg_amt_prior` | Feature store | User's historical mean amount |
    | `uid_std_amt_prior` | Feature store | User's historical volatility |
    | `amt_zscore_user` | Feature store | How unusual this amount is for the user |
    | `time_since_prev_uid` | Feature store | Seconds since last transaction |
    | `card*_freq` | Feature store | Train-fit frequency encoding |
    """)

    st.markdown("---")

    st.subheader("End-to-End Inference Pipeline")

    st.subheader("End-to-End Inference Pipeline")

    pipe_col1, pipe_col2, pipe_col3, pipe_col4, pipe_col5 = st.columns(5)

    with pipe_col1:
        st.markdown("""
        <div style="background:#f1f5f9;padding:1rem;border-radius:8px;text-align:center;height:110px;">
            <div style="font-size:1.5rem;">📥</div>
            <div style="font-weight:700;color:#0f172a;margin-top:.5rem;">Transaction</div>
            <div style="font-size:.75rem;color:#64748b;">Kafka stream</div>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col2:
        st.markdown("""
        <div style="background:#f1f5f9;padding:1rem;border-radius:8px;text-align:center;height:110px;">
            <div style="font-size:1.5rem;">🗄️</div>
            <div style="font-weight:700;color:#0f172a;margin-top:.5rem;">Feature Store</div>
            <div style="font-size:.75rem;color:#64748b;">Redis / Feast</div>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col3:
        st.markdown("""
        <div style="background:#dbeafe;padding:1rem;border-radius:8px;text-align:center;height:110px;">
            <div style="font-size:1.5rem;">🧠</div>
            <div style="font-weight:700;color:#0f172a;margin-top:.5rem;">LightGBM</div>
            <div style="font-size:.75rem;color:#64748b;">ONNX runtime</div>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col4:
        st.markdown("""
        <div style="background:#f1f5f9;padding:1rem;border-radius:8px;text-align:center;height:110px;">
            <div style="font-size:1.5rem;">📐</div>
            <div style="font-weight:700;color:#0f172a;margin-top:.5rem;">Isotonic Calibration</div>
            <div style="font-size:.75rem;color:#64748b;">Probability mapping</div>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col5:
        st.markdown("""
        <div style="background:#dcfce7;padding:1rem;border-radius:8px;text-align:center;height:110px;">
            <div style="font-size:1.5rem;">✅</div>
            <div style="font-weight:700;color:#0f172a;margin-top:.5rem;">Decision Engine</div>
            <div style="font-size:.75rem;color:#64748b;">Approve / Review / Block</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:1.2rem;margin-top:.5rem;'>"
        "➜ &nbsp;&nbsp;&nbsp; ➜ &nbsp;&nbsp;&nbsp; ➜ &nbsp;&nbsp;&nbsp; ➜</div>",
        unsafe_allow_html=True
    )

st.markdown("---")

st.subheader("Operational Considerations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Latency Budget**")
    st.markdown("""
    - Target: **< 100 ms** end-to-end
    - LightGBM inference: ~2 ms
    - Feature store lookup: ~10–30 ms
    - Network overhead: ~20 ms
    - Total: comfortably under budget
    """)

    st.markdown("**Model Refresh**")
    st.markdown("""
    - Retrain **weekly** on rolling 3-month window
    - Monitor drift via PSI on key features
    - Auto-rollback if val PR-AUC drops > 5%
    """)

with col2:
    st.markdown("**Calibration Monitoring**")
    st.markdown("""
    - Track Brier score on live traffic
    - Re-fit isotonic regression monthly
    - Alert if calibration error > 0.05
    """)

    st.markdown("**Analyst Feedback Loop**")
    st.markdown("""
    - Analysts mark each alert: **confirmed fraud** or **false positive**
    - Feedback feeds back into next training batch
    - This is the single biggest lever for long-term improvement
    """)

st.markdown("---")

st.info(
    "**Key Result:** At the top 1% of risk scores, precision is **84.8%** "
    "(a 24× lift over the 3.5% base rate). This is the metric that matters "
    "for a fraud operations team — not PR-AUC."
)