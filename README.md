# Enterprise Fraud Detection Pipeline

An end-to-end Machine Learning pipeline for financial fraud detection, built on the IEEE-CIS dataset.

## Pipeline Architecture
1. **`01_load_and_split.py`**: Loads raw data and performs strict chronological train/val/test splitting to prevent future-data leakage.
2. **`02_feature_engineering.py`**: Creates temporal aggregations, frequency encodings, and user-profile features.
3. **`03_train_model.py`**: Trains LightGBM with early stopping and applies Isotonic Regression for probability calibration.
4. **`04_evaluate.py`**: Generates PR/ROC curves, reliability diagrams, and performs business cost-optimization (FN/FP costs).
5. **`05_app.py`**: Streamlit dashboard for interactive threshold optimization and production deployment visualization.

## Key Results
- **Test PR-AUC:** 0.5177
- **Test ROC-AUC:** 0.8984
- **Calibrated Brier Score:** 0.0227

## How to Run
1. Place raw Kaggle data in `data/raw/`
2. Run the pipeline sequentially: `python src/01_load_and_split.py`, etc.
3. Launch the dashboard: `streamlit run src/05_app.py`