"""
Financial Market Analytics — Main Pipeline
Run this file to execute the full end-to-end pipeline.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator    import generate_all_assets
from src.feature_engineering import engineer_features
from src.ml_models         import run_training
from src.dashboard         import build_dashboard

import pandas as pd


def run_pipeline():
    t0 = time.time()

    print("=" * 60)
    print("  FINANCIAL MARKET ANALYTICS PIPELINE")
    print("=" * 60)

    # ── Step 1: Data Generation
    print("\n[1/4] Generating synthetic market data...")
    df_raw = generate_all_assets()
    os.makedirs("data", exist_ok=True)
    df_raw.to_csv("data/market_data.csv", index=False)
    print(f"      {len(df_raw):,} rows | {df_raw['ticker'].nunique()} assets")

    # ── Step 2: Feature Engineering
    print("\n[2/4] Engineering technical indicators & features...")
    df_feat = engineer_features(df_raw)
    df_feat.to_csv("data/market_features.csv", index=False)
    print(f"      {df_feat.shape[1]} columns engineered")

    # ── Step 3: ML Training
    print("\n[3/4] Training ML models...")
    results = run_training(df_feat)

    # Save feature importance for best model
    best_model = max(
        results["metrics"],
        key=lambda m: m["roc_auc"]
    )["model"]
    fi_df = results["feature_importance"].get(best_model, pd.DataFrame())
    if not fi_df.empty:
        fi_df.to_csv("data/feature_importance.csv", index=False)

    metrics_df = pd.read_csv("data/model_metrics.csv")

    # ── Step 4: Dashboard
    print("\n[4/4] Building interactive Plotly dashboard...")
    build_dashboard(df_feat, metrics_df, fi_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Reports saved to: ./reports/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_pipeline()
