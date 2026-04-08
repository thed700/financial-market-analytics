"""
Machine Learning Models for Market Direction Prediction
Uses Random Forest, Gradient Boosting, and a simple ensemble.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import warnings
import json

warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct_b",
    "sma_7", "sma_21", "sma_50",
    "ema_14", "ema_21",
    "atr_pct",
    "volume_ratio",
    "daily_return", "log_return",
    "price_range", "body_size", "is_bullish",
    "volatility_5d", "volatility_10d", "volatility_21d",
    "momentum_5d", "momentum_10d", "momentum_21d",
]
TARGET_COL = "target_direction_5d"


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Chronological train/val/test split (no look-ahead bias)."""
    df = df.sort_values("date")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def prepare_xy(df: pd.DataFrame):
    """Extract features and target, drop NaNs."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    subset = df[cols + [TARGET_COL]].dropna()
    X = subset[cols]
    y = subset[TARGET_COL].astype(int)
    return X, y


def build_models():
    """Return dict of model pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=20,
                random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )),
        ]),
    }


def evaluate_model(name, model, X_test, y_test) -> dict:
    """Return metrics dict for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "precision_up": round(
            classification_report(y_test, y_pred, output_dict=True)["1"]["precision"], 4
        ),
        "recall_up": round(
            classification_report(y_test, y_pred, output_dict=True)["1"]["recall"], 4
        ),
    }
    return metrics


def get_feature_importance(model_pipeline, feature_names) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    clf = model_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()
    
    return pd.DataFrame({
        "feature": feature_names,
        "importance": imp
    }).sort_values("importance", ascending=False)


def run_training(df: pd.DataFrame) -> dict:
    """Full training pipeline. Returns fitted models + metrics."""
    train_df, val_df, test_df = time_series_split(df)
    
    X_train, y_train = prepare_xy(train_df)
    X_val,   y_val   = prepare_xy(val_df)
    X_test,  y_test  = prepare_xy(test_df)
    
    # Combine train+val for final model
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    
    models = build_models()
    results = {"metrics": [], "models": {}, "feature_importance": {}}
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_trainval, y_trainval)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results["metrics"].append(metrics)
        results["models"][name] = pipeline
        
        # Feature importance
        fi = get_feature_importance(pipeline, X_train.columns.tolist())
        results["feature_importance"][name] = fi
        
        print(f"  Accuracy: {metrics['accuracy']:.2%} | AUC: {metrics['roc_auc']:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame(results["metrics"])
    metrics_df.to_csv("data/model_metrics.csv", index=False)
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/market_features.csv", parse_dates=["date"])
    results = run_training(df)
