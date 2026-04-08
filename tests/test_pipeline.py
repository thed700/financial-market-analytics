"""
Unit Tests for Financial Market Analytics
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from src.data_generator      import generate_ohlcv, generate_all_assets
from src.feature_engineering import engineer_features, add_rsi, add_macd


# ── Data Generator Tests ──────────────────────────────────────────────

class TestDataGenerator:
    def test_ohlcv_shape(self):
        df = generate_ohlcv("TEST", 100.0, "2023-01-01", days=50)
        assert len(df) == 50
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_high_gte_low(self):
        df = generate_ohlcv("TEST", 100.0, "2023-01-01", days=200)
        assert (df["high"] >= df["low"]).all()

    def test_positive_prices(self):
        df = generate_ohlcv("TEST", 50.0, "2023-01-01", days=200)
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all(), f"{col} has non-positive values"

    def test_positive_volume(self):
        df = generate_ohlcv("TEST", 100.0, "2023-01-01", days=100)
        assert (df["volume"] > 0).all()

    def test_generate_all_assets_tickers(self):
        df = generate_all_assets()
        assert "BTC-USD" in df["ticker"].values
        assert "AAPL" in df["ticker"].values
        assert df["ticker"].nunique() == 8


# ── Feature Engineering Tests ─────────────────────────────────────────

class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self):
        df = generate_all_assets()
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def test_rsi_bounds(self, sample_df):
        df = add_rsi(sample_df.copy())
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_columns(self, sample_df):
        df = add_macd(sample_df.copy())
        for col in ["macd", "macd_signal", "macd_hist"]:
            assert col in df.columns

    def test_feature_count(self, sample_df):
        df = engineer_features(sample_df.copy())
        # Should have at least 40 columns after feature engineering
        assert df.shape[1] >= 40, f"Only {df.shape[1]} columns found"

    def test_no_negative_atr(self, sample_df):
        from src.feature_engineering import add_atr
        df = add_atr(sample_df.copy())
        atr = df["atr"].dropna()
        assert (atr >= 0).all()

    def test_target_binary(self, sample_df):
        df = engineer_features(sample_df.copy())
        targets = df["target_direction_5d"].dropna().unique()
        assert set(targets).issubset({0, 1, 0.0, 1.0})


# ── Integration Test ───────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_runs(self):
        """Smoke test: full pipeline from data gen to features."""
        df = generate_all_assets()
        df = engineer_features(df)
        assert len(df) > 0
        assert "rsi_14" in df.columns
        assert "macd" in df.columns
        assert "volatility_21d" in df.columns

    def test_no_inf_values(self):
        df = generate_all_assets()
        df = engineer_features(df)
        numeric = df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric).any().any(), "Inf values found in features"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
