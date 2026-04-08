"""
Technical Indicators & Feature Engineering
Computes a comprehensive set of technical indicators for financial time series.
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA and EMA indicators."""
    for window in [7, 14, 21, 50, 200]:
        df[f"sma_{window}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"ema_{window}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    def compute_rsi(series):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df[f"rsi_{window}"] = df.groupby("ticker")["close"].transform(compute_rsi)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD: 12-26-9 standard."""
    def compute_macd(series):
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal
        return macd_line, signal, histogram
    
    for ticker, group in df.groupby("ticker"):
        idx = group.index
        macd, signal, hist = compute_macd(group["close"])
        df.loc[idx, "macd"] = macd.values
        df.loc[idx, "macd_signal"] = signal.values
        df.loc[idx, "macd_hist"] = hist.values
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    def compute_bb(series):
        sma = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        width = (upper - lower) / sma
        pct_b = (series - lower) / (upper - lower + 1e-10)
        return upper, lower, width, pct_b
    
    for ticker, group in df.groupby("ticker"):
        idx = group.index
        upper, lower, width, pct_b = compute_bb(group["close"])
        df.loc[idx, "bb_upper"] = upper.values
        df.loc[idx, "bb_lower"] = lower.values
        df.loc[idx, "bb_width"] = width.values
        df.loc[idx, "bb_pct_b"] = pct_b.values
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range (volatility measure)."""
    def compute_atr(group):
        high_low = group["high"] - group["low"]
        high_close = (group["high"] - group["close"].shift()).abs()
        low_close = (group["low"] - group["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window, min_periods=1).mean()
    
    df["atr"] = df.groupby("ticker", group_keys=False).apply(compute_atr)
    df["atr_pct"] = df["atr"] / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based indicators."""
    df["volume_sma_20"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1)
    
    # On-Balance Volume
    def compute_obv(group):
        direction = np.sign(group["close"].diff().fillna(0))
        return (direction * group["volume"]).cumsum()
    
    df["obv"] = df.groupby("ticker", group_keys=False).apply(compute_obv)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price-derived features."""
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df.groupby("ticker")["close"].shift(1))
    df["price_range"] = (df["high"] - df["low"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    df["body_size"] = (df["close"] - df["open"]).abs() / df["close"]
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    
    # Rolling statistics
    for window in [5, 10, 21]:
        df[f"volatility_{window}d"] = df.groupby("ticker")["log_return"].transform(
            lambda x: x.rolling(window).std() * np.sqrt(252)
        )
        df[f"momentum_{window}d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.pct_change(window)
        )
    
    return df


def add_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Future return as target for ML."""
    df[f"future_return_{horizon}d"] = df.groupby("ticker")["close"].transform(
        lambda x: x.shift(-horizon) / x - 1
    )
    df[f"target_direction_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    print("Engineering features...")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    df = add_target_variable(df)
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/market_data.csv", parse_dates=["date"])
    featured = engineer_features(raw)
    featured.to_csv("data/market_features.csv", index=False)
    print(featured.head())
