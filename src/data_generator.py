"""
Financial Market Data Generator
Generates realistic synthetic OHLCV data for stocks and crypto assets
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


def generate_ohlcv(
    ticker: str,
    start_price: float,
    start_date: str,
    days: int = 730,
    volatility: float = 0.02,
    trend: float = 0.0002,
    volume_base: int = 1_000_000,
) -> pd.DataFrame:
    """Generate realistic OHLCV data using Geometric Brownian Motion."""
    dates = pd.date_range(start=start_date, periods=days, freq="B")
    
    # GBM simulation
    returns = np.random.normal(trend, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Add regime changes (market crashes/booms)
    for _ in range(random.randint(2, 5)):
        if days <= 100:
            break
        crash_day = random.randint(50, days - 50)
        crash_magnitude = random.choice([-0.15, -0.10, 0.12, 0.18])
        prices[crash_day:] *= (1 + crash_magnitude)
    
    records = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_range = close * random.uniform(0.005, volatility * 2)
        open_price = close * (1 + random.uniform(-volatility, volatility))
        high = max(open_price, close) + daily_range * random.uniform(0.1, 0.5)
        low = min(open_price, close) - daily_range * random.uniform(0.1, 0.5)
        
        # Volume with seasonality
        vol_multiplier = 1 + 0.5 * np.sin(2 * np.pi * i / 252)
        volume = int(volume_base * vol_multiplier * random.uniform(0.5, 2.0))
        
        records.append({
            "date": date,
            "ticker": ticker,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
        })
    
    return pd.DataFrame(records)


def generate_all_assets() -> pd.DataFrame:
    """Generate data for multiple assets."""
    assets = [
        # Stocks
        {"ticker": "AAPL", "start_price": 150.0, "volatility": 0.018, "trend": 0.0003, "volume_base": 80_000_000},
        {"ticker": "TSLA", "start_price": 200.0, "volatility": 0.040, "trend": 0.0002, "volume_base": 50_000_000},
        {"ticker": "MSFT", "start_price": 280.0, "volatility": 0.016, "trend": 0.0004, "volume_base": 30_000_000},
        {"ticker": "NVDA", "start_price": 220.0, "volatility": 0.035, "trend": 0.0008, "volume_base": 40_000_000},
        {"ticker": "GOOGL", "start_price": 130.0, "volatility": 0.019, "trend": 0.0003, "volume_base": 25_000_000},
        # Crypto
        {"ticker": "BTC-USD", "start_price": 28000.0, "volatility": 0.045, "trend": 0.0005, "volume_base": 20_000_000_000},
        {"ticker": "ETH-USD", "start_price": 1800.0, "volatility": 0.055, "trend": 0.0004, "volume_base": 10_000_000_000},
        {"ticker": "SOL-USD", "start_price": 22.0, "volatility": 0.065, "trend": 0.0006, "volume_base": 1_000_000_000},
    ]
    
    dfs = []
    for asset in assets:
        df = generate_ohlcv(
            ticker=asset["ticker"],
            start_price=asset["start_price"],
            start_date="2022-01-01",
            days=730,
            volatility=asset["volatility"],
            trend=asset["trend"],
            volume_base=asset["volume_base"],
        )
        df["asset_type"] = "crypto" if "USD" in asset["ticker"] else "stock"
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    df = generate_all_assets()
    df.to_csv("data/market_data.csv", index=False)
    print(f"Generated {len(df):,} rows across {df['ticker'].nunique()} assets")
    print(df.groupby("ticker")[["close", "volume"]].last())
