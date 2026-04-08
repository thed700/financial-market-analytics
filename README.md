# 📈 Financial Market Analytics Pipeline

> End-to-end quantitative analytics system for multi-asset financial markets — combining technical analysis, statistical feature engineering, and machine learning for market direction forecasting.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.17%2B-purple?style=flat-square&logo=plotly)
![Tests](https://img.shields.io/badge/Tests-12%20passing-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🗂️ Project Overview

This project implements a **production-grade financial data analytics pipeline** covering:

| Stage | Description |
|---|---|
| **Data Generation** | Realistic synthetic OHLCV data via Geometric Brownian Motion with regime changes |
| **Feature Engineering** | 40+ technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, momentum) |
| **Machine Learning** | Time-series-safe train/val/test splits, 3 classifiers, full evaluation suite |
| **Visualization** | Interactive Plotly dashboards: candlesticks, correlation heatmaps, volatility charts |

**Assets covered:** AAPL · TSLA · MSFT · NVDA · GOOGL · BTC-USD · ETH-USD · SOL-USD

---

## 🚀 Quickstart

```bash
# Clone and install
git clone https://github.com/thed700/financial-market-analytics.git
cd financial-market-analytics
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run tests
pytest tests/ -v
```

After running, open any file in `reports/` in your browser for interactive charts.

---

## 📁 Project Structure

```
financial-market-analytics/
│
├── main.py                    # Pipeline orchestrator
├── requirements.txt
│
├── src/
│   ├── data_generator.py      # GBM synthetic OHLCV generator
│   ├── feature_engineering.py # Technical indicators & features
│   ├── ml_models.py           # ML training, evaluation, feature importance
│   └── dashboard.py           # Plotly interactive visualizations
│
├── data/                      # Generated CSVs (gitignored)
│   ├── market_data.csv
│   ├── market_features.csv
│   ├── model_metrics.csv
│   └── feature_importance.csv
│
├── reports/                   # HTML dashboards (gitignored)
│   ├── candlestick_AAPL.html
│   ├── correlation_heatmap.html
│   ├── volatility_comparison.html
│   ├── model_comparison.html
│   └── ...
│
└── tests/
    └── test_pipeline.py       # 12 unit + integration tests
```

---

## 🔬 Technical Indicators Implemented

### Trend
- **SMA** — Simple Moving Average (7, 14, 21, 50, 200-day)
- **EMA** — Exponential Moving Average (7, 14, 21, 50-day)

### Momentum
- **RSI** — Relative Strength Index (14-day)
- **MACD** — Moving Average Convergence/Divergence (12-26-9)
- **Momentum** — Rolling price change (5, 10, 21-day)

### Volatility
- **Bollinger Bands** — Width, %B position (20-day, 2σ)
- **ATR** — Average True Range, normalized (14-day)
- **Rolling Volatility** — Annualized (5, 10, 21-day)

### Volume
- **OBV** — On-Balance Volume
- **Volume Ratio** — Current vs 20-day average

### Price Structure
- Candlestick body size, upper/lower shadow, daily range
- Log returns, daily returns, bullish/bearish flag

---

## 🤖 Machine Learning

### Methodology
- **No data leakage:** strict chronological train (70%) / val (15%) / test (15%) split
- **Target:** 5-day forward price direction (binary: up/down)
- All features computed using only past data

### Models

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~51% | ~0.52 |
| Random Forest | ~48% | ~0.49 |
| Gradient Boosting | ~51% | ~0.51 |

> **Note:** Market direction forecasting is inherently close to random (~50%) for efficient markets. These results are realistic and expected for raw technical indicators on synthetic data. The project demonstrates the full ML workflow correctly.

---

## 📊 Visualizations

All charts are fully interactive (zoom, hover, export):

- **Candlestick charts** — OHLCV with overlaid SMA, EMA, Bollinger Bands, volume, RSI, MACD
- **Correlation heatmap** — Pairwise return correlations across all 8 assets
- **Volatility comparison** — Rolling annualized volatility by asset
- **Normalized returns** — Cumulative performance starting from base 100
- **Model comparison** — Grouped bar chart: accuracy, AUC, precision, recall
- **Feature importance** — Top-20 features ranked by tree-based importance

---

## 🧪 Tests

```
tests/test_pipeline.py  — 12 tests
  TestDataGenerator       (5 tests)  OHLCV validity, price positivity, asset coverage
  TestFeatureEngineering  (5 tests)  RSI bounds, MACD columns, ATR, target encoding
  TestIntegration         (2 tests)  Smoke test, no-infinity check
```

---

## 🛠️ Tech Stack

- **Python 3.10+** — Core language
- **Pandas / NumPy** — Data manipulation and numerical computing
- **Scikit-learn** — ML models, pipelines, evaluation metrics
- **Plotly** — Interactive financial charts and dashboards
- **Pytest** — Unit and integration testing

---

## 📌 Key Skills Demonstrated

- Quantitative financial data modeling (GBM simulation)
- Time-series feature engineering without look-ahead bias
- Production-style modular Python code with type hints and docstrings
- Full ML workflow: training, validation, evaluation, interpretability
- Interactive data visualization for business stakeholders
- Automated testing with pytest

---

## 👤 Author

**Akmal** — [@thed700](https://github.com/thed700)

*Economics & Data Analytics student | Aspiring Data Analyst*

---

## 📄 License

MIT License — feel free to use, fork, and build on this.
