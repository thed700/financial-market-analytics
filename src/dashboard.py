"""
Interactive Financial Dashboard
Plotly-based multi-panel dashboard with candlesticks, indicators, and ML insights.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")


DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"


def candlestick_chart(df: pd.DataFrame, ticker: str, last_n: int = 120) -> go.Figure:
    """Full OHLCV candlestick with volume, RSI, and MACD panels."""
    t = df[df["ticker"] == ticker].tail(last_n).copy()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        vertical_spacing=0.02,
        subplot_titles=("", "", "RSI (14)", "MACD"),
    )

    # ── Candlestick
    colors_up   = [ACCENT_GREEN if c >= o else ACCENT_RED for c, o in zip(t["close"], t["open"])]
    colors_down = [ACCENT_GREEN if c >= o else ACCENT_RED for c, o in zip(t["close"], t["open"])]

    fig.add_trace(go.Candlestick(
        x=t["date"], open=t["open"], high=t["high"],
        low=t["low"], close=t["close"],
        increasing_line_color=ACCENT_GREEN,
        decreasing_line_color=ACCENT_RED,
        name="OHLC",
    ), row=1, col=1)

    # SMAs
    for col, color, width in [("sma_21", ACCENT_BLUE, 1.5), ("sma_50", ACCENT_ORANGE, 1.5)]:
        if col in t.columns:
            fig.add_trace(go.Scatter(
                x=t["date"], y=t[col], name=col.upper(),
                line=dict(color=color, width=width), opacity=0.85,
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in t.columns:
        fig.add_trace(go.Scatter(
            x=t["date"], y=t["bb_upper"], name="BB Upper",
            line=dict(color=ACCENT_PURPLE, width=0.8, dash="dot"), opacity=0.6,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t["date"], y=t["bb_lower"], name="BB Lower",
            line=dict(color=ACCENT_PURPLE, width=0.8, dash="dot"), opacity=0.6,
            fill="tonexty", fillcolor="rgba(188,140,255,0.05)",
        ), row=1, col=1)

    # ── Volume bars
    fig.add_trace(go.Bar(
        x=t["date"], y=t["volume"],
        name="Volume",
        marker_color=[ACCENT_GREEN if c >= o else ACCENT_RED
                      for c, o in zip(t["close"], t["open"])],
        opacity=0.7,
    ), row=2, col=1)

    # ── RSI
    if "rsi_14" in t.columns:
        fig.add_trace(go.Scatter(
            x=t["date"], y=t["rsi_14"], name="RSI 14",
            line=dict(color=ACCENT_BLUE, width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=ACCENT_RED,  opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=ACCENT_GREEN, opacity=0.5, row=3, col=1)

    # ── MACD
    if "macd" in t.columns:
        colors_hist = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in t["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=t["date"], y=t["macd_hist"], name="MACD Hist",
            marker_color=colors_hist, opacity=0.7,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=t["date"], y=t["macd"], name="MACD",
            line=dict(color=ACCENT_BLUE, width=1.3),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=t["date"], y=t["macd_signal"], name="Signal",
            line=dict(color=ACCENT_ORANGE, width=1.3),
        ), row=4, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> — Technical Analysis Dashboard", font=dict(size=20, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="monospace"),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR, borderwidth=1, font_size=10),
        xaxis_rangeslider_visible=False,
        height=900,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    for i in range(1, 5):
        fig.update_xaxes(row=i, col=1, gridcolor=GRID_COLOR, zeroline=False)
        fig.update_yaxes(row=i, col=1, gridcolor=GRID_COLOR, zeroline=False)

    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix of asset returns."""
    pivot = df.pivot_table(index="date", columns="ticker", values="daily_return")
    corr = pivot.corr().round(3)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, ACCENT_RED], [0.5, PANEL_BG], [1, ACCENT_GREEN]],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        hovertemplate="<b>%{y} vs %{x}</b><br>Corr: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Asset Return Correlations</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        height=550,
        margin=dict(l=80, r=40, t=60, b=80),
    )
    return fig


def volatility_comparison(df: pd.DataFrame) -> go.Figure:
    """Rolling 21-day annualized volatility across all assets."""
    fig = go.Figure()
    palette = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE,
               ACCENT_RED, "#79c0ff", "#7ee787", "#ffa657"]
    
    for i, (ticker, group) in enumerate(df.groupby("ticker")):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=group["date"], y=group["volatility_21d"],
            name=ticker, line=dict(color=color, width=1.5),
            hovertemplate=f"<b>{ticker}</b><br>Vol: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="<b>21-Day Rolling Volatility (Annualized)</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        yaxis=dict(tickformat=".0%", gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR),
        height=480,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def model_metrics_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing ML model performance."""
    metrics = ["accuracy", "roc_auc", "precision_up", "recall_up"]
    colors  = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.replace("_", " ").title(),
            x=metrics_df["model"],
            y=metrics_df[metric],
            marker_color=color,
            text=metrics_df[metric].apply(lambda v: f"{v:.2%}"),
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(text="<b>ML Model Performance Comparison</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        barmode="group",
        yaxis=dict(tickformat=".0%", gridcolor=GRID_COLOR, range=[0, 1.1]),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR),
        height=480,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def feature_importance_chart(fi_df: pd.DataFrame, model_name: str) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    top20 = fi_df.head(20).sort_values("importance")
    
    fig = go.Figure(go.Bar(
        x=top20["importance"], y=top20["feature"],
        orientation="h",
        marker=dict(
            color=top20["importance"],
            colorscale=[[0, PANEL_BG], [0.5, ACCENT_BLUE], [1, ACCENT_GREEN]],
        ),
        text=top20["importance"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"<b>Feature Importance — {model_name}</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        height=600,
        margin=dict(l=160, r=80, t=60, b=40),
    )
    return fig


def normalized_returns_chart(df: pd.DataFrame) -> go.Figure:
    """Cumulative normalized return: base 100."""
    fig = go.Figure()
    palette = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE,
               ACCENT_RED, "#79c0ff", "#7ee787", "#ffa657"]
    
    for i, (ticker, group) in enumerate(df.groupby("ticker")):
        group = group.sort_values("date")
        norm = (group["close"] / group["close"].iloc[0]) * 100
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=group["date"], y=norm, name=ticker,
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{ticker}</b><br>Return: %{{y:.1f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", line_color=TEXT_COLOR, opacity=0.3)
    fig.update_layout(
        title=dict(text="<b>Normalized Price Performance (Base = 100)</b>", font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR),
        height=500,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def build_dashboard(df: pd.DataFrame, metrics_df: pd.DataFrame, fi_df: pd.DataFrame):
    """Generate and save all dashboard charts."""
    import os
    os.makedirs("reports", exist_ok=True)

    print("Building dashboard charts...")

    # 1. Candlestick for each ticker
    for ticker in df["ticker"].unique():
        fig = candlestick_chart(df, ticker)
        fig.write_html(f"reports/candlestick_{ticker.replace('-', '_')}.html")
        print(f"  Saved: reports/candlestick_{ticker.replace('-', '_')}.html")

    # 2. Correlation heatmap
    corr_fig = correlation_heatmap(df)
    corr_fig.write_html("reports/correlation_heatmap.html")
    print("  Saved: reports/correlation_heatmap.html")

    # 3. Volatility comparison
    vol_fig = volatility_comparison(df)
    vol_fig.write_html("reports/volatility_comparison.html")
    print("  Saved: reports/volatility_comparison.html")

    # 4. Normalized returns
    ret_fig = normalized_returns_chart(df)
    ret_fig.write_html("reports/normalized_returns.html")
    print("  Saved: reports/normalized_returns.html")

    # 5. Model metrics
    if not metrics_df.empty:
        model_fig = model_metrics_chart(metrics_df)
        model_fig.write_html("reports/model_comparison.html")
        print("  Saved: reports/model_comparison.html")

    # 6. Feature importance
    if not fi_df.empty:
        fi_fig = feature_importance_chart(fi_df, "Gradient Boosting")
        fi_fig.write_html("reports/feature_importance.html")
        print("  Saved: reports/feature_importance.html")

    print("Dashboard complete!")


if __name__ == "__main__":
    df = pd.read_csv("data/market_features.csv", parse_dates=["date"])
    try:
        metrics_df = pd.read_csv("data/model_metrics.csv")
    except FileNotFoundError:
        metrics_df = pd.DataFrame()
    
    build_dashboard(df, metrics_df, pd.DataFrame())
