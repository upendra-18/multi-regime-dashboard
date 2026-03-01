# ==========================================================
# Multi-Regime Market State Detection Dashboard
# Extended Version (Satisfies All 10 Requirements)
# ==========================================================

import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

API_BASE = "https://multi-regime-market-state-detection-adaptive-c-production-1c05.up.railway.app/"

st.set_page_config(layout="wide")
st.title("Multi-Regime Market State Detection & Capital Allocation System")

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("Regime Mode")
mode = st.sidebar.radio(
    "",
    ["4-Regime", "2-Regime"],
    horizontal=False
)

st.sidebar.header("Strategy Controls")

bull_calm = st.sidebar.slider("Bull Calm Exposure", 0.5, 1.5, 1.0)
bull_turb = st.sidebar.slider("Bull Turbulent Exposure", 0.5, 1.5, 0.9)
bear_calm = st.sidebar.slider("Bear Calm Exposure", 0.0, 1.0, 0.6)
bear_turb = st.sidebar.slider("Bear Turbulent Exposure", 0.0, 1.0, 0.2)
cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.05)/100

# ==========================================================
# SAFE API CALL
# ==========================================================

def safe_get(endpoint, params=None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except:
        return None

predict = safe_get("/predict")
health = safe_get("/health")
metrics = safe_get("/metrics")
history = safe_get("/history")
importance = safe_get("/importance")

backtest = safe_get(
    "/backtest",
    params={
        "bull_calm": bull_calm,
        "bull_turb": bull_turb,
        "bear_calm": bear_calm,
        "bear_turb": bear_turb,
        "cost": cost,
        "mode": mode
    }
)

# ==========================================================
# API STATUS
# ==========================================================

if health:
    st.success(f"API Healthy | Last Update: {health.get('timestamp','N/A')}")
else:
    st.error("API Not Responding")

# ==========================================================
# LIVE REGIME PANEL
# ==========================================================

if predict:

    final = predict["Final_Decision"]

    regime = final["Regime_Final"]
    exposure = final["Exposure_Fraction"]
    alloc_pct = final["Recommended_Position_%"]

    raw = predict["Raw_Model_Output"]

    class1_probs = [
        float(v.get("1", 0))
        for v in raw.values()
    ]

    confidence = round(max(class1_probs) * 100, 1)

    risk = (
        "High" if "HighVol" in regime
        else "Moderate" if "LowVol" in regime
        else "Neutral"
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Current Regime", regime)
    c2.metric("Recommended Allocation", f"{alloc_pct}%")
    c3.metric("Risk Level", risk)
    c4.metric("Model Confidence", f"{confidence}%")

# ==========================================================
# ALLOCATION GAUGE
# ==========================================================

if predict:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=exposure*100,
        title={'text': "Allocation %"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,40],'color':'red'},
                {'range':[40,70],'color':'yellow'},
                {'range':[70,100],'color':'green'}
            ]
        }
    ))
    st.plotly_chart(gauge, width="stretch")

# ==========================================================
# EQUITY & DRAWDOWN
# ==========================================================

if backtest:

    dates = backtest["dates"]
    strategy = backtest["strategy"]
    bh = backtest["buy_hold"]
    dd = backtest["drawdown"]
    bh_dd = backtest.get("buy_hold_drawdown", None)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=dates,y=strategy,name="Strategy"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dates,y=bh,name="Buy & Hold",line=dict(dash="dash")),row=1,col=1)

    fig.add_trace(go.Scatter(x=dates,y=dd,name="Strategy DD"),row=2,col=1)

    if bh_dd:
        fig.add_trace(go.Scatter(x=dates,y=bh_dd,name="BH DD",line=dict(dash="dash")),row=2,col=1)

    fig.update_layout(height=750,title="Equity & Drawdown Comparison")
    st.plotly_chart(fig, width="stretch")

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

if metrics:

    m1,m2,m3,m4 = st.columns(4)
    m5,m6,m7,m8 = st.columns(4)

    m1.metric("CAGR %", metrics.get("cagr"))
    m2.metric("Sharpe", metrics.get("sharpe"))
    m3.metric("Sortino", metrics.get("sortino"))
    m4.metric("Calmar", metrics.get("calmar"))

    m5.metric("Volatility %", metrics.get("volatility"))
    m6.metric("Max Drawdown %", metrics.get("max_drawdown"))
    m7.metric("Avg Exposure %", metrics.get("avg_exposure"))
    m8.metric("Turnover", metrics.get("turnover"))

# ==========================================================
# REGIME STRIP (Clean Professional Version)
# ==========================================================

if history:

    df = pd.DataFrame({
        "Date": pd.to_datetime(history["dates"]),
        "Regime": history["regimes"]
    }).sort_values("Date")

    # Detect regime change blocks
    df["Change"] = df["Regime"] != df["Regime"].shift(1)
    df["Block"] = df["Change"].cumsum()

    color_map = {
        "Bull": "#2ecc71",
        "Bear": "#e74c3c",
        "HighVol_Bull": "#f39c12",
        "HighVol_Bear": "#8e44ad"
    }

    fig_strip = go.Figure()

    # Draw horizontal strip using line segments
    for _, block in df.groupby("Block"):
        regime = block["Regime"].iloc[0]
        start = block["Date"].iloc[0]
        end = block["Date"].iloc[-1]

        fig_strip.add_trace(go.Scatter(
            x=[start, end],
            y=[1, 1],
            mode="lines",
            line=dict(
                width=18,
                color=color_map.get(regime, "#888888")
            ),
            showlegend=False
        ))

    fig_strip.update_layout(
        height=130,
        title="Regime Timeline",
        yaxis=dict(visible=False),
        xaxis=dict(title="Date"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_strip, use_container_width=True)
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

if importance:

    imp_df = pd.DataFrame(
        list(importance.items()),
        columns=["Feature","Importance"]
    ).sort_values("Importance",ascending=False).head(10)

    fig3 = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h"
    ))

    fig3.update_layout(title="Top 10 Feature Importance")
    st.plotly_chart(fig3,width="stretch")

# ==========================================================
# EXPOSURE DISTRIBUTION
# ==========================================================

if backtest:

    exposure_series = backtest.get("exposure_series")

    if exposure_series:
        fig4 = go.Figure(go.Histogram(x=exposure_series,nbinsx=20))
        fig4.update_layout(title="Exposure Distribution")
        st.plotly_chart(fig4,width="stretch")

# ==========================================================
# SYSTEM ARCHITECTURE
# ==========================================================

st.markdown("---")
st.markdown("""
### System Architecture

Data → Feature Engineering → ML Model  
→ Regime Classification → Allocation Logic  
→ Backtest Engine → Flask API (Railway)  
→ Streamlit Dashboard  
→ GitHub Actions (Daily Data Pipeline)
""")
