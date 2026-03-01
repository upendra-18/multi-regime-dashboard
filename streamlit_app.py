# ==========================================================
# Multi-Regime Market State Detection Dashboard
# Full Production Version (API Driven)
# ==========================================================

import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

API_BASE = "https://multi-regime-market-state-detection-adaptive-c-production-b216.up.railway.app/"  

st.set_page_config(layout="wide")
st.title("Multi-Regime Market State Detection & Capital Allocation System")

# ==========================================================
# SIDEBAR STRATEGY CONTROLS
# ==========================================================

st.sidebar.header("Strategy Controls")

bull_calm = st.sidebar.slider("Bull Calm Exposure", 0.5, 1.5, 1.0)
bull_turb = st.sidebar.slider("Bull Turbulent Exposure", 0.5, 1.5, 0.9)
bear_calm = st.sidebar.slider("Bear Calm Exposure", 0.0, 1.0, 0.6)
bear_turb = st.sidebar.slider("Bear Turbulent Exposure", 0.0, 1.0, 0.2)
cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.05)/100

# ==========================================================
# FETCH API DATA
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
        "cost": cost
    }
)

# ==========================================================
# HEALTH INDICATOR
# ==========================================================

if health:
    st.success(f"API Healthy | {health['timestamp']}")
else:
    st.error("API Not Responding")

# ==========================================================
# LIVE REGIME PANEL
# ==========================================================

if predict:

    final = predict["Final_Decision"]
    raw = predict["Raw_Model_Output"]

    regime = final["Regime_Final"]
    exposure = final["Exposure_Fraction"]
    alloc_pct = final["Recommended_Position_%"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Regime", regime)
    col2.metric("Recommended Allocation", f"{alloc_pct}%")

    risk = "High" if "HighVol" in regime else "Moderate" if "LowVol" in regime else "Neutral"
    col3.metric("Risk Level", risk)

    confidence = round(sum(raw.values())/len(raw)*100, 1)
    col4.metric("Model Confidence", f"{confidence}%")

# ==========================================================
# ALLOCATION GAUGE
# ==========================================================

if predict:

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=exposure*100,
        title={'text': "Current Allocation %"},
        gauge={
            'axis': {'range': [0,100]},
            'steps': [
                {'range': [0,40], 'color': "red"},
                {'range': [40,70], 'color': "yellow"},
                {'range': [70,100], 'color': "green"},
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

# ==========================================================
# EQUITY + DRAWDOWN
# ==========================================================

if backtest:

    dates = backtest["dates"]
    strategy = backtest["strategy"]
    bh = backtest["buy_hold"]
    dd = backtest["drawdown"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=dates, y=strategy, name="Strategy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bh, name="Buy & Hold", line=dict(dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=dd, name="Drawdown"), row=2, col=1)

    fig.update_layout(height=800, title="Equity Curve & Drawdown")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

if metrics:

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("CAGR %", metrics["cagr"])
    m2.metric("Sharpe", metrics["sharpe"])
    m3.metric("Volatility %", metrics["volatility"])
    m4.metric("Max Drawdown %", metrics["max_drawdown"])

# ==========================================================
# REGIME TIMELINE
# ==========================================================

if history:

    df = pd.DataFrame({
        "Date": history["dates"],
        "Regime": history["regimes"]
    })

    color_map = {
        "Bull": "green",
        "Bear": "red",
        "HighVol_Bull": "orange",
        "HighVol_Bear": "purple"
    }

    df["Color"] = df["Regime"].map(color_map)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df["Date"],
        y=[1]*len(df),
        mode="markers",
        marker=dict(color=df["Color"], size=6),
        showlegend=False
    ))

    fig2.update_layout(
        height=200,
        title="Regime Timeline",
        yaxis=dict(showticklabels=False)
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

if importance:

    imp_df = pd.DataFrame(
        list(importance.items()),
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False).head(10)

    fig3 = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h"
    ))

    fig3.update_layout(title="Top 10 Feature Importance")
    st.plotly_chart(fig3, use_container_width=True)

# ==========================================================
# ARCHITECTURE SECTION
# ==========================================================

st.markdown("---")
st.markdown("""
### System Architecture

Data → Feature Engineering → ML Models  
→ Regime Classification → Capital Allocation  
→ Backtesting Engine → Flask API (Railway)  
→ Streamlit Dashboard  
→ GitHub Actions (Daily Data Pipeline)
""")
