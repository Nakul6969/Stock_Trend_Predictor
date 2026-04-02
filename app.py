import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from model import train_model, predict_signal, add_features, FEATURE_COLS

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense — ML Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark navy theme (matches your Apex Screener palette)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a1520;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1923 !important;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Cards */
.card {
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent { border-left: 3px solid #38bdf8; }

/* Signal badges */
.signal-buy {
    display:inline-block; padding:10px 28px;
    background:linear-gradient(135deg,#065f46,#047857);
    color:#6ee7b7; font-size:1.6rem; font-weight:700;
    border-radius:10px; border:1px solid #047857;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px;
}
.signal-sell {
    display:inline-block; padding:10px 28px;
    background:linear-gradient(135deg,#7f1d1d,#b91c1c);
    color:#fca5a5; font-size:1.6rem; font-weight:700;
    border-radius:10px; border:1px solid #b91c1c;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px;
}
.signal-hold {
    display:inline-block; padding:10px 28px;
    background:linear-gradient(135deg,#1c3149,#1e3a5f);
    color:#7dd3fc; font-size:1.6rem; font-weight:700;
    border-radius:10px; border:1px solid #38bdf8;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px;
}

/* Metric tiles */
.metric-tile {
    background:#0f1923; border:1px solid #1e2d3d;
    border-radius:10px; padding:16px 20px; text-align:center;
}
.metric-label { font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-size:1.5rem; font-weight:700; color:#38bdf8; font-family:'JetBrains Mono',monospace; }
.metric-value.green  { color:#34d399; }
.metric-value.red    { color:#f87171; }
.metric-value.white  { color:#e2e8f0; }

/* Page title */
.page-title {
    font-family:'DM Serif Display',serif;
    font-size:2.4rem; color:#f1f5f9;
    margin-bottom:4px;
}
.page-sub { color:#64748b; font-size:0.9rem; margin-bottom:28px; }

/* Progress bar override */
.stProgress > div > div { background-color:#38bdf8 !important; }

/* Button */
div.stButton > button {
    background: linear-gradient(135deg,#0369a1,#0284c7);
    color: white; border: none; border-radius: 8px;
    padding: 10px 28px; font-weight:600;
    font-family:'DM Sans',sans-serif;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity:0.85; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
POPULAR_STOCKS = {
    "🇮🇳 Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "WIPRO.NS", "SBIN.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "ADANIENT.NS",
    ],
    "🇺🇸 US Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "NFLX", "JPM", "V",
    ],
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def color_signal(sig):
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵"}.get(sig, "")


def fmt_price(v):
    return f"₹{v:,.2f}" if v else "—"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockSense")
    st.markdown("*ML-powered next-day prediction*")
    st.divider()

    st.markdown("**Select Stock**")
    category = st.selectbox("Category", list(POPULAR_STOCKS.keys()), label_visibility="collapsed")
    ticker_choice = st.selectbox("Stock", POPULAR_STOCKS[category], label_visibility="collapsed")

    custom = st.text_input("Or enter custom ticker", placeholder="e.g. ONGC.NS")
    ticker = custom.upper().strip() if custom.strip() else ticker_choice

    st.divider()
    st.markdown("**Settings**")
    period = st.select_slider("Training period", ["6mo", "1y", "2y", "3y", "5y"], value="2y")
    chart_days = st.slider("Chart lookback (days)", 30, 180, 90)

    st.divider()
    run_btn = st.button("🚀 Analyze & Predict", use_container_width=True)

    st.markdown("""
    <div style='margin-top:24px; font-size:0.75rem; color:#475569;'>
    Model: RF + GBT Ensemble<br>
    Features: 17 technical indicators<br>
    Signal: BUY ≥60% | SELL ≤40%
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📊 StockSense ML Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Next-day price direction prediction using ensemble machine learning</div>', unsafe_allow_html=True)

if not run_btn:
    # Landing state
    st.markdown("""
    <div class="card card-accent">
    <h3 style='color:#38bdf8; font-family:"DM Serif Display",serif; margin-top:0'>How it works</h3>
    <p style='color:#94a3b8; margin:0'>
    Select a stock from the sidebar, choose your training period, then click <strong style='color:#e2e8f0'>Analyze & Predict</strong>.<br><br>
    The model trains a <strong style='color:#38bdf8'>Random Forest + Gradient Boosting</strong> ensemble on 17 technical indicators
    (RSI, MACD, Bollinger Bands, Moving Averages, Volatility, Volume signals) and predicts whether
    tomorrow's price will be <strong style='color:#34d399'>UP ↑</strong> or <strong style='color:#f87171'>DOWN ↓</strong>,
    generating a <strong>BUY / SELL / HOLD</strong> signal.
    </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    tips = [
        ("🎯", "17 Features", "RSI, MACD, BB, MA crossovers, volume ratio & more"),
        ("🤖", "Ensemble Model", "RandomForest + GradientBoosting soft-vote ensemble"),
        ("⚡", "Live Data", "Real-time data via yfinance — always up to date"),
    ]
    for col, (icon, title, desc) in zip(cols, tips):
        col.markdown(f"""
        <div class="card" style="text-align:center;">
        <div style="font-size:2rem">{icon}</div>
        <div style="font-weight:600; color:#e2e8f0; margin:8px 0 4px">{title}</div>
        <div style="font-size:0.82rem; color:#64748b">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ─── Data fetch ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**..."):
    df_raw = fetch_data(ticker, period)

if df_raw.empty:
    st.error(f"❌ Could not fetch data for **{ticker}**. Check the ticker symbol.")
    st.stop()

# ─── Train model ─────────────────────────────────────────────────────────────
progress = st.progress(0, text="Training model...")
time.sleep(0.1)

try:
    model, scaler, accuracy, importances = train_model(df_raw)
    progress.progress(70, text="Generating prediction...")
    signal, prob_up, next_price = predict_signal(model, df_raw)
    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()
except Exception as e:
    progress.empty()
    st.error(f"Training failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CARD
# ─────────────────────────────────────────────────────────────────────────────
last_close = float(df_raw["Close"].iloc[-1])
prev_close = float(df_raw["Close"].iloc[-2])
day_chg    = (last_close - prev_close) / prev_close * 100
chg_color  = "green" if day_chg >= 0 else "red"
chg_arrow  = "▲" if day_chg >= 0 else "▼"

sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(signal, "signal-hold")
sig_emoji  = {"BUY": "↑ UP", "SELL": "↓ DOWN", "HOLD": "→ NEUTRAL"}.get(signal, "HOLD")

st.markdown(f"""
<div class="card card-accent">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
    <div>
      <div style="font-size:0.8rem; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px">
        Tomorrow's Signal for <span style="color:#38bdf8">{ticker}</span>
      </div>
      <div class="{sig_class}">{color_signal(signal)} {signal}</div>
      <div style="margin-top:10px; font-size:0.85rem; color:#94a3b8">
        Predicted direction: <strong style="color:#e2e8f0">{sig_emoji}</strong> &nbsp;|&nbsp;
        Confidence: <strong style="color:#38bdf8">{prob_up*100:.1f}%</strong> probability of price going UP
      </div>
    </div>
    <div style="display:flex; gap:16px; flex-wrap:wrap;">
      <div class="metric-tile">
        <div class="metric-label">Last Close</div>
        <div class="metric-value white">₹{last_close:,.2f}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Day Change</div>
        <div class="metric-value {chg_color}">{chg_arrow} {abs(day_chg):.2f}%</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Est. Next Close</div>
        <div class="metric-value">₹{next_price:,.2f}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">{accuracy*100:.1f}%</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — Price + Volume + RSI
# ─────────────────────────────────────────────────────────────────────────────
df_feat  = add_features(df_raw).dropna()
df_chart = df_feat.tail(chart_days).copy()

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.25, 0.20],
    vertical_spacing=0.03,
    subplot_titles=("Price & Moving Averages", "Volume", "RSI (14)")
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df_chart.index,
    open=df_chart["Open"], high=df_chart["High"],
    low=df_chart["Low"],  close=df_chart["Close"],
    name="OHLC",
    increasing_line_color="#34d399", decreasing_line_color="#f87171",
    increasing_fillcolor="#065f46",  decreasing_fillcolor="#7f1d1d",
), row=1, col=1)

# MAs
for ma, color in [("MA_5","#fbbf24"), ("MA_20","#38bdf8"), ("MA_50","#a78bfa")]:
    fig.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart[ma], name=ma.replace("_"," "),
        line=dict(color=color, width=1.2), opacity=0.85
    ), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["BB_upper"],
    line=dict(color="#64748b", width=1, dash="dot"), name="BB Upper", showlegend=False
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["BB_lower"],
    line=dict(color="#64748b", width=1, dash="dot"), name="BB Lower",
    fill="tonexty", fillcolor="rgba(100,116,139,0.06)", showlegend=False
), row=1, col=1)

# BUY/SELL signal annotation on last bar
arrow_color = {"BUY": "#34d399", "SELL": "#f87171", "HOLD": "#38bdf8"}[signal]
arrow_sym   = {"BUY": "triangle-up", "SELL": "triangle-down", "HOLD": "circle"}[signal]
fig.add_trace(go.Scatter(
    x=[df_chart.index[-1]], y=[df_chart["Close"].iloc[-1]],
    mode="markers+text",
    marker=dict(symbol=arrow_sym, size=16, color=arrow_color, line=dict(color="white", width=1)),
    text=[f"  {signal}"], textposition="top right",
    textfont=dict(color=arrow_color, size=13, family="JetBrains Mono"),
    name=f"Signal: {signal}", showlegend=True,
), row=1, col=1)

# Volume bars
vol_colors = ["#34d399" if r >= 0 else "#f87171" for r in df_chart["Return_1d"]]
fig.add_trace(go.Bar(
    x=df_chart.index, y=df_chart["Volume"],
    marker_color=vol_colors, name="Volume", opacity=0.7,
), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["RSI"],
    line=dict(color="#a78bfa", width=1.5), name="RSI",
), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="#f87171", opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="#34d399", opacity=0.5, row=3, col=1)
fig.add_hrect(y0=30, y1=70, fillcolor="rgba(56,189,248,0.04)", line_width=0, row=3, col=1)

# Layout
fig.update_layout(
    paper_bgcolor="#0a1520", plot_bgcolor="#0f1923",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    legend=dict(bgcolor="rgba(15,25,35,0.8)", bordercolor="#1e2d3d", borderwidth=1,
                font=dict(size=11)),
    margin=dict(l=0, r=0, t=40, b=0),
    height=620,
    xaxis_rangeslider_visible=False,
)
fig.update_xaxes(gridcolor="#1e2d3d", showgrid=True)
fig.update_yaxes(gridcolor="#1e2d3d", showgrid=True)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE + PROBABILITY GAUGE
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Feature Importance")
    top_feats  = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    feat_names = [f[0] for f in top_feats]
    feat_vals  = [f[1] for f in top_feats]

    fig_imp = go.Figure(go.Bar(
        x=feat_vals[::-1], y=feat_names[::-1],
        orientation="h",
        marker=dict(
            color=feat_vals[::-1],
            colorscale=[[0,"#1e3a5f"],[0.5,"#38bdf8"],[1,"#0ea5e9"]],
            showscale=False,
        ),
        text=[f"{v*100:.1f}%" for v in feat_vals[::-1]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_imp.update_layout(
        paper_bgcolor="#0f1923", plot_bgcolor="#0f1923",
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(l=0, r=50, t=10, b=10),
        height=280,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(gridcolor="#1e2d3d"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Prediction Confidence")

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_up * 100,
        number=dict(suffix="%", font=dict(color="#38bdf8", size=36, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#475569",
                      tickfont=dict(color="#64748b"), nticks=5),
            bar=dict(color=arrow_color, thickness=0.3),
            bgcolor="#0a1520",
            borderwidth=0,
            steps=[
                dict(range=[0,  40], color="#7f1d1d"),
                dict(range=[40, 60], color="#1e3a5f"),
                dict(range=[60,100], color="#065f46"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=50),
        ),
        title=dict(text="Prob. Price UP Tomorrow", font=dict(color="#94a3b8", size=13)),
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0f1923", plot_bgcolor="#0f1923",
        height=240, margin=dict(l=20, r=20, t=30, b=10),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:4px;">
      <span style="background:#7f1d1d; color:#fca5a5; padding:4px 12px; border-radius:20px; font-size:0.78rem;">≤40% → SELL</span>
      <span style="background:#1e3a5f; color:#7dd3fc; padding:4px 12px; border-radius:20px; font-size:0.78rem;">40–60% → HOLD</span>
      <span style="background:#065f46; color:#6ee7b7; padding:4px 12px; border-radius:20px; font-size:0.78rem;">≥60% → BUY</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RECENT SIGNALS TABLE (last 30 days backtest)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 Recent Signals (Last 30 Days — Backtest)")

df_bt = df_feat.tail(35).copy()
proba_list = []
for i in range(5, len(df_bt)):
    subset = df_raw.iloc[:-(len(df_bt) - i)]
    if len(subset) < 60:
        proba_list.append(0.5)
        continue
    try:
        m2, _, _, _ = train_model(subset)
        _, p, _ = predict_signal(m2, subset)
        proba_list.append(p)
    except Exception:
        proba_list.append(0.5)

df_display = df_bt.tail(len(proba_list)).copy()
df_display["Prob_UP"]   = proba_list
df_display["Signal"]    = df_display["Prob_UP"].apply(
    lambda p: "🟢 BUY" if p >= 0.6 else ("🔴 SELL" if p <= 0.4 else "🔵 HOLD")
)
df_display["Next_Actual"] = df_display["Close"].shift(-1)
df_display["Actual_Dir"] = (df_display["Next_Actual"] > df_display["Close"]).map(
    {True: "↑ UP", False: "↓ DOWN"}
)
df_display["Conf"] = (df_display["Prob_UP"] * 100).round(1).astype(str) + "%"

table = df_display[["Close", "Signal", "Conf", "Actual_Dir"]].dropna().tail(20)
table.index = table.index.strftime("%d %b %Y")
table.columns = ["Close Price", "Signal", "Confidence", "Actual (Next Day)"]
table["Close Price"] = table["Close Price"].apply(lambda x: f"₹{x:,.2f}")

st.dataframe(table, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.78rem; margin-top:40px; padding:20px 0; border-top:1px solid #1e2d3d;'>
⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Not financial advice. Do your own research before investing.<br>
Built with Python · scikit-learn · yfinance · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)