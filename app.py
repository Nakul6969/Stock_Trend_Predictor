import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time

from model import train_model, predict_signal, add_features, FEATURE_COLS

# Set up matplotlib style for dark black & white theme
plt.rcParams.update({
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
    'axes.edgecolor': '#262626',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#a3a3a3',
    'ytick.color': '#a3a3a3',
    'text.color': '#ffffff',
    'grid.color': '#171717',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5
})

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
# CUSTOM CSS — Dark Black & White Monochromatic Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #000000;
    color: #e5e5e5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #262626;
}
section[data-testid="stSidebar"] * { color: #d4d4d4 !important; }

/* Cards */
.card {
    background: #0d0d0d;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent { border-left: 3px solid #ffffff; }

/* Signal badges - soft actionable color highlights */
.signal-buy {
    display:inline-block; padding:10px 28px;
    background: #0f2d1e;
    color: #22c55e; font-size:1.6rem; font-weight:700;
    border-radius:8px; border:1px solid #1b4d32;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px;
}
.signal-sell {
    display:inline-block; padding:10px 28px;
    background: #3d1616;
    color: #ef4444; font-size:1.6rem; font-weight:700;
    border-radius:8px; border:1px solid #632020;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px;
}
.signal-hold {
    display:inline-block; padding:10px 28px;
    background: #1a1a1a;
    color: #a3a3a3; font-size:1.6rem; font-weight:700;
    border-radius:8px; border:1px solid #333333;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px;
}

/* Metric tiles */
.metric-tile {
    background: #0d0d0d; border: 1px solid #262626;
    border-radius: 8px; padding: 16px 20px; text-align: center;
    min-width: 140px;
}
.metric-label { font-size:0.75rem; color:#737373; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-size:1.4rem; font-weight:700; color:#ffffff; font-family:'JetBrains Mono',monospace; }
.metric-value.white { color:#ffffff; }
.metric-value.grey { color:#a3a3a3; }
.metric-value.green { color:#22c55e !important; }
.metric-value.red { color:#ef4444 !important; }

/* Page title */
.page-title {
    font-family:'DM Serif Display',serif;
    font-size:2.4rem; color:#ffffff;
    margin-bottom:4px;
}
.page-sub { color:#737373; font-size:0.9rem; margin-bottom:28px; }

/* Progress bar override */
.stProgress > div > div { background-color:#ffffff !important; }

/* Button */
div.stButton > button, div.stButton > button * {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif;
    transition: opacity 0.2s;
}
div.stButton > button {
    padding: 10px 28px;
    width: 100%;
}
div.stButton > button:hover, div.stButton > button:hover *, div.stButton > button:active, div.stButton > button:focus {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
    opacity: 0.9;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
POPULAR_STOCKS = {
    "🇺🇸 US Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "NFLX", "JPM", "V",
    ],
    "🇮🇳 Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "WIPRO.NS", "SBIN.NS", "TATAMOTORS.NS", "BAJFINANCE.NS", "ADANIENT.NS",
    ],
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def color_signal(sig):
    return {"BUY": "▲", "SELL": "▼", "HOLD": "■"}.get(sig, "")

def fmt_price(v, ticker=""):
    symbol = "$"
    if ticker.endswith(".NS"):
        symbol = "₹"
    return f"{symbol}{v:,.2f}" if v is not None else "—"

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
    run_btn = st.button("🚀 Analyze & Predict")

    st.markdown("""
    <div style='margin-top:24px; font-size:0.75rem; color:#737373;'>
    Model: Random Forest Classifier<br>
    Features: 7 technical indicators<br>
    Signal: BUY ≥60% | SELL ≤40%
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📊 StockSense ML Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Next-day price direction prediction using machine learning</div>', unsafe_allow_html=True)

if not run_btn:
    # Landing state
    st.markdown("""
    <div class="card card-accent">
    <h3 style='color:#ffffff; font-family:"DM Serif Display",serif; margin-top:0'>How it works</h3>
    <p style='color:#a3a3a3; margin:0'>
    Select a stock from the sidebar, choose your training period, then click <strong style='color:#ffffff'>Analyze & Predict</strong>.<br><br>
    The model trains a <strong style='color:#ffffff'>Random Forest Classifier</strong> on 7 core technical indicators
    (RSI, MA ratios, price returns, and volume ratios) and predicts whether
    tomorrow's price will be <strong style='color:#ffffff'>UP ↑</strong> or <strong style='color:#ffffff'>DOWN ↓</strong>,
    generating a <strong>BUY / SELL / HOLD</strong> signal.
    </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    tips = [
        ("🎯", "7 Indicators", "RSI, SMA ratios, crossovers, volume ratios"),
        ("🤖", "Random Forest", "A robust tree classifier built for daily predictions"),
        ("⚡", "Live Data", "Real-time stock data fetched dynamically via yfinance"),
    ]
    for col, (icon, title, desc) in zip(cols, tips):
        col.markdown(f"""
        <div class="card" style="text-align:center; height: 100%;">
        <div style="font-size:2rem">{icon}</div>
        <div style="font-weight:600; color:#ffffff; margin:8px 0 4px">{title}</div>
        <div style="font-size:0.82rem; color:#737373">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─── Data fetch ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}..."):
    df_raw = fetch_data(ticker, period)

if df_raw.empty:
    st.error(f"❌ Could not fetch data for {ticker}. Check the ticker symbol.")
    st.stop()

# ─── Train model ─────────────────────────────────────────────────────────────
progress = st.progress(0, text="Training model...")
time.sleep(0.1)

try:
    models, accuracies, scaler, importances = train_model(df_raw)
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_accuracy = accuracies[best_model_name]
    
    progress.progress(70, text="Generating prediction...")
    signal, prob_up, next_price = predict_signal(best_model, scaler, df_raw)
    progress.progress(100, text="Done!")
    time.sleep(0.2)
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
sig_emoji  = {"BUY": "▲ UP", "SELL": "▼ DOWN", "HOLD": "■ NEUTRAL"}.get(signal, "HOLD")
progress_color = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#a3a3a3"}.get(signal, "#ffffff")

st.markdown(f"""
<div class="card card-accent">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
    <div>
      <div style="font-size:0.8rem; color:#737373; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px">
        Tomorrow's Signal for <span style="color:#ffffff">{ticker}</span>
      </div>
      <div class="{sig_class}">{color_signal(signal)} {signal}</div>
      <div style="margin-top:10px; font-size:0.85rem; color:#a3a3a3">
        Predicted direction: <strong style="color:#ffffff">{sig_emoji}</strong> &nbsp;|&nbsp;
        Confidence: <strong style="color:#ffffff">{prob_up*100:.1f}%</strong>
      </div>
    </div>
    <div style="display:flex; gap:16px; flex-wrap:wrap;">
      <div class="metric-tile">
        <div class="metric-label">Last Close</div>
        <div class="metric-value white">{fmt_price(last_close, ticker)}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Day Change</div>
        <div class="metric-value {chg_color}">{chg_arrow} {abs(day_chg):.2f}%</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Est. Next Close</div>
        <div class="metric-value">{fmt_price(next_price, ticker)}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Accuracy ({best_model_name})</div>
        <div class="metric-value">{best_accuracy*100:.1f}%</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — Price + Volume (Matplotlib Candlesticks)
# ─────────────────────────────────────────────────────────────────────────────
df_feat  = add_features(df_raw).dropna()
df_chart = df_feat.tail(chart_days).copy()

fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]},
    figsize=(12, 6.5)
)
fig.subplots_adjust(hspace=0.08)

# Configure subplots styling for seamless card integration
for ax in [ax1, ax2]:
    ax.set_facecolor('#0d0d0d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#262626')
    ax.spines['bottom'].set_color('#262626')
    ax.tick_params(colors='#a3a3a3', labelsize=9)
    ax.grid(True, color='#262626', linestyle=':', linewidth=0.5)

# 1. Draw Candlesticks on ax1
body_width = 0.6
# Green candles (Close >= Open)
up = df_chart[df_chart["Close"] >= df_chart["Open"]]
ax1.vlines(up.index, up["Low"], up["High"], color="#22c55e", linewidth=1.0)
ax1.bar(up.index, up["Close"] - up["Open"], body_width, bottom=up["Open"], color="#22c55e", edgecolor="#22c55e", linewidth=0)

# Red candles (Close < Open)
down = df_chart[df_chart["Close"] < df_chart["Open"]]
ax1.vlines(down.index, down["Low"], down["High"], color="#ef4444", linewidth=1.0)
ax1.bar(down.index, down["Open"] - down["Close"], body_width, bottom=down["Close"], color="#ef4444", edgecolor="#ef4444", linewidth=0)

# 2. Draw Moving Averages
if "MA_5" in df_chart.columns:
    ax1.plot(df_chart.index, df_chart["MA_5"], color="#ffffff", linestyle="--", label="MA 5", linewidth=1.0)
if "MA_20" in df_chart.columns:
    ax1.plot(df_chart.index, df_chart["MA_20"], color="#a3a3a3", linestyle=":", label="MA 20", linewidth=1.0)

ax1.set_title(f"{ticker} Price & Moving Averages", loc="left", fontsize=11, fontweight="bold", color="#ffffff", pad=10)
ax1.legend(facecolor="#0d0d0d", edgecolor="#262626")

# 3. Draw Volume on ax2 (matching green/red candle colors)
vol_colors = ["#22c55e" if r >= 0 else "#ef4444" for r in df_chart["Return_1d"]]
ax2.bar(df_chart.index, df_chart["Volume"], color=vol_colors, width=0.8, alpha=0.8)
ax2.set_title("Volume", loc="left", fontsize=9, fontweight="bold", color="#a3a3a3", pad=5)

# Formatting Date axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.autofmt_xdate()
fig.patch.set_facecolor('#0d0d0d')

st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE + CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Feature Importance")
    
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0] for f in top_feats]
    feat_vals = [f[1] for f in top_feats]

    fig_imp, ax_imp = plt.subplots(figsize=(6, 3.2))
    ax_imp.barh(feat_names[::-1], feat_vals[::-1], color="#ffffff", edgecolor="#262626", height=0.6)
    
    # Clean spines
    for spine in ['top', 'right', 'bottom']:
        ax_imp.spines[spine].set_visible(False)
    ax_imp.spines['left'].set_color('#262626')
    ax_imp.grid(axis='x', linestyle='--', linewidth=0.5, color='#171717')
    ax_imp.tick_params(axis='both', which='both', length=0)
    ax_imp.tick_params(colors='#a3a3a3', labelsize=8)
    fig_imp.patch.set_facecolor('#0d0d0d')
    ax_imp.set_facecolor('#0d0d0d')
    
    fig_imp.tight_layout()
    st.pyplot(fig_imp)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Prediction Confidence")
    st.write("")
    st.markdown(f"""
    <div style="text-align: center; margin: 15px 0;">
        <div style="font-size: 3rem; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace;">
            {prob_up*100:.1f}%
        </div>
        <div style="color: #737373; font-size: 0.85rem; margin-top: 5px; margin-bottom: 25px;">
            Probability of price going UP tomorrow
        </div>
        <div style="background-color: #262626; border-radius: 10px; height: 10px; width: 100%; overflow: hidden;">
            <div style="background-color: {progress_color}; height: 100%; width: {prob_up*100}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:16px;">
      <span style="border: 1px solid #632020; background:#3d1616; color:#ef4444; padding:4px 12px; border-radius:4px; font-size:0.78rem; font-weight:600;">≤40% → SELL</span>
      <span style="border: 1px solid #333333; background:#1a1a1a; color:#a3a3a3; padding:4px 12px; border-radius:4px; font-size:0.78rem; font-weight:600;">40–60% → HOLD</span>
      <span style="border: 1px solid #1b4d32; background:#0f2d1e; color:#22c55e; padding:4px 12px; border-radius:4px; font-size:0.78rem; font-weight:600;">≥60% → BUY</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Model Comparison & Selection")

col_comp1, col_comp2, col_comp3 = st.columns(3)
model_names = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
cols = [col_comp1, col_comp2, col_comp3]

for name, col in zip(model_names, cols):
    acc = accuracies[name]
    is_active = (name == best_model_name)
    status_border = "border: 1px solid #22c55e;" if is_active else "border: 1px solid #262626;"
    status_label = "<div style='color: #22c55e; font-size: 0.75rem; font-weight: bold; margin-top: 5px;'>● ACTIVE (BEST ACCURACY)</div>" if is_active else "<div style='color: #737373; font-size: 0.75rem; margin-top: 5px;'>INACTIVE</div>"
    
    col.markdown(f"""
    <div class="card" style="{status_border} text-align: center; height: 100%;">
        <div style="font-size: 0.8rem; color: #a3a3a3; text-transform: uppercase; letter-spacing: 0.5px;">{name}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 10px 0 5px;">
            {acc*100:.1f}%
        </div>
        {status_label}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#525252; font-size:0.78rem; margin-top:40px; padding:20px 0; border-top:1px solid #262626;'>
⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Not financial advice. Do your own research before investing.<br>
Built with Python · scikit-learn · yfinance · Streamlit · Matplotlib
</div>
""", unsafe_allow_html=True)
