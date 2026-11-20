import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

from model import (
    train_model,
    predict_signal,
    predict_all_models,
    compute_technical_insights,
    add_features,
    FEATURE_COLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION (Zero Emojis, Sidebar Collapsed & Hidden)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI — Institutional Quantitative Predictor",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLING — Obsidian Black & White Gradient Theme (Fully Adaptive)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Root Variables */
:root {
    --bg-main: #050507;
    --bg-card: #0d0d11;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.22);
    --text-primary: #ffffff;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent-silver: #e4e4e7;
    --bullish-green: #22c55e;
    --bearish-red: #ef4444;
}

/* Eliminate Sidebar Completely */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
header button[kind="header"] {
    display: none !important;
}

/* Global App Surface */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #050507 !important;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255, 255, 255, 0.07), transparent),
        radial-gradient(circle at 100% 0%, rgba(255, 255, 255, 0.03), transparent 40%),
        linear-gradient(180deg, #09090c 0%, #030304 100%) !important;
    color: #e4e4e7 !important;
    min-height: 100vh;
    overflow-x: hidden !important;
}

/* Adaptive Container Spacing */
.block-container {
    max-width: 1400px !important;
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: clamp(0.75rem, 3vw, 2rem) !important;
    padding-right: clamp(0.75rem, 3vw, 2rem) !important;
}

/* Header & Toolbar */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer {
    visibility: hidden !important;
}

/* Adaptive Glassmorphic Cards */
.quant-card {
    background: linear-gradient(145deg, rgba(22, 22, 28, 0.75) 0%, rgba(10, 10, 14, 0.90) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: clamp(14px, 2.5vw, 24px);
    margin-bottom: 16px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: border-color 0.25s ease, transform 0.25s ease;
}
.quant-card:hover {
    border-color: rgba(255, 255, 255, 0.16);
}

.card-gradient-top {
    position: relative;
    overflow: hidden;
}
.card-gradient-top::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.45) 50%, transparent 100%);
}

/* Typography Gradient Headers */
.gradient-headline {
    font-size: clamp(1.4rem, 3.2vw, 2.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(180deg, #FFFFFF 0%, #A1A1AA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.gradient-subline {
    color: #71717a;
    font-size: clamp(0.78rem, 1.4vw, 0.92rem);
    font-weight: 400;
    margin-top: 4px;
}

/* Top Control Bar */
.control-bar {
    background: linear-gradient(180deg, rgba(24, 24, 30, 0.85) 0%, rgba(12, 12, 16, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: clamp(14px, 2vw, 22px);
    margin-bottom: 20px;
    box-shadow: 0 12px 36px -12px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

/* Metric Display Tiles */
.metric-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: clamp(10px, 1.6vw, 16px);
    text-align: left;
    transition: all 0.2s ease;
}
.metric-box:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.16);
}
.metric-box-title {
    font-size: clamp(0.65rem, 1vw, 0.72rem);
    color: #71717a;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-box-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.15rem, 2vw, 1.45rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}

/* Adaptive Responsive Grid Systems */
.hero-layout-grid {
    display: grid;
    grid-template-columns: 1.1fr 1.9fr;
    gap: 20px;
    align-items: center;
}
.hero-metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}
.scorecard-adaptive-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
}
.ensemble-adaptive-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}
.boundaries-adaptive-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

/* Media Queries for Tablet and Mobile Viewports */
@media (max-width: 1024px) {
    .scorecard-adaptive-grid {
        grid-template-columns: repeat(3, 1fr);
    }
    .boundaries-adaptive-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 868px) {
    .hero-layout-grid {
        grid-template-columns: 1fr;
    }
    .ensemble-adaptive-grid {
        grid-template-columns: 1fr;
    }
}
@media (max-width: 580px) {
    .scorecard-adaptive-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .hero-metrics-grid {
        grid-template-columns: 1fr;
    }
    .boundaries-adaptive-grid {
        grid-template-columns: 1fr;
    }
}

/* Verdict Badges (Using Clean Unicode Symbols, Zero Emojis) */
.badge-strong-buy, .badge-buy {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.16) 0%, rgba(20, 83, 45, 0.25) 100%);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.35);
    border-radius: 10px;
    padding: 8px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.1rem, 2vw, 1.45rem);
    font-weight: 800;
    letter-spacing: 0.06em;
    box-shadow: 0 0 24px rgba(34, 197, 94, 0.2);
}
.badge-strong-sell, .badge-sell {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(127, 29, 29, 0.25) 100%);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: 10px;
    padding: 8px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.1rem, 2vw, 1.45rem);
    font-weight: 800;
    letter-spacing: 0.06em;
    box-shadow: 0 0 24px rgba(239, 68, 68, 0.2);
}
.badge-hold {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(113, 113, 122, 0.18) 100%);
    color: #e4e4e7;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 10px;
    padding: 8px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1.1rem, 2vw, 1.45rem);
    font-weight: 800;
    letter-spacing: 0.06em;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.08);
}

/* Insight Rationale Pill */
.insight-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.insight-pill-bull {
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}
.insight-pill-bear {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.insight-pill-neutral {
    background: rgba(255, 255, 255, 0.08);
    color: #d4d4d8;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

/* Streamlit Inputs Overrides */
div[data-baseweb="select"] > div {
    background-color: #0d0d12 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] * {
    color: #e4e4e7 !important;
}
div[data-baseweb="input"] > div {
    background-color: #0d0d12 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
input.st-bc {
    color: #ffffff !important;
}

/* ───────────────────────────────────────────────────────────────────────── */
/* REVOLUTIONIZED SLIDER UI — Sleek Black & White Gradient */
/* ───────────────────────────────────────────────────────────────────────── */
div[data-testid="stSlider"] {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
div[data-testid="stSlider"] label {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: #a1a1aa !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
/* Track rail background */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: #18181f !important;
    height: 6px !important;
    border-radius: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}
/* Active filled portion */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #52525b 0%, #ffffff 100%) !important;
    height: 6px !important;
    border-radius: 4px !important;
}
/* Slider thumb / handle */
div[data-testid="stSlider"] div[role="slider"] {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.5) !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-testid="stSlider"] div[role="slider"]:hover {
    transform: scale(1.15) !important;
    box-shadow: 0 0 18px rgba(255, 255, 255, 0.8) !important;
}
/* Thumb value tooltip */
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: #18181f !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] {
    color: #71717a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ───────────────────────────────────────────────────────────────────────── */
/* REVOLUTIONIZED TABS UI — Luxury Segmented Pill Switcher (No Red Line) */
/* ───────────────────────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    display: inline-flex !important;
    background: #09090d !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    margin-bottom: 16px !important;
    width: auto !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}
/* Eliminate default Streamlit red underline slider */
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    display: none !important;
}
div[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 7px 16px !important;
    color: #71717a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
div[data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05) !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(180deg, #ffffff 0%, #e4e4e7 100%) !important;
    color: #050507 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(255, 255, 255, 0.18) !important;
}

/* Primary Action Button — High Contrast Monochromatic Glow */
div.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #D4D4D8 100%) !important;
    color: #050507 !important;
    font-weight: 800 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #FFFFFF 0%, #F4F4F5 100%) !important;
    box-shadow: 0 0 28px rgba(255, 255, 255, 0.4) !important;
    transform: translateY(-1px);
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(1px);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #050507;
}
::-webkit-scrollbar-thumb {
    background: #27272a;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3f3f46;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP 50+ US STOCKS & TOP 50+ INDIAN NIFTY STOCKS (With Clean Descriptions)
# ─────────────────────────────────────────────────────────────────────────────
POPULAR_STOCKS_US = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corp."),
    ("NVDA", "NVIDIA Corp."),
    ("GOOGL", "Alphabet Inc. (Class A)"),
    ("AMZN", "Amazon.com Inc."),
    ("META", "Meta Platforms Inc."),
    ("TSLA", "Tesla Inc."),
    ("BRK-B", "Berkshire Hathaway"),
    ("LLY", "Eli Lilly & Co."),
    ("AVGO", "Broadcom Inc."),
    ("JPM", "JPMorgan Chase & Co."),
    ("V", "Visa Inc."),
    ("UNH", "UnitedHealth Group"),
    ("XOM", "Exxon Mobil Corp."),
    ("MA", "Mastercard Inc."),
    ("JNJ", "Johnson & Johnson"),
    ("PG", "Procter & Gamble"),
    ("HD", "Home Depot Inc."),
    ("COST", "Costco Wholesale"),
    ("ABBV", "AbbVie Inc."),
    ("MRK", "Merck & Co."),
    ("NFLX", "Netflix Inc."),
    ("AMD", "Advanced Micro Devices"),
    ("CRM", "Salesforce Inc."),
    ("BAC", "Bank of America"),
    ("CVX", "Chevron Corp."),
    ("KO", "Coca-Cola Co."),
    ("PEP", "PepsiCo Inc."),
    ("WMT", "Walmart Inc."),
    ("QCOM", "Qualcomm Inc."),
    ("TMO", "Thermo Fisher Scientific"),
    ("LIN", "Linde plc"),
    ("CSCO", "Cisco Systems"),
    ("ACN", "Accenture plc"),
    ("MCD", "McDonald's Corp."),
    ("ABT", "Abbott Laboratories"),
    ("ORCL", "Oracle Corp."),
    ("IBM", "IBM Corp."),
    ("GE", "GE Aerospace"),
    ("DIS", "Walt Disney Co."),
    ("INTU", "Intuit Inc."),
    ("CAT", "Caterpillar Inc."),
    ("VZ", "Verizon Communications"),
    ("TXN", "Texas Instruments"),
    ("PM", "Philip Morris International"),
    ("AMAT", "Applied Materials"),
    ("NOW", "ServiceNow Inc."),
    ("BKNG", "Booking Holdings"),
    ("ISRG", "Intuitive Surgical"),
    ("CMG", "Chipotle Mexican Grill"),
    ("UBER", "Uber Technologies"),
    ("SPY", "SPDR S&P 500 ETF"),
    ("QQQ", "Invesco QQQ Trust ETF"),
]

POPULAR_STOCKS_IN = [
    ("RELIANCE.NS", "Reliance Industries"),
    ("TCS.NS", "Tata Consultancy Services"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("BHARTIARTL.NS", "Bharti Airtel"),
    ("ICICIBANK.NS", "ICICI Bank"),
    ("INFY.NS", "Infosys"),
    ("SBIN.NS", "State Bank of India"),
    ("ITC.NS", "ITC Ltd"),
    ("HINDUNILVR.NS", "Hindustan Unilever"),
    ("LT.NS", "Larsen & Toubro"),
    ("BAJFINANCE.NS", "Bajaj Finance"),
    ("HCLTECH.NS", "HCL Technologies"),
    ("MARUTI.NS", "Maruti Suzuki"),
    ("SUNPHARMA.NS", "Sun Pharmaceutical"),
    ("TATAMOTORS.NS", "Tata Motors"),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank"),
    ("AXISBANK.NS", "Axis Bank"),
    ("ONGC.NS", "Oil & Natural Gas Corp"),
    ("NTPC.NS", "NTPC Ltd"),
    ("TITAN.NS", "Titan Company"),
    ("POWERGRID.NS", "Power Grid Corp"),
    ("ADANIENT.NS", "Adani Enterprises"),
    ("ADANIPORTS.NS", "Adani Ports & SEZ"),
    ("COALINDIA.NS", "Coal India"),
    ("BAJAJFINSV.NS", "Bajaj Finserv"),
    ("TATASTEEL.NS", "Tata Steel"),
    ("M&M.NS", "Mahindra & Mahindra"),
    ("ASIANPAINT.NS", "Asian Paints"),
    ("ULTRACEMCO.NS", "UltraTech Cement"),
    ("WIPRO.NS", "Wipro Ltd"),
    ("NESTLEIND.NS", "Nestle India"),
    ("JSWSTEEL.NS", "JSW Steel"),
    ("GRASIM.NS", "Grasim Industries"),
    ("TECHM.NS", "Tech Mahindra"),
    ("DRREDDY.NS", "Dr. Reddy's Laboratories"),
    ("CIPLA.NS", "Cipla Ltd"),
    ("HINDALCO.NS", "Hindalco Industries"),
    ("BPCL.NS", "Bharat Petroleum"),
    ("SBILIFE.NS", "SBI Life Insurance"),
    ("BRITANNIA.NS", "Britannia Industries"),
    ("TATACONSUM.NS", "Tata Consumer Products"),
    ("EICHERMOT.NS", "Eicher Motors"),
    ("APOLLOHOSP.NS", "Apollo Hospitals"),
    ("DIVISLAB.NS", "Divi's Laboratories"),
    ("HEROMOTOCO.NS", "Hero MotoCorp"),
    ("INDUSINDBK.NS", "IndusInd Bank"),
    ("BAJAJ-AUTO.NS", "Bajaj Auto"),
    ("TRENT.NS", "Trent Ltd"),
    ("BEL.NS", "Bharat Electronics"),
    ("SHREECEM.NS", "Shree Cement"),
    ("ZOMATO.NS", "Zomato Ltd"),
    ("JIOFIN.NS", "Jio Financial Services"),
]

STOCK_DICT_US = dict(POPULAR_STOCKS_US)
STOCK_DICT_IN = dict(POPULAR_STOCKS_IN)
ALL_STOCK_NAMES = {**STOCK_DICT_US, **STOCK_DICT_IN}

CATEGORY_MAP = {
    "US Equities (S&P 500 / Nasdaq Top 50+)": [s[0] for s in POPULAR_STOCKS_US],
    "Indian Equities (NSE Nifty Top 50+)": [s[0] for s in POPULAR_STOCKS_IN],
    "Custom Symbol / Ticker": []
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data using yfinance with cleaned single-level column headers."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df is not None and not df.empty:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def fmt_price(v, ticker=""):
    """Format numeric price with dynamic currency prefix ($ or ₹)."""
    if v is None or np.isnan(v):
        return "—"
    symbol = "₹" if ticker.endswith(".NS") else "$"
    return f"{symbol}{v:,.2f}"

def render_html(html_str: str, *args, **kwargs):
    """Safely render HTML without triggering CommonMark indented code block parsing."""
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)

# Initialize session state for ticker selection
if "active_ticker" not in st.session_state:
    st.session_state["active_ticker"] = "AAPL"
if "trigger_analysis" not in st.session_state:
    st.session_state["trigger_analysis"] = True

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION HEADER (Zero Emojis, No Sidebar)
# ─────────────────────────────────────────────────────────────────────────────
header_col1, header_col2 = st.columns([2.8, 1.2])

with header_col1:
    render_html("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
        <div style="background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%); width:30px; height:30px; border-radius:7px; display:flex; align-items:center; justify-content:center; color:#000000; font-weight:900; font-size:1.05rem; box-shadow:0 0 16px rgba(255,255,255,0.25);">
            ▲
        </div>
        <div>
            <span class="gradient-headline">StockSense AI</span>
            <span style="font-size:0.72rem; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#d4d4d8; padding:2px 8px; border-radius:12px; font-weight:700; margin-left:8px; vertical-align:middle; font-family:'JetBrains Mono',monospace;">v2.5 QUANT PRO</span>
        </div>
    </div>
    <div class="gradient-subline">Institutional machine learning price direction forecasting, multi-model consensus, and quantitative factor analytics.</div>
    """)

with header_col2:
    render_html("""
    <div style="text-align:right; padding-top:6px;">
        <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); border-radius:20px; padding:5px 14px; font-size:0.76rem; font-family:'JetBrains Mono',monospace; color:#a1a1aa;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#22c55e; box-shadow:0 0 8px #22c55e;"></span>
            SYSTEM OPERATIONAL
        </span>
    </div>
    """)

st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# TOP CONTROL COMMAND BAR (Asset & Parameter Configuration — Quick Assets Removed)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="control-bar">', unsafe_allow_html=True)

curr_active = st.session_state.get("active_ticker", "AAPL")
if curr_active.endswith(".NS") or curr_active in CATEGORY_MAP["Indian Equities (NSE Nifty Top 50+)"]:
    cat_default_idx = 1
elif curr_active in CATEGORY_MAP["US Equities (S&P 500 / Nasdaq Top 50+)"]:
    cat_default_idx = 0
else:
    cat_default_idx = 2

# Main Control Inputs Grid
c1, c2, c3, c4, c5 = st.columns([1.5, 1.8, 1.1, 1.2, 1.3])

with c1:
    market_category = st.selectbox(
        "Market Category",
        list(CATEGORY_MAP.keys()),
        index=cat_default_idx,
        label_visibility="visible"
    )

with c2:
    if market_category == "Custom Symbol / Ticker":
        custom_input = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("active_ticker", "AAPL"),
            placeholder="e.g. AMD, BTC-USD, SPY",
            help="Enter any valid ticker symbol from Yahoo Finance"
        ).upper().strip()
        selected_ticker = custom_input if custom_input else "AAPL"
    else:
        stock_list = CATEGORY_MAP[market_category]
        default_stock_idx = stock_list.index(curr_active) if curr_active in stock_list else 0
        
        selected_ticker = st.selectbox(
            "Select Stock Asset",
            stock_list,
            index=default_stock_idx,
            format_func=lambda x: f"{x} — {ALL_STOCK_NAMES.get(x, '')}" if x in ALL_STOCK_NAMES else x
        )

# Update session active ticker
st.session_state["active_ticker"] = selected_ticker

with c3:
    period = st.selectbox(
        "Training History",
        ["1y", "2y", "3y", "5y"],
        index=1,
        help="Historical lookback dataset used to fit and calibrate machine learning models"
    )

with c4:
    chart_lookback = st.selectbox(
        "Chart Horizon",
        [30, 60, 90, 120, 180],
        index=2,
        format_func=lambda x: f"{x} Sessions",
        help="Number of trading sessions displayed in charts"
    )

with c5:
    st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
    analyze_btn = st.button("RUN PREDICTION", key="btn_run_main", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Determine execution state
ticker_changed = ("last_analyzed_ticker" in st.session_state and st.session_state["last_analyzed_ticker"] != selected_ticker)
should_run = analyze_btn or st.session_state.get("trigger_analysis", False) or ticker_changed

if not should_run:
    # ─── Landing Showcase State (Zero Emojis) ───
    render_html("""
    <div class="quant-card card-gradient-top" style="text-align:center; padding: 40px 20px;">
        <div style="font-size: 2.2rem; font-weight:900; margin-bottom: 10px; color:#ffffff; font-family:'JetBrains Mono',monospace;">▲ // STOCKSENSE</div>
        <div class="gradient-headline" style="font-size: 1.8rem; margin-bottom: 8px;">Institutional Machine Learning Directional Predictor</div>
        <p style="color:#a1a1aa; max-width:680px; margin:0 auto 28px; font-size:0.92rem; line-height:1.6;">
            Select any stock ticker from over 100+ US and Indian benchmark assets above and click <strong style="color:#ffffff;">RUN PREDICTION</strong>. 
            The engine trains Random Forest, Gradient Boosting, and Logistic Regression on 7 core quantitative momentum factors.
        </p>
    </div>
    """)
    
    st.write("")
    c_a, c_b, c_c = st.columns(3)
    features_info = [
        ("[01] ENSEMBLE TRIANGULATION", "Random Forest, Gradient Boosting, and Logistic Regression benchmarked simultaneously with automated consensus."),
        ("[02] 7 QUANTITATIVE FACTORS", "RSI(14), dual moving-average crossovers, 1D/5D momentum velocity, and volume surge expansion."),
        ("[03] INSTITUTIONAL DELIVERY", "Actionable verdict, directional confidence probabilities, volatility stop boundaries, and executive theses.")
    ]
    for col, (title, desc) in zip([c_a, c_b, c_c], features_info):
        with col:
            render_html(f"""
            <div class="quant-card" style="height:100%;">
                <div style="color:#ffffff; font-weight:700; font-size:0.92rem; font-family:'JetBrains Mono',monospace; margin-bottom:8px;">{title}</div>
                <div style="color:#71717a; font-size:0.85rem; line-height:1.5;">{desc}</div>
            </div>
            """)
    st.stop()

# Record last analyzed ticker & reset trigger flag
st.session_state["last_analyzed_ticker"] = selected_ticker
st.session_state["trigger_analysis"] = False

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING & PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Querying quantitative feeds for {selected_ticker}..."):
    df_raw = fetch_data(selected_ticker, period)

if df_raw is None or df_raw.empty or len(df_raw) < 30:
    st.error(f"[ERROR] Insufficient market data returned for '{selected_ticker}'. Please verify the ticker symbol or select a longer training period.")
    st.stop()

# ─── Model Training & Evaluation ───
progress_placeholder = st.empty()
with progress_placeholder.container():
    pbar = st.progress(25, text=f"Calibrating quantitative models on {selected_ticker} ({period} history)...")
    time.sleep(0.05)

try:
    models, accuracies, scaler, importances = train_model(df_raw)
    pbar.progress(70, text="Generating ensemble predictions and computing factor indicators...")
    
    model_predictions, consensus_signal, avg_prob_up, next_price = predict_all_models(models, scaler, df_raw)
    tech_insights = compute_technical_insights(df_raw)
    
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    best_model_call = model_predictions.get(best_model_name, {}).get("signal", "HOLD")
    best_model_prob = model_predictions.get(best_model_name, {}).get("prob_up", 0.5)
    
    pbar.progress(100, text="Execution complete.")
    time.sleep(0.1)
    progress_placeholder.empty()

except Exception as ex:
    progress_placeholder.empty()
    st.error(f"[ERROR] Execution error during model training: {ex}")
    st.stop()

# Key Price Metrics
last_close = tech_insights["last_close"]
prev_close = tech_insights["prev_close"]
day_chg = tech_insights["day_chg"]
chg_color = "#22c55e" if day_chg >= 0 else "#ef4444"
chg_arrow = "▲" if day_chg >= 0 else "▼"
chg_sign = "+" if day_chg >= 0 else ""

# Primary Signal Classification
display_signal = consensus_signal
badge_class = {
    "STRONG BUY": "badge-strong-buy",
    "BUY": "badge-buy",
    "HOLD": "badge-hold",
    "SELL": "badge-sell",
    "STRONG SELL": "badge-strong-sell"
}.get(display_signal, "badge-hold")

direction_text = "▲ UPWARD CONTINUATION" if avg_prob_up >= 0.55 else ("▼ DOWNWARD CONSOLIDATION" if avg_prob_up <= 0.45 else "■ NEUTRAL CHOP")
prob_pct = avg_prob_up * 100
prob_bar_color = "#22c55e" if avg_prob_up >= 0.55 else ("#ef4444" if avg_prob_up <= 0.45 else "#e4e4e7")

company_full_name = ALL_STOCK_NAMES.get(selected_ticker, "")
ticker_label_display = f"{selected_ticker} — {company_full_name}" if company_full_name else selected_ticker

# ─────────────────────────────────────────────────────────────────────────────
# 1. HERO EXECUTIVE INTELLIGENCE CARD (Actionable Verdict & Core Metrics)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="quant-card card-gradient-top">', unsafe_allow_html=True)

render_html(f"""
<div class="hero-layout-grid">
    <div>
        <div style="font-size:0.74rem; color:#71717a; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:8px; font-family:'JetBrains Mono',monospace;">
            // TOMORROW'S PREDICTED VERDICT • <span style="color:#ffffff;">{ticker_label_display}</span>
        </div>
        <div style="margin-bottom:14px;">
            <span class="{badge_class}">
                <span>{chg_arrow if 'BUY' in display_signal else ('▼' if 'SELL' in display_signal else '■')}</span>
                <span>{display_signal}</span>
            </span>
        </div>
        <div style="font-size:0.86rem; color:#a1a1aa; margin-bottom:12px;">
            Expected Trajectory: <strong style="color:#ffffff;">{direction_text}</strong>
        </div>
        <div style="margin-top:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.76rem; font-family:'JetBrains Mono',monospace; color:#a1a1aa; margin-bottom:5px;">
                <span>PROBABILITY (UP)</span>
                <span style="color:#ffffff; font-weight:700;">{prob_pct:.1f}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 8px; width: 100%; overflow: hidden;">
                <div style="background: {prob_bar_color}; width: {prob_pct}%; height: 100%; border-radius: 6px; transition: width 0.5s ease;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#71717a; margin-top:4px; font-family:'JetBrains Mono',monospace;">
                <span>0% BEARISH</span>
                <span>50% NEUTRAL</span>
                <span>100% BULLISH</span>
            </div>
        </div>
    </div>
    <div>
        <div class="hero-metrics-grid">
            <div class="metric-box">
                <div class="metric-box-title">LAST SPOT CLOSE</div>
                <div class="metric-box-val">{fmt_price(last_close, selected_ticker)}</div>
                <div style="font-size:0.76rem; color:{chg_color}; font-weight:600; margin-top:4px; font-family:'JetBrains Mono',monospace;">
                    {chg_arrow} {chg_sign}{day_chg:.2f}% (1D)
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">MOMENTUM TARGET CLOSE</div>
                <div class="metric-box-val">{fmt_price(next_price, selected_ticker)}</div>
                <div style="font-size:0.76rem; color:{'#22c55e' if next_price >= last_close else '#ef4444'}; font-weight:600; margin-top:4px; font-family:'JetBrains Mono',monospace;">
                    {'+' if next_price >= last_close else ''}{((next_price-last_close)/last_close)*100:.2f}% Expected Velocity
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">TOP MODEL ACCURACY</div>
                <div class="metric-box-val">{best_accuracy*100:.1f}%</div>
                <div style="font-size:0.76rem; color:#a1a1aa; margin-top:4px;">
                    Champion: <span style="color:#ffffff; font-weight:600;">{best_model_name}</span>
                </div>
            </div>
            <div class="metric-box">
                <div class="metric-box-title">DAILY VOLATILITY (ATR 14)</div>
                <div class="metric-box-val">{fmt_price(tech_insights['atr'], selected_ticker)}</div>
                <div style="font-size:0.76rem; color:#a1a1aa; margin-top:4px;">
                    Expected Swing: <span style="color:#ffffff; font-weight:600;">±{(tech_insights['atr']/last_close)*100:.2f}%</span>
                </div>
            </div>
        </div>
    </div>
</div>
""")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. AI QUANTITATIVE SYNTHESIS & INSIGHTS RATIONALE (Zero Emojis)
# ─────────────────────────────────────────────────────────────────────────────
rsi_val = tech_insights["rsi"]
vol_rat = tech_insights["vol_ratio"]
ma_stat = tech_insights["ma_posture"]
ma_dist = tech_insights["price_to_ma20"]

if "BUY" in display_signal:
    thesis_title = "Bullish Structural Expansion"
    thesis_summary = (
        f"The ensemble indicates favorable upward momentum for {selected_ticker} tomorrow with an average confidence of {prob_pct:.1f}%. "
        f"Price action is consolidating above the 20-day moving average ({ma_dist:+.2f}% distance) with an RSI of {rsi_val:.1f} ({tech_insights['rsi_status']}). "
        f"Volume participation sits at {vol_rat:.2f}x its 5-day baseline, reflecting sustained buyer support."
    )
    catalyst_text = f"Healthy momentum accumulation with {best_model_name} registering {best_model_prob*100:.1f}% upside probability."
    invalidation_level = f"{fmt_price(tech_insights['support_20d'], selected_ticker)} (20-day dynamic support floor)"
    action_note = "Favorable risk-to-reward ratio for long positioning with trailing stop below recent swing low."
    thesis_badge = '<span class="insight-pill insight-pill-bull">[POSITIVE REGIME]</span>'
elif "SELL" in display_signal:
    thesis_title = "Bearish Structural Exhaustion"
    thesis_summary = (
        f"Quantitative models forecast downward or defensive pressure for {selected_ticker} over the next session. "
        f"Current price is positioned {ma_dist:+.2f}% relative to the 20-day SMA under {ma_stat}. "
        f"RSI reads {rsi_val:.1f} ({tech_insights['rsi_status']}), suggesting vulnerable short-term structure."
    )
    catalyst_text = f"Downside probability is elevated at {(100-prob_pct):.1f}%, with primary model consensus pointing toward capital preservation."
    invalidation_level = f"{fmt_price(tech_insights['resistance_20d'], selected_ticker)} (20-day dynamic resistance ceiling)"
    action_note = "Elevated risk environment. Consider locking gains or utilizing protective stops."
    thesis_badge = '<span class="insight-pill insight-pill-bear">[DEFENSIVE REGIME]</span>'
else:
    thesis_title = "Neutral Compression / Range-Bound"
    thesis_summary = (
        f"{selected_ticker} is displaying balanced two-way price discovery without decisive directional edge ({prob_pct:.1f}% probability). "
        f"RSI stands at {rsi_val:.1f} ({tech_insights['rsi_status']}) with price hovering near the 20-DMA ({ma_dist:+.2f}%). "
        f"Volume ratio of {vol_rat:.2f}x indicates average equilibrium without breakout conviction."
    )
    catalyst_text = "Conflicting model weights indicate consolidation between immediate support and resistance boundaries."
    invalidation_level = f"Band: {fmt_price(tech_insights['support_20d'], selected_ticker)} – {fmt_price(tech_insights['resistance_20d'], selected_ticker)}"
    action_note = "Awaiting confirmed directional breakout before initiating high-conviction positions."
    thesis_badge = '<span class="insight-pill insight-pill-neutral">[RANGEBOUND]</span>'

render_html(f"""
<div class="quant-card" style="border-left: 3px solid #ffffff;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-weight:700; color:#ffffff; font-size:1.0rem; letter-spacing:0.04em; font-family:'JetBrains Mono',monospace;">// QUANTITATIVE SYNTHESIS & INSIGHTS</span>
        </div>
        <div>{thesis_badge}</div>
    </div>
    <div style="color:#e4e4e7; font-size:0.92rem; line-height:1.6; margin-bottom:16px;">
        <strong style="color:#ffffff;">{thesis_title}:</strong> {thesis_summary}
    </div>
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);">
        <div style="font-size:0.83rem;">
            <span style="color:#71717a; font-weight:700; text-transform:uppercase; font-size:0.70rem; letter-spacing:0.06em; font-family:'JetBrains Mono',monospace;">[PRIMARY CATALYST]</span><br>
            <span style="color:#d4d4d8;">{catalyst_text}</span>
        </div>
        <div style="font-size:0.83rem;">
            <span style="color:#71717a; font-weight:700; text-transform:uppercase; font-size:0.70rem; letter-spacing:0.06em; font-family:'JetBrains Mono',monospace;">[INVALIDATION LEVEL]</span><br>
            <span style="color:#d4d4d8; font-family:'JetBrains Mono',monospace;">{invalidation_level}</span>
        </div>
        <div style="font-size:0.83rem;">
            <span style="color:#71717a; font-weight:700; text-transform:uppercase; font-size:0.70rem; letter-spacing:0.06em; font-family:'JetBrains Mono',monospace;">[TACTICAL PERSPECTIVE]</span><br>
            <span style="color:#d4d4d8;">{action_note}</span>
        </div>
    </div>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERACTIVE CHART STUDIO (Zero Emojis, Enhanced Segmented Switcher)
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<div class="quant-card" style="margin-bottom:12px;">
    <div style="font-weight:700; color:#ffffff; font-size:1.02rem; font-family:'JetBrains Mono',monospace;">// TECHNICAL CHART STUDIO</div>
    <div style="color:#71717a; font-size:0.8rem; margin-top:2px;">Interactive price action, trend envelopes, volume profile, and oscillator dynamics</div>
</div>
""")

df_feat = add_features(df_raw).dropna()
df_chart = df_feat.tail(chart_lookback).copy()

tab_plotly, tab_rsi, tab_static = st.tabs([
    "Interactive Candlestick + Volume",
    "Momentum & RSI Oscillator",
    "High-Precision Vector Plot"
])

with tab_plotly:
    fig_p = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig_p.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart["Open"],
            high=df_chart["High"],
            low=df_chart["Low"],
            close=df_chart["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#ef4444",
            decreasing_fillcolor="#ef4444",
        ),
        row=1, col=1
    )

    # Moving Averages
    if "MA_5" in df_chart.columns:
        fig_p.add_trace(
            go.Scatter(
                x=df_chart.index, y=df_chart["MA_5"],
                name="MA 5 (Fast)",
                line=dict(color="#ffffff", width=1.5),
                hoverinfo="y+name"
            ),
            row=1, col=1
        )
    if "MA_20" in df_chart.columns:
        fig_p.add_trace(
            go.Scatter(
                x=df_chart.index, y=df_chart["MA_20"],
                name="MA 20 (Medium)",
                line=dict(color="#a1a1aa", width=1.2, dash="dash"),
                hoverinfo="y+name"
            ),
            row=1, col=1
        )

    # Bollinger Bands
    rolling_20_std = df_chart["Close"].rolling(20).std()
    bb_hi = df_chart["MA_20"] + (2 * rolling_20_std)
    bb_lo = df_chart["MA_20"] - (2 * rolling_20_std)
    fig_p.add_trace(
        go.Scatter(
            x=df_chart.index, y=bb_hi,
            name="Upper Band",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    fig_p.add_trace(
        go.Scatter(
            x=df_chart.index, y=bb_lo,
            name="Lower Band",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            fill="tonexty",
            fillcolor="rgba(255,255,255,0.03)",
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )

    # Volume Bars
    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(df_chart["Close"], df_chart["Open"])
    ]
    fig_p.add_trace(
        go.Bar(
            x=df_chart.index,
            y=df_chart["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.75
        ),
        row=2, col=1
    )

    # Volume MA20
    vol_ma = df_chart["Volume"].rolling(20).mean()
    fig_p.add_trace(
        go.Scatter(
            x=df_chart.index, y=vol_ma,
            name="Vol MA20",
            line=dict(color="#ffffff", width=1.0),
            hoverinfo="y+name"
        ),
        row=2, col=1
    )

    fig_p.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090d",
        plot_bgcolor="#09090d",
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#a1a1aa"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="#18181f",
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor="#18181f",
            showgrid=True,
            zeroline=False,
            side="right"
        ),
        xaxis2=dict(
            gridcolor="#18181f",
            showgrid=True,
            zeroline=False,
        ),
        yaxis2=dict(
            gridcolor="#18181f",
            showgrid=True,
            zeroline=False,
            side="right"
        ),
        hovermode="x unified"
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab_rsi:
    fig_rsi = go.Figure()

    fig_rsi.add_trace(
        go.Scatter(
            x=df_chart.index,
            y=df_chart["RSI"],
            name="RSI (14)",
            line=dict(color="#ffffff", width=1.8),
        )
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)", annotation_position="top right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold (30)", annotation_position="bottom right")
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="#52525b")

    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(34, 197, 94, 0.08)", line_width=0)

    fig_rsi.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090d",
        plot_bgcolor="#09090d",
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
        yaxis=dict(range=[10, 90], gridcolor="#18181f", side="right"),
        xaxis=dict(gridcolor="#18181f"),
        hovermode="x"
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab_static:
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(12, 6.2)
    )
    fig.subplots_adjust(hspace=0.06)

    for ax in [ax1, ax2]:
        ax.set_facecolor('#09090d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#27272a')
        ax.spines['bottom'].set_color('#27272a')
        ax.tick_params(colors='#a1a1aa', labelsize=8.5)
        ax.grid(True, color='#18181f', linestyle=':', linewidth=0.5)

    body_width = 0.6
    up_m = df_chart[df_chart["Close"] >= df_chart["Open"]]
    ax1.vlines(up_m.index, up_m["Low"], up_m["High"], color="#22c55e", linewidth=1.0)
    ax1.bar(up_m.index, up_m["Close"] - up_m["Open"], body_width, bottom=up_m["Open"], color="#22c55e", edgecolor="#22c55e", linewidth=0)

    down_m = df_chart[df_chart["Close"] < df_chart["Open"]]
    ax1.vlines(down_m.index, down_m["Low"], down_m["High"], color="#ef4444", linewidth=1.0)
    ax1.bar(down_m.index, down_m["Open"] - down_m["Close"], body_width, bottom=down_m["Close"], color="#ef4444", edgecolor="#ef4444", linewidth=0)

    if "MA_5" in df_chart.columns:
        ax1.plot(df_chart.index, df_chart["MA_5"], color="#ffffff", linestyle="-", label="MA 5", linewidth=1.2)
    if "MA_20" in df_chart.columns:
        ax1.plot(df_chart.index, df_chart["MA_20"], color="#a1a1aa", linestyle="--", label="MA 20", linewidth=1.0)

    ax1.set_title(f"{selected_ticker} — High Precision Candlestick Vector", loc="left", fontsize=10, fontweight="bold", color="#ffffff", pad=8)
    ax1.legend(facecolor="#09090d", edgecolor="#27272a", fontsize=8)

    v_cols = ["#22c55e" if r >= 0 else "#ef4444" for r in df_chart["Return_1d"]]
    ax2.bar(df_chart.index, df_chart["Volume"], color=v_cols, width=0.8, alpha=0.8)
    ax2.set_title("Trading Volume", loc="left", fontsize=8.5, fontweight="bold", color="#a1a1aa", pad=4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    fig.patch.set_facecolor('#09090d')

    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 4. QUANTITATIVE INDICATOR SCORECARD (Adaptive Grid, Zero Emojis)
# ─────────────────────────────────────────────────────────────────────────────
rsi_badge_color = "#22c55e" if rsi_val >= 55 else ("#ef4444" if rsi_val <= 45 else "#e4e4e7")
ma_spr = tech_insights["ma_spread"]
ma_col = "#22c55e" if ma_spr >= 0 else "#ef4444"
p_ma20 = tech_insights["price_to_ma20"]
p_col = "#22c55e" if p_ma20 >= 0 else "#ef4444"
v_rat = tech_insights["vol_ratio"]
v_col = "#22c55e" if v_rat >= 1.1 else ("#ef4444" if v_rat <= 0.8 else "#e4e4e7")
r_5d = tech_insights["ret_5d"]
r_col = "#22c55e" if r_5d >= 0 else "#ef4444"
pct_hi = tech_insights["pct_from_high"]

scorecard_html = f"""
<div class="quant-card">
    <div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:14px; font-family:'JetBrains Mono',monospace;">
        // QUANTITATIVE FACTOR SCORECARD
    </div>
    <div class="scorecard-adaptive-grid">
        <div class="metric-box">
            <div class="metric-box-title">RSI (14)</div>
            <div class="metric-box-val">{rsi_val:.1f}</div>
            <div style="font-size:0.7rem; color:{rsi_badge_color}; font-weight:700; margin-top:4px;">
                {tech_insights['rsi_status'].split('(')[0]}
            </div>
        </div>
        <div class="metric-box">
            <div class="metric-box-title">MA5 vs MA20</div>
            <div class="metric-box-val" style="color:{ma_col};">{ma_spr:+.2f}%</div>
            <div style="font-size:0.7rem; color:#a1a1aa; font-weight:700; margin-top:4px;">
                {'Golden Posture' if ma_spr >= 0 else 'Death Posture'}
            </div>
        </div>
        <div class="metric-box">
            <div class="metric-box-title">Price vs 20-DMA</div>
            <div class="metric-box-val" style="color:{p_col};">{p_ma20:+.2f}%</div>
            <div style="font-size:0.7rem; color:#a1a1aa; font-weight:700; margin-top:4px;">
                {'Above 20-DMA' if p_ma20 >= 0 else 'Below 20-DMA'}
            </div>
        </div>
        <div class="metric-box">
            <div class="metric-box-title">Volume Surge</div>
            <div class="metric-box-val">{v_rat:.2f}x</div>
            <div style="font-size:0.7rem; color:{v_col}; font-weight:700; margin-top:4px;">
                {'Elevated' if v_rat >= 1.1 else ('Quiet' if v_rat <= 0.8 else 'Baseline')}
            </div>
        </div>
        <div class="metric-box">
            <div class="metric-box-title">5-Day Velocity</div>
            <div class="metric-box-val" style="color:{r_col};">{r_5d:+.2f}%</div>
            <div style="font-size:0.7rem; color:#a1a1aa; font-weight:700; margin-top:4px;">
                Weekly Momentum
            </div>
        </div>
        <div class="metric-box">
            <div class="metric-box-title">Cycle Peak Off</div>
            <div class="metric-box-val">{pct_hi:.1f}%</div>
            <div style="font-size:0.7rem; color:#a1a1aa; font-weight:700; margin-top:4px;">
                Peak: {fmt_price(tech_insights['high_52w'], selected_ticker)}
            </div>
        </div>
    </div>
</div>
"""
render_html(scorecard_html)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL ENSEMBLE BENCHMARK & CONSENSUS ARENA
# ─────────────────────────────────────────────────────────────────────────────
model_keys = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
cards_html = []

for m_name in model_keys:
    m_acc = accuracies.get(m_name, 0.0)
    m_data = model_predictions.get(m_name, {"signal": "HOLD", "prob_up": 0.5})
    m_sig = m_data["signal"]
    m_p_up = m_data["prob_up"] * 100
    is_top = (m_name == best_model_name)
    
    border_style = "border: 1px solid #ffffff; box-shadow: 0 0 16px rgba(255,255,255,0.12);" if is_top else "border: 1px solid rgba(255,255,255,0.08);"
    top_badge = '<span style="background:rgba(255,255,255,0.12); color:#ffffff; border:1px solid rgba(255,255,255,0.25); border-radius:4px; font-size:0.65rem; padding:2px 6px; font-weight:700; font-family:\'JetBrains Mono\',monospace;">[CHAMPION MODEL]</span>' if is_top else '<span style="color:#71717a; font-size:0.65rem; font-weight:600; font-family:\'JetBrains Mono\',monospace;">[BENCHMARK]</span>'
    
    sig_color = "#22c55e" if m_sig == "BUY" else ("#ef4444" if m_sig == "SELL" else "#e4e4e7")
    sig_glyph = "▲" if m_sig == "BUY" else ("▼" if m_sig == "SELL" else "■")
    
    card_str = f"""<div style="{border_style} background: rgba(255,255,255,0.02); border-radius: 12px; padding: 18px 20px; text-align: left; height: 100%;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<span style="font-size:0.82rem; font-weight:700; color:#ffffff;">{m_name}</span>
{top_badge}
</div>
<div style="font-size:1.8rem; font-family:'JetBrains Mono',monospace; font-weight:800; color:{sig_color}; margin: 4px 0;">
{sig_glyph} {m_sig}
</div>
<div style="font-size:0.78rem; color:#a1a1aa; margin-bottom:12px;">
Validation Accuracy: <strong style="color:#ffffff;">{m_acc*100:.1f}%</strong>
</div>
<div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 6px; width: 100%; overflow: hidden; margin-bottom:6px;">
<div style="background: {sig_color}; width: {m_p_up}%; height: 100%;"></div>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#71717a; font-family:'JetBrains Mono',monospace;">
<span>Up: {m_p_up:.1f}%</span>
<span>Down: {(100-m_p_up):.1f}%</span>
</div>
</div>"""
    cards_html.append(card_str)

arena_html = f"""
<div class="quant-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div>
            <div style="font-weight:700; color:#ffffff; font-size:1.02rem; font-family:'JetBrains Mono',monospace;">// ENSEMBLE BENCHMARK ARENA</div>
            <div style="color:#71717a; font-size:0.8rem; margin-top:2px;">Independent directional predictions and test set accuracy across 3 algorithmic engines</div>
        </div>
    </div>
    <div class="ensemble-adaptive-grid">
        {"".join(cards_html)}
    </div>
</div>
"""
render_html(arena_html)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE & FACTOR GLOSSARY
# ─────────────────────────────────────────────────────────────────────────────
fi_col1, fi_col2 = st.columns([1.2, 1.0])

with fi_col1:
    render_html("""
    <div class="quant-card" style="margin-bottom:10px;">
        <div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:4px; font-family:'JetBrains Mono',monospace;">
            // QUANTITATIVE FEATURE IMPORTANCE
        </div>
        <div style="color:#71717a; font-size:0.8rem;">
            Gini impurity feature contributions derived from the Random Forest ensemble
        </div>
    </div>
    """)

    sorted_imps = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feat_labels = [k.replace('_', ' ') for k, v in sorted_imps]
    feat_scores = [v * 100 for k, v in sorted_imps]

    fig_f = go.Figure()
    fig_f.add_trace(
        go.Bar(
            x=feat_scores[::-1],
            y=feat_labels[::-1],
            orientation="h",
            marker=dict(
                color=feat_scores[::-1],
                colorscale=[[0, "#27272a"], [0.5, "#a1a1aa"], [1.0, "#ffffff"]],
                line=dict(color="#18181f", width=1)
            ),
            text=[f"{s:.1f}%" for s in feat_scores[::-1]],
            textposition="auto",
            textfont=dict(color="#000000", size=10, family="JetBrains Mono"),
        )
    )
    fig_f.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=10, b=10),
        height=280,
        xaxis=dict(showgrid=True, gridcolor="#18181f", zeroline=False, title="% Relative Weight"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_f, use_container_width=True)

with fi_col2:
    render_html("""
    <div class="quant-card" style="height:100%;">
        <div style="font-weight:700; color:#ffffff; font-size:1.02rem; margin-bottom:4px; font-family:'JetBrains Mono',monospace;">
            // FACTOR SPECIFICATIONS & GLOSSARY
        </div>
        <div style="color:#71717a; font-size:0.8rem; margin-bottom:12px;">
            How these core metrics guide the machine's decision
        </div>
        <div style="font-size:0.8rem; color:#a1a1aa; display:flex; flex-direction:column; gap:10px;">
            <div>
                <strong style="color:#ffffff;">1. RSI (14):</strong> Measures overbought (>70) and oversold (<30) speed of directional price changes.
            </div>
            <div>
                <strong style="color:#ffffff;">2. MA5 / MA20 Ratio:</strong> Gauges short-term trend extension over the medium-term 20-day baseline.
            </div>
            <div>
                <strong style="color:#ffffff;">3. Return 1D & 5D:</strong> Recent directional momentum velocity that acts as immediate trend persistence.
            </div>
            <div>
                <strong style="color:#ffffff;">4. Volume Ratio:</strong> Ratio of today's volume against the 5-day average to detect institutional accumulation.
            </div>
        </div>
    </div>
    """)

# ─────────────────────────────────────────────────────────────────────────────
# 7. INSTITUTIONAL EXECUTION BOUNDARIES & DISCLAIMER FOOTER
# ─────────────────────────────────────────────────────────────────────────────
render_html(f"""
<div class="quant-card" style="margin-top:20px; background:linear-gradient(180deg, rgba(18,18,24,0.6) 0%, rgba(8,8,12,0.9) 100%);">
    <div style="font-size:0.74rem; color:#71717a; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:10px; font-family:'JetBrains Mono',monospace;">
        // TACTICAL EXECUTION BOUNDARIES ({selected_ticker})
    </div>
    <div class="boundaries-adaptive-grid">
        <div>
            <div style="font-size:0.72rem; color:#71717a;">20-DAY RESISTANCE</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#ffffff; font-weight:700;">
                {fmt_price(tech_insights['resistance_20d'], selected_ticker)}
            </div>
        </div>
        <div>
            <div style="font-size:0.72rem; color:#71717a;">20-DAY SUPPORT</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#ffffff; font-weight:700;">
                {fmt_price(tech_insights['support_20d'], selected_ticker)}
            </div>
        </div>
        <div>
            <div style="font-size:0.72rem; color:#71717a;">DYNAMIC 1-ATR STOP BUFFER</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#ffffff; font-weight:700;">
                {fmt_price(last_close - tech_insights['atr'], selected_ticker)}
            </div>
        </div>
        <div>
            <div style="font-size:0.72rem; color:#71717a;">BOLLINGER VOLATILITY RANGE</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:#a1a1aa; font-weight:600;">
                {fmt_price(tech_insights['bb_lower'], selected_ticker)} – {fmt_price(tech_insights['bb_upper'], selected_ticker)}
            </div>
        </div>
    </div>
</div>
""")

render_html("""
<div style='text-align:center; color:#52525b; font-size:0.74rem; margin-top:36px; padding:24px 0; border-top:1px solid rgba(255,255,255,0.06); line-height:1.6; font-family:"Plus Jakarta Sans",sans-serif;'>
    <strong>INSTITUTIONAL RESEARCH NOTICE:</strong> StockSense AI is engineered strictly for quantitative exploration and mathematical backtesting. 
    It does not constitute financial, investment, or brokerage advice. Past model backtest accuracy is not indicative of future market returns.<br>
    Powered by <strong>Scikit-Learn · Plotly · yfinance · Streamlit</strong> | Obsidian Monochromatic Engine
</div>
""")
