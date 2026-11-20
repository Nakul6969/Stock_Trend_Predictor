import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Core indicators to keep features minimal and easy to understand
FEATURE_COLS = [
    "Return_1d",
    "Return_5d",
    "MA5_ratio",
    "MA20_ratio",
    "MA5_MA20_cross",
    "RSI",
    "Volume_ratio"
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer a simplified set of technical indicator features."""
    df = df.copy()

    # Price returns
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)

    # Moving averages
    df["MA_5"]  = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    df["MA5_ratio"]  = df["Close"] / df["MA_5"]
    df["MA20_ratio"] = df["Close"] / df["MA_20"]
    df["MA5_MA20_cross"] = (df["MA_5"] > df["MA_20"]).astype(int)

    # Relative Strength Index (RSI 14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volume Ratio
    df["Volume_MA5"]   = df["Volume"].rolling(5).mean()
    df["Volume_ratio"] = df["Volume"] / (df["Volume_MA5"] + 1e-9)

    # Target: 1 if next-day Close is higher, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df

def train_model(df: pd.DataFrame):
    """
    Train Random Forest, Gradient Boosting, and Logistic Regression models.
    Returns (models, accuracies, scaler, importances).
    """
    df = add_features(df).dropna()

    X = df[FEATURE_COLS].values
    y = df["Target"].values

    # Sequential split for time series backtest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))

    # 2. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test_scaled))

    # 3. Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))

    models = {
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Logistic Regression": lr
    }

    accuracies = {
        "Random Forest": rf_acc,
        "Gradient Boosting": gb_acc,
        "Logistic Regression": lr_acc
    }

    # Extract RF feature importances
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))

    return models, accuracies, scaler, importances

def predict_signal(model, scaler, df: pd.DataFrame):
    """
    Given a trained model, scaler, and recent stock data, return:
      - signal   : "BUY" | "SELL" | "HOLD"
      - prob_up  : float (probability of price going up tomorrow)
      - next_price_est : float (simple momentum-based estimate)
    """
    df = add_features(df).dropna()
    if df.empty:
        return "HOLD", 0.5, None

    last_row = df[FEATURE_COLS].values[-1].reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)
    
    prob     = model.predict_proba(last_row_scaled)[0]
    prob_up  = float(prob[1])

    # Momentum-based next price estimation
    last_close = float(df["Close"].iloc[-1])
    avg_return = float(df["Return_1d"].tail(5).mean())
    next_price = last_close * (1 + avg_return)

    if prob_up >= 0.60:
        signal = "BUY"
    elif prob_up <= 0.40:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, prob_up, next_price

def predict_all_models(models, scaler, df: pd.DataFrame):
    """
    Generate predictions, probabilities, and consensus across all trained models.
    Returns:
      - model_predictions: dict mapping model name to {signal, prob_up, prob_down}
      - consensus_signal: "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
      - avg_prob_up: float (average confidence across models)
      - next_price: float (momentum-based estimate)
    """
    df_feat = add_features(df).dropna()
    if df_feat.empty:
        return {}, "HOLD", 0.5, 0.0

    last_row = df_feat[FEATURE_COLS].values[-1].reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)

    model_predictions = {}
    probs = []
    signals = []

    for name, model in models.items():
        prob = model.predict_proba(last_row_scaled)[0]
        prob_up = float(prob[1])
        probs.append(prob_up)

        if prob_up >= 0.60:
            sig = "BUY"
        elif prob_up <= 0.40:
            sig = "SELL"
        else:
            sig = "HOLD"

        signals.append(sig)
        model_predictions[name] = {
            "signal": sig,
            "prob_up": prob_up,
            "prob_down": float(prob[0])
        }

    avg_prob_up = float(np.mean(probs))
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count == 3:
        consensus_signal = "STRONG BUY"
    elif buy_count == 2:
        consensus_signal = "BUY"
    elif sell_count == 3:
        consensus_signal = "STRONG SELL"
    elif sell_count == 2:
        consensus_signal = "SELL"
    else:
        consensus_signal = "HOLD"

    last_close = float(df_feat["Close"].iloc[-1])
    avg_return = float(df_feat["Return_1d"].tail(5).mean())
    next_price = last_close * (1 + avg_return)

    return model_predictions, consensus_signal, avg_prob_up, next_price

def compute_technical_insights(df: pd.DataFrame) -> dict:
    """
    Extract comprehensive technical indicator values and health diagnostics.
    """
    df_feat = add_features(df).dropna()
    if df_feat.empty:
        return {}

    last = df_feat.iloc[-1]
    prev = df_feat.iloc[-2] if len(df_feat) > 1 else last

    last_close = float(last["Close"])
    prev_close = float(prev["Close"])
    day_chg = ((last_close - prev_close) / prev_close) * 100

    # 5-day return
    ret_5d = float(last["Return_5d"]) * 100 if "Return_5d" in last else 0.0

    # RSI (14)
    rsi = float(last["RSI"]) if "RSI" in last else 50.0
    if rsi >= 70:
        rsi_status = "Overbought (High Risk)"
        rsi_tag = "BEARISH_DIVERGENCE"
    elif rsi <= 30:
        rsi_status = "Oversold (Rebound Potential)"
        rsi_tag = "BULLISH_OPPORTUNITY"
    elif rsi >= 55:
        rsi_status = "Healthy Bullish Momentum"
        rsi_tag = "BULLISH"
    elif rsi <= 45:
        rsi_status = "Bearish Weakness"
        rsi_tag = "BEARISH"
    else:
        rsi_status = "Neutral Equilibrium"
        rsi_tag = "NEUTRAL"

    # Moving Average 5 & 20
    ma5 = float(last["MA_5"]) if "MA_5" in last else last_close
    ma20 = float(last["MA_20"]) if "MA_20" in last else last_close
    ma_spread = ((ma5 - ma20) / ma20) * 100
    price_to_ma20 = ((last_close - ma20) / ma20) * 100
    ma_posture = "Golden Posture (MA5 > MA20)" if ma5 >= ma20 else "Death Posture (MA5 < MA20)"

    # Volume dynamics
    vol_ratio = float(last["Volume_ratio"]) if "Volume_ratio" in last else 1.0
    if vol_ratio >= 1.5:
        vol_status = "High Surge (Institutional Activity)"
    elif vol_ratio >= 1.1:
        vol_status = "Above Average (Accumulation)"
    elif vol_ratio <= 0.7:
        vol_status = "Low Volume (Weak Participation)"
    else:
        vol_status = "Average Baseline"

    # Bollinger Bands (20-day, 2 std)
    rolling_std = float(df_feat["Close"].tail(20).std()) if len(df_feat) >= 20 else 0.0
    bb_upper = ma20 + (2 * rolling_std)
    bb_lower = ma20 - (2 * rolling_std)

    # ATR (14)
    high_low = df_feat["High"] - df_feat["Low"]
    high_prev_close = (df_feat["High"] - df_feat["Close"].shift(1)).abs()
    low_prev_close = (df_feat["Low"] - df_feat["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean()) if len(tr) >= 14 else rolling_std

    # High / Low ranges
    high_52w = float(df["High"].max())
    low_52w = float(df["Low"].min())
    pct_from_high = ((last_close - high_52w) / high_52w) * 100
    pct_from_low = ((last_close - low_52w) / low_52w) * 100

    # 20-day Support / Resistance
    recent_20 = df_feat.tail(20)
    support_20d = float(recent_20["Low"].min())
    resistance_20d = float(recent_20["High"].max())

    return {
        "last_close": last_close,
        "prev_close": prev_close,
        "day_chg": day_chg,
        "ret_5d": ret_5d,
        "rsi": rsi,
        "rsi_status": rsi_status,
        "rsi_tag": rsi_tag,
        "ma5": ma5,
        "ma20": ma20,
        "ma_spread": ma_spread,
        "price_to_ma20": price_to_ma20,
        "ma_posture": ma_posture,
        "vol_ratio": vol_ratio,
        "vol_status": vol_status,
        "bb_upper": float(bb_upper),
        "bb_lower": float(bb_lower),
        "atr": atr,
        "support_20d": support_20d,
        "resistance_20d": resistance_20d,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low,
    }