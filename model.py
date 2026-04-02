import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer technical indicator features from OHLCV data."""
    df = df.copy()

    # ── Price-based features ──────────────────────────────────────────────────
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_3d"] = df["Close"].pct_change(3)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["MA_5"]  = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    df["MA5_ratio"]  = df["Close"] / df["MA_5"]
    df["MA10_ratio"] = df["Close"] / df["MA_10"]
    df["MA20_ratio"] = df["Close"] / df["MA_20"]
    df["MA5_MA20_cross"] = (df["MA_5"] > df["MA_20"]).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["Volatility_5d"]  = df["Return_1d"].rolling(5).std()
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["High_Low_range"] = (df["High"] - df["Low"]) / df["Close"]

    # ── RSI (14-period) ───────────────────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid  = df["Close"].rolling(20).mean()
    bb_std  = df["Close"].rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_pos"]   = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["Volume_MA5"]   = df["Volume"].rolling(5).mean()
    df["Volume_ratio"] = df["Volume"] / (df["Volume_MA5"] + 1e-9)

    # ── Target: 1 if next-day close is higher, else 0 ─────────────────────────
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


FEATURE_COLS = [
    "Return_1d", "Return_3d", "Return_5d", "Return_10d",
    "MA5_ratio", "MA10_ratio", "MA20_ratio", "MA5_MA20_cross",
    "Volatility_5d", "Volatility_10d", "High_Low_range",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "BB_pos", "Volume_ratio",
]


def train_model(df: pd.DataFrame):
    """
    Train an ensemble of RF + GBT on historical data.
    Returns (model, scaler, accuracy, feature_importances).
    """
    df = add_features(df).dropna()

    X = df[FEATURE_COLS].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    gbt = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)

    # Soft-vote ensemble
    rf_prob  = rf.predict_proba(X_test)[:, 1]
    gbt_prob = gbt.predict_proba(X_test)[:, 1]
    ensemble_prob = (rf_prob + gbt_prob) / 2
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, ensemble_pred)

    # Use RF feature importances
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))

    # Return a wrapper that gives ensemble probabilities
    class EnsembleModel:
        def __init__(self, rf, gbt, scaler):
            self.rf     = rf
            self.gbt    = gbt
            self.scaler = scaler

        def predict_proba(self, X):
            Xs = self.scaler.transform(X)
            p1 = self.rf.predict_proba(Xs)[:, 1]
            p2 = self.gbt.predict_proba(Xs)[:, 1]
            prob_up = (p1 + p2) / 2
            return np.column_stack([1 - prob_up, prob_up])

    model = EnsembleModel(rf, gbt, scaler)

    return model, scaler, acc, importances


def predict_signal(model, df: pd.DataFrame):
    """
    Given a trained model and recent OHLCV df, return:
      - signal   : "BUY" | "SELL" | "HOLD"
      - prob_up  : float  (probability price goes up tomorrow)
      - next_price_est : float (simple regression estimate)
    """
    df = add_features(df).dropna()
    if df.empty:
        return "HOLD", 0.5, None

    last_row = df[FEATURE_COLS].values[-1].reshape(1, -1)
    prob     = model.predict_proba(last_row)[0]
    prob_up  = float(prob[1])

    # Simple next-price estimate using recent momentum
    last_close   = float(df["Close"].iloc[-1])
    avg_return   = float(df["Return_1d"].tail(5).mean())
    next_price   = last_close * (1 + avg_return)

    if prob_up >= 0.60:
        signal = "BUY"
    elif prob_up <= 0.40:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, prob_up, next_price
