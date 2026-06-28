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
