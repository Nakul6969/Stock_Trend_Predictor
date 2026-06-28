# 📈 StockSense ML Predictor

StockSense is a clean, modern, and high-performance machine learning dashboard designed to predict next-day stock price directions. Using live financial data, it engineers core technical indicators and trains three distinct machine learning models—selecting the most accurate one to generate buy, sell, or hold signals.

Built with a premium, high-contrast dark black & white UI, the application features custom-rendered Matplotlib candlestick charts, volume dynamics, feature importance evaluation, and model comparison.

---

## ✨ Key Features

* 🚀 **Real-Time Data**: Dynamic stock data acquisition from the Yahoo Finance API (`yfinance`) with support for custom ticker symbols.
* 🤖 **Multi-Model Pipeline**: Parallel training of three classifiers:
  1. **Random Forest Classifier**
  2. **Gradient Boosting Classifier**
  3. **Logistic Regression** (with automated Standardization scaling)
* 🎯 **Dynamic Model Selection**: Evaluates all models on a time-series test split and automatically uses the model with the **highest accuracy** to issue predictions.
* 📊 **Professional Candlestick Charting**: Custom Matplotlib charting displaying red/green wicks and bodies integrated cleanly into the dark theme.
* 📈 **Matched Volume Subplot**: Color-coded volume bars (green/red) indicating buy/sell pressure.
* 🔍 **Feature Importance**: Grayscale evaluation of the technical indicators that influence the decision-making process.
* ⚡ **Model Comparison Panel**: Side-by-side card grid comparing model accuracies, highlighting the active predictor.
* 🎨 **Premium Aesthetic UI**: Custom CSS formatting featuring a pure black canvas, clean card modules, and colored signal badges.

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Data Sources**: [yfinance](https://github.com/ranaroussi/yfinance)
* **Data Manipulation**: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
* **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (Ensemble methods, Linear models)
* **Visualization**: [Matplotlib](https://matplotlib.org/)

---

## 📂 Project Structure

```text
├── app.py          # Streamlit frontend layout, custom styling, and chart renderings
└── model.py        # Feature engineering pipeline, model training, and prediction signals
