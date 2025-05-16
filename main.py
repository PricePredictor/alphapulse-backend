# ======================
# main.py
# ======================
# FastAPI backend using pre-trained models saved from train_models.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBRegressor
import ta

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Endpoint: Predict with XGBoost ----------
@app.get("/predict-xgb")
def predict_xgb(ticker: str = "AAPL"):
    try:
        model = joblib.load("xgb_model.pkl")
        df = yf.download(ticker, period="3mo", interval="1d")

        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        latest = df.dropna().iloc[-1]
        input_features = [[latest['SMA_10'], latest['SMA_50'], latest['RSI']]]
        prediction = model.predict(input_features)[0]
        return {"ticker": ticker, "predicted_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Endpoint: Predict with LSTM ----------
@app.get("/predict-lstm")
def predict_lstm(ticker: str = "AAPL", sequence_length: int = 50):
    try:
        model = tf.keras.models.load_model("lstm_model.h5")
        scaler = joblib.load("lstm_scaler.save")

        df = yf.download(ticker, period="3mo", interval="1d")
        close_prices = df[['Close']].values
        scaled = scaler.transform(close_prices)

        X_test = [scaled[-sequence_length:]]
        X_test = np.array(X_test).reshape(1, sequence_length, 1)

        predicted_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        return {"ticker": ticker, "predicted_price": round(predicted_price, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Endpoint: Backtesting Strategy ----------
@app.get("/backtest")
def backtest(ticker: str = "AAPL", short_window: int = 10, long_window: int = 50):
    df = yf.download(ticker, period="6mo", interval="1d")
    df['SMA_Short'] = ta.trend.sma_indicator(df['Close'], window=short_window)
    df['SMA_Long'] = ta.trend.sma_indicator(df['Close'], window=long_window)
    df.dropna(inplace=True)

    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
    df.loc[df['SMA_Short'] <= df['SMA_Long'], 'Signal'] = -1

    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']

    cumulative = (1 + df['Strategy_Return']).cumprod()
    total_return = cumulative.iloc[-1] - 1
    sharpe = (df['Strategy_Return'].mean() / df['Strategy_Return'].std()) * np.sqrt(252)

    return {
        "ticker": ticker,
        "total_return_percent": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "last_price": round(df['Close'].iloc[-1], 2),
        "start_date": df.index[0].strftime('%Y-%m-%d'),
        "end_date": df.index[-1].strftime('%Y-%m-%d')
    }
