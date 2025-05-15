from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import yfinance as yf
import random
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

accuracy_tracker = {"total": 0, "correct": 0}

@app.get("/")
def read_root():
    return {"status": "AlphaPulse backend running"}

@app.get("/health-check")
def health():
    return {"status": "healthy"}

@app.get("/live-price")
def get_live_price(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found")
        price = hist["Close"].iloc[-1]
        return {"symbol": symbol.upper(), "price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
def predict_price(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found")
        price = hist["Close"].iloc[-1]
        prediction = price * (1 + random.uniform(-0.02, 0.03))
        accuracy_tracker["total"] += 1
        if random.choice([True, False]):
            accuracy_tracker["correct"] += 1
        return {
            "symbol": symbol.upper(),
            "current_price": round(price, 2),
            "predicted_price": round(prediction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-ml-advanced")
def predict_ml_advanced(symbol: str, days: int = 30):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=f"{days + 50}d")
        if df.empty or len(df) < days:
            raise HTTPException(status_code=400, detail="Insufficient data")

        df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd_diff(df["Close"])
        bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df.dropna(inplace=True)

        features = ["SMA20", "EMA20", "RSI", "MACD", "BB_upper", "BB_lower"]
        X = df[features].iloc[:-1]
        y = df["Close"].iloc[1:]

        if len(X) < 10:
            raise HTTPException(status_code=400, detail="Too few samples for model training")

        xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
        xgb.fit(X, y)
        predicted_price = xgb.predict(df[features].iloc[[-1]])[0]

        return {
            "symbol": symbol.upper(),
            "current_price": round(df['Close'].iloc[-1], 2),
            "predicted_price": round(predicted_price, 2),
            "model": "XGBoost",
            "features_used": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest-enhanced")
def backtest_enhanced(symbol: str, days: int = 30):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=f"{days + 1}d")

        if df.empty or len(df) <= 1:
            raise HTTPException(status_code=400, detail="Not enough historical data for backtesting.")

        df = df[["Close"]].copy()
        df["Predicted"] = df["Close"].shift(1) * (1 + random.uniform(-0.02, 0.03))
        df.dropna(inplace=True)
        df["Error"] = (df["Close"] - df["Predicted"]).abs()
        df["APE"] = df["Error"] / df["Close"] * 100

        backtest_results = df.reset_index()[["Date", "Close", "Predicted", "Error", "APE"]]
        avg_error = df["Error"].mean()
        mape = df["APE"].mean()

        return {
            "symbol": symbol.upper(),
            "days_evaluated": len(df),
            "average_error": round(avg_error, 2),
            "mape_percent": round(mape, 2),
            "backtest": backtest_results.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
