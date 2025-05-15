# This file will be the upgraded `main.py` to support training, prediction, logging, and dashboard-ready APIs.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

def fetch_data(symbol: str, days: int = 60):
    df = yf.Ticker(symbol).history(period=f"{days}d")
    if df.empty or len(df) < 21:
        raise HTTPException(status_code=400, detail="Insufficient historical data")

    df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

@app.post("/train-model")
def train_model(symbol: str = "AAPL"):
    df = fetch_data(symbol, 180)
    features = ["SMA20", "EMA20", "RSI", "MACD", "BB_upper", "BB_lower"]
    df["target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    X = df[features]
    y = df["target"]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(objective='reg:squarederror')
    }
    scores = {}

    for name, model in models.items():
        model.fit(X, y)
        score = model.score(X, y)
        scores[name] = score
        joblib.dump(model, f"{model_dir}/{symbol}_{name}.joblib")

    return {"symbol": symbol, "scores": scores}

@app.get("/predict-model")
def predict_model(symbol: str = "AAPL", model_name: str = "XGBoost"):
    df = fetch_data(symbol)
    features = ["SMA20", "EMA20", "RSI", "MACD", "BB_upper", "BB_lower"]
    X = df[features]

    model_path = f"{model_dir}/{symbol}_{model_name}.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not trained. Please run /train-model")

    model = joblib.load(model_path)
    prediction = model.predict(X.iloc[[-1]])[0]
    return {
        "symbol": symbol,
        "model": model_name,
        "features": features,
        "current_price": round(df["Close"].iloc[-1], 2),
        "predicted_price": round(prediction, 2)
    }

@app.get("/")
def health():
    return {"status": "AlphaPulse backend live with ML models."}
