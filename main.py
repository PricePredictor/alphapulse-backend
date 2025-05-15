from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

accuracy_logs = {
    "Linear Regression": [],
    "Random Forest": [],
    "SVR": [],
    "XGBoost": []
}

def fetch_features(symbol: str, days: int):
    stock = yf.Ticker(symbol)
    df = stock.history(period=f"{days + 1}d")
    if df.empty or len(df) <= days:
        raise ValueError("Insufficient data")

    df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df.dropna(inplace=True)

    feature_cols = ["SMA20", "EMA20", "RSI", "MACD", "BB_upper", "BB_lower"]
    return df[feature_cols + ["Close"]], feature_cols

def predict_model(df, features, model, model_name):
    X = df[features].iloc[:-1]
    y = df["Close"].iloc[1:]
    model.fit(X, y)
    last_row = df[features].iloc[[-1]]
    prediction = model.predict(last_row)[0]
    actual = df["Close"].iloc[-1]
    error = abs(prediction - actual)
    accuracy_logs[model_name].append({
        "date": str(df.index[-1].date()),
        "actual": round(actual, 2),
        "predicted": round(prediction, 2),
        "error": round(error, 2)
    })
    return round(actual, 2), round(prediction, 2)

@app.get("/predict-lr")
def predict_lr(symbol: str, days: int = 60):
    try:
        df, features = fetch_features(symbol, days)
        model = LinearRegression()
        current, pred = predict_model(df, features, model, "Linear Regression")
        return {"symbol": symbol, "model": "Linear Regression", "current_price": current, "predicted_price": pred, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-rf")
def predict_rf(symbol: str, days: int = 60):
    try:
        df, features = fetch_features(symbol, days)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        current, pred = predict_model(df, features, model, "Random Forest")
        return {"symbol": symbol, "model": "Random Forest", "current_price": current, "predicted_price": pred, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-svr")
def predict_svr(symbol: str, days: int = 60):
    try:
        df, features = fetch_features(symbol, days)
        model = SVR(kernel='rbf')
        current, pred = predict_model(df, features, model, "SVR")
        return {"symbol": symbol, "model": "SVR", "current_price": current, "predicted_price": pred, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-xgb")
def predict_xgb(symbol: str, days: int = 60):
    try:
        df, features = fetch_features(symbol, days)
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        current, pred = predict_model(df, features, model, "XGBoost")
        return {"symbol": symbol, "model": "XGBoost", "current_price": current, "predicted_price": pred, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accuracy-logs")
def get_accuracy_logs():
    return accuracy_logs
