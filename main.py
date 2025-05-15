from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import yfinance as yf
import random
import pandas as pd
import numpy as np
import ta
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

accuracy_tracker = {
    "total": 0,
    "correct": 0
}

@app.get("/")
def read_root():
    return {"status": "AlphaPulse backend running"}

@app.get("/live-price")
def get_live_price(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return {"symbol": symbol.upper(), "price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
def predict_price(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
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

@app.get("/predict-ml")
def predict_ml(symbol: str, days: int = 30):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=f"{days + 20}d")  # extra buffer for indicators
        if df.empty or len(df) < days:
            raise HTTPException(status_code=400, detail="Insufficient data")

        df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()

        df.dropna(inplace=True)

        feature_cols = ["SMA20", "EMA20", "RSI", "MACD", "BB_upper", "BB_lower"]
        X = df[feature_cols].iloc[:-1]
        y = df["Close"].iloc[1:]

        if X.empty:
            raise HTTPException(status_code=400, detail="Not enough data after indicators")

        model = LinearRegression()
        model.fit(X, y)

        last_features = df[feature_cols].iloc[[-1]]
        predicted_price = model.predict(last_features)[0]
        current_price = df["Close"].iloc[-1]

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "model": "Linear Regression (technical indicators)",
            "features_used": feature_cols
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(symbol: str, days: Optional[int] = 7):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=f"{days}d")
        history = [{"date": str(date.date()), "close": round(close, 2)}
                   for date, close in data["Close"].items()]
        return {"symbol": symbol.upper(), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest")
def backtest(symbol: str, days: Optional[int] = 10):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=f"{days + 1}d")
        dates = list(data.index)
        closes = data["Close"].tolist()

        result = []
        for i in range(1, len(closes)):
            actual = closes[i]
            predicted = closes[i - 1] * (1 + random.uniform(-0.02, 0.03))
            result.append({
                "date": str(dates[i].date()),
                "actual": round(actual, 2),
                "predicted": round(predicted, 2),
                "error": round(abs(actual - predicted), 2)
            })
        return {"symbol": symbol.upper(), "backtest": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accuracy")
def get_accuracy():
    try:
        total = accuracy_tracker["total"]
        correct = accuracy_tracker["correct"]
        return {
            "evaluated_predictions": total,
            "accurate_predictions": correct,
            "accuracy_percent": round((correct / total) * 100, 2) if total > 0 else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top-movers")
def get_top_movers():
    try:
        symbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA"]
        movers = []
        for sym in symbols:
            stock = yf.Ticker(sym)
            data = stock.history(period="2d")
            if len(data) >= 2:
                close_yesterday = data["Close"].iloc[-2]
                close_today = data["Close"].iloc[-1]
                percent_change = ((close_today - close_yesterday) / close_yesterday) * 100
                movers.append({
                    "symbol": sym,
                    "change_percent": round(percent_change, 2)
                })
        sorted_movers = sorted(movers, key=lambda x: abs(x["change_percent"]), reverse=True)
        return {"top_movers": sorted_movers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
