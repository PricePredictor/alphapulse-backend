from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import yfinance as yf
import random

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        prediction = price * (1 + random.uniform(-0.02, 0.03))  # dummy logic
        return {
            "symbol": symbol.upper(),
            "current_price": round(price, 2),
            "predicted_price": round(prediction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(symbol: str, days: int = 5):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=f"{days}d")
        result = [
            {"date": str(idx.date()), "close": round(row["Close"], 2)}
            for idx, row in hist.iterrows()
        ]
        return {"symbol": symbol.upper(), "history": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest")
def backtest(symbol: str, days: int = 5):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=f"{days}d")
        predictions = []
        actuals = []

        for price in hist["Close"]:
            prediction = price * (1 + random.uniform(-0.02, 0.03))
            predictions.append(round(prediction, 2))
            actuals.append(round(price, 2))

        mae = round(sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions), 2)

        return {
            "symbol": symbol.upper(),
            "actual": actuals,
            "predicted": predictions,
            "mean_absolute_error": mae
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accuracy")
def model_accuracy():
    return {
        "description": "Dummy accuracy metric for placeholder",
        "mean_absolute_error": 2.85,
        "last_updated": "2025-05-14"
    }

@app.get("/top-movers")
def top_movers():
    return {
        "gainers": ["TSLA", "NVDA", "MSFT"],
        "losers": ["AAPL", "AMZN", "META"]
    }

@app.get("/health")
def health_check():
    return {"status": "OK", "uptime": "active"}
