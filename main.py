from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import yfinance as yf
import random
from datetime import datetime, timedelta

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
        prediction = price * (1 + random.uniform(-0.02, 0.03))  # dummy prediction logic
        return {
            "symbol": symbol.upper(),
            "current_price": round(price, 2),
            "predicted_price": round(prediction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_price_history(symbol: str, days: int = 7):
    try:
        stock = yf.Ticker(symbol)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        hist = stock.history(start=start_date, end=end_date)
        history_data = [
            {"date": date.strftime("%Y-%m-%d"), "close": round(close, 2)}
            for date, close in zip(hist.index, hist["Close"])
        ]
        return {"symbol": symbol.upper(), "history": history_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
