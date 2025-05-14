
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
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
        prediction = price * (1 + random.uniform(-0.02, 0.03))  # dummy prediction logic
        return {
            "symbol": symbol.upper(),
            "current_price": round(price, 2),
            "predicted_price": round(prediction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
