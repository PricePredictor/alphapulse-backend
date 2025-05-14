from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
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

@app.get("/")
def read_root():
    return {"status": "AlphaPulse backend running with real ML model"}

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
        hist = stock.history(period="60d")
        if hist.empty or len(hist) < 30:
            raise ValueError("Not enough historical data to make a prediction.")

        hist = hist[["Close"]].dropna()
        hist["Day"] = np.arange(len(hist))

        # Linear regression on 'Day' vs 'Close'
        X = hist[["Day"]]
        y = hist["Close"]
        model = LinearRegression()
        model.fit(X, y)

        next_day = [[X.values[-1][0] + 1]]
        predicted_price = model.predict(next_day)[0]
        current_price = y.iloc[-1]

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "model": "Linear Regression on 60-day trend"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))