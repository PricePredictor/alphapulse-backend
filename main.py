# ======================
# main.py
# ======================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Initialize app
app = FastAPI()

# Enable CORS for all origins (relax in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    xgb_model: XGBRegressor = joblib.load("xgb_model.pkl")
    lstm_model = load_model("lstm_model.h5")
    lstm_scaler: MinMaxScaler = joblib.load("lstm_scaler.save")
    rf_model = joblib.load("random_forest.pkl")
    lgb_model = joblib.load("lightgbm.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# -------------------------
# Unified Predict Endpoint
# -------------------------
@app.get("/predict")
def predict_price(ticker: str, model_type: str = "xgb"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        close = df[['Close']].squeeze()
        df['SMA_10'] = SMAIndicator(close=close, window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        df.dropna(inplace=True)

        X = df[['SMA_10', 'SMA_50', 'RSI']].copy()
        X.columns = ['SMA10', 'SMA50', 'RSI']
        last_row = X.iloc[-1:].values

        if model_type == "xgb":
            prediction = xgb_model.predict(last_row)[0]
        elif model_type == "rf":
            prediction = rf_model.predict(last_row)[0]
        elif model_type == "lgb":
            prediction = lgb_model.predict(last_row)[0]
        elif model_type == "lstm":
            scaled = lstm_scaler.transform(df["Close"].values.reshape(-1, 1))
            last_sequence = scaled[-50:].reshape(1, 50, 1)
            prediction = lstm_scaler.inverse_transform(lstm_model.predict(last_sequence))[0][0]
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing model type")

        return {
            "ticker": ticker.upper(),
            "model": model_type.upper(),
            "predicted_price": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
