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
from ta.trend import sma_indicator
from ta.momentum import rsi

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
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# -------------------------
# Health & root endpoints
# -------------------------
@app.get("/")
def root():
    return {"message": "AlphaPulse FastAPI backend is live."}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# XGBoost Prediction
# -------------------------
@app.get("/predict-xgb")
def predict_xgb(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="3mo", interval="1d")
        df['SMA_10'] = sma_indicator(df['Close'].squeeze(), window=10)
        df['SMA_50'] = sma_indicator(df['Close'].squeeze(), window=50)
        df['RSI'] = rsi(df['Close'].squeeze(), window=14)
        df.dropna(inplace=True)

        features = df[['SMA_10', 'SMA_50', 'RSI']].values[-1].reshape(1, -1)
        prediction = xgb_model.predict(features)[0]

        return {
            "ticker": ticker,
            "model": "XGBoost",
            "prediction": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# LSTM Prediction
# -------------------------
@app.get("/predict-lstm")
def predict_lstm(ticker: str = "AAPL", sequence_length: int = 50):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        close_prices = df[['Close']].values

        scaled = lstm_scaler.transform(close_prices)

        if len(scaled) < sequence_length:
            raise HTTPException(status_code=400, detail="Not enough data for LSTM prediction.")

        X = scaled[-sequence_length:].reshape(1, sequence_length, 1)
        predicted_scaled = lstm_model.predict(X)
        predicted_price = lstm_scaler.inverse_transform(predicted_scaled)[0][0]

        return {
            "ticker": ticker,
            "model": "LSTM",
            "prediction": float(predicted_price)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# PREDICT ALL Prediction
# -------------------------

@app.get("/predict-all")
def predict_all(ticker: str = "AAPL", sequence_length: int = 50):
    try:
        # --- XGBoost prediction ---
        df_xgb = yf.download(ticker, period="3mo", interval="1d")
        df_xgb['SMA_10'] = sma_indicator(df_xgb['Close'].squeeze(), window=10)
        df_xgb['SMA_50'] = sma_indicator(df_xgb['Close'].squeeze(), window=50)
        df_xgb['RSI'] = rsi(df_xgb['Close'].squeeze(), window=14)
        df_xgb.dropna(inplace=True)

        xgb_features = df_xgb[['SMA_10', 'SMA_50', 'RSI']].values[-1].reshape(1, -1)
        xgb_prediction = xgb_model.predict(xgb_features)[0]

        # --- LSTM prediction ---
        df_lstm = yf.download(ticker, period="6mo", interval="1d")
        close_prices = df_lstm[['Close']].values
        scaled = lstm_scaler.transform(close_prices)

        if len(scaled) < sequence_length:
            raise HTTPException(status_code=400, detail="Not enough data for LSTM prediction.")

        X_lstm = scaled[-sequence_length:].reshape(1, sequence_length, 1)
        predicted_scaled = lstm_model.predict(X_lstm)
        lstm_prediction = lstm_scaler.inverse_transform(predicted_scaled)[0][0]

        return {
            "ticker": ticker,
            "predictions": {
                "XGBoost": float(xgb_prediction),
                "LSTM": float(lstm_prediction)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# History
# -------------------------
@app.get("/history")
def get_history(ticker: str, period: str = "1mo", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data returned from Yahoo Finance.")

        if 'Date' not in df.columns:
            df.reset_index(inplace=True)

        if 'Date' not in df.columns:
            raise HTTPException(status_code=500, detail="Date column missing in returned data.")

        df['Date'] = df['Date'].astype(str)  # ensure JSON serializable
        history = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient="records")

        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# BackTest
# -------------------------

@app.get("/backtest")
def backtest(ticker: str = "AAPL", model: str = "xgb", start: str = "2023-01-01", end: str = "2023-03-01"):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d")
        df.dropna(inplace=True)

        if model.lower() == "xgb":
            df['SMA_10'] = sma_indicator(df['Close'].squeeze(), window=10)
            df['SMA_50'] = sma_indicator(df['Close'].squeeze(), window=50)
            df['RSI'] = rsi(df['Close'].squeeze(), window=14)
            df.dropna(inplace=True)

            predictions = []
            actuals = []

            for i in range(len(df)):
                if i < 50:
                    continue
                row = df.iloc[i]
                features = df[['SMA_10', 'SMA_50', 'RSI']].iloc[i].values.reshape(1, -1)
                pred = xgb_model.predict(features)[0]
                predictions.append(pred)
                actuals.append(row['Close'])

        elif model.lower() == "lstm":
            close_prices = df[['Close']].values
            scaled = lstm_scaler.transform(close_prices)

            sequence_length = 50
            predictions = []
            actuals = []

            for i in range(sequence_length, len(scaled)):
                X = scaled[i-sequence_length:i].reshape(1, sequence_length, 1)
                pred_scaled = lstm_model.predict(X)
                pred = lstm_scaler.inverse_transform(pred_scaled)[0][0]
                predictions.append(pred)
                actuals.append(close_prices[i][0])

        else:
            raise HTTPException(status_code=400, detail="Model must be 'xgb' or 'lstm'")

        mse = mean_squared_error(actuals, predictions)
        return {
            "ticker": ticker,
            "model": model.upper(),
            "start": start,
            "end": end,
            "mse": round(mse, 4),
            "n_predictions": len(predictions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


