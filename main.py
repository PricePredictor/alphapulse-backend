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

# Enable CORS
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

# ====================================
# 1. Predict Single Model
# ====================================
@app.get("/predict")
def predict_price(ticker: str, model_type: str = "xgb"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        close = df['Close']
        df['SMA10'] = SMAIndicator(close, window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(close, window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close, window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        last_row = features.iloc[-1:].values

        if model_type == "xgb":
            prediction = xgb_model.predict(last_row)[0]
        elif model_type == "rf":
            prediction = rf_model.predict(last_row)[0]
        elif model_type == "lgb":
            prediction = lgb_model.predict(last_row)[0]
        elif model_type == "lstm":
            scaled = lstm_scaler.transform(close.values.reshape(-1, 1))
            last_seq = scaled[-50:].reshape(1, 50, 1)
            prediction = lstm_scaler.inverse_transform(lstm_model.predict(last_seq))[0][0]
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        return {
            "ticker": ticker.upper(),
            "model": model_type.upper(),
            "predicted_price": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================
# 2. Backtest All Models
# ====================================
@app.get("/backtest-multi")
def backtest_multi(ticker: str = "AAPL", start: str = "2023-01-01", end: str = "2023-03-01"):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d")
        df['SMA10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        y_true = df['Close'].values.ravel()
        results = {}

        results["XGBoost"] = {
            "mse": round(mean_squared_error(y_true, xgb_model.predict(features)), 4)
        }

        results["RandomForest"] = {
            "mse": round(mean_squared_error(y_true, rf_model.predict(features)), 4)
        }

        results["LightGBM"] = {
            "mse": round(mean_squared_error(y_true, lgb_model.predict(features)), 4)
        }

        # LSTM
        scaled = lstm_scaler.transform(df['Close'].values.reshape(-1, 1))
        seq_len = 50
        preds_lstm, actuals = [], []

        for i in range(seq_len, len(scaled)):
            X_seq = scaled[i - seq_len:i].reshape(1, seq_len, 1)
            pred = lstm_model.predict(X_seq, verbose=0)
            preds_lstm.append(lstm_scaler.inverse_transform(pred)[0][0])
            actuals.append(df['Close'].values[i])

        results["LSTM"] = {
            "mse": round(mean_squared_error(actuals, preds_lstm), 4)
        }

        return {
            "ticker": ticker.upper(),
            "start": start,
            "end": end,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================
# 3. Predict Ensemble (Simple + Weighted)
# ====================================
@app.get("/predict-ensemble")
def predict_ensemble(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        close = df['Close']
        df['SMA10'] = SMAIndicator(close, window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(close, window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close, window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        last_row = features.iloc[-1:].values
        y_true = df['Close'].values.ravel()

        predictions = {}
        mse_scores = {}

        # XGBoost
        pred_xgb_all = xgb_model.predict(features)
        predictions["XGBoost"] = float(xgb_model.predict(last_row)[0])
        mse_scores["XGBoost"] = mean_squared_error(y_true[-len(pred_xgb_all):], pred_xgb_all)

        # Random Forest
        pred_rf_all = rf_model.predict(features)
        predictions["RandomForest"] = float(rf_model.predict(last_row)[0])
        mse_scores["RandomForest"] = mean_squared_error(y_true[-len(pred_rf_all):], pred_rf_all)

        # LightGBM
        pred_lgb_all = lgb_model.predict(features)
        predictions["LightGBM"] = float(lgb_model.predict(last_row)[0])
        mse_scores["LightGBM"] = mean_squared_error(y_true[-len(pred_lgb_all):], pred_lgb_all)

        # LSTM
        scaled = lstm_scaler.transform(close.values.reshape(-1, 1))
        seq_len = 50
        preds_lstm = []
        actuals = []

        for i in range(seq_len, len(scaled)):
            seq = scaled[i - seq_len:i].reshape(1, seq_len, 1)
            pred = lstm_model.predict(seq, verbose=0)
            preds_lstm.append(lstm_scaler.inverse_transform(pred)[0][0])
            actuals.append(close.values[i])

        pred_lstm_final = lstm_model.predict(scaled[-seq_len:].reshape(1, seq_len, 1))
        predictions["LSTM"] = float(lstm_scaler.inverse_transform(pred_lstm_final)[0][0])
        mse_scores["LSTM"] = mean_squared_error(actuals, preds_lstm)

        # Ensemble averages
        ensemble_avg = np.mean(list(predictions.values()))
        weights = {k: 1 / v for k, v in mse_scores.items()}
        total_wt = sum(weights.values())
        weights = {k: w / total_wt for k, w in weights.items()}
        ensemble_weighted = sum(predictions[k] * weights[k] for k in predictions)

        return {
            "ticker": ticker.upper(),
            "predictions": {k: round(v, 2) for k, v in predictions.items()},
            "weights": {k: round(w, 4) for k, w in weights.items()},
            "mse_scores": {k: round(mse_scores[k], 4) for k in mse_scores},
            "ensemble_avg": round(ensemble_avg, 2),
            "ensemble_weighted": round(ensemble_weighted, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================
# 4. Accuracy Multi-Model
# ====================================
@app.get("/accuracy-multi")
def accuracy_multi(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df['SMA10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        y_true = df['Close'].values.ravel()
        results = {}

        results["XGBoost"] = round(mean_squared_error(y_true, xgb_model.predict(features)), 4)
        results["RandomForest"] = round(mean_squared_error(y_true, rf_model.predict(features)), 4)
        results["LightGBM"] = round(mean_squared_error(y_true, lgb_model.predict(features)), 4)

        scaled = lstm_scaler.transform(df['Close'].values.reshape(-1, 1))
        seq_len = 50
        preds_lstm = []
        actuals = []

        for i in range(seq_len, len(scaled)):
            X_seq = scaled[i-seq_len:i].reshape(1, seq_len, 1)
            pred = lstm_model.predict(X_seq, verbose=0)
            preds_lstm.append(lstm_scaler.inverse_transform(pred)[0][0])
            actuals.append(df['Close'].values[i])

        results["LSTM"] = round(mean_squared_error(actuals, preds_lstm), 4)

        return {
            "ticker": ticker.upper(),
            "accuracy": results,
            "n_samples": len(y_true)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
