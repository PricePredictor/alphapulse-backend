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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    xgb_model: XGBRegressor = joblib.load("xgb_model.pkl")
    lstm_model = load_model("lstm_model.h5")
    lstm_scaler: MinMaxScaler = joblib.load("lstm_scaler.save")
    rf_model = joblib.load("random_forest.pkl")
    lgb_model = joblib.load("lightgbm.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# =================== /predict ===================
@app.get("/predict")
def predict_price(ticker: str, model_type: str = "xgb"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        close = df['Close']
        df['SMA10'] = SMAIndicator(close=close, window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(close=close, window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
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
            scaled = lstm_scaler.transform(df["Close"].values.reshape(-1, 1))
            last_sequence = scaled[-50:].reshape(1, 50, 1)
            prediction = lstm_scaler.inverse_transform(lstm_model.predict(last_sequence))[0][0]
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        return {
            "ticker": ticker.upper(),
            "model": model_type.upper(),
            "predicted_price": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================== /backtest-multi ===================
@app.get("/backtest-multi")
def backtest_multi(ticker: str = "AAPL", start: str = "2023-01-01", end: str = "2023-03-01"):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d")
        df['SMA10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        y_true = df['Close'].values.ravel()
        results = {}

        results["XGBoost"] = {
            "mse": round(mean_squared_error(y_true, xgb_model.predict(features)), 4),
            "n_predictions": len(features)
        }

        results["RandomForest"] = {
            "mse": round(mean_squared_error(y_true, rf_model.predict(features)), 4),
            "n_predictions": len(features)
        }

        results["LightGBM"] = {
            "mse": round(mean_squared_error(y_true, lgb_model.predict(features)), 4),
            "n_predictions": len(features)
        }

        # LSTM
        scaled_close = lstm_scaler.transform(df[['Close']].values)
        sequence_length = 50
        preds_lstm = []
        actual_lstm = []

        for i in range(sequence_length, len(scaled_close)):
            X_seq = scaled_close[i-sequence_length:i].reshape(1, sequence_length, 1)
            pred_scaled = lstm_model.predict(X_seq, verbose=0)
            pred = lstm_scaler.inverse_transform(pred_scaled)[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df['Close'].values[i])

        results["LSTM"] = {
            "mse": round(mean_squared_error(np.array(actual_lstm), np.array(preds_lstm)), 4),
            "n_predictions": len(preds_lstm)
        }

        return {
            "ticker": ticker,
            "start": start,
            "end": end,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================== /predict-ensemble ===================
@app.get("/predict-ensemble")
def predict_ensemble(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        close = df['Close']
        df['SMA10'] = SMAIndicator(close=close, window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(close=close, window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        last_row = features.iloc[-1:].values
        close_prices = df['Close'].values

        preds = {}
        mse_scores = {}

        preds['XGBoost'] = float(xgb_model.predict(last_row)[0])
        mse_scores['XGBoost'] = mean_squared_error(close_prices[-len(features):], xgb_model.predict(features))

        preds['RandomForest'] = float(rf_model.predict(last_row)[0])
        mse_scores['RandomForest'] = mean_squared_error(close_prices[-len(features):], rf_model.predict(features))

        preds['LightGBM'] = float(lgb_model.predict(last_row)[0])
        mse_scores['LightGBM'] = mean_squared_error(close_prices[-len(features):], lgb_model.predict(features))

        # LSTM
        scaled_close = lstm_scaler.transform(close_prices.reshape(-1, 1))
        X_lstm = []
        y_lstm = []
        sequence_length = 50
        for i in range(sequence_length, len(scaled_close)):
            X_lstm.append(scaled_close[i-sequence_length:i])
            y_lstm.append(close_prices[i])
        X_lstm = np.array(X_lstm).reshape(-1, sequence_length, 1)
        y_lstm = np.array(y_lstm)
        preds_lstm = lstm_model.predict(X_lstm, verbose=0)
        preds_lstm_inv = lstm_scaler.inverse_transform(preds_lstm).squeeze()
        preds['LSTM'] = float(lstm_scaler.inverse_transform(lstm_model.predict(scaled_close[-sequence_length:].reshape(1, 50, 1)))[0][0])
        mse_scores['LSTM'] = mean_squared_error(y_lstm, preds_lstm_inv)

        # Ensemble
        ensemble_avg = round(np.mean(list(preds.values())), 2)
        weights = {k: 1/v for k, v in mse_scores.items()}
        total = sum(weights.values())
        weights = {k: w/total for k, w in weights.items()}
        ensemble_weighted = round(sum(preds[k] * weights[k] for k in preds), 2)

        return {
            "ticker": ticker.upper(),
            "predictions": {k: round(v, 2) for k, v in preds.items()},
            "weights": {k: round(weights[k], 4) for k in weights},
            "mse_scores": {k: round(mse_scores[k], 4) for k in mse_scores},
            "ensemble_avg": ensemble_avg,
            "ensemble_weighted": ensemble_weighted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =================== /accuracy-multi ===================
@app.get("/accuracy-multi")
def accuracy_multi(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df['SMA10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA10', 'SMA50', 'RSI']]
        y_true = df['Close'].values.ravel()
        results = {}

        results["XGBoost"] = round(mean_squared_error(y_true, xgb_model.predict(features)), 4)
        results["RandomForest"] = round(mean_squared_error(y_true, rf_model.predict(features)), 4)
        results["LightGBM"] = round(mean_squared_error(y_true, lgb_model.predict(features)), 4)

        scaled_close = lstm_scaler.transform(df[['Close']].values)
        sequence_length = 50
        preds_lstm = []
        actual_lstm = []

        for i in range(sequence_length, len(scaled_close)):
            X_seq = scaled_close[i-sequence_length:i].reshape(1, sequence_length, 1)
            pred_scaled = lstm_model.predict(X_seq, verbose=0)
            pred = lstm_scaler.inverse_transform(pred_scaled)[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df['Close'].values[i])

        results["LSTM"] = round(mean_squared_error(np.array(actual_lstm), np.array(preds_lstm)), 4)

        return {
            "ticker": ticker.upper(),
            "accuracy": results,
            "n_samples": len(y_true)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
