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

# -------------------------
# /predict
# -------------------------
@app.get("/predict")
def predict_price(ticker: str, model_type: str = "xgb"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df.dropna(inplace=True)

        features = df[["SMA_10", "SMA_50", "RSI"]].copy()
        features.rename(columns={"SMA_10": "SMA10", "SMA_50": "SMA50"}, inplace=True)
        features.columns = features.columns.str.strip()
        last_row = features.iloc[-1:].values

        if model_type == "xgb":
            prediction = float(xgb_model.predict(last_row).reshape(-1)[0])
        elif model_type == "rf":
            prediction = float(rf_model.predict(last_row).reshape(-1)[0])
        elif model_type == "lgb":
            prediction = float(lgb_model.predict(last_row).reshape(-1)[0])
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


# -------------------------
# /backtest-multi
# -------------------------
@app.get("/backtest-multi")
def backtest_multi(ticker: str = "AAPL", start: str = "2023-01-01", end: str = "2023-03-01"):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d")
        df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df.dropna(inplace=True)

        features = df[["SMA_10", "SMA_50", "RSI"]].copy()
        features.rename(columns={"SMA_10": "SMA10", "SMA_50": "SMA50"}, inplace=True)
        features.columns = features.columns.str.strip()
        y_true = df["Close"].values.reshape(-1)

        results = {}

        results["XGBoost"] = {
            "mse": round(mean_squared_error(y_true, xgb_model.predict(features).reshape(-1)), 4),
            "n_predictions": len(y_true)
        }

        results["RandomForest"] = {
            "mse": round(mean_squared_error(y_true, rf_model.predict(features).reshape(-1)), 4),
            "n_predictions": len(y_true)
        }

        results["LightGBM"] = {
            "mse": round(mean_squared_error(y_true, lgb_model.predict(features).reshape(-1)), 4),
            "n_predictions": len(y_true)
        }

        # LSTM
        scaled = lstm_scaler.transform(df["Close"].values.reshape(-1, 1))
        preds_lstm, actual_lstm = [], []
        for i in range(50, len(scaled)):
            seq = scaled[i - 50:i].reshape(1, 50, 1)
            pred = lstm_scaler.inverse_transform(lstm_model.predict(seq))[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df["Close"].values[i])
        results["LSTM"] = {
            "mse": round(mean_squared_error(np.array(actual_lstm), np.array(preds_lstm)), 4),
            "n_predictions": len(preds_lstm)
        }

        return {
            "ticker": ticker.upper(),
            "start": start,
            "end": end,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# /predict-ensemble
# -------------------------
@app.get("/predict-ensemble")
def predict_ensemble(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df.dropna(inplace=True)

        features = df[["SMA_10", "SMA_50", "RSI"]].copy()
        features.rename(columns={"SMA_10": "SMA10", "SMA_50": "SMA50"}, inplace=True)
        features.columns = features.columns.str.strip()
        last_row = features.iloc[-1:].values
        y_true = df["Close"].values.reshape(-1)

        predictions = {}
        mse_scores = {}

        predictions["XGBoost"] = float(xgb_model.predict(last_row).reshape(-1)[0])
        mse_scores["XGBoost"] = mean_squared_error(y_true, xgb_model.predict(features).reshape(-1))

        predictions["RandomForest"] = float(rf_model.predict(last_row).reshape(-1)[0])
        mse_scores["RandomForest"] = mean_squared_error(y_true, rf_model.predict(features).reshape(-1))

        predictions["LightGBM"] = float(lgb_model.predict(last_row).reshape(-1)[0])
        mse_scores["LightGBM"] = mean_squared_error(y_true, lgb_model.predict(features).reshape(-1))

        scaled = lstm_scaler.transform(df["Close"].values.reshape(-1, 1))
        preds_lstm, actual_lstm = [], []
        for i in range(50, len(scaled)):
            seq = scaled[i - 50:i].reshape(1, 50, 1)
            pred = lstm_scaler.inverse_transform(lstm_model.predict(seq))[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df["Close"].values[i])
        mse_scores["LSTM"] = mean_squared_error(np.array(actual_lstm), np.array(preds_lstm))

        last_seq = scaled[-50:].reshape(1, 50, 1)
        predictions["LSTM"] = float(lstm_scaler.inverse_transform(lstm_model.predict(last_seq))[0][0])

        ensemble_avg = round(np.mean(list(predictions.values())), 2)
        inv_mses = {k: 1 / v for k, v in mse_scores.items()}
        total = sum(inv_mses.values())
        weights = {k: inv_mses[k] / total for k in inv_mses}
        ensemble_weighted = round(sum(predictions[k] * weights[k] for k in predictions), 2)

        return {
            "ticker": ticker.upper(),
            "predictions": {k: round(v, 2) for k, v in predictions.items()},
            "weights": {k: round(weights[k], 4) for k in weights},
            "mse_scores": {k: round(mse_scores[k], 4) for k in mse_scores},
            "ensemble_avg": ensemble_avg,
            "ensemble_weighted": ensemble_weighted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# /accuracy-multi
# -------------------------
@app.get("/accuracy-multi")
def accuracy_multi(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df.dropna(inplace=True)

        features = df[["SMA_10", "SMA_50", "RSI"]].copy()
        features.rename(columns={"SMA_10": "SMA10", "SMA_50": "SMA50"}, inplace=True)
        features.columns = features.columns.str.strip()
        y_true = df["Close"].values.reshape(-1)

        results = {
            "XGBoost": round(mean_squared_error(y_true, xgb_model.predict(features).reshape(-1)), 4),
            "RandomForest": round(mean_squared_error(y_true, rf_model.predict(features).reshape(-1)), 4),
            "LightGBM": round(mean_squared_error(y_true, lgb_model.predict(features).reshape(-1)), 4)
        }

        # LSTM
        scaled = lstm_scaler.transform(df["Close"].values.reshape(-1, 1))
        preds_lstm, actual_lstm = [], []
        for i in range(50, len(scaled)):
            seq = scaled[i - 50:i].reshape(1, 50, 1)
            pred = lstm_scaler.inverse_transform(lstm_model.predict(seq))[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df["Close"].values[i])
        results["LSTM"] = round(mean_squared_error(np.array(actual_lstm), np.array(preds_lstm)), 4)

        return {
            "ticker": ticker.upper(),
            "accuracy": results,
            "n_samples": len(y_true)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
